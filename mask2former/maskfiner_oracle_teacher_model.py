# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random

import math
from einops import rearrange

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.criterion_downsampled import SetCriterionDownSample
from .modeling.criterion_mixed import SetCriterionMix
from .modeling.criterion_mixed_oracle import SetCriterionMixOracle
from .modeling.matcher import HungarianMatcher
from .modeling.matcher_downsampled import HungarianMatcherDownSample
from .modeling.matcher_mixed import HungarianMatcherMix
from .modeling.meta_arch.build import build_mask_predictor_indexed


@META_ARCH_REGISTRY.register()
class MaskFinerOracleTeacher(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        mask_predictors: nn.ModuleList,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        oracle_teacher_ratio: float,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.mask_predictors = mask_predictors
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.oracle_teacher_ratio = oracle_teacher_ratio

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        mask_predictors = nn.ModuleList()
        for i in range(cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES):
            mask_predictor = build_mask_predictor_indexed(cfg, i)
            mask_predictors.append(mask_predictor)

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FINER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FINER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FINER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FINER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FINER.MASK_WEIGHT
        upsampling_weight = cfg.MODEL.MASK_FINER.UPSAMPLING_WEIGHT

        # building criterion
        matcher = HungarianMatcherMix(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FINER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            if cfg.MODEL.MASK_FINER.MASK_DECODER_ALL_LEVELS:
                dec_layers = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES + sum(cfg.MODEL.MASK_FINER.DEC_LAYERS)
            else:
                dec_layers = cfg.MODEL.MASK_FINER.DEC_LAYERS[-1] + 1
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

            up_layers = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES - 1
            up_dict = {}
            for i in range(up_layers):
                up_dict.update({"loss_upsampling_{}".format(i): upsampling_weight})
            weight_dict.update(up_dict)
        else:
            up_dict = {"loss_upsampling": upsampling_weight}
            weight_dict.update(up_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterionMixOracle(
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FINER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FINER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FINER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "mask_predictors": mask_predictors,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FINER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FINER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FINER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FINER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FINER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FINER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FINER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FINER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FINER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FINER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "oracle_teacher_ratio": cfg.MODEL.MASK_FINER.ORACLE_TEACHER_RATIO
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = None
        features_pos = None
        upsampling_mask = None
        disagreement_masks_pred = []
        disagreement_masks_oracle = []
        outputs = {}
        outputs['pred_masks'] = None
        outputs['pred_logits'] = None
        outputs['aux_outputs'] = []
        outputs['upsampling_outputs'] = []

        sem_seg_gt = [x["sem_seg"].to(self.device) for x in batched_inputs]
        sem_seg_gt, target_pad = self.prepare_oracle_targets(sem_seg_gt, images)

        upsampling_targets = []

        for l_idx in range(len(self.mask_predictors)):
            outs, features, features_pos = self.mask_predictors[l_idx](images.tensor, l_idx, features, features_pos, upsampling_mask)
            if l_idx < len(self.mask_predictors) - 1:
                upsampling_mask_pred = outs["upsampling_mask_{}".format(l_idx)]
                #print("Original upsampling mask shape for layer {} is {}".format(l_idx, upsampling_mask.shape))
                if l_idx == 0:
                    upsampling_mask_oracle = self.generate_initial_oracle_upsampling_mask_edge(sem_seg_gt, target_pad)
                else:
                    upsampling_mask_oracle = self.generate_subsequent_oracle_upsampling_mask_edge(sem_seg_gt, features_pos,
                                                                                             l_idx, target_pad)
                if self.training and random.random() < self.oracle_teacher_ratio:
                    upsampling_mask = upsampling_mask_oracle
                    #print("Upsampling mask in layer {} chosen from oracle, shape is {}".format(l_idx, upsampling_mask.shape))
                else:
                    upsampling_mask = upsampling_mask_pred
                    #print("Upsampling mask in layer {} chosen from pred, shape is {}".format(l_idx, upsampling_mask.shape))

                upsampling_targets.append(upsampling_mask_oracle)

                dm_pred = {}
                dm_oracle = {}
                dm_pred["disagreement_mask_pred_{}".format(l_idx)] = upsampling_mask_pred
                dm_pred["disagreement_mask_pred_pos_{}".format(l_idx)] = features_pos
                disagreement_masks_pred.append(dm_pred)
                dm_oracle["disagreement_mask_oracle_{}".format(l_idx)] = upsampling_mask_oracle
                dm_oracle["disagreement_mask_oracle_pos_{}".format(l_idx)] = features_pos
                disagreement_masks_oracle.append(dm_oracle)

                outputs['upsampling_outputs'].append(outs["upsampling_mask_{}".format(l_idx)])

            outputs['aux_outputs'] = outputs['aux_outputs'] + outs['aux_outputs']
        outputs['pred_logits'] = outs['pred_logits']
        outputs['pred_masks'] = outs['pred_masks']

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            losses = self.criterion(outputs, targets, upsampling_targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs
            i = 0
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})


                for level, dmp in enumerate(disagreement_masks_pred):
                    dis_mask = dmp["disagreement_mask_pred_{}".format(level)][i]
                    dis_mask_pos = dmp["disagreement_mask_pred_pos_{}".format(level)][i]
                    top_scale = int(dis_mask_pos[:,0].max())
                    disagreement_map = torch.zeros(images.tensor.shape[-2], images.tensor.shape[-1], device=dis_mask.device)
                    disagreement_map = self.create_disagreement_map(disagreement_map, dis_mask, dis_mask_pos, level, top_scale)
                    processed_results[-1]["disagreement_mask_pred_{}".format(level)] = disagreement_map.cpu()

                for level, dmp in enumerate(disagreement_masks_oracle):
                    dis_mask = dmp["disagreement_mask_oracle_{}".format(level)][i]
                    dis_mask_pos = dmp["disagreement_mask_oracle_pos_{}".format(level)][i]
                    top_scale = int(dis_mask_pos[:,0].max())
                    disagreement_map = torch.zeros(images.tensor.shape[-2], images.tensor.shape[-1], device=dis_mask.device)
                    disagreement_map = self.create_disagreement_map(disagreement_map, dis_mask, dis_mask_pos, level, top_scale)
                    processed_results[-1]["disagreement_mask_oracle_{}".format(level)] = disagreement_map.cpu()

                i += 1

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets


    def prepare_oracle_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        pad_height_width = []
        #print("image shape for preparation is: {}".format(images.tensor.shape))
        for targets_per_image in targets:
            h_pad_n = h_pad - targets_per_image.shape[0]
            w_pad_n = w_pad - targets_per_image.shape[1]
            pad_height_width.append((h_pad_n, w_pad_n))
            # pad gt
            #print("target shape for preparation is: {}".format(targets_per_image.shape))
            padded_masks = torch.zeros((h_pad, w_pad), dtype=targets_per_image.dtype, device=targets_per_image.device)
            padded_masks = padded_masks + 254
            padded_masks[: targets_per_image.shape[0], : targets_per_image.shape[1]] = targets_per_image
            new_targets.append(padded_masks)
            #print("padded shape is {}".format(padded_masks.shape))
        return new_targets, pad_height_width

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def create_disagreement_map(self, disagreement_map, dis_mask, dis_mask_pos, level, scale):
        min_pos, max_pos = self.get_min_max_position(dis_mask_pos[:,1:], disagreement_map.shape[1], disagreement_map.shape[0])
        #print("Min pos at level {} for all: {}".format(level, min_pos))
        #print("Max pos at level {} for all: {}".format(level, max_pos))
        dis_mask_at_scale, dis_pos_at_scale = self.get_disagreement_mask_and_pos_at_scale(dis_mask, dis_mask_pos, scale)
        min_pos, max_pos = self.get_min_max_position(dis_pos_at_scale, disagreement_map.shape[1], disagreement_map.shape[0])
        #print("Min pos at level {} for scale {}: {}".format(level, scale, min_pos))
        #print("Max pos at level {} for scale {}: {}".format(level, scale, max_pos))
        dis_mask_top, dis_pos_top = self.get_top_disagreement_mask_and_pos(dis_mask_at_scale, dis_pos_at_scale, level)
        min_pos, max_pos = self.get_min_max_position(dis_pos_top, disagreement_map.shape[1], disagreement_map.shape[0])
        #print("Min pos at level {} for top scale {}: {}".format(level, scale, min_pos))
        #print("Max pos at level {} for top scale {}: {}".format(level, scale, max_pos))
        pos_at_org_scale = dis_pos_top * self.mask_predictors[0].backbone.min_patch_size
        patch_size = self.mask_predictors[level].backbone.patch_sizes[scale]

        dis_mask_top = dis_mask_top.unsqueeze(1).expand(-1, patch_size ** 2).reshape(-1)

        new_coords = torch.stack(torch.meshgrid(torch.arange(0, patch_size), torch.arange(0, patch_size)))
        new_coords = new_coords.permute(1, 2, 0).transpose(0, 1).reshape(-1, 2).to(dis_pos_top.device)
        pos_at_org_scale = pos_at_org_scale.unsqueeze(1) + new_coords
        pos_at_org_scale = pos_at_org_scale.reshape(-1, 2)
        #print("pos_to_split shape before: {}".format(pos_to_split.shape))
        #print("max x pos before: {}".format(pos_to_split[...,0].max()))
        #print("max y pos before: {}".format(pos_to_split[...,1].max()))

        x_pos = pos_at_org_scale[...,0].long()
        y_pos = pos_at_org_scale[...,1].long()
        #print("max x pos: {}".format(x_pos.max()))
        #print("max y pos: {}".format(y_pos.max()))
        #print("pred_map_low_res shape: {}".format(pred_map_low_res.shape))
        disagreement_map[y_pos, x_pos] = 255 #dis_mask_at_scale
        return disagreement_map


    def get_min_max_position(self, pos, width, height):
        max_y = height // self.mask_predictors[0].backbone.min_patch_size
        max_x = width // self.mask_predictors[0].backbone.min_patch_size
        #print("Max x: {}".format(max_x))
        #print("Max y: {}".format(max_y))
        #print("Pos max x: {}".format(pos[:,0].max().item()))
        #print("Pos max y: {}".format(pos[:, 1].max().item()))
        assert max_x >= pos[:,0].max()
        assert max_y >= pos[:, 1].max()
        pos_flat = pos[:,0] + max_x * pos[:,1]
        min_val, min_indice = torch.min(pos_flat, dim=0)
        max_val, max_indice = torch.max(pos_flat, dim=0)

        min_pos = pos[min_indice]
        max_pos = pos[max_indice]

        return min_pos, max_pos


    def get_disagreement_mask_and_pos_at_scale(self, dis_mask, dis_mask_pos, scale):
        n_scale_idx = torch.where(dis_mask_pos[:, 0] == scale)
        dis_pos_at_scale = dis_mask_pos[n_scale_idx][:,1:]
        dis_mask_at_scale = dis_mask[n_scale_idx]

        return dis_mask_at_scale, dis_pos_at_scale

    def get_top_disagreement_mask_and_pos(self, dis_mask, dis_mask_pos, level):
        if level == len(self.mask_predictors) - 1:
            k_top = int(dis_mask.shape[0] * self.mask_predictors[0].backbone.upscale_ratio)
        else:
            k_top = int(dis_mask.shape[0] * self.mask_predictors[level + 1].backbone.upscale_ratio)
        sorted_scores, sorted_indices = torch.sort(dis_mask, dim=0, descending=False)
        top_indices = sorted_indices[-k_top:]
        top_dis_mask = dis_mask.gather(dim=0, index=top_indices)
        top_dis_mask_pos = dis_mask_pos.gather(dim=0, index=top_indices.unsqueeze(-1).expand(-1, 2))

        return top_dis_mask, top_dis_mask_pos

    def get_upsampled_mask_and_pos(self, dis_mask, dis_mask_pos, scale):
        n_scale_idx = torch.where(dis_mask_pos[:, 0] == scale)
        dis_pos_at_scale = dis_mask_pos[n_scale_idx][:,1:]
        dis_mask_at_scale = dis_mask[n_scale_idx]

        return dis_mask_at_scale, dis_pos_at_scale

    def generate_initial_oracle_upsampling_mask_gini(self, targets):
        patch_size = self.mask_predictors[0].backbone.patch_size
        disagreement_map = []
        for batch in range(len(targets)):
            targets_batch = targets[batch].squeeze()
            #print("Initial oracle target shape: {}".format(targets_batch.shape))
            H, W = targets_batch.shape
            #targets_batch = self.fix_borders(targets_batch)
            targets_patched = rearrange(targets_batch, '(hp ph) (wp pw) -> (hp wp) (ph pw)', ph=patch_size,
                                        pw=patch_size, hp=H // patch_size, wp=W // patch_size)
            #print("Initial patched target shape: {}".format(targets_patched.shape))
            targets_shifted = (targets_patched.byte() + 2).long()
            histogram = torch.nn.functional.one_hot(targets_shifted, num_classes=152).sum(dim=1)
            histogram = histogram[:, 1:]
            histogram = torch.div(histogram, histogram.sum(dim=1).unsqueeze(1))
            #print("Initial histogram shape: {}".format(targets_batch.shape))
            disagreement = 1 - self.gini(histogram.float())
            disagreement[(targets_shifted == 0).all(dim=1)] = 0
            #print("Initial disagreement shape: {}".format(disagreement.shape))
            disagreement_map.append(disagreement)
        disagreement_map_tensor = torch.stack(disagreement_map)
        #print("Initial disagreement map shape: {}".format(disagreement_map_tensor.shape))
        return disagreement_map_tensor

    def generate_initial_oracle_upsampling_mask_edge(self, targets, targets_pad):
        patch_size = self.mask_predictors[0].backbone.patch_size
        disagreement_map = []
        for batch in range(len(targets)):
            targets_batch = targets[batch].squeeze()
            targets_shifted = (targets_batch.byte() + 2).long()
            pad_h, pad_w = targets_pad[batch]
            border_mask = self.get_ignore_mask(targets_shifted, pad_h, pad_w)
            edge_mask = self.compute_edge_mask_with_ignores(targets_shifted, border_mask)
            disagreement = self.count_edges_per_patch_masked(edge_mask, patch_size=patch_size)
            disagreement_map.append(disagreement)
        disagreement_map = torch.stack(disagreement_map).float()
        disagreement_map = (disagreement_map - disagreement_map.mean(dim=1, keepdim=True)) / (disagreement_map.var(dim=1, keepdim=True) + 1e-6).sqrt()
        #print("Initial disagreement map shape: {}".format(disagreement_map_tensor.shape))
        return disagreement_map

    def generate_subsequent_oracle_upsampling_mask_gini(self, targets, pos, level):
        B,N,C = pos.shape
        patch_size = self.mask_predictors[level].backbone.patch_size
        disagreement_map = []
        #pos_level = self.get_pos_at_scale(pos, level)
        #print("Subsequent pos shape: {}".format(pos.shape))
        for batch in range(B):
            targets_batch = targets[batch].squeeze()
            #targets_batch = self.fix_borders(targets_batch)
            #print("Subsequent oracle target shape: {}".format(targets_batch.shape))
            pos_batch = pos[batch][:,1:]
            p_org = (pos_batch * self.mask_predictors[level].backbone.min_patch_size).long()
            patch_coords = torch.stack(torch.meshgrid(torch.arange(0, patch_size), torch.arange(0, patch_size)))
            patch_coords = patch_coords.permute(1, 2, 0).transpose(0, 1).reshape(-1, 2).to(pos.device)
            pos_patches = p_org.unsqueeze(1) + patch_coords.unsqueeze(0)
            pos_patches = pos_patches.view(-1, 2)
            x_pos = pos_patches[..., 0].long()
            y_pos = pos_patches[..., 1].long()
            targets_patched = targets_batch[y_pos, x_pos]
            targets_patched = rearrange(targets_patched, '(n p) -> n p', n=N)
            #print("Subsequent targets_patched shape: {}".format(targets_patched.shape))
            targets_shifted = (targets_patched.byte() + 2).long()
            histogram = torch.nn.functional.one_hot(targets_shifted, num_classes=152).sum(dim=1)
            histogram = histogram[:, 1:]
            histogram = torch.div(histogram, histogram.sum(dim=1).unsqueeze(1))
            #print("Subsequent histogram shape: {}".format(histogram.shape))
            disagreement = 1 - self.gini(histogram.float())
            #print("Subsequent disagreement shape: {}".format(disagreement.shape))
            disagreement[pos[batch][:, 0] != level] = 0
            disagreement[(targets_shifted == 0).all(dim=1)] = 0
            disagreement_map.append(disagreement)
        disagreement_map_tensor = torch.stack(disagreement_map)

        #print("Subsequent disagreement map shape: {}".format(disagreement_map_tensor.shape))
        return disagreement_map_tensor

    def generate_subsequent_oracle_upsampling_mask_edge(self, targets, pos, level, targets_pad):
        B,N,C = pos.shape
        patch_size = self.mask_predictors[level].backbone.patch_size
        initial_patch_size = self.mask_predictors[0].backbone.patch_size
        disagreement_map = []
        #pos_level = self.get_pos_at_scale(pos, level)
        #print("Subsequent pos shape: {}".format(pos.shape))
        for batch in range(B):
            targets_batch = targets[batch].squeeze()
            targets_shifted = (targets_batch.byte() + 2).long()
            pad_h, pad_w = targets_pad[batch]
            border_mask = self.get_ignore_mask(targets_shifted, pad_h, pad_w)
            edge_mask = self.compute_edge_mask_with_ignores(targets_shifted, border_mask)

            pos_batch = pos[batch][:,1:]
            p_org = (pos_batch * self.mask_predictors[level].backbone.min_patch_size).long()
            patch_coords = torch.stack(torch.meshgrid(torch.arange(0, patch_size), torch.arange(0, patch_size)))
            patch_coords = patch_coords.permute(1, 2, 0).transpose(0, 1).reshape(-1, 2).to(pos.device)
            pos_patches = p_org.unsqueeze(1) + patch_coords.unsqueeze(0)
            pos_patches = pos_patches.view(-1, 2)
            x_pos = pos_patches[..., 0].long()
            y_pos = pos_patches[..., 1].long()

            edge_mask_patched = edge_mask[y_pos, x_pos]
            edge_mask_patched = rearrange(edge_mask_patched, '(n ph pw) -> n ph pw', n=N, ph=patch_size, pw=patch_size)
            #print("Subsequent targets_patched shape: {}".format(targets_patched.shape))

            disagreement = edge_mask_patched.sum(dim=(1, 2))
            disagreement = disagreement / 2**((level - pos[batch][:, 0]) * 2) # Rescaling targets based on patch size
            disagreement_map.append(disagreement)
        disagreement_map = torch.stack(disagreement_map).float()
        disagreement_map = (disagreement_map - disagreement_map.mean(dim=1, keepdim=True)) / (disagreement_map.var(dim=1, keepdim=True) + 1e-6).sqrt()

        #print("Subsequent disagreement map shape: {}".format(disagreement_map_tensor.shape))
        return disagreement_map


    def count_edge_pixels_per_patch(self, patches):
        B, PH, PW = patches.shape
        # Ignores edges on borders of the image.
        edge_top = patches[:, 1:, :] != patches[:, :-1, :]
        edge_bottom = edge_top
        edge_left = patches[:, :, 1:] != patches[:, :, :-1]
        edge_right = edge_left

        edge_mask = torch.zeros_like(patches, dtype=torch.bool)
        edge_mask[:, 1:, :] |= edge_top
        edge_mask[:, :-1, :] |= edge_bottom
        edge_mask[:, :, 1:] |= edge_left
        edge_mask[:, :, :-1] |= edge_right

        edge_pixel_counts = edge_mask.view(B, -1).sum(dim=1)

        return edge_pixel_counts

    def count_edges_per_patch_masked(self, edge_mask, patch_size):
        H, W = edge_mask.shape
        P = patch_size
        patches = edge_mask.view(H // P, P, W // P, P).permute(0, 2, 1, 3)
        patches = patches.reshape(-1, P, P)
        return patches.sum(dim=(1, 2))

    def get_ignore_mask(self, label_map, pad_h, pad_w, border_size=5):
        H, W = label_map.shape
        usable_h = H - pad_h
        usable_w = W - pad_w

        ignore_mask = (label_map == 0)
        border_mask = torch.zeros_like(label_map, dtype=torch.bool)
        border_mask[:border_size, :usable_w] = True
        border_mask[usable_h - border_size:usable_h, :usable_w] = True
        border_mask[:usable_h, :border_size] = True
        border_mask[:usable_h, usable_w - border_size:usable_w] = True

        class1_mask = (label_map == 1)
        ignore_mask |= class1_mask & border_mask
        return ignore_mask

    def compute_edge_mask_with_ignores(self, label_map, ignore_mask):
        H, W = label_map.shape
        edge_mask = torch.zeros_like(label_map, dtype=torch.bool)

        # Top neighbor (i, j) vs (i-1, j)
        valid = (~ignore_mask[1:, :]) & (~ignore_mask[:-1, :])
        diff = label_map[1:, :] != label_map[:-1, :]
        edge_mask[1:, :] |= valid & diff

        # Bottom neighbor
        valid = (~ignore_mask[:-1, :]) & (~ignore_mask[1:, :])
        diff = label_map[:-1, :] != label_map[1:, :]
        edge_mask[:-1, :] |= valid & diff

        # Left neighbor
        valid = (~ignore_mask[:, 1:]) & (~ignore_mask[:, :-1])
        diff = label_map[:, 1:] != label_map[:, :-1]
        edge_mask[:, 1:] |= valid & diff

        # Right neighbor
        valid = (~ignore_mask[:, :-1]) & (~ignore_mask[:, 1:])
        diff = label_map[:, :-1] != label_map[:, 1:]
        edge_mask[:, :-1] |= valid & diff

        return edge_mask

    def gini(self, class_counts):
        mad = torch.abs(class_counts.unsqueeze(1) - class_counts.unsqueeze(2)).mean(dim=(1, 2))
        rmad = mad / class_counts.mean(dim=1)
        g = 0.5 * rmad
        return g

    def get_pos_at_scale(self, pos, scale):
        B, _, _ = pos.shape
        b_scale_idx, n_scale_idx = torch.where(pos[:, :, 0] == scale)
        coords_at_curr_scale = pos[b_scale_idx, n_scale_idx, :]
        coords_at_curr_scale = rearrange(coords_at_curr_scale, '(b n) p -> b n p', b=B).contiguous()

        return coords_at_curr_scale

    def fix_borders(self, targets, border=5, pad_val=254):
        H, W = targets.shape
        targets[0:border, :][targets[0:border, :] == 0] = pad_val
        targets[H - border:, :][targets[H - border:, :] == 0] = pad_val
        targets[:, 0:border][targets[:, 0:border] == 0] = pad_val
        targets[:, W - border:][targets[:, W - border:] == 0] = pad_val

        return targets