# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

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
from .modeling.matcher import HungarianMatcher
from .modeling.meta_arch.build import build_mask_predictor_indexed


@META_ARCH_REGISTRY.register()
class MaskFiner(nn.Module):
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

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FINER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES + sum(cfg.MODEL.MASK_FINER.DEC_LAYERS)
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
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
        disagreement_masks = []
        outputs = {}
        outputs['pred_masks'] = None
        outputs['pred_logits'] = None
        outputs['aux_outputs'] = []
        for l_idx in range(len(self.mask_predictors)):
            outs, features, features_pos, upsampling_mask = self.mask_predictors[l_idx](images.tensor, l_idx, features, features_pos, upsampling_mask)

            dm = {}
            dm["disagreement_mask_{}".format(l_idx)] = upsampling_mask
            B, _, _ = features_pos.shape
            b_scale_idx, n_scale_idx = torch.where(features_pos[:, :, 0] == l_idx)
            dm_pos = features_pos[b_scale_idx, n_scale_idx, :]
            dm_pos = rearrange(dm_pos, '(b n) p -> b n p', b=B).contiguous()
            dm["disagreement_mask_pos_{}".format(l_idx)] = dm_pos
            disagreement_masks.append(dm)

            outputs['aux_outputs'] = outputs['aux_outputs'] + outs['aux_outputs']
        outputs['pred_logits'] = outs['pred_logits']
        outputs['pred_masks'] = outs['pred_masks']

        '''
        print(outputs.keys())
        print(outputs['pred_masks'].shape)
        print(outputs['pred_logits'].shape)
        print(len(outputs['aux_outputs']))
        '''

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            losses = self.criterion(outputs, targets)

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



                for level, dmp in enumerate(disagreement_masks):
                    dis_mask = dmp["disagreement_mask_{}".format(level)][i]
                    dis_mask_pos = dmp["disagreement_mask_pos_{}".format(level)][i]
                    n_scales = int(dis_mask_pos[:,0].max() + 1)
                    disagreement_map = torch.zeros(images.tensor.shape[-2], images.tensor.shape[-1], device=dis_mask.device)
                    for scale in range(n_scales):
                        disagreement_map = self.create_disagreement_map(disagreement_map, dis_mask, dis_mask_pos, level, scale)
                    processed_results[-1]["disagreement_mask_{}".format(level)] = disagreement_map.cpu()

                i += 1

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r_no_pp = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r_no_pp, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r
                    processed_results[-1]["sem_seg_no_pp"] = r_no_pp

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
        dis_mask_at_scale, dis_pos_at_scale = self.get_disagreement_mask_and_pos_at_scale(dis_mask, dis_mask_pos, scale)
        pos_at_org_scale = dis_pos_at_scale * self.mask_predictors[0].backbone.min_patch_size
        patch_size = self.mask_predictors[level].backbone.patch_sizes[scale]

        dis_mask_at_scale = dis_mask_at_scale.unsqueeze(1).expand(-1, patch_size ** 2).reshape(-1)

        new_coords = torch.stack(torch.meshgrid(torch.arange(0, patch_size), torch.arange(0, patch_size)))
        new_coords = new_coords.view(2,-1).permute(1,0).to(dis_pos_at_scale.device)
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
        disagreement_map[y_pos, x_pos] = dis_mask_at_scale
        return disagreement_map


    def get_disagreement_mask_and_pos_at_scale(self, dis_mask, dis_mask_pos, scale):
        n_scale_idx = torch.where(dis_mask_pos[:, 0] == scale)
        dis_pos_at_scale = dis_mask_pos[n_scale_idx][:,1:]
        dis_mask_at_scale = dis_mask[n_scale_idx]

        return dis_mask_at_scale, dis_pos_at_scale
