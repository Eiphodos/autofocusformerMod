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
class MaskFinerOracleTeacherSW(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone,
        sem_seg_head,
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
        test_sw_stride: Tuple[int],
        test_sw_crop_size: Tuple[int]
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
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
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
        self.test_sw_stride = test_sw_stride
        self.test_sw_crop_size = test_sw_crop_size
        self.n_classes = self.sem_seg_head.num_classes

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference
        print("Successfully built MaskFinerOracleTeacherBB model!")

    @classmethod
    def from_config(cls, cfg):
        print("Building MaskFinerOracleTeacherBB model...")
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

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
            cfg.MODEL.MR_SEM_SEG_HEAD.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FINER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FINER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FINER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
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
            "test_sw_stride": cfg.TEST.SW_STRIDE,
            "test_sw_crop_size": cfg.TEST.SW_CROP_SIZE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if self.training:
            return self.forward_train(batched_inputs)
        else:
            return self.forward_eval(batched_inputs)

    def forward_eval(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        #print("Image shape in eval is {}".format(images.tensor.shape))

        sem_seg_gt = None
        target_pad = None

        h_stride, w_stride = self.test_sw_stride
        h_crop, w_crop = self.test_sw_crop_size
        h_img, w_img = (images.tensor.shape[-2], images.tensor.shape[-1])
        batch_size = len(images)
        out_channels = self.n_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = images.tensor.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = images.tensor.new_zeros((batch_size, 1, h_img, w_img))
        processed_results = []

        #print("Output prediction shape in eval is {}".format(preds.shape))

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = images.tensor[:, :, y1:y2, x1:x2]
                #print("Crop shape in eval is {}".format(crop_img.shape))

                features = self.backbone(crop_img, sem_seg_gt, target_pad)
                outputs = self.sem_seg_head(features)


                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]

                #print("Mask pred shape in eval before interpolation is {}".format(mask_pred_results.shape))
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(h_crop, w_crop),
                    mode="bilinear",
                    align_corners=False,
                )
                #print("Mask pred shape in eval after interpolation is {}".format(mask_pred_results.shape))

                i = 0
                for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                    mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
                ):
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)

                    preds[i, :, :, :] += F.pad(r, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                    count_mat[i, :, y1:y2, x1:x2] += 1
                    i += 1

                del outputs
        assert (count_mat == 0).sum() == 0
        seg_probs = preds / count_mat
        for b in range(seg_probs.shape[0]):
            processed_results.append({})
            processed_results[-1]["sem_seg"] = seg_probs[b]
        return processed_results

    def forward_train(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        #print("Image shape in train is {}".format(images.tensor.shape))

        if self.panoptic_on:
            key = "instances"
        elif self.semantic_on:
            key = "sem_seg"
        elif self.instance_on:
            key = "instances"
        else:
            raise Exception("No label key found in batched inputs")
        sem_seg_gt = [x[key].to(self.device) for x in batched_inputs]
        sem_seg_gt, target_pad = self.prepare_oracle_targets(sem_seg_gt, images)

        features = self.backbone(images.tensor, sem_seg_gt, target_pad)
        outputs = self.sem_seg_head(features)

        disagreement_masks_pred = []
        disagreement_masks_oracle = []
        upsampling_targets = []
        outputs['upsampling_outputs'] = []
        for i in range(self.backbone.n_scales - 1):
            upsampling_mask_pred = features['upsampling_mask_pred_{}'.format(i)]
            upsampling_mask_oracle = features['upsampling_mask_oracle_{}'.format(i)]
            upsampling_mask_pos = features['upsampling_mask_pos_{}'.format(i)]

            #Prepare target and output
            upsampling_targets.append(upsampling_mask_oracle)
            outputs['upsampling_outputs'].append(upsampling_mask_pred)

            # For visualizations
            dm_pred = {}
            dm_oracle = {}
            dm_pred["disagreement_mask_pred_{}".format(i)] = upsampling_mask_pred
            dm_pred["disagreement_mask_pred_pos_{}".format(i)] = upsampling_mask_pos
            disagreement_masks_pred.append(dm_pred)
            dm_oracle["disagreement_mask_oracle_{}".format(i)] = upsampling_mask_oracle
            dm_oracle["disagreement_mask_oracle_pos_{}".format(i)] = upsampling_mask_pos
            disagreement_masks_oracle.append(dm_oracle)

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
            if self.panoptic_on or self.instance_on:
                gt_masks = targets_per_image.gt_masks
                h_pad_n = h_pad - gt_masks.shape[1]
                w_pad_n = w_pad - gt_masks.shape[2]
                if gt_masks.shape[0] == 0:
                    padded_masks = torch.zeros((1, h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                else:
                    padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype,
                                               device=gt_masks.device)
                    padded_masks = padded_masks + 254
                    padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                    padded_masks = padded_masks.int().argmax(dim=0)
            elif self.semantic_on:
                h_pad_n = h_pad - targets_per_image.shape[0]
                w_pad_n = w_pad - targets_per_image.shape[1]
                padded_masks = torch.zeros((h_pad, w_pad), dtype=targets_per_image.dtype,
                                           device=targets_per_image.device)
                padded_masks = padded_masks + 254
                padded_masks[: targets_per_image.shape[0], : targets_per_image.shape[1]] = targets_per_image
            new_targets.append(padded_masks)
            pad_height_width.append((h_pad_n, w_pad_n))
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
        dis_mask_at_scale, dis_pos_at_scale = self.get_disagreement_mask_and_pos_at_scale(dis_mask, dis_mask_pos, scale)
        dis_mask_top, dis_pos_top = self.get_top_disagreement_mask_and_pos(dis_mask_at_scale, dis_pos_at_scale, level)
        pos_at_org_scale = dis_pos_top * self.backbone.backbones[0].min_patch_size
        patch_size = self.backbone.backbones[level].patch_sizes[scale]

        new_coords = torch.stack(torch.meshgrid(torch.arange(0, patch_size), torch.arange(0, patch_size)))
        new_coords = new_coords.permute(1, 2, 0).transpose(0, 1).reshape(-1, 2).to(dis_pos_top.device)
        pos_at_org_scale = pos_at_org_scale.unsqueeze(1) + new_coords
        pos_at_org_scale = pos_at_org_scale.reshape(-1, 2)

        x_pos = pos_at_org_scale[...,0].long()
        y_pos = pos_at_org_scale[...,1].long()
        disagreement_map[y_pos, x_pos] = 255 #dis_mask_at_scale
        return disagreement_map


    def get_min_max_position(self, pos, width, height):
        max_y = height // self.backbone.backbones[0].min_patch_size
        max_x = width // self.backbone.backbones[0].min_patch_size
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
        N = dis_mask.shape[0]
        upscale_ratio = self.backbone.backbones[level + 1].upscale_ratio
        if self.backbone.dynamic_up_ratios:
            up_trshld = self.backbone.backbones[level + 1].dynamic_up_threshold
            if not self.training:
                upscale_ratio = 1.0
            dyn_ratio = min(((dis_mask > up_trshld).sum(-1) / N).max(), upscale_ratio)
            dyn_ratio = max(dyn_ratio, 0.1)
            k_top = int(N * dyn_ratio)
        else:
            k_top = int(N * upscale_ratio)
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