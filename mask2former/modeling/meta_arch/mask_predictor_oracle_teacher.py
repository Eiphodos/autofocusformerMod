# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone

from ..transformer_decoder.build_maskfiner_decoder import build_transformer_decoder
from ..pixel_decoder.msdeformattn_pc_maskfiner import build_pixel_decoder
from ..backbone.build import build_backbone_indexed


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@SEM_SEG_HEADS_REGISTRY.register()
class MaskPredictorOracleTeacher(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )


    @configurable
    def __init__(
        self,
        backbone: Backbone,
        pixel_decoder: nn.Module,
        mask_decoder,
        num_classes: int,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        hidden_dim: int = 256,
        final_layer: bool = False,
        mask_decoder_all_levels: bool = True):
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.mask_decoder = mask_decoder
        self.ignore_value = ignore_value
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.final_layer = final_layer
        self.mask_decoder_all_levels = mask_decoder_all_levels

        if not self.final_layer:
            self.upsample_out = MLP(hidden_dim, hidden_dim*2, 1, num_layers=3)

    @classmethod
    def from_config(cls, cfg, layer_index):
        final_layer = (layer_index == (cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES - 1))
        mask_decoder_all_levels = cfg.MODEL.MASK_FINER.MASK_DECODER_ALL_LEVELS
        backbone = build_backbone_indexed(cfg, layer_index)
        bb_output_shape = backbone.output_shape()
        pixel_decoder = build_pixel_decoder(cfg, layer_index, bb_output_shape)
        if final_layer or mask_decoder_all_levels:
            mask_decoder_input_dim = cfg.MODEL.MR_SEM_SEG_HEAD.CONVS_DIM[layer_index]
            mask_decoder = build_transformer_decoder(cfg, layer_index, mask_decoder_input_dim, mask_classification=True)
        else:
            mask_decoder = None
        return {
            "backbone": backbone,
            "pixel_decoder": pixel_decoder,
            "mask_decoder": mask_decoder,
            "loss_weight": cfg.MODEL.MR_SEM_SEG_HEAD.LOSS_WEIGHT,
            "ignore_value": cfg.MODEL.MR_SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.MR_SEM_SEG_HEAD.NUM_CLASSES,
            "hidden_dim": cfg.MODEL.MR_SEM_SEG_HEAD.CONVS_DIM[layer_index],
            "final_layer": final_layer,
            "mask_decoder_all_levels": mask_decoder_all_levels
        }

    def forward(self, im, scale, features, features_pos, upsampling_mask):
        return self.layers(im, scale, features, features_pos, upsampling_mask)

    def layers(self, im, scale, features, features_pos, upsampling_mask):
        features = self.backbone(im, scale, features, features_pos, upsampling_mask)
        mask_features, mf_pos, multi_scale_features, multi_scale_poss, ms_scale, finest_input_shape, input_shapes = self.pixel_decoder.forward_features(features)
        if self.final_layer or self.mask_decoder_all_levels:
            predictions = self.mask_decoder(multi_scale_features, multi_scale_poss, mask_features, mf_pos, finest_input_shape, input_shapes)
        else:
            predictions = {"aux_outputs": []}
        all_pos = torch.cat(multi_scale_poss, dim=1)
        all_scale = torch.cat(ms_scale, dim=1)
        pos_scale = torch.cat([all_scale.unsqueeze(2), all_pos], dim=2)
        all_feat = torch.cat(multi_scale_features, dim=1)

        if not self.final_layer:
            predictions["upsampling_mask_{}".format(scale)] = self.upsample_out(all_feat).squeeze(-1)

        return predictions, all_feat, pos_scale
