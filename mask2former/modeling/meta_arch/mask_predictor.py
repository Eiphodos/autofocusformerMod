# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone

from ..transformer_decoder.maskfiner_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.msdeformattn_pc_maskfiner import build_pixel_decoder
from ..backbone.build import build_backbone_indexed


@META_ARCH_REGISTRY.register()
class MaskPredictor(nn.Module):

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
        mask_decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.mask_decoder = mask_decoder

    @classmethod
    def from_config(cls, cfg, layer_index):
        backbone = build_backbone_indexed(cfg, layer_index)
        bb_output_shape = backbone.get_output_shape()
        pixel_decoder = build_pixel_decoder(cfg, layer_index, input_shape=bb_output_shape)
        mask_decoder_input_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM[layer_index]
        mask_decoder = build_transformer_decoder(cfg, layer_index, mask_decoder_input_dim, mask_classification=True)
        return {
            "backbone": backbone,
            "pixel_decoder": pixel_decoder,
            "mask_decoder": mask_decoder
        }

    def forward(self, im, scale, features, features_pos, upsampling_mask):
        features = self.backbone(im, scale, features, features_pos, upsampling_mask)
        mask_features, mf_pos, multi_scale_features, multi_scale_poss, ms_scale = self.pixel_decoder.forward_features(features)
        predictions, upsampling_mask = self.mask_decoder(multi_scale_features, multi_scale_poss, mask_features, mf_pos)
        all_pos = torch.stack(multi_scale_poss, dim=1)
        all_scale = torch.stack(ms_scale, dim=1)
        pos_scale = torch.cat([all_scale, all_pos], dim=1)
        return predictions, torch.stack(multi_scale_features, dim=1), pos_scale, upsampling_mask
