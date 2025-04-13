# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone

from ..transformer_decoder.build_maskfiner_decoder import build_transformer_decoder
from ..pixel_decoder.msdeformattn_pc_maskfiner import build_pixel_decoder
from ..backbone.build import build_backbone_indexed


@BACKBONE_REGISTRY.register()
class OracleTeacherBackbone(nn.Module, Backbone)):

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
        mask_decoder: nn.Module,
        num_classes: int,
        loss_weight: float = 1.0,
        ignore_value: int = -1):
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.mask_decoder = mask_decoder
        self.ignore_value = ignore_value
        self.loss_weight = loss_weight
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, layer_index):
        backbone = build_backbone_indexed(cfg, layer_index)
        bb_output_shape = backbone.output_shape()
        pixel_decoder = build_pixel_decoder(cfg, layer_index, bb_output_shape)
        mask_decoder_input_dim = cfg.MODEL.MR_SEM_SEG_HEAD.CONVS_DIM[layer_index]
        mask_decoder = build_transformer_decoder(cfg, layer_index, mask_decoder_input_dim, mask_classification=True)
        return {
            "backbone": backbone,
            "pixel_decoder": pixel_decoder,
            "mask_decoder": mask_decoder,
            "loss_weight": cfg.MODEL.MR_SEM_SEG_HEAD.LOSS_WEIGHT,
            "ignore_value": cfg.MODEL.MR_SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.MR_SEM_SEG_HEAD.NUM_CLASSES,
        }

    def forward(self, im, scale, features, features_pos, upsampling_mask):
        return self.layers(im, scale, features, features_pos, upsampling_mask)
    def layers(self, im, scale, features, features_pos, upsampling_mask):
        features = self.backbone(im, scale, features, features_pos, upsampling_mask)
        mask_features, mf_pos, multi_scale_features, multi_scale_poss, ms_scale, finest_input_shape, input_shapes = self.pixel_decoder.forward_features(features)
        predictions, upsampling_mask = self.mask_decoder(multi_scale_features, multi_scale_poss, mask_features, mf_pos, finest_input_shape, input_shapes)
        all_pos = torch.cat(multi_scale_poss, dim=1)
        all_scale = torch.cat(ms_scale, dim=1)
        pos_scale = torch.cat([all_scale.unsqueeze(2), all_pos], dim=2)
        all_feat = torch.cat(multi_scale_features, dim=1)

        '''
        all_pos = []
        all_feat = []
        all_scale = []
        for out_feat in self.backbone._out_features:
            feat = features[out_feat]
            pos = features[out_feat + '_pos']
            scale = features[out_feat + '_scale']
            all_pos.append(pos)
            all_feat.append(feat)
            all_scale.append(scale)
        all_pos = torch.cat(all_pos, dim=1)
        all_scale = torch.cat(all_scale, dim=1)
        pos_scale = torch.cat([all_scale.unsqueeze(2), all_pos], dim=2)
        all_feat = torch.cat(all_feat, dim=1)
        '''
        return predictions, all_feat, pos_scale, upsampling_mask



@BACKBONE_REGISTRY.register()
class OracleTeacherBackbone(MROTB, Backbone):
    def __init__(self, cfg):

        all_backbones = []

        for i in range(cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES):
            bb = build_backbone_indexed(cfg, i)
            all_backbones.append(bb)

        if layer_index == 0:
            in_chans = 3
        else:
            in_chans = cfg.MODEL.MR_SEM_SEG_HEAD.CONVS_DIM[layer_index - 1]
        image_size = cfg.INPUT.CROP.SIZE
        n_scales = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES
        keep_old_scale = cfg.MODEL.MR.KEEP_OLD_SCALE
        add_image_data_to_all = cfg.MODEL.MR.ADD_IMAGE_DATA_TO_ALL
        min_patch_size = cfg.MODEL.MR.PATCH_SIZES[-1]

        patch_sizes = cfg.MODEL.MR.PATCH_SIZES[:layer_index + 1]
        embed_dim = cfg.MODEL.MR.EMBED_DIM[layer_index]
        depths = cfg.MODEL.MR.DEPTHS[layer_index]
        num_heads = cfg.MODEL.MR.NUM_HEADS[layer_index]
        drop_rate = cfg.MODEL.MR.DROP_RATE[layer_index]
        drop_path_rate = cfg.MODEL.MR.DROP_PATH_RATE[layer_index]
        attn_drop_rate = cfg.MODEL.MR.ATTN_DROP_RATE[layer_index]
        split_ratio = cfg.MODEL.MR.SPLIT_RATIO[layer_index]
        mlp_ratio = cfg.MODEL.MR.MLP_RATIO[layer_index]
        cluster_size = cfg.MODEL.MR.CLUSTER_SIZE[layer_index]
        nbhd_size = cfg.MODEL.MR.NBHD_SIZE[layer_index]
        upscale_ratio = cfg.MODEL.MR.UPSCALE_RATIO[layer_index]




        super().__init__(
            image_size=image_size,
            patch_sizes=patch_sizes,
            n_layers=depths,
            d_model=embed_dim,
            n_heads=num_heads,
            dropout=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            mlp_ratio=mlp_ratio,
            split_ratio=split_ratio,
            channels=in_chans,
            cluster_size=cluster_size,
            nbhd_size=nbhd_size,
            n_scales=n_scales,
            min_patch_size=min_patch_size,
            upscale_ratio=upscale_ratio,
            keep_old_scale=keep_old_scale,
            scale=layer_index,
            add_image_data_to_all=add_image_data_to_all
        )

        self._out_features = cfg.MODEL.MR.OUT_FEATURES[-(layer_index+1):]

        self._in_features_channels = cfg.MODEL.MR.EMBED_DIM[layer_index - 1]

        #self._out_feature_strides = { "res{}".format(layer_index+2): cfg.MODEL.MRNB.PATCH_SIZES[layer_index]}
        self._out_feature_strides = {"res{}".format(n_scales + 1 - i): cfg.MODEL.MRML.PATCH_SIZES[i] for i in range(layer_index + 1)}
        #print("backbone strides: {}".format(self._out_feature_strides))
        #self._out_feature_channels = { "res{}".format(i+2): list(reversed(self.num_features))[i] for i in range(num_scales)}
        self._out_feature_channels = {"res{}".format(n_scales + 1 - i): embed_dim for i in range(layer_index + 1)}
        #print("backbone channels: {}".format(self._out_feature_channels))

    def forward(self, x, scale, features, features_pos, upsampling_mask):
        """
        Args:
            x: Tensor of shape (B,C,H,W)
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"MRML takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        y = super().forward(x, scale, features, features_pos, upsampling_mask)
        return y

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }