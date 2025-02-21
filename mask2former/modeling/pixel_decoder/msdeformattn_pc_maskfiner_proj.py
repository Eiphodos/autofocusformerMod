#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import numpy as np
from typing import Callable, Dict, List, Optional, Union
import math

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_
from torch.cuda.amp import autocast
from einops import rearrange

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


def build_pixel_decoder(cfg, layer_index, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, layer_index, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model




@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnPixelDecoderMaskFinerProj(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
        shepard_power: float,
        shepard_power_learnable: bool,
        maskformer_num_feature_levels: int
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
            transformer_in_features: list of feature names into the deformable MSDETR
            common_stride: the stride of the finest feature map; outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
            shepard_power: the power used in deformable attn interpolation
            shepard_power_learnable: whether to make the power learnable
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]

        input_proj_list = []
        # from low resolution to high resolution (res5 -> res2)
        for in_channels in self.feature_channels[::-1]:
            input_proj_list.append(nn.Sequential(
                nn.Linear(in_channels, conv_dim, bias=True),
                nn.LayerNorm(conv_dim)
            ))
        self.input_proj = nn.ModuleList(input_proj_list)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.mask_dim = mask_dim
        self.mask_features = nn.Linear(
            conv_dim,
            mask_dim
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = maskformer_num_feature_levels  # always use 3 scales

    @classmethod
    def from_config(cls, cfg, layer_index, input_shape):
        pix_dec_in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES[-(layer_index + 1):]
        all_transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        all_dtf_len = len(all_transformer_in_features)
        if layer_index == 0:
            trans_dec_in_feat = all_transformer_in_features[-1]
        else:
            trans_dec_in_feat = all_transformer_in_features[(all_dtf_len - layer_index):]
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in pix_dec_in_features
        }
        m_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM[layer_index]
        ret["conv_dim"] = m_dim
        ret["mask_dim"] = cfg.MODEL.MASK_FINER.MASK_DIM[layer_index]
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.SEM_SEG_HEAD.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.SEM_SEG_HEAD.NHEADS[layer_index]
        ret["transformer_dim_feedforward"] = int(m_dim * cfg.MODEL.SEM_SEG_HEAD.MLP_RATIO[layer_index])
        ret["transformer_enc_layers"] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS[layer_index]
        ret["transformer_in_features"] = trans_dec_in_feat
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        ret['shepard_power'] = cfg.MODEL.MASK_FINER.SHEPARD_POWER / 2.0  # since the distances are already squared
        ret['shepard_power_learnable'] = cfg.MODEL.MASK_FINER.SHEPARD_POWER_LEARNABLE
        ret['maskformer_num_feature_levels'] = cfg.MODEL.MASK_FINER.DECODER_LEVELS[layer_index]
        return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        """
        Args
            features - a dictionary of a list of point clouds with their features, positions and canvas sizes
        """
        min_spatial_shape = features['min_spatial_shape']
        poss = []
        scaless = []
        spatial_shapes = []
        min_spatial_shapes = []
        out = []
        # Reverse feature maps into top-down order (from low to high resolution) res5 to res3
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f].float()
            pos = features[f+"_pos"].float()
            scales = features[f + "_scale"].float()
            spatial_shape = features[f+"_spatial_shape"]
            out.append(self.input_proj[idx](x))
            poss.append(pos)
            scaless.append(scales)
            spatial_shapes.append(spatial_shape)
            min_spatial_shapes.append(min_spatial_shape)

        multi_scale_features = []

        num_cur_levels = 0
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        mf = torch.cat(out, dim=1)
        mf_pos = torch.cat(poss, dim=1)

        return self.mask_features(mf), mf_pos, multi_scale_features, poss[:self.maskformer_num_feature_levels], scaless[:self.maskformer_num_feature_levels], spatial_shapes[-1]
