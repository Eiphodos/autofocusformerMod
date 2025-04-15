# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict
import random

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


class MROTB(nn.Module):
    def __init__(self, backbones, backbone_dims, out_dim, oracle_teacher_ratio, all_out_features):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.out_dim = out_dim
        self.backbone_dims = backbone_dims
        self.oracle_teacher_ratio = oracle_teacher_ratio
        self.all_out_features = all_out_features
        self.all_out_features_scales = {k: len(all_out_features) - i - 1 for i, k in enumerate(all_out_features)}

        upsamplers = []
        for i in range(len(self.backbones) - 1):
            upsample_out = MLP(backbone_dims[i], backbone_dims[i] * 2, 1, num_layers=3)
            upsamplers.append(upsample_out)
        self.upsamplers = nn.ModuleList(upsamplers)

        feat_projs = []
        for i in range(len(self.backbones)):
            scale_projs = []
            for j in range(len(self.backbones[i]._out_features) - 1):
                f_proj = nn.Linear(backbone_dims[i], backbone_dims[j])
                scale_projs.append(f_proj)
            scale_projs = nn.ModuleList(scale_projs)
            feat_projs.append(scale_projs)
        self.feat_proj = nn.ModuleList(feat_projs)


    def forward(self, im, sem_seg_gt, target_pad):
        upsampling_mask = None
        features = None
        features_pos = None
        outs = {}
        for scale in range(len(self.backbones)):
            output = self.backbones[scale](im, scale, features, features_pos, upsampling_mask)
            all_out_features = self.backbones[scale]._out_features
            all_feat = []
            all_scale = []
            all_pos = []
            all_ss = []
            for i, f in enumerate(all_out_features):
                feat = output[f]
                feat_pos = output[f + '_pos']
                feat_scale = output[f + '_scale']
                feat_ss = output[f + '_spatial_shape']
                curr_scale = self.all_out_features_scales[f]

                if f + '_pos' in outs:
                    assert (outs[f + '_pos'] == feat_pos).all()
                    outs[f] = outs[f] + self.feat_proj[scale][curr_scale](feat)
                else:
                    outs[f] = feat
                    outs[f + '_pos'] = feat_pos
                    outs[f + '_scale'] = feat_scale
                    outs[f + '_spatial_shape'] = feat_ss

                all_feat.append(feat)
                all_pos.append(feat_pos)
                all_scale.append(feat_scale)
                all_ss.append(feat_ss)
            if scale == 0:
                upsampling_mask_oracle = self.generate_initial_oracle_upsampling_mask_edge(sem_seg_gt, target_pad)
            else:
                upsampling_mask_oracle = self.generate_subsequent_oracle_upsampling_mask_edge(sem_seg_gt, all_pos[0],
                                                                                              scale, target_pad)
            if scale < len(self.backbones) - 1:
                upsampling_mask_pred = self.upsamplers[scale](all_feat[0])
                outs['upsampling_mask_{}'.format(scale)] = upsampling_mask_pred

            if self.training and random.random() < self.oracle_teacher_ratio:
                upsampling_mask = upsampling_mask_oracle
            else:
                upsampling_mask = upsampling_mask_pred


            all_pos = torch.cat(all_pos, dim=1)
            all_scale = torch.cat(all_scale, dim=1)
            features_pos = torch.cat([all_scale.unsqueeze(2), all_pos], dim=2)
            features = torch.cat(all_feat, dim=1)
        outs['min_spatial_shape'] = output['min_spatial_shape']
        return outs



@BACKBONE_REGISTRY.register()
class OracleTeacherBackbone(MROTB, Backbone):
    def __init__(self, cfg):

        all_backbones = []
        n_scales = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES
        for i in range(n_scales):
            bb = build_backbone_indexed(cfg, i)
            all_backbones.append(bb)
        out_dim = cfg.MODEL.MR_SEM_SEG_HEAD.CONVS_DIM[-1]

        super().__init__(
            backbones=all_backbones,
            backbone_dims=cfg.MODEL.MR.EMBED_DIM,
            out_dim=out_dim,
            oracle_teacher_ratio=cfg.MODEL.MASK_FINER.ORACLE_TEACHER_RATIO,
            all_out_features=cfg.MODEL.MR.OUT_FEATURES
        )

        self._out_features = cfg.MODEL.MR.OUT_FEATURES

        self._in_features_channels = cfg.MODEL.MR.EMBED_DIM

        #self._out_feature_strides = { "res{}".format(layer_index+2): cfg.MODEL.MRNB.PATCH_SIZES[layer_index]}
        self._out_feature_strides = {"res{}".format(n_scales + 1 - i): cfg.MODEL.MRML.PATCH_SIZES[i] for i in range(n_scales)}
        #print("backbone strides: {}".format(self._out_feature_strides))
        #self._out_feature_channels = { "res{}".format(i+2): list(reversed(self.num_features))[i] for i in range(num_scales)}
        self._out_feature_channels = {"res{}".format(n_scales + 1 - i): cfg.MODEL.MR.EMBED_DIM[i] for i in range(n_scales)}
        #print("backbone channels: {}".format(self._out_feature_channels))

    def forward(self, x, sem_seg_gt, target_pad):
        """
        Args:
            x: Tensor of shape (B,C,H,W)
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"MRML takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        y = super().forward(x, sem_seg_gt, target_pad)
        return y

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }