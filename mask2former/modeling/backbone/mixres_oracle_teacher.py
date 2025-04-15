# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict
import random
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec, build_backbone

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
    def __init__(self, backbones, backbone_dims, out_dim, oracle_teacher_ratio, all_out_features, n_scales):
        super().__init__()
        self.backbones = nn.ModuleList(backbones)
        self.out_dim = out_dim
        self.backbone_dims = backbone_dims
        self.oracle_teacher_ratio = oracle_teacher_ratio
        self.all_out_features = all_out_features
        self.all_out_features_scales = {k: len(all_out_features) - i - 1 for i, k in enumerate(all_out_features)}
        self.n_scales = n_scales

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

                print("Output {} for scale {}: feat_shape: {}, pos_shape: {}, scale_shape: {}, spatial_shape: {}".format(f, scale, feat.shape, feat_pos.shape, feat_scale.shape, feat_ss))


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
                outs['upsampling_mask_pred_{}'.format(scale)] = upsampling_mask_pred
                outs['upsampling_mask_oracle_{}'.format(scale)] = upsampling_mask_oracle
                outs['upsampling_mask_pos_{}'.format(scale)] = all_pos[0]

            print("Training is {}".format(int(self.training)))
            if self.training and random.random() < self.oracle_teacher_ratio:
                upsampling_mask = upsampling_mask_oracle
            else:
                upsampling_mask = upsampling_mask_pred

            print("Upsampling mask for scale {}: pred: {}, oracle: {}".format(scale,upsampling_mask_pred.shape, upsampling_mask_oracle))


            all_pos = torch.cat(all_pos, dim=1)
            all_scale = torch.cat(all_scale, dim=1)
            features_pos = torch.cat([all_scale.unsqueeze(2), all_pos], dim=2)
            features = torch.cat(all_feat, dim=1)
        outs['min_spatial_shape'] = output['min_spatial_shape']
        for k, v in outs.items():
            if type(v) == torch.Tensor:
                print("Outs {} has shape {}".format(k, v.shape))
            else:
                print("Outs {} is {}".format(k, v))
        return outs



@BACKBONE_REGISTRY.register()
class OracleTeacherBackbone(MROTB, Backbone):
    def __init__(self, cfg, input_shape):

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
            all_out_features=cfg.MODEL.MR.OUT_FEATURES,
            n_scales=n_scales
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


    def generate_initial_oracle_upsampling_mask_edge(self, targets, targets_pad):
        patch_size = self.backbones[0].patch_size
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


    def generate_subsequent_oracle_upsampling_mask_edge(self, targets, pos, level, targets_pad):
        B,N,C = pos.shape
        patch_size = self.backbones[level].patch_size
        initial_patch_size = self.backbones[0].patch_size
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
            p_org = (pos_batch * self.backbones[0].min_patch_size).long()
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

    def count_edges_per_patch_masked(self, edge_mask, patch_size):
        H, W = edge_mask.shape
        P = patch_size
        patches = edge_mask.view(H // P, P, W // P, P).permute(0, 2, 1, 3)
        patches = patches.reshape(-1, P, P)
        return patches.sum(dim=(1, 2))