# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


def get_2dpos_of_curr_ps_in_min_ps(height, width, patch_size, min_patch_size, scale):
    patches_coords = torch.meshgrid(torch.arange(0, width // min_patch_size, patch_size // min_patch_size), torch.arange(0, height // min_patch_size, patch_size // min_patch_size), indexing='ij')
    patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
    patches_coords = patches_coords.permute(1, 2, 0)
    patches_coords = patches_coords.transpose(0, 1)
    patches_coords = patches_coords.reshape(-1, 2)
    n_patches = patches_coords.shape[0]

    scale_lvl = torch.tensor([[scale]] * n_patches)
    patches_scale_coords = torch.cat([scale_lvl, patches_coords], dim=1)
    return patches_scale_coords


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class CNVNXT2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, patch_sizes=[32],
                 depth=2, dim=512, scale=0, min_patch_size=4,
                 n_scales=4, upscale_ratio=0.5
                 ):
        super().__init__()
        self.depths = depth
        self.patch_size = patch_sizes[-1]
        self.patch_sizes = patch_sizes
        self.scale = scale
        self.min_patch_size = min_patch_size
        self.n_scales = n_scales
        self.upscale_ratio = upscale_ratio
        self.pe_layer = PositionEmbeddingSine(dim // 2, normalize=True)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=self.patch_size, stride=self.patch_size),
            LayerNorm(dim, eps=1e-6, data_format="channels_first")
        )
        self.stage = nn.Sequential(
            *[Block(dim=dim, drop_path=0.0) for j in range(depth)]
        )


        self.norm = nn.LayerNorm(dim, eps=1e-6)  # final norm layer

        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x, scale, features, features_pos, upsampling_mask):
        B, _, H, W = x.shape
        patched_im_size = (H // self.patch_size, W // self.patch_size)
        min_patched_im_size = (H // self.min_patch_size, W // self.min_patch_size)

        x = self.stem(x)
        x = self.stage(x)

        x = x.flatten(2).transpose(1, 2)
        pos = get_2dpos_of_curr_ps_in_min_ps(H, W, self.patch_size, self.min_patch_size, scale).to('cuda')
        pos = pos.repeat(B, 1, 1)
        pos_embed = self.pe_layer(pos[:, :, 1:])
        x = x + pos_embed
        x = self.norm(x)

        outs = {}
        out_name = self._out_features[0]
        outs[out_name] = x
        outs[out_name + "_pos"] = pos[:,:,1:]  # torch.div(pos_scale, 2 ** (self.n_scales - s - 1), rounding_mode='trunc')
        outs[out_name + "_spatial_shape"] = patched_im_size
        outs[out_name + "_scale"] = pos[:, :, 0]
        outs["min_spatial_shape"] = min_patched_im_size

        return outs

    def forward(self, x, scale, features, features_pos, upsampling_mask):
        x = self.forward_features(x, scale, features, features_pos, upsampling_mask)
        return x


@BACKBONE_REGISTRY.register()
class ConvNeXtV2(CNVNXT2, Backbone):
    def __init__(self, cfg, layer_index):
        in_chans = 3
        n_scales = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES
        min_patch_size = cfg.MODEL.MR.PATCH_SIZES[-1]

        #patch_size = cfg.MODEL.MR.PATCH_SIZES[layer_index]
        patch_sizes = cfg.MODEL.MR.PATCH_SIZES[:layer_index + 1]
        embed_dim = cfg.MODEL.MR.EMBED_DIM[layer_index]
        depths = cfg.MODEL.MR.DEPTHS[layer_index]
        mlp_ratio = cfg.MODEL.MR.MLP_RATIO[layer_index]
        num_heads = cfg.MODEL.MR.NUM_HEADS[layer_index]
        drop_rate = cfg.MODEL.MR.DROP_RATE[layer_index]
        drop_path_rate = cfg.MODEL.MR.DROP_PATH_RATE[layer_index]
        split_ratio = cfg.MODEL.MR.SPLIT_RATIO[layer_index]
        upscale_ratio = cfg.MODEL.MR.UPSCALE_RATIO[layer_index]

        super().__init__(
            in_chans=in_chans,
            patch_sizes=patch_sizes,
            depth=depths,
            dim=embed_dim,
            scale=layer_index,
            min_patch_size=min_patch_size,
            n_scales=n_scales,
            upscale_ratio=upscale_ratio
        )

        self._out_features = cfg.MODEL.MR.OUT_FEATURES[-(layer_index+1):]
        out_index = (n_scales - 1) + 2
        self._out_feature_strides = {"res{}".format(out_index): cfg.MODEL.MR.PATCH_SIZES[layer_index]}
        # self._out_feature_strides = {"res{}".format(i + 2): cfg.MODEL.MRML.PATCH_SIZES[-1] for i in range(num_scales)}
        # print("backbone strides: {}".format(self._out_feature_strides))
        # self._out_feature_channels = { "res{}".format(i+2): list(reversed(self.num_features))[i] for i in range(num_scales)}
        self._out_feature_channels = {"res{}".format(out_index): embed_dim}
        # print("backbone channels: {}".format(self._out_feature_channels))

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