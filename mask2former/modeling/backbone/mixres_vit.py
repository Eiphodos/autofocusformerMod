"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
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


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DownSampleConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.b_norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.b_norm(x)

        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Parameter):
        nn.init.trunc_normal_(m, std=0.02)

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, channels):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, channels):
        super().__init__()

        self.patch_size = patch_size

        n_layers = int(torch.log2(torch.tensor([patch_size])).item())
        conv_layers = []
        emb_dims = [int(embed_dim // 2**(n_layers -1 - i)) for i in range(n_layers) ]
        emb_dim_list = [channels] + emb_dims
        for i in range(n_layers):
            conv = DownSampleConvBlock(emb_dim_list[i], emb_dim_list[i + 1])
            conv_layers.append(conv)
        self.conv_layers = nn.Sequential(*conv_layers)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, im):
        x = self.conv_layers(im).flatten(2).transpose(1, 2)
        x = self.out_norm(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(
            self,
            n_blocks,
            dim,
            n_heads,
            dim_ff,
            dropout=0.0,
            drop_path_rate=0.0,
    ):
        super().__init__()

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, dim_ff, dropout, dpr[i]) for i in range(n_blocks)]
        )

    def forward(self, x):
        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x)
        return x


class MRVIT(nn.Module):
    def __init__(
            self,
            patch_sizes,
            n_layers,
            d_model,
            n_heads,
            mlp_ratio=4.0,
            dropout=0.0,
            drop_path_rate=0.0,
            channels=3,
            split_ratio=4,
            n_scales=2,
            min_patch_size=4,
            upscale_ratio=0.5
    ):
        super().__init__()
        self.patch_size = patch_sizes[-1]
        self.patch_sizes = patch_sizes
        self.patch_embed = OverlapPatchEmbedding(
            self.patch_size,
            d_model,
            channels,
        )
        self.patch_size = self.patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_scales = n_scales
        self.min_patch_size = min_patch_size
        self.upscale_ratio = upscale_ratio

        num_features = d_model
        self.num_features = num_features

        # Pos Embs
        self.pe_layer = PositionEmbeddingSine(d_model // 2, normalize=True)
        dim_ff = int(d_model * mlp_ratio)
        # transformer layers
        self.layers = TransformerLayer(n_layers, d_model, n_heads, dim_ff, dropout, drop_path_rate)

        #nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pre_logits = nn.Identity()
        self.norm_out = nn.LayerNorm(d_model)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, scale, features, features_pos, upsampling_mask):
        B, _, H, W = im.shape
        PS = self.patch_size
        x = self.patch_embed(im)
        patched_im_size = (H // PS, W // PS)
        min_patched_im_size = (H // self.min_patch_size, W // self.min_patch_size)

        pos = get_2dpos_of_curr_ps_in_min_ps(H, W, PS, self.min_patch_size, scale).to('cuda')
        pos = pos.repeat(B, 1, 1)
        #print("Encoder pos max x: {}, max y: {}, and all pos: {}".format(pos[:, :, 0].max(), pos[:, :, 1].max(), pos))
        #self.test_pos_cover_and_overlap(pos[0], H, W, scale)
        pos_embed = self.pe_layer(pos[:,:,1:])
        x = x + pos_embed

        x = self.layers(x)

        outs = {}
        out_name = self._out_features[0]
        outs[out_name] = self.norm_out(x)
        outs[out_name + "_pos"] = pos[:,:,1:]  # torch.div(pos_scale, 2 ** (self.n_scales - s - 1), rounding_mode='trunc')
        outs[out_name + "_spatial_shape"] = patched_im_size
        outs[out_name + "_scale"] = pos[:, :, 0]
        outs["min_spatial_shape"] = min_patched_im_size
        return outs


@BACKBONE_REGISTRY.register()
class MixResViT(MRVIT, Backbone):
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
            patch_sizes=patch_sizes,
            n_layers=depths,
            d_model=embed_dim,
            n_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=drop_rate,
            drop_path_rate=drop_path_rate,
            split_ratio=split_ratio,
            channels=in_chans,
            n_scales=n_scales,
            min_patch_size=min_patch_size,
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

    def test_pos_cover_and_overlap(self, pos, im_h, im_w, scale_max):
        print("Testing position cover and overlap in level {}".format(scale_max))
        pos_true = torch.meshgrid(torch.arange(0, im_w), torch.arange(0, im_h), indexing='ij')
        pos_true = torch.stack([pos_true[0], pos_true[1]]).permute(1, 2, 0).view(-1, 2).to(pos.device).half()

        all_pos = []

        for s in range(scale_max + 1):
            n_scale_idx = torch.where(pos[:, 0] == s)
            pos_at_scale = pos[n_scale_idx[0].long(), 1:]
            pos_at_org_scale = pos_at_scale*self.min_patch_size
            patch_size = self.patch_sizes[s]
            new_coords = torch.stack(torch.meshgrid(torch.arange(0, patch_size), torch.arange(0, patch_size)))
            new_coords = new_coords.view(2, -1).permute(1, 0).to(pos.device)
            pos_at_org_scale = pos_at_org_scale.unsqueeze(1) + new_coords
            pos_at_org_scale = pos_at_org_scale.reshape(-1, 2)
            all_pos.append(pos_at_org_scale)

        all_pos = torch.cat(all_pos).half()

        print("Computing cover in level {}".format(scale_max))
        cover = torch.tensor([all(torch.any(i == all_pos, dim=0)) for i in pos_true])
        print("Finished computing cover in level {}".format(scale_max))
        if not all(cover):
            print("Total pos map is not covered in level {}, missing {} positions".format(scale_max, sum(~cover)))
            missing = pos_true[~cover]
            print("Missing positions: {}".format(missing))
        print("Computing duplicates in level {}".format(scale_max))
        dupli_unq, dupli_idx, dupli_counts = torch.unique(all_pos, dim=0, return_counts=True, return_inverse=True)
        if len(dupli_counts) > len(all_pos):
            print("Found {} duplicate posses in level {}".format(sum(dupli_counts > 1), scale_max))
        print("Finished computing duplicates in level {}".format(scale_max))

        return True
