"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import math
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from ..transformer_decoder.position_encoding import PositionEmbeddingSine

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


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


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = rearrange(x.transpose(1, 2), 'b c (h w) -> b c h w', b=B, c=C, h=H, w=W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, dw_conv=False, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        if dw_conv:
            self.dwconv = DWConv(hidden_dim)
        self.dw_conv = dw_conv
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, h, w):
        x = self.fc1(x)
        if self.dw_conv:
            x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, h, w):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        if torch.isnan(attn).any():
            print("NaNs detected in q-k-attn in ViT")
        attn = attn.softmax(dim=-1)
        if torch.isnan(attn).any():
            print("NaNs detected in softmax-attn in ViT")
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if torch.isnan(attn).any():
            print("NaNs detected in v-attn in ViT")
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""

    def __init__(self, *args, rope_theta=10.0, rope_mixed=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.rope_mixed = rope_mixed

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

            freqs = init_2d_freqs(
                dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta,
                rotate=True
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs, requires_grad=True)

            t_x, t_y = init_t_xy(end_x=14, end_y=14)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=self.dim // self.num_heads, theta=rope_theta)
            freqs_cis = self.compute_cis(end_x=14, end_y=14)
            self.freqs_cis = freqs_cis

    def forward(self, x, h, w):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ###### Apply rotary position embedding
        #w = h = math.sqrt(x.shape[1] - 1)
        if self.rope_mixed:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            if self.freqs_t_x.shape[0] != x.shape[1]:
                t_x, t_y = init_t_xy(end_x=w, end_y=h)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        else:
            freqs_cis = self.freqs_cis
            if self.freqs_cis.shape[0] != x.shape[1]:
                freqs_cis = self.compute_cis(end_x=w, end_y=h)
            freqs_cis = freqs_cis.to(x.device)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        #########

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, layer_scale=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # layer_scale code copied from https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/a2cfef599fffd36d058a5a4cfdbd81c008e1c349/classification/nat.py
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float] and layer_scale > 0:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, h, w):
        y = self.attn(self.norm1(x), h, w)
        if not self.layer_scale:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x), h, w))
        else:
            x = x + self.drop_path(self.gamma1 * y)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), h, w))
        if torch.isnan(x).any():
            print("NaNs detected in ff-attn in ViT")
        return x

class DownSampleConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.g_norm = nn.GroupNorm(1, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if torch.isnan(x).any():
            print("NaNs detected after conv in PE in first ViT")
        x = self.relu(x)
        if torch.isnan(x).any():
            print("NaNs detected after relu in PE in first ViT")
        x = self.g_norm(x)
        if torch.isnan(x).any():
            print("NaNs detected after group_norm in PE in first ViT")

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
            drop_path_rate=[0.0],
            layer_scale=0.0
    ):
        super().__init__()

        # transformer blocks
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, dim_ff, dropout, drop_path_rate[i], layer_scale) for i in range(n_blocks)]
        )

    def forward(self, x, h, w):
        for blk_idx in range(len(self.blocks)):
            x = self.blocks[blk_idx](x, h, w)
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
            drop_path_rate=[0.0],
            channels=3,
            split_ratio=4,
            n_scales=2,
            min_patch_size=4,
            upscale_ratio=0.0,
            first_layer=True,
            layer_scale=0.0
    ):
        super().__init__()
        self.patch_size = patch_sizes[-1]
        self.patch_sizes = patch_sizes

        self.patch_size = self.patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_scales = n_scales
        self.min_patch_size = min_patch_size
        self.upscale_ratio = upscale_ratio
        self.first_layer = first_layer

        num_features = d_model
        self.num_features = num_features

        if self.first_layer:
            # Pos Embs
            self.pe_layer = PositionEmbeddingSine(d_model // 2, normalize=True)
            self.patch_embed = OverlapPatchEmbedding(
                self.patch_size,
                d_model,
                channels,
            )
        else:
            self.token_norm = nn.LayerNorm(channels)
            if channels != d_model:
                self.token_projection = nn.Linear(channels, d_model)
            else:
                self.token_projection = nn.Identity()
        dim_ff = int(d_model * mlp_ratio)
        # transformer layers
        self.layers = TransformerLayer(n_layers, d_model, n_heads, dim_ff, dropout, drop_path_rate, layer_scale)

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
        if torch.isnan(im).any():
            print("NaNs detected in input image in ViT in scale {}".format(scale))
        PS = self.patch_size
        patched_im_size = (H // PS, W // PS)
        min_patched_im_size = (H // self.min_patch_size, W // self.min_patch_size)

        if self.first_layer:
            x = self.patch_embed(im)
            if torch.isnan(x).any():
                print("NaNs detected in patch-embedded features in ViT in scale {}".format(scale))
            pos = get_2dpos_of_curr_ps_in_min_ps(H, W, PS, self.min_patch_size, scale).to('cuda')
            pos = pos.repeat(B, 1, 1)
            #print("Encoder pos max x: {}, max y: {}, and all pos: {}".format(pos[:, :, 0].max(), pos[:, :, 1].max(), pos))
            #self.test_pos_cover_and_overlap(pos[0], H, W, scale)
            pos_embed = self.pe_layer(pos[:,:,1:])
            x = x + pos_embed
            if torch.isnan(x).any():
                print("NaNs detected in pos-embedded features in ViT in scale {}".format(scale))
        else:
            features = self.token_norm(features)
            x = self.token_projection(features)
            pos = features_pos
            if torch.isnan(x).any():
                print("NaNs detected in projected features in ViT in scale {}".format(scale))

        x = self.layers(x, h=patched_im_size[0], w=patched_im_size[1])

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
        print("Building MixResViT model...")
        if layer_index == 0:
            in_chans = 3
            first_layer = True
        else:
            in_chans = cfg.MODEL.MR.EMBED_DIM[layer_index - 1]
            first_layer = False
        n_scales = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES
        min_patch_size = cfg.MODEL.MR.PATCH_SIZES[n_scales - 1]
        n_layers = len(cfg.MODEL.MR.EMBED_DIM)
        if layer_index >= n_scales:
            scale = n_layers - layer_index - 1
            patch_sizes = cfg.MODEL.MR.PATCH_SIZES[layer_index:]
            down = True
            in_chans = sum(cfg.MODEL.MR.EMBED_DIM[-(layer_index+1):-(n_layers - layer_index)])
        else:
            scale = layer_index
            patch_sizes = cfg.MODEL.MR.PATCH_SIZES[:layer_index + 1]
            down = False
        embed_dim = cfg.MODEL.MR.EMBED_DIM[layer_index]
        depths = cfg.MODEL.MR.DEPTHS[layer_index]
        mlp_ratio = cfg.MODEL.MR.MLP_RATIO[layer_index]
        num_heads = cfg.MODEL.MR.NUM_HEADS[layer_index]
        drop_rate = cfg.MODEL.MR.DROP_RATE[layer_index]
        split_ratio = cfg.MODEL.MR.SPLIT_RATIO[layer_index]
        upscale_ratio = cfg.MODEL.MR.UPSCALE_RATIO[layer_index]
        layer_scale = cfg.MODEL.MR.LAYER_SCALE

        drop_path_rate = cfg.MODEL.MR.DROP_PATH_RATE
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(cfg.MODEL.MR.DEPTHS))]
        drop_path = dpr[sum(cfg.MODEL.MR.DEPTHS[:layer_index]):sum(cfg.MODEL.MR.DEPTHS[:layer_index + 1])]

        super().__init__(
            patch_sizes=patch_sizes,
            n_layers=depths,
            d_model=embed_dim,
            n_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=drop_rate,
            drop_path_rate=drop_path,
            split_ratio=split_ratio,
            channels=in_chans,
            n_scales=n_scales,
            min_patch_size=min_patch_size,
            upscale_ratio=upscale_ratio,
            first_layer=first_layer,
            layer_scale=layer_scale
        )

        if down:
            self._out_features = cfg.MODEL.MR.OUT_FEATURES[-(n_layers - layer_index):]
            self._in_features_channels = in_chans
            self._out_feature_strides = {"res{}".format(n_scales + 1 - i): cfg.MODEL.MR.PATCH_SIZES[i] for i in
                                         range(n_layers - layer_index)}
            self._out_feature_channels = {"res{}".format(n_scales + 1 - i): embed_dim for i in
                                          range(n_layers - layer_index)}
        else:
            self._out_features = cfg.MODEL.MR.OUT_FEATURES[-(layer_index+1):]
            out_index = (n_scales - 1) + 2
            self._out_feature_strides = {"res{}".format(out_index): cfg.MODEL.MR.PATCH_SIZES[layer_index]}
            # self._out_feature_strides = {"res{}".format(i + 2): cfg.MODEL.MRML.PATCH_SIZES[-1] for i in range(num_scales)}
            # print("backbone strides: {}".format(self._out_feature_strides))
            # self._out_feature_channels = { "res{}".format(i+2): list(reversed(self.num_features))[i] for i in range(num_scales)}
            self._out_feature_channels = {"res{}".format(out_index): embed_dim}
            # print("backbone channels: {}".format(self._out_feature_channels))
            self._in_features_channels = in_chans

        print("Successfully built MixResViT model with {} out_features, {} strides, {} out channels and {} in channels".format(
            self._out_features, self._out_feature_strides, self._out_feature_channels, self._in_features_channels))


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
