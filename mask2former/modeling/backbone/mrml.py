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
    patches_coords = torch.meshgrid(torch.arange(0, height // min_patch_size, patch_size // min_patch_size),
                                    torch.arange(0, width // min_patch_size, patch_size // min_patch_size),
                                    indexing='ij')
    patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
    patches_coords = patches_coords.permute(1, 2, 0)
    patches_coords = patches_coords.view(-1, 2)
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

    def forward(self, x, mask=None):
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

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DownSampleConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        #self.instance_norm = nn.InstanceNorm2d(out_dim, affine=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        #x = self.instance_norm(x)

        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        n_layers = int(torch.log2(torch.tensor([patch_size])).item())
        conv_layers = []
        emb_dim_list = [channels] + [embed_dim] * (n_layers - 1)
        for i in range(n_layers):
            conv = DownSampleConvBlock(emb_dim_list[i], embed_dim)
            conv_layers.append(conv)
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.conv_layers(im).flatten(2).transpose(1, 2)
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


class MRML(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            n_layers,
            d_model,
            n_heads,
            dropout=0.0,
            drop_path_rate=0.0,
            channels=1,
            split_ratio=4,
            n_scales=2,
            upscale_ratio=0.25
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbedding(
            image_size,
            patch_size,
            d_model[0],
            channels,
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.split_ratio = split_ratio
        self.n_scales = n_scales
        self.upscale_ratio = upscale_ratio
        self.min_patch_size = patch_size // (2 ** (n_scales - 1))

        num_features = d_model
        self.num_features = num_features

        # Pos Embs
        self.rel_pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.split_ratio, d_model[i])) for i in range(n_scales - 1)])
        self.pe_layer = PositionEmbeddingSine(d_model[0] // 2, normalize=True)
        #self.scale_embs = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model[i])) for i in range(n_scales - 1)])

        # transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(n_layers[i], d_model[i], n_heads[i], d_model[i] * 4, dropout, drop_path_rate) for i in
             range(len(n_layers))]
        )

        # Downsamplers
        self.downsamplers = nn.ModuleList([nn.Linear(d_model[i], d_model[i + 1]) for i in range(n_scales - 1)])

        # Split layers
        self.splits = nn.ModuleList(
            [nn.Linear(d_model[i], d_model[i] * self.split_ratio) for i in range(n_scales - 1)]
        )

        # Metaloss predictions
        self.metalosses = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model[i], d_model[i]),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model[i]),
            nn.Linear(d_model[i], 1)) for i in range(n_scales - 1)])

        self.high_res_patchers = nn.ModuleList(
            [nn.Conv2d(channels, d_model[i - 1], kernel_size=patch_size // (2 ** i), stride=patch_size // (2 ** i)) for
             i in
             range(1, len(n_layers))])

        #nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def divide_tokens_to_split_and_keep(self, tokens_at_curr_scale, patches_scale_coords_curr_scale, curr_scale):
        k_split = int(tokens_at_curr_scale.shape[1] * self.upscale_ratio)
        k_keep = int(tokens_at_curr_scale.shape[1] - k_split)
        pred_meta_loss = self.metalosses[curr_scale](tokens_at_curr_scale.detach()).squeeze(2)
        tkv, tki = torch.topk(pred_meta_loss, k=k_split, dim=1, sorted=False)
        bkv, bki = torch.topk(pred_meta_loss, k=k_keep, dim=1, sorted=False, largest=False)

        batch_indices_k = torch.arange(tokens_at_curr_scale.shape[0]).unsqueeze(1).repeat(1, k_keep)
        batch_indices_s = torch.arange(tokens_at_curr_scale.shape[0]).unsqueeze(1).repeat(1, k_split)

        tokens_to_keep = tokens_at_curr_scale[batch_indices_k, bki]
        tokens_to_split = tokens_at_curr_scale[batch_indices_s, tki]
        coords_to_keep = patches_scale_coords_curr_scale[batch_indices_k, bki]
        coords_to_split = patches_scale_coords_curr_scale[batch_indices_s, tki]

        return tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep, pred_meta_loss

    def divide_tokens_coords_on_scale(self, tokens, patches_scale_coords, curr_scale):
        B, _, _ = tokens.shape
        b_scale_idx, n_scale_idx = torch.where(patches_scale_coords[:, :, 0] == curr_scale)
        coords_at_curr_scale = patches_scale_coords[b_scale_idx, n_scale_idx, :]
        coords_at_curr_scale = rearrange(coords_at_curr_scale, '(b n) p -> b n p', b=B).contiguous()
        tokens_at_curr_scale = tokens[b_scale_idx, n_scale_idx, :]
        tokens_at_curr_scale = rearrange(tokens_at_curr_scale, '(b n) c -> b n c', b=B).contiguous()

        b_scale_idx, n_scale_idx = torch.where(patches_scale_coords[:, :, 0] != curr_scale)
        coords_at_older_scales = patches_scale_coords[b_scale_idx, n_scale_idx, :]
        coords_at_older_scales = rearrange(coords_at_older_scales, '(b n) p -> b n p', b=B).contiguous()
        tokens_at_older_scale = tokens[b_scale_idx, n_scale_idx, :]
        tokens_at_older_scale = rearrange(tokens_at_older_scale, '(b n) c -> b n c', b=B).contiguous()

        return tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales

    def split_tokens(self, tokens_to_split, curr_scale):
        x_splitted = self.splits[curr_scale](tokens_to_split)
        x_splitted = rearrange(x_splitted, 'b n (s d) -> b n s d', s=self.split_ratio).contiguous()
        x_splitted = x_splitted + self.scale_embs[curr_scale]# + self.rel_pos_embs[curr_scale]
        x_splitted = rearrange(x_splitted, 'b n s d -> b (n s) d', s=self.split_ratio).contiguous()
        return x_splitted

    def split_coords(self, coords_to_split, patch_size, curr_scale):
        batch_size = coords_to_split.shape[0]
        new_scale = curr_scale + 1
        new_coord_ratio = 2 ** (self.n_scales - new_scale - 1)
        a = torch.stack([coords_to_split[:, :, 1], coords_to_split[:, :, 2]], dim=2)
        b = torch.stack([coords_to_split[:, :, 1] + new_coord_ratio, coords_to_split[:, :, 2]], dim=2)
        c = torch.stack([coords_to_split[:, :, 1], coords_to_split[:, :, 2] + new_coord_ratio], dim=2)
        d = torch.stack([coords_to_split[:, :, 1] + new_coord_ratio, coords_to_split[:, :, 2] + new_coord_ratio], dim=2)

        new_coords_2dim = torch.stack([a, b, c, d], dim=2)
        new_coords_2dim = rearrange(new_coords_2dim, 'b n s c -> b (n s) c', s=self.split_ratio, c=2).contiguous()

        scale_lvl = torch.tensor([new_scale] * new_coords_2dim.shape[1])
        scale_lvl = scale_lvl.repeat(batch_size, 1)
        scale_lvl = scale_lvl.to('cuda').int().unsqueeze(2)
        patches_scale_coords = torch.cat([scale_lvl, new_coords_2dim], dim=2)

        return patches_scale_coords

    def add_high_res_feat(self, tokens, coords, curr_scale, image):
        patched_im = self.high_res_patchers[curr_scale](image)
        b = torch.arange(coords.shape[0]).unsqueeze(-1).expand(-1, coords.shape[1])
        x = torch.div(coords[..., 0], 2 ** (self.n_scales - curr_scale - 2), rounding_mode='trunc')
        y = torch.div(coords[..., 1], 2 ** (self.n_scales - curr_scale - 2), rounding_mode='trunc')
        patched_im = patched_im[b, :, x, y]
        tokens = tokens + patched_im

        return tokens

    def split_input(self, tokens, patches_scale_coords, curr_scale, patch_size, im):
        tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales = self.divide_tokens_coords_on_scale(
            tokens, patches_scale_coords, curr_scale)
        meta_loss_coords = coords_at_curr_scale[:, :, 1:]
        tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep, pred_meta_loss = self.divide_tokens_to_split_and_keep(
            tokens_at_curr_scale, coords_at_curr_scale, curr_scale)
        tokens_after_split = self.split_tokens(tokens_to_split, curr_scale)
        coords_after_split = self.split_coords(coords_to_split, patch_size, curr_scale)

        tokens_after_split = self.add_high_res_feat(tokens_after_split, coords_after_split[:, :, 1:], curr_scale, im)

        all_tokens = torch.cat([tokens_at_older_scale, tokens_to_keep, tokens_after_split], dim=1)
        all_coords = torch.cat([coords_at_older_scales, coords_to_keep, coords_after_split], dim=1)

        return all_tokens, all_coords, pred_meta_loss, meta_loss_coords

    def forward(self, im):
        B, _, H, W = im.shape
        PS = self.patch_size
        x = self.patch_embed(im)
        patched_im_size = (H // PS, W // PS)
        org_patched_im_size = patched_im_size
        patches_scale_coords = get_2dpos_of_curr_ps_in_min_ps(H, W, PS, self.min_patch_size, 0).to('cuda')
        patches_scale_coords = patches_scale_coords.repeat(B, 1, 1)
        pos_embed = self.pe_layer(patches_scale_coords[:,:,1:])
        x = x + pos_embed
        outs = {}
        for l_idx in range(len(self.layers)):
            out_idx = self.n_scales - l_idx + 1
            x = self.layers[l_idx](x)
            outs["res{}_spatial_shape".format(out_idx)] = patched_im_size
            if l_idx < self.n_scales - 1:
                x, patches_scale_coords, meta_loss, meta_loss_coord = self.split_input(x, patches_scale_coords, l_idx,
                                                                                       patched_im_size[0], im)
                PS /= 2
                patched_im_size = (H // PS, W // PS)
                x = self.downsamplers[l_idx](x)
                outs["metaloss{}".format(l_idx)] = meta_loss
                outs["metaloss{}_pos".format(l_idx)] = meta_loss_coord

        for s in range(self.n_scales):
            out_idx = self.n_scales - s + 1
            b_scale_idx, n_scale_idx = torch.where(patches_scale_coords[:,:,0] == s)
            pos_scale = patches_scale_coords[b_scale_idx, n_scale_idx, 1:]
            pos_scale = rearrange(pos_scale, '(b n) p -> b n p', b=B).contiguous()
            out_scale = x[b_scale_idx, n_scale_idx, :]
            out_scale = rearrange(out_scale, '(b n) c -> b n c', b=B).contiguous()
            outs["res{}".format(out_idx)] = out_scale
            outs["res{}_pos".format(out_idx)] = pos_scale
            #outs["res{}_spatial_shape".format(out_idx)] = org_patched_im_size
        '''
        for k, v in outs.items():
            if "spatial_shape" in k:
                print("AFF Model - Key: {}, Value: {}".format(k, v))
            else:
                print("AFF Model -  Key: {}, Value shape: {}, Value min: {}, Value max: {}".format(k, v.shape, v.min(),
                                                                                            v.max()))
        '''
        return outs


@BACKBONE_REGISTRY.register()
class MixResMetaLoss(MRML, Backbone):
    def __init__(self, cfg, input_shape):

        in_chans = 3
        initial_patch_size = cfg.MODEL.MRML.PATCH_SIZES[0]
        embed_dim = cfg.MODEL.MRML.EMBED_DIM
        depths = cfg.MODEL.MRML.DEPTHS
        num_heads = cfg.MODEL.MRML.NUM_HEADS
        drop_rate = cfg.MODEL.MRML.DROP_RATE
        drop_path_rate = cfg.MODEL.MRML.DROP_PATH_RATE
        split_ratio = cfg.MODEL.MRML.SPLIT_RATIO
        num_scales = cfg.MODEL.MRML.NUM_SCALES
        image_size = cfg.INPUT.CROP.SIZE
        upscale_ratio = cfg.MODEL.MRML.UPSCALE_RATIO

        super().__init__(
            image_size=image_size,
            patch_size=initial_patch_size,
            n_layers=depths,
            d_model=embed_dim,
            n_heads=num_heads,
            dropout=drop_rate,
            drop_path_rate=drop_path_rate,
            split_ratio=split_ratio,
            n_scales=num_scales,
            channels=in_chans,
            upscale_ratio=upscale_ratio
        )

        self._out_features = cfg.MODEL.MRML.OUT_FEATURES

        self._out_feature_strides = { "res{}".format(i+2): list(reversed(cfg.MODEL.MRML.PATCH_SIZES))[i] for i in range(num_scales)}
        #print("backbone strides: {}".format(self._out_feature_strides))
        #self._out_feature_channels = { "res{}".format(i+2): list(reversed(self.num_features))[i] for i in range(num_scales)}
        self._out_feature_channels = {"res{}".format(i + 2): self.num_features[-1] for i in range(num_scales)}
        #print("backbone channels: {}".format(self._out_feature_channels))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B,C,H,W)
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"MRML takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        y = super().forward(x)
        return y

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }
