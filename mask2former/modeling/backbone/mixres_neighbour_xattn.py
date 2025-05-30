"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import math
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from einops import rearrange
from ..transformer_decoder.position_encoding import PositionEmbeddingSine

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from ..clusten import CLUSTENQKFunction, CLUSTENAVFunction
from .point_utils import knn_keops, space_filling_cluster

# assumes largest input resolution is 2048 x 2048
rel_pos_width = 2048 // 4 - 1
table_width = 2 * rel_pos_width + 1

pre_hs = torch.arange(table_width).float()-rel_pos_width
pre_ws = torch.arange(table_width).float()-rel_pos_width
pre_ys, pre_xs = torch.meshgrid(pre_hs, pre_ws)  # table_width x table_width

# expanded relative position lookup table
dis_table = (pre_ys**2 + pre_xs**2) ** 0.5
sin_table = pre_ys / dis_table
cos_table = pre_xs / dis_table
pre_table = torch.stack([pre_xs, pre_ys, dis_table, sin_table, cos_table], dim=2)  # table_width x table_width x 5
pre_table[torch.bitwise_or(pre_table.isnan(), pre_table.isinf()).nonzero(as_tuple=True)] = 0
pre_table = pre_table.reshape(-1, 5)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



def divide_tensor_on_scale(token_tensor, pos_tensor, scale):
    B, _, _ = token_tensor.shape
    b_scale_idx, n_scale_idx = torch.where(pos_tensor[:, :, 0] == scale)
    tokens_at_curr_scale = token_tensor[b_scale_idx, n_scale_idx, :]
    tokens_at_curr_scale = rearrange(tokens_at_curr_scale, '(b n) c -> b n c', b=B).contiguous()

    b_scale_idx, n_scale_idx = torch.where(pos_tensor[:, :, 0] != scale)
    tokens_at_older_scale = token_tensor[b_scale_idx, n_scale_idx, :]
    tokens_at_older_scale = rearrange(tokens_at_older_scale, '(b n) c -> b n c', b=B).contiguous()

    return tokens_at_curr_scale, tokens_at_older_scale

def get_2dpos_of_ps(height, width, patch_size):
    patches_coords = torch.meshgrid(torch.arange(0, width // patch_size),
                                    torch.arange(0, height // patch_size),
                                    indexing='ij')
    patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
    patches_coords = patches_coords.permute(1, 2, 0)
    patches_coords = patches_coords.view(-1, 2)
    return patches_coords


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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClusterAttention(nn.Module):
    """
    Performs local attention on nearest clusters

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2*dim)
        self.softmax = nn.Softmax(dim=-1)

        self.blank_k = nn.Parameter(torch.randn(dim))
        self.blank_v = nn.Parameter(torch.randn(dim))

        self.pos_embed = nn.Linear(self.pos_dim+3, num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn):
        """
        Args:
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            global_attn - bool, whether to perform global attention
        """

        b, n, c = feat.shape
        c_ = c // self.num_heads
        d = self.pos_dim
        assert c == self.dim, "dim does not accord to input"
        h = self.num_heads

        # get qkv
        q = self.q(feat)  # b x n x c
        q = q * self.scale
        kv = self.kv(feat)  # b x n x 2c

        # get attention
        if not global_attn:
            nbhd_size = member_idx.shape[-1]
            m = nbhd_size
            q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)
            kv = kv.view(b, n, h, 2, c_).permute(3, 0, 2, 1, 4)  # 2 x b x h x n x c_
            key, v = kv[0], kv[1]
            attn = CLUSTENQKFunction.apply(q, key, member_idx)  # b x h x n x m
            mask = cluster_mask
            if mask is not None:
                mask = mask.reshape(b, 1, n, m)
        else:
            q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)  # b x h x n x c_
            kv = kv.view(b, n, h, 2, c_).permute(3, 0, 2, 1, 4)  # 2 x b x h x n x c_
            key, v = kv[0], kv[1]
            attn = q @ key.transpose(-1, -2)  # b x h x n x n
            mask = None

        # position embedding
        global pre_table
        if not pre_table.is_cuda:
            pre_table = pre_table.to(pe_idx.device)
        pe_table = self.pos_embed(pre_table)  # 111 x 111 x h

        pe_shape = pe_idx.shape
        pos_embed = pe_table.gather(index=pe_idx.view(-1, 1).expand(-1, h), dim=0).reshape(*(pe_shape), h).permute(0, 3, 1, 2)

        attn = attn + pos_embed

        if mask is not None:
            attn = attn + (1-mask)*(-100)

        # blank token
        blank_attn = (q * self.blank_k.reshape(1, h, 1, c_)).sum(-1, keepdim=True)  # b x h x n x 1
        attn = torch.cat([attn, blank_attn], dim=-1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        blank_attn = attn[..., -1:]
        attn = attn[..., :-1]
        blank_v = blank_attn * self.blank_v.reshape(1, h, 1, c_)  # b x h x n x c_

        # aggregate v
        if global_attn:
            feat = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, c)
            feat = feat + blank_v.permute(0, 2, 1, 3).reshape(b, n, c)
        else:
            feat = CLUSTENAVFunction.apply(attn, v, member_idx).permute(0, 2, 1, 3).reshape(b, n, c)
            feat = feat + blank_v.permute(0, 2, 1, 3).reshape(b, n, c)

        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        return feat

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'



class CrossAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=True):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)



class ClusterXATransformerBlock(nn.Module):
    r""" Cluster Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0., layer_scale=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.x_attn = CrossAttention(dim, num_heads)
        N_steps = dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.c_attn = ClusterAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # layer_scale code copied from https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/a2cfef599fffd36d058a5a4cfdbd81c008e1c349/classification/nat.py
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float] and layer_scale > 0:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, feat, image_feat, feat_pos, member_idx, cluster_mask, pe_idx, global_attn, scale):
        """
        Args:
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            global_attn - bool, whether to perform global attention
        """

        b, n, c = feat.shape
        assert c == self.dim, "dim does not accord to input"



        # Divide all tokens into high-res and low-res to keep order
        feat_high, feat_low = divide_tensor_on_scale(feat, feat_pos, scale)

        # Reorder to ensure order is consistent
        member_idx_high, member_idx_low = divide_tensor_on_scale(member_idx, feat_pos, scale)
        member_idx = torch.cat([member_idx_low, member_idx_high], dim=1)
        if cluster_mask is not None:
            cluster_mask_high, cluster_mask_low = divide_tensor_on_scale(cluster_mask, feat_pos, scale)
            cluster_mask = torch.cat([cluster_mask_low, cluster_mask_high], dim=1)
        pe_idx_high, pe_idx_low = divide_tensor_on_scale(pe_idx, feat_pos, scale)
        pe_idx = torch.cat([pe_idx_low, pe_idx_high], dim=1)
        feat_pos_high, feat_pos_low = divide_tensor_on_scale(feat_pos, feat_pos, scale)
        feat_pos = torch.cat([feat_pos_low, feat_pos_high], dim=1)

        # Do cross attention between image features and high-res features
        feat_high = self.x_attn(feat_high, image_feat, memory_mask=None, memory_key_padding_mask=None, pos=None,
                           query_pos=None)

        # Return all tokens again in identical order.
        feat = torch.cat([feat_low, feat_high], dim=1)


        shortcut = feat
        feat = self.norm1(feat)

        # cluster attention
        feat = self.c_attn(feat=feat,
                         member_idx=member_idx,
                         cluster_mask=cluster_mask,
                         pe_idx=pe_idx,
                         global_attn=global_attn)

        # FFN
        if not self.layer_scale:
            feat = shortcut + self.drop_path(feat)
            feat_mlp = self.mlp(self.norm2(feat))
            feat = feat + self.drop_path(feat_mlp)
        else:
            feat = shortcut + self.drop_path(self.gamma1 * feat)
            feat_mlp = self.mlp(self.norm2(feat))
            feat = feat + self.drop_path(self.gamma2 * feat_mlp)

        return feat, feat_pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"

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

    def forward(self, im):
        x = self.conv_layers(im)
        return x


class BasicLayer(nn.Module):
    """ AutoFocusFormer layer for one stage.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        cluster_size (int): Cluster size.
        nbhd_size (int): Neighbor size. If larger than or equal to number of tokens, perform global attention;
                            otherwise, rounded to the nearest multiples of cluster_size.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        layer_scale (float, optional): Layer scale initial parameter. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, cluster_size, nbhd_size,
                 depth, num_heads, mlp_ratio,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale=0.0):

        super().__init__()
        self.dim = dim
        self.nbhd_size = nbhd_size
        self.cluster_size = cluster_size
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            ClusterXATransformerBlock(dim=dim,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    layer_scale=layer_scale,
                                    norm_layer=norm_layer)
            for i in range(depth)])


        # cache the clustering result for the first feature map since it is on grid
        self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = None, None, None, None, None

    def forward(self, pos, feat, image_feat, h, w, on_grid, scale):
        """
        Args:
            pos - b x n x 2, token positions
            feat - b x n x c, token features
            h,w - max height and width of token positions
            on_grid - bool, whether the tokens are still on grid; True for the first feature map
            stride - int, "stride" of the current token set; starts with 2, then doubles in each stage
        """
        pos_scale = pos[:, :, 0]
        pos = pos[:, :, 1:]
        b, n, d = pos.shape
        if not isinstance(b, int):
            b, n, d = b.item(), n.item(), d.item()  # make the flop analyzer happy
        c = feat.shape[2]
        assert self.cluster_size > 0, 'self.cluster_size must be positive'

        if self.nbhd_size >= n:
            global_attn = True
            member_idx, cluster_mask = None, None
        else:
            global_attn = False
            k = int(math.ceil(n / float(self.cluster_size)))  # number of clusters
            nnc = min(int(round(self.nbhd_size / float(self.cluster_size))), k)  # number of nearest clusters
            nbhd_size = self.cluster_size * nnc
            self.nbhd_size = nbhd_size  # if not global attention, then nbhd size is rounded to nearest multiples of cluster


        if global_attn:
            rel_pos = (pos[:, None, :, :]+rel_pos_width) - pos[:, :, None, :]  # b x n x n x d
        else:
            if k == n:
                cluster_mean_pos = pos
                member_idx = torch.arange(n, device=feat.device).long().reshape(1, n, 1).expand(b, -1, -1)  # b x n x 1
                cluster_mask = None
            else:
                if on_grid and self.training:
                    if self.cluster_mean_pos is None:
                        self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = space_filling_cluster(pos, self.cluster_size, h, w, no_reorder=False)
                    pos, cluster_mean_pos, member_idx, cluster_mask = self.pos[:b], self.cluster_mean_pos[:b], self.member_idx[:b], self.cluster_mask
                    feat = feat[torch.arange(b).to(feat.device).repeat_interleave(n), self.reorder[:b].view(-1)].reshape(b, n, c)
                    pos_scale = pos_scale[torch.arange(b).to(pos_scale.device).repeat_interleave(n), self.reorder[:b].view(-1)].reshape(b, n, 1)
                    if cluster_mask is not None:
                        cluster_mask = cluster_mask[:b]
                else:
                    pos, cluster_mean_pos, member_idx, cluster_mask, reorder = space_filling_cluster(pos, self.cluster_size, h, w, no_reorder=False)
                    feat = feat[torch.arange(b).to(feat.device).repeat_interleave(n), reorder.view(-1)].reshape(b, n, c)
                    pos_scale = pos_scale[torch.arange(b).to(pos_scale.device).repeat_interleave(n), reorder.view(-1)].reshape(b, n, 1)

            assert member_idx.shape[1] == k and member_idx.shape[2] == self.cluster_size, "member_idx shape incorrect!"

            nearest_cluster = knn_keops(pos, cluster_mean_pos, nnc)  # b x n x nnc

            m = self.cluster_size
            member_idx = member_idx.gather(index=nearest_cluster.view(b, -1, 1).expand(-1, -1, m), dim=1).reshape(b, n, nbhd_size)  # b x n x nnc*m
            if cluster_mask is not None:
                cluster_mask = cluster_mask.gather(index=nearest_cluster.view(b, -1, 1).expand(-1, -1, m), dim=1).reshape(b, n, nbhd_size)
            pos_ = pos.gather(index=member_idx.view(b, -1, 1).expand(-1, -1, d), dim=1).reshape(b, n, nbhd_size, d)
            rel_pos = pos_ - (pos.unsqueeze(2)-rel_pos_width)  # b x n x nbhd_size x d

        rel_pos = rel_pos.clamp(0, table_width-1)
        pe_idx = (rel_pos[..., 1] * table_width + rel_pos[..., 0]).long()

        pos = torch.cat([pos_scale, pos], dim=2)

        for i_blk in range(len(self.blocks)):
            blk = self.blocks[i_blk]
            feat, pos = blk(feat=feat,
                       image_feat=image_feat,
                       feat_pos=pos,
                       member_idx=member_idx,
                       cluster_mask=cluster_mask,
                       pe_idx=pe_idx,
                       global_attn=global_attn,
                       scale=scale)

        return pos, feat

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class MRNBXA(nn.Module):
    def __init__(
            self,
            image_size,
            patch_sizes,
            n_layers,
            d_model,
            n_heads,
            dropout=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            channels=1,
            mlp_ratio=4.0,
            split_ratio=4,
            n_scales=4,
            cluster_size=8,
            nbhd_size=[48, 48, 48, 48],
            layer_scale=0.0,
            min_patch_size=4,
            upscale_ratio=0.25
    ):
        super().__init__()
        self.patch_size = patch_sizes[-1]
        self.image_size = image_size
        self.patch_sizes = patch_sizes
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.mlp_ratio = mlp_ratio
        self.split_ratio = split_ratio
        self.n_scales = n_scales
        self.min_patch_size = min_patch_size
        self.cluster_size = cluster_size,
        self.nbhd_size = nbhd_size
        self.upscale_ratio = upscale_ratio

        num_features = d_model
        self.num_features = num_features

        # Pos Embs
        self.pe_layer = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.scale_emb = nn.Parameter(torch.randn(1, 1, d_model))

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        norm_layer = nn.LayerNorm


        # transformer layers
        self.layers = BasicLayer(dim=int(d_model),
                               cluster_size=cluster_size,
                               nbhd_size=nbhd_size,
                               depth=n_layers,
                               num_heads=n_heads,
                               mlp_ratio=mlp_ratio,
                               drop=dropout,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr,
                               norm_layer=norm_layer,
                               layer_scale=layer_scale,
                               )

        self.image_patch_projection = nn.Linear(3 * (self.patch_size**2), d_model)
        self.high_res_norm1 = nn.LayerNorm(d_model)
        self.high_res_mlp = Mlp(in_features=d_model, out_features=d_model, hidden_features=d_model, act_layer=nn.LeakyReLU)

        self.token_projection = nn.Linear(channels, d_model)

        '''
        # add a norm layer for each output
        for i_layer in range(len(n_layers)):
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
        '''
        self.norm_out = nn.LayerNorm(d_model)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)


    def divide_tokens_to_split_and_keep(self, feat_at_curr_scale, pos_at_curr_scale, upsampling_mask):
        B, N, C = feat_at_curr_scale.shape
        k_split = int(feat_at_curr_scale.shape[1] * self.upscale_ratio)
        k_bottom = 0 #k_split // 2
        k_top = k_split #- k_bottom
        k_keep = int(feat_at_curr_scale.shape[1] - k_split)

        sorted_scores, sorted_indices = torch.sort(upsampling_mask, dim=1, descending=False)

        #bottom_indices = sorted_indices[:, :k_bottom]
        mid_indices = sorted_indices[:, k_bottom:-k_top]
        top_indices = sorted_indices[:, -k_top:]
        #bot_top_indices = torch.cat((bottom_indices, top_indices), dim=1)

        tokens_to_split = feat_at_curr_scale.gather(dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, C))
        tokens_to_keep = feat_at_curr_scale.gather(dim=1, index=mid_indices.unsqueeze(-1).expand(-1, -1, C))

        coords_to_split = pos_at_curr_scale.gather(dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, 3))
        coords_to_keep = pos_at_curr_scale.gather(dim=1, index=mid_indices.unsqueeze(-1).expand(-1, -1, 3))

        return tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep


    def divide_feat_pos_on_scale(self, tokens, patches_scale_coords, curr_scale, upsampling_mask):
        B, _, _ = tokens.shape
        b_scale_idx, n_scale_idx = torch.where(patches_scale_coords[:, :, 0] == curr_scale)
        coords_at_curr_scale = patches_scale_coords[b_scale_idx, n_scale_idx, :]
        coords_at_curr_scale = rearrange(coords_at_curr_scale, '(b n) p -> b n p', b=B).contiguous()
        tokens_at_curr_scale = tokens[b_scale_idx, n_scale_idx, :]
        tokens_at_curr_scale = rearrange(tokens_at_curr_scale, '(b n) c -> b n c', b=B).contiguous()
        upsampling_mask_curr = upsampling_mask[b_scale_idx, n_scale_idx]
        upsampling_mask_curr = rearrange(upsampling_mask_curr, '(b n) -> b n', b=B).contiguous()

        b_scale_idx, n_scale_idx = torch.where(patches_scale_coords[:, :, 0] != curr_scale)
        coords_at_older_scales = patches_scale_coords[b_scale_idx, n_scale_idx, :]
        coords_at_older_scales = rearrange(coords_at_older_scales, '(b n) p -> b n p', b=B).contiguous()
        tokens_at_older_scale = tokens[b_scale_idx, n_scale_idx, :]
        tokens_at_older_scale = rearrange(tokens_at_older_scale, '(b n) c -> b n c', b=B).contiguous()

        return tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales, upsampling_mask_curr


    def split_features(self, tokens_to_split):
        b, n, c = tokens_to_split.shape
        x_splitted = tokens_to_split.unsqueeze(2).expand(b, n, self.split_ratio, c)
        x_splitted = x_splitted + self.scale_emb
        x_splitted = rearrange(x_splitted, 'b n s d -> b (n s) d', s=self.split_ratio).contiguous()
        return x_splitted

    def split_pos(self, pos_to_split, curr_scale):
        batch_size = pos_to_split.shape[0]
        new_coord_ratio = 2 ** (self.n_scales - curr_scale - 1)
        a = torch.stack([pos_to_split[:, :, 1], pos_to_split[:, :, 2]], dim=2)
        b = torch.stack([pos_to_split[:, :, 1] + new_coord_ratio, pos_to_split[:, :, 2]], dim=2)
        c = torch.stack([pos_to_split[:, :, 1], pos_to_split[:, :, 2] + new_coord_ratio], dim=2)
        d = torch.stack([pos_to_split[:, :, 1] + new_coord_ratio, pos_to_split[:, :, 2] + new_coord_ratio], dim=2)

        new_pos_2dim = torch.stack([a, b, c, d], dim=2)
        new_pos_2dim = rearrange(new_pos_2dim, 'b n s c -> b (n s) c', s=self.split_ratio, c=2).contiguous()

        scale_lvl = torch.tensor([curr_scale] * new_pos_2dim.shape[1])
        scale_lvl = scale_lvl.repeat(batch_size, 1)
        scale_lvl = scale_lvl.to(pos_to_split.device).int().unsqueeze(2)
        patches_scale_pos = torch.cat([scale_lvl, new_pos_2dim], dim=2)

        return patches_scale_pos


    def upsample_features(self, im, scale, features, features_pos, upsampling_mask):

        features = self.token_projection(features)

        old_scale = scale - 1
        feat_curr, pos_curr, feat_old, pos_old, upsampling_mask_curr = self.divide_feat_pos_on_scale(
            features, features_pos, old_scale, upsampling_mask)
        feat_to_split, pos_to_split, feat_to_keep, pos_to_keep = self.divide_tokens_to_split_and_keep(
            feat_curr, pos_curr, upsampling_mask_curr)
        feat_after_split = self.split_features(feat_to_split)
        pos_after_split = self.split_pos(pos_to_split, scale)

        pos_emb_split_tokens = self.pe_layer(pos_after_split[:,:,1:])
        feat_after_split = feat_after_split + pos_emb_split_tokens

        image_features = self.get_image_features(im, pos_after_split[:, :, 1:])
        image_features = image_features + pos_emb_split_tokens

        all_feat = torch.cat([feat_old, feat_to_keep, feat_after_split], dim=1)
        all_pos = torch.cat([pos_old, pos_to_keep, pos_after_split], dim=1)

        return all_feat, all_pos, image_features


    def get_image_features(self, im, pos):
        b, n, _ = pos.shape
        patch_coords = torch.stack(torch.meshgrid(torch.arange(0, self.patch_size), torch.arange(0, self.patch_size)))
        patch_coords = patch_coords.permute(1, 2, 0).transpose(0, 1).reshape(-1, 2).to(pos.device)
        patch_coords = patch_coords.repeat(b, 1, 1)
        pos_patches = pos.unsqueeze(2) + patch_coords.unsqueeze(1)
        pos_patches = pos_patches.view(b, -1, 2)
        x_pos = pos_patches[..., 0].long()
        y_pos = pos_patches[..., 1].long()
        b_ = torch.arange(b).unsqueeze(-1).expand(-1, pos_patches.shape[1])
        im_high = im[b_, :, y_pos, x_pos]
        im_high = rearrange(im_high, 'b (n p) c -> b n (p c)', b=b, n=n, c=3)
        im_high_feats = self.image_patch_projection(im_high)
        im_high_feats = self.high_res_norm1(im_high_feats)
        im_high_feats = self.high_res_mlp(im_high_feats)

        return im_high_feats



    def forward(self, im, scale, features, features_pos, upsampling_mask):
        B, _, H, W = im.shape
        PS = self.patch_size
        min_patched_im_size = (H // self.min_patch_size, W // self.min_patch_size)


        x, pos, image_feat = self.upsample_features(im, scale, features, features_pos, upsampling_mask)
        pos, x = self.layers(pos, x, image_feat, h=min_patched_im_size[0], w=min_patched_im_size[1], on_grid=False, scale=scale)
        #success = self.test_pos_cover_and_overlap(pos[0], H, W, scale)
        outs = {}
        for s in range(scale + 1):
            out_idx = self.n_scales - s + 1
            patched_im_size = (H // self.patch_sizes[s], W // self.patch_sizes[s])
            b_scale_idx, n_scale_idx = torch.where(pos[:,:,0] == s)
            pos_scale = pos[b_scale_idx, n_scale_idx, :]
            pos_scale = rearrange(pos_scale, '(b n) p -> b n p', b=B).contiguous()
            out_scale = x[b_scale_idx, n_scale_idx, :]
            out_scale = rearrange(out_scale, '(b n) c -> b n c', b=B).contiguous()
            outs["res{}".format(out_idx)] = self.norm_out(out_scale)
            outs["res{}_pos".format(out_idx)] = pos_scale[:,:,1:]
            outs["res{}_scale".format(out_idx)] = pos_scale[:, :, 0]
            outs["res{}_spatial_shape".format(out_idx)] = patched_im_size
        outs["min_spatial_shape"] = min_patched_im_size
        return outs


@BACKBONE_REGISTRY.register()
class MixResNeighbourXAttn(MRNBXA, Backbone):
    def __init__(self, cfg, layer_index):

        if layer_index == 0:
            in_chans = 3
        else:
            in_chans = cfg.MODEL.MR_SEM_SEG_HEAD.CONVS_DIM[layer_index - 1]
        image_size = cfg.INPUT.CROP.SIZE
        n_scales = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES
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
            upscale_ratio=upscale_ratio
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

    def get_top_disagreement_mask_and_pos(self, dis_mask, dis_mask_pos, level):
        k_top = int(dis_mask.shape[0] * self.mask_predictors[level].backbone.upscale_ratio)
        sorted_scores, sorted_indices = torch.sort(dis_mask, dim=1, descending=False)

        top_indices = sorted_indices[:, -k_top:]

        top_dis_mask = dis_mask.gather(dim=1, index=top_indices)
        top_dis_mask_pos = dis_mask_pos.gather(dim=1, index=top_indices.unsqueeze(-1).expand(-1, 2))

        return top_dis_mask, top_dis_mask_pos




