"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import math
import torch
import torch.nn as nn
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

def get_2dpos_of_curr_ps_in_min_ps(height, width, patch_size, min_patch_size, scale):
    patches_coords = torch.meshgrid(torch.arange(0, width // min_patch_size, patch_size // min_patch_size),
                                    torch.arange(0, height // min_patch_size, patch_size // min_patch_size),
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


class ClusterTransformerBlock(nn.Module):
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
        self.attn = ClusterAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # layer_scale code copied from https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/a2cfef599fffd36d058a5a4cfdbd81c008e1c349/classification/nat.py
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float] and layer_scale > 0:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

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
        assert c == self.dim, "dim does not accord to input"

        shortcut = feat
        feat = self.norm1(feat)

        # cluster attention
        feat = self.attn(feat=feat,
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

        return feat

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"

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
            ClusterTransformerBlock(dim=dim,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    layer_scale=layer_scale,
                                    norm_layer=norm_layer)
            for i in range(depth)])


        # cache the clustering result for the first feature map since it is on grid
        self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = None, None, None, None, None

    def forward(self, pos, feat, h, w, on_grid):
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

        for i_blk in range(len(self.blocks)):
            blk = self.blocks[i_blk]
            feat = blk(feat=feat,
                       member_idx=member_idx,
                       cluster_mask=cluster_mask,
                       pe_idx=pe_idx,
                       global_attn=global_attn)
        pos = torch.cat([pos_scale, pos], dim=2)
        return pos, feat

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


class MRMLNB(nn.Module):
    def __init__(
            self,
            image_size,
            patch_size,
            n_layers,
            d_model,
            n_heads,
            dropout=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            channels=1,
            mlp_ratio=4.0,
            split_ratio=4,
            n_scales=2,
            upscale_ratio=0.25,
            cluster_size=8,
            nbhd_size=[48, 48, 48, 48],
            layer_scale=0.0
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
        self.mlp_ratio = mlp_ratio
        self.split_ratio = split_ratio
        self.n_scales = n_scales
        self.upscale_ratio = upscale_ratio
        self.min_patch_size = patch_size // (2 ** (n_scales - 1))
        self.cluster_size = cluster_size,
        self.nbhd_size = nbhd_size

        num_features = d_model
        self.num_features = num_features

        # Pos Embs
        self.rel_pos_embs = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.split_ratio, d_model[i])) for i in range(n_scales - 1)])
        #self.pe_layer = PositionEmbeddingSine(d_model[0] // 2, normalize=True)
        self.scale_embs = nn.ParameterList([nn.Parameter(torch.randn(1, 1, d_model[i])) for i in range(n_scales - 1)])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(n_layers))]
        norm_layer = nn.LayerNorm


        # transformer layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(n_layers)):
            layer = BasicLayer(dim=int(d_model[i_layer]),
                               cluster_size=cluster_size,
                               nbhd_size=nbhd_size[i_layer],
                               depth=n_layers[i_layer],
                               num_heads=n_heads[i_layer],
                               mlp_ratio=mlp_ratio,
                               drop=dropout,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(n_layers[:i_layer]):sum(n_layers[:i_layer + 1])],
                               norm_layer=norm_layer,
                               layer_scale=layer_scale,
                               )
            self.layers.append(layer)

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

        '''
        # add a norm layer for each output
        for i_layer in range(len(n_layers)):
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
        '''
        self.norm_out = nn.LayerNorm(d_model[-1])
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
        x_splitted = x_splitted + self.rel_pos_embs[curr_scale] + self.scale_embs[curr_scale]
        x_splitted = rearrange(x_splitted, 'b n s d -> b (n s) d', s=self.split_ratio).contiguous()
        return x_splitted

    def split_coords(self, coords_to_split, curr_scale):
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
        scale_lvl = scale_lvl.to(coords_to_split.device).int().unsqueeze(2)
        patches_scale_coords = torch.cat([scale_lvl, new_coords_2dim], dim=2)

        return patches_scale_coords

    def add_high_res_feat(self, tokens, coords, curr_scale, image):
        patched_im = self.high_res_patchers[curr_scale](image)
        b = torch.arange(coords.shape[0]).unsqueeze(-1).expand(-1, coords.shape[1])
        x = torch.div(coords[..., 0], 2 ** (self.n_scales - curr_scale - 2), rounding_mode='trunc').long()
        y = torch.div(coords[..., 1], 2 ** (self.n_scales - curr_scale - 2), rounding_mode='trunc').long()
        patched_im = patched_im[b, :, y, x]
        tokens = tokens + patched_im

        return tokens

    def split_input(self, tokens, patches_scale_coords, curr_scale, im):
        tokens_at_curr_scale, coords_at_curr_scale, tokens_at_older_scale, coords_at_older_scales = self.divide_tokens_coords_on_scale(
            tokens, patches_scale_coords, curr_scale)
        meta_loss_coords = coords_at_curr_scale[:, :, 1:]
        tokens_to_split, coords_to_split, tokens_to_keep, coords_to_keep, pred_meta_loss = self.divide_tokens_to_split_and_keep(
            tokens_at_curr_scale, coords_at_curr_scale, curr_scale)
        tokens_after_split = self.split_tokens(tokens_to_split, curr_scale)
        coords_after_split = self.split_coords(coords_to_split, curr_scale)

        tokens_after_split = self.add_high_res_feat(tokens_after_split, coords_after_split[:, :, 1:], curr_scale, im)

        all_tokens = torch.cat([tokens_at_older_scale, tokens_to_keep, tokens_after_split], dim=1)
        all_coords = torch.cat([coords_at_older_scales, coords_to_keep, coords_after_split], dim=1)

        return all_tokens, all_coords, pred_meta_loss, meta_loss_coords

    def forward(self, im):
        B, _, H, W = im.shape
        PS = self.patch_size
        x = self.patch_embed(im)
        patched_im_size = (H // PS, W // PS)
        min_patched_im_size = (H // self.min_patch_size, W // self.min_patch_size)
        patches_scale_coords = get_2dpos_of_curr_ps_in_min_ps(H, W, PS, self.min_patch_size, 0).to('cuda')
        patches_scale_coords = patches_scale_coords.repeat(B, 1, 1)
        #pos_embed = self.pe_layer(patches_scale_coords[:,:,1:])
        #x = x + pos_embed
        outs = {}
        #print("Feature shape after PE: {}".format(x.shape))
        #print("Pos h max after PE: {}".format(patches_scale_coords[:, :, 1].max()))
        #print("Pos w max after PE: {}".format(patches_scale_coords[:, :, 2].max()))
        for l_idx in range(len(self.layers)):
            out_idx = self.n_scales - l_idx + 1
            patches_scale_coords, x = self.layers[l_idx](patches_scale_coords, x, h=min_patched_im_size[0],
                                                         w=min_patched_im_size[1], on_grid=l_idx == 0)
            #print("Feature shape after layer {}: {}".format(l_idx, x.shape))
            #print("Pos shape after layer {}: {}".format(l_idx, patches_scale_coords.shape))
            #print("Feature is contiguous after layer {}: {}".format(l_idx, x.is_contiguous()))
            #print("Pos is contiguous after layer {}: {}".format(l_idx, patches_scale_coords.is_contiguous()))
            #outs["res{}_spatial_shape".format(out_idx)] = patched_im_size
            if l_idx < self.n_scales - 1:
                x, patches_scale_coords, meta_loss, meta_loss_coord = self.split_input(x, patches_scale_coords, l_idx, im)
                #print("Feature shape after split in layer {}: {}".format(l_idx, x.shape))
                #print("Pos shape after split in layer {}: {}".format(l_idx, patches_scale_coords.shape))
                #print("Feature is contiguous after split in layer {}: {}".format(l_idx, x.is_contiguous()))
                #print("Pos is contiguous after split in layer {}: {}".format(l_idx, patches_scale_coords.is_contiguous()))
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
            outs["res{}".format(out_idx)] = self.norm_out(out_scale)
            outs["res{}_pos".format(out_idx)] = pos_scale #torch.div(pos_scale, 2 ** (self.n_scales - s - 1), rounding_mode='trunc')
            outs["res{}_spatial_shape".format(out_idx)] = min_patched_im_size
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
class MixResMetaLossNeighbour(MRMLNB, Backbone):
    def __init__(self, cfg, input_shape):

        in_chans = 3
        initial_patch_size = cfg.MODEL.MRML.PATCH_SIZES[0]
        embed_dim = cfg.MODEL.MRML.EMBED_DIM
        depths = cfg.MODEL.MRML.DEPTHS
        num_heads = cfg.MODEL.MRML.NUM_HEADS
        drop_rate = cfg.MODEL.MRML.DROP_RATE
        drop_path_rate = cfg.MODEL.MRML.DROP_PATH_RATE
        attn_drop_rate = cfg.MODEL.MRML.ATTN_DROP_RATE
        split_ratio = cfg.MODEL.MRML.SPLIT_RATIO
        mlp_ratio = cfg.MODEL.MRML.MLP_RATIO
        num_scales = cfg.MODEL.MRML.NUM_SCALES
        image_size = cfg.INPUT.CROP.SIZE
        upscale_ratio = cfg.MODEL.MRML.UPSCALE_RATIO

        cluster_size = cfg.MODEL.MRML.CLUSTER_SIZE
        nbhd_size = cfg.MODEL.MRML.NBHD_SIZE

        super().__init__(
            image_size=image_size,
            patch_size=initial_patch_size,
            n_layers=depths,
            d_model=embed_dim,
            n_heads=num_heads,
            dropout=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            mlp_ratio=mlp_ratio,
            split_ratio=split_ratio,
            n_scales=num_scales,
            channels=in_chans,
            upscale_ratio=upscale_ratio,
            cluster_size=cluster_size,
            nbhd_size=nbhd_size
        )

        self._out_features = cfg.MODEL.MRML.OUT_FEATURES

        self._out_feature_strides = { "res{}".format(i+2): list(reversed(cfg.MODEL.MRML.PATCH_SIZES))[i] for i in range(num_scales)}
        #self._out_feature_strides = {"res{}".format(i + 2): cfg.MODEL.MRML.PATCH_SIZES[-1] for i in range(num_scales)}
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
