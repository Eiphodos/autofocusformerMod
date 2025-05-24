# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Adapted for AutoFocusFormer by Ziwen 2023

import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable

from .position_encoding import PositionEmbeddingSine
from ..backbone.point_utils import upsample_feature_shepard, upsample_tokens_fixed_scales, hierarchical_upsample_ordered

from .build_maskfiner_decoder import TRANSFORMER_DECODER_REGISTRY

def fix_pos_no_bias(pos, current_ss, finest_ss):
    ret = pos.clone()
    ss_ratio_h = finest_ss[0] / current_ss[0]
    ss_ratio_w = finest_ss[1] / current_ss[1]
    shift_value_h = (ss_ratio_h / 2) - 0.5
    shift_value_w = (ss_ratio_w / 2) - 0.5
    ret[:, :, 0] = ret[:, :, 0] + shift_value_w
    ret[:, :, 1] = ret[:, :, 1] + shift_value_h

    return ret

def scale_pos(last_pos, last_ss, cur_ss, no_bias=False):
    """
    Scales the positions from last_ss scale to cur_ss scale.
    Args:
        last_pos - ... x 2, 2D positions
        *_ss - (h,w), height and width
        no_bias - bool, if True, move the positions to the center of the grid and then scale,
                        so that there is no bias toward the upperleft corner
    Returns:
        res - ... x 2, scaled 2D positions
    """
    if last_ss[0] == cur_ss[0] and last_ss[1] == cur_ss[1]:
        return last_pos
    last_h, last_w = last_ss
    cur_h, cur_w = cur_ss
    h_ratio = cur_h / last_h
    w_ratio = cur_w / last_w
    ret = last_pos.clone()
    if no_bias:
        ret += 0.5
    ret[..., 0] *= w_ratio
    ret[..., 1] *= h_ratio
    if no_bias:
        ret -= 0.5
    return ret

'''
def build_transformer_decoder(cfg, layer_index, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FINER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, layer_index, in_channels, mask_classification)
'''

def point2img(x, pos, mask_size=None):
    '''
    x - b x q x n
    pos - b x n x 2
    '''
    if x.shape[0] != pos.shape[0]:
        pos = pos.repeat(x.shape[0]//pos.shape[0], 1, 1)
    b, q, n = x.shape
    pos = pos.long()
    if mask_size is None:
        h = pos[:, :, 1].max().item() + 1
        w = pos[:, :, 0].max().item() + 1
    else:
        h, w = mask_size
    assert h*w == n, "h*w != n in point2img!"
    pos_idx = pos[:, :, 1] * w + pos[:, :, 0]  # b x n
    ret = torch.zeros(b, q, h*w, device=x.device, dtype=x.dtype)
    ret.scatter_(index=pos_idx.unsqueeze(1).expand(-1, q, -1), dim=2, src=x)
    ret = ret.reshape(b, q, h, w)
    return ret


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="lrelu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

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

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="lrelu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

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


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="lrelu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "lrelu":
        return F.leaky_relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskFinerTransformerDecoderOracleTeacher(nn.Module):

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
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
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
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        num_decoder_levels: int,
        final_layer: bool
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.final_layer = final_layer

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = num_decoder_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Linear(in_channels, hidden_dim))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        print("Successfully built MultiScaleMaskFinerTransformerDecoderOracleTeacher model!")


    @classmethod
    def from_config(cls, cfg, layer_index, in_channels, mask_classification):
        print("Building MultiScaleMaskFinerTransformerDecoderOracleTeacher model...")
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.MR_SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FINER.HIDDEN_DIM[layer_index]
        ret["num_queries"] = cfg.MODEL.MASK_FINER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FINER.NHEADS[layer_index]
        ret["dim_feedforward"] = cfg.MODEL.MASK_FINER.DIM_FEEDFORWARD[layer_index]

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FINER.DEC_LAYERS[layer_index] >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FINER.DEC_LAYERS[layer_index] - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FINER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FINER.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.MASK_FINER.MASK_DIM[layer_index]
        ret["num_decoder_levels"] = cfg.MODEL.MASK_FINER.DECODER_LEVELS[layer_index]
        final_layer = (layer_index + 1) == cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES
        ret["final_layer"] = final_layer
        return ret


    def forward(self, x, pos, mask_features, mf_pos, finest_input_shape, input_shapes):
        '''
        x - [b x n x c]
        pos - [b x n x 2]
        mask_feature - b x n x c
        mf_pos - b x n x 2
        '''
        # x is a list of multi-scale feature
        finest_inp_feat_shape = input_shapes[-1]

        tokens_per_scale = [tx.shape[1] for tx in x]
        #mask_features, finest_pos = hierarchical_upsample_ordered(mask_features, torch.cat(pos, dim=1), tokens_per_scale, finest_input_shape)

        x = x[:self.num_feature_levels]
        pos = pos[:self.num_feature_levels]
        input_shapes = input_shapes[:self.num_feature_levels]
        assert len(x) == self.num_feature_levels
        src = []
        pos_emb = []

        if len(pos) == 1 and pos[0].shape == mf_pos.shape and (pos[0] == mf_pos).all():
            masked_attn = False
        else:
            masked_attn = True


        # scale positions to finest input positions
        b, _, _ = x[0].shape
        poss_scaled = []
        #print("Mask feature max pos before scaling: {}".format(mf_pos.max()))
        mf_pos_scaled = scale_pos(mf_pos, finest_input_shape, finest_inp_feat_shape)
        #print("Mask feature max pos after scaling: {}".format(mf_pos_scaled.max()))
        i = 0
        for p, inp_shape in zip(pos, input_shapes):
            #print("Feature {} max pos before scaling: {}".format(i, p.max()))
            fixed_pos = fix_pos_no_bias(p, inp_shape, finest_input_shape)
            pos_scaled = scale_pos(fixed_pos, finest_input_shape, finest_inp_feat_shape)
            #print("Feature {} max pos after scaling: {}".format(i, pos_scaled.max()))
            poss_scaled.append(pos_scaled)
            i += 1
        finest_pos = torch.stack(torch.meshgrid(torch.arange(0, finest_inp_feat_shape[1]), torch.arange(0, finest_inp_feat_shape[0]), indexing='ij')).permute(1, 2, 0).transpose(0, 1).reshape(-1, 2)
        finest_pos = finest_pos.to(mf_pos.device).repeat(b, 1, 1)

        for i in range(self.num_feature_levels):
            pos_emb.append(self.pe_layer(poss_scaled[i]))
            src.append(self.input_proj[i](x[i]) + self.level_embed.weight[i][None, None, :])

            # b x n x c to n x b x c
            pos_emb[-1] = pos_emb[-1].permute(1, 0, 2)
            src[-1] = src[-1].permute(1, 0, 2)

        # prediction heads on learnable query features
        _, b, _ = src[0].shape
        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, b, 1)
        predictions_class = []
        predictions_mask = []
        if torch.isnan(finest_pos).any():
            print("NaNs detected in finest_pos")
        #mask_features = upsample_feature_shepard(finest_pos, mf_pos_scaled, mask_features)
        #outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, finest_pos, poss_scaled[0], masked_attn)  # b x q x nc, b x q x n, b*h x q x n
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, mf_pos_scaled, poss_scaled[0], masked_attn)  # b x q x nc, b x q x n, b*h x q x n
        #pos_indices = find_pos_indices_in_pos(finest_pos, mf_pos_scaled)
        outputs_mask = upsample_feature_shepard(finest_pos, mf_pos_scaled, outputs_mask.permute(0, 2, 1)).permute(0, 2, 1)
        outputs_mask = point2img(outputs_mask, finest_pos)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)


        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if masked_attn:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos_emb[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            #outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, finest_pos, poss_scaled[(i + 1) % self.num_feature_levels], masked_attn)  # b x q x nc, b x q x n, b*h x q x n
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, mf_pos_scaled, poss_scaled[(i + 1) % self.num_feature_levels], masked_attn)  # b x q x nc, b x q x n, b*h x q x n
            outputs_mask = upsample_feature_shepard(finest_pos, mf_pos_scaled, outputs_mask.permute(0, 2, 1)).permute(0, 2, 1)
            outputs_mask = point2img(outputs_mask, finest_pos)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1
        if self.final_layer:
            out = {
                'pred_logits': predictions_class[-1],
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(
                    predictions_class[:-1] if self.mask_classification else None, predictions_mask[:-1]
                )
            }
        else:
            out = {
                'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask
                )
            }
        return out

    def forward_prediction_heads(self, output, mask_features, mf_pos, target_pos, masked_attn):
        '''
        output - q x b x c
        mask_features - b x n x c
        '''
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # b x q x c'
        outputs_class = self.class_embed(decoder_output)  # b x q x nc
        mask_embed = self.mask_embed(decoder_output)  # b x q x c
        outputs_mask = mask_embed @ mask_features.permute(0, 2, 1)  # b x q x n
        #print("Mask output shape before upsampling: {}".format(outputs_mask.shape))
        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        # attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        if masked_attn:
            if torch.isnan(target_pos).any():
                print("NaNs detected in target_pos")
            attn_mask = upsample_feature_shepard(target_pos, mf_pos, outputs_mask.permute(0, 2, 1)).permute(0, 2, 1)  # b x q x n
            attn_mask = (attn_mask.sigmoid().unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()  # b*h x q x n
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None
        #print("Final Attn Mask output shape: {}".format(attn_mask.shape))
        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class, outputs_seg_masks)
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks]


    def create_disagreement_mask(self, outputs_mask, outputs_class):
        b, q, n = outputs_mask.shape
        cls_i = outputs_class.argmax(dim=-1)
        disagreement_mask = torch.zeros(b, n, requires_grad=True).to(outputs_mask.device)
        for b in range(cls_i.shape[0]):
            cls_unique = torch.unique(cls_i)
            disagreement_mask_b = torch.zeros(n, len(cls_unique)).to(outputs_mask.device)
            for i, c in enumerate(cls_unique):
                batch_cls_mask = outputs_mask[b, cls_i[b] == c].sum(dim=0)
                batch_cls_mask = torch.sigmoid(batch_cls_mask)
                disagreement_mask_b[:, i] = batch_cls_mask
            disagreement_mask[b, :] = self.gini(disagreement_mask_b)

        return disagreement_mask

    def gini(self, disagreement_mask):
        mad = torch.abs(disagreement_mask.unsqueeze(1) - disagreement_mask.unsqueeze(2)).mean(dim=(1, 2))
        rmad = mad / disagreement_mask.mean(dim=1)
        g = 0.5 * rmad
        return g

    def zero_edges(self, disagreement_mask, disagreement_pos, max_height, max_width):
        disagreement_mask[disagreement_pos[..., 0] == 0] = 0
        disagreement_mask[disagreement_pos[..., 1] == 0] = 0
        disagreement_mask[disagreement_pos[..., 0] == (max_width - 1)] = 0
        disagreement_mask[disagreement_pos[..., 1] == (max_height - 1)] = 0

        return disagreement_mask


    def create_disagreement_mask2(self, outputs_mask, outputs_class):
        pred = outputs_mask.permute(0, 2, 1) @ outputs_class
        pred  = F.softmax(pred, dim=-1)
        pred_max = pred.max(dim=-1)[0]
        disagreement_mask = 1 - pred_max
        return disagreement_mask


    def create_disagreement_mask3(self, outputs_mask, outputs_class, pos, scale):
        b = torch.arange(pos.shape[0]).unsqueeze(-1).expand(-1, pos.shape[1])
        pos_x = torch.div(pos[..., 0], 2 ** (4 - scale), rounding_mode='trunc').long()
        pos_y = torch.div(pos[..., 1], 2 ** (4 - scale), rounding_mode='trunc').long()
        mask_tokens = outputs_mask[b, :, pos_y, pos_x].permute(0, 2, 1)

        b, q, n = mask_tokens.shape
        cls_i = outputs_class.argmax(dim=-1)
        disagreement_mask = torch.zeros(b, n, requires_grad=True).to(mask_tokens.device)
        for b in range(cls_i.shape[0]):
            for c in cls_i[b].unique():
                batch_cls_mask = torch.sigmoid(mask_tokens[b, cls_i[b] == c].sum(dim=0))
                batch_cls_mask = (batch_cls_mask > 0.5).int()
                disagreement_mask[b] = disagreement_mask[b] + batch_cls_mask
        #print("Number of unique classes in sample 0: {}".format(len(cls_i[0].unique())))
        return disagreement_mask

    def create_disagreement_mask4(self, outputs_mask):
        disagreement_mask = outputs_mask.sum(dim=1)
        return disagreement_mask

    def find_pos_org_order(self, pos_org, pos_shuffled):
        dists = torch.cdist(pos_org.float(), pos_shuffled.float(), p=2)  # Manhattan distance
        pos_indices = torch.argmin(dists, dim=2)  # (B, N_)

        return pos_indices