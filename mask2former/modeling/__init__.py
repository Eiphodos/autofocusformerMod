# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted for AutoFocusFormer by Ziwen 2023

from .backbone.aff import AutoFocusFormer
from .backbone.mrml import MixResMetaLoss
from .backbone.mrml_neighbour import MixResMetaLossNeighbour
from .backbone.mrmean import MixResMeanAct
from .backbone.mixres_neighbour import MixResNeighbour
from .backbone.mixres_neighbour_xattn import MixResNeighbourXAttn
from .backbone.mixres_vit import MixResViT
from .backbone.convnextv2 import ConvNeXtV2
from .backbone.mixres_oracle_teacher import OracleTeacherBackbone
from .backbone.mixres_up_down import UpDownBackbone
from .backbone.swin import D2SwinTransformer

from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoderSwin
from .pixel_decoder.msdeformattn_pc_maskfiner_oracle_teacher import MSDeformAttnPixelDecoderMaskFinerOracleTeacher
from .pixel_decoder.proj_maskfiner_oracle_teacher import ProjMaskFinerOracleTeacher
from .pixel_decoder.msdeformattn_pc_maskfiner_proj import MSDeformAttnPixelDecoderMaskFinerProj
from .pixel_decoder.msdeformattn_pc_maskfiner import MSDeformAttnPixelDecoderMaskFiner
from .pixel_decoder.msdeformattn_pc_maskfiner_hierup import MSDeformAttnPixelDecoderMaskFinerHierUp
from .pixel_decoder.msdeformattn_pc import MSDeformAttnPixelDecoder
from .pixel_decoder.msdeformattn_up_pc import MSDeformAttnPixelDecoderUp
from .meta_arch.mask_predictor import MaskPredictor
from .meta_arch.mask_predictor_oracle_teacher import MaskPredictorOracleTeacher
from .meta_arch.mask_former_head import MaskFormerHead
from .meta_arch.mask_former_head_swin import MaskFormerHeadSwin
from .meta_arch.mask_finer_ot_head import MaskFinerOTHead