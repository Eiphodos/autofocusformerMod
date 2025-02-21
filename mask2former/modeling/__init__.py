# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted for AutoFocusFormer by Ziwen 2023

from .backbone.aff import AutoFocusFormer
from .backbone.mrml import MixResMetaLoss
from .backbone.mrml_neighbour import MixResMetaLossNeighbour
from .backbone.mrmean import MixResMeanAct
from .backbone.mixres_neighbour import MixResNeighbour
from .backbone.mixres_vit import MixResViT

from .pixel_decoder.msdeformattn_pc_maskfiner_proj import MSDeformAttnPixelDecoderMaskFinerProj
from .pixel_decoder.msdeformattn_pc_maskfiner import MSDeformAttnPixelDecoderMaskFiner
from .pixel_decoder.msdeformattn_pc import MSDeformAttnPixelDecoder
from .pixel_decoder.msdeformattn_up_pc import MSDeformAttnPixelDecoderUp
from .meta_arch.mask_predictor import MaskPredictor
from .meta_arch.mask_former_head import MaskFormerHead