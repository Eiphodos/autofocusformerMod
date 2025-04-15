# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict

from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.build_maskfiner_decoder import build_transformer_decoder
from ..pixel_decoder.build import build_pixel_decoder_indexed


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFinerOTHead(nn.Module):

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
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.num_classes = num_classes

        print("Successfully built MaskFinerOTHead model!")

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        print("Building MaskFinerOTHead model...")
        final_indx = cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES - 1
        mask_decoder_input_dim = cfg.MODEL.MR_SEM_SEG_HEAD.CONVS_DIM[-1]

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "loss_weight": cfg.MODEL.MR_SEM_SEG_HEAD.LOSS_WEIGHT,
            "ignore_value": cfg.MODEL.MR_SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.MR_SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder_indexed(cfg, final_indx, input_shape),
            "transformer_predictor": build_transformer_decoder(cfg, final_indx, mask_decoder_input_dim, mask_classification=True),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, mf_pos, multi_scale_features, multi_scale_poss, ms_scale, finest_input_shape, input_shapes = self.pixel_decoder.forward_features(features)
        predictions = self.mask_decoder(multi_scale_features, multi_scale_poss, mask_features, mf_pos, finest_input_shape, input_shapes)
        return predictions
