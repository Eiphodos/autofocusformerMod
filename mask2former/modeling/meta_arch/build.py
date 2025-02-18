# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


def build_mask_predictor_indexed(cfg, layer_index):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    mask_predictor = cfg.MODEL.SEM_SEG_HEAD.NAME
    model = SEM_SEG_HEADS_REGISTRY.get(mask_predictor)(cfg, layer_index)
    return model