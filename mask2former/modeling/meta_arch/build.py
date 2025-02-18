# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")  # noqa F401 isort:skip
SEM_SEG_HEADS_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_mask_predictor_indexed(cfg, layer_index):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """

    mask_predictor = cfg.MODEL.SEM_SEG_HEAD.NAME
    model = SEM_SEG_HEADS_REGISTRY.get(mask_predictor)(cfg, layer_index)
    _log_api_usage("modeling.meta_arch." + mask_predictor)
    return model