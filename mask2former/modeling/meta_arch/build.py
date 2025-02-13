# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
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

    mask_predictor = cfg.MODEL.MASK_PREDICTOR
    model = META_ARCH_REGISTRY.get(mask_predictor)(cfg, layer_index)
    _log_api_usage("modeling.meta_arch." + mask_predictor)
    return model