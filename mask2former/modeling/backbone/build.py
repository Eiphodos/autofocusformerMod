# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone import Backbone

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_backbone_indexed(cfg, layer_index=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """

    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name == "MixRes":
        backbone_name = cfg.MODEL.BACKBONE.MR.NAME[layer_index]
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, layer_index)
    assert isinstance(backbone, Backbone)
    return backbone