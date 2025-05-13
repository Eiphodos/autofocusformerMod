# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone import Backbone
from detectron2.modeling import BACKBONE_REGISTRY

def build_backbone_indexed(cfg, layer_index=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """

    backbone_name = cfg.MODEL.BACKBONE.NAME
    if backbone_name in ["MixRes", "OracleTeacherBackbone", "UpDownBackbone"]:
        backbone_name = cfg.MODEL.MR.NAME[layer_index]
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, layer_index)
    assert isinstance(backbone, Backbone)
    return backbone