from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

def build_pixel_decoder_indexed(cfg, layer_index, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.MR_SEM_SEG_HEAD.PIXEL_DECODER_NAME[layer_index]
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, layer_index, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model