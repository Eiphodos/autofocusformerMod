# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted for AutoFocusFormer by Ziwen 2023

from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.BETAS = (0.9, 0.999)
    cfg.SOLVER.EPSILON = 1e-8

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # Only used by MetaLoss version
    cfg.MODEL.MASK_FORMER.METALOSS_WEIGHT = 5.0

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MSDeformAttnPixelDecoder"
    cfg.MODEL.SEM_SEG_HEAD.FPN_COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 150
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0


    # autofocusformer backbone
    cfg.MODEL.AFF = CN()
    cfg.MODEL.AFF.EMBED_DIM = [32, 128, 256, 384]
    cfg.MODEL.AFF.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.AFF.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.AFF.MLP_RATIO = 2.0
    cfg.MODEL.AFF.CLUSTER_SIZE = 8
    cfg.MODEL.AFF.NBHD_SIZE = [48, 48, 48, 48]
    cfg.MODEL.AFF.LAYER_SCALE = 0.0
    cfg.MODEL.AFF.ALPHA = 4.0
    cfg.MODEL.AFF.DS_RATE = 0.25
    cfg.MODEL.AFF.RESERVE = True
    cfg.MODEL.AFF.DROP_RATE = 0.0
    cfg.MODEL.AFF.ATTN_DROP_RATE = 0.0
    cfg.MODEL.AFF.DROP_PATH_RATE = 0.3
    cfg.MODEL.AFF.PATCH_NORM = True
    cfg.MODEL.AFF.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.AFF.SHEPARD_POWER = 6.0
    cfg.MODEL.AFF.SHEPARD_POWER_LEARNABLE = True


    # metaloss backbone
    cfg.MODEL.MRML = CN()
    cfg.MODEL.MRML.EMBED_DIM = [32, 128, 256, 384]
    cfg.MODEL.MRML.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.MRML.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.MRML.PATCH_SIZES = [32, 16, 8, 4]
    cfg.MODEL.MRML.SPLIT_RATIO = 4
    cfg.MODEL.MRML.UPSCALE_RATIO = 0.25
    cfg.MODEL.MRML.MLP_RATIO = 4.0
    cfg.MODEL.MRML.NUM_SCALES = 4
    cfg.MODEL.MRML.DROP_RATE = 0.0
    cfg.MODEL.MRML.DROP_PATH_RATE = 0.0
    cfg.MODEL.MRML.ATTN_DROP_RATE = 0.0
    cfg.MODEL.MRML.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.MRML.CLUSTER_SIZE = 8
    cfg.MODEL.MRML.NBHD_SIZE = [48, 48, 48, 48]

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75



    # Maskfiner config

    cfg.MODEL.MASK_FINER = CN()

    # loss
    cfg.MODEL.MASK_FINER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FINER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FINER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FINER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FINER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FINER.UPSAMPLING_WEIGHT = 10

    # transformer config
    cfg.MODEL.MASK_FINER.NHEADS = [8, 8, 8, 8]
    cfg.MODEL.MASK_FINER.DROPOUT = 0.1
    cfg.MODEL.MASK_FINER.DIM_FEEDFORWARD = [2048, 2048, 2048, 2048]
    cfg.MODEL.MASK_FINER.ENC_LAYERS = [0, 0, 0, 0]
    cfg.MODEL.MASK_FINER.DEC_LAYERS = [4, 7, 10, 10]
    cfg.MODEL.MASK_FINER.DECODER_LEVELS = [1, 2, 3, 3]
    cfg.MODEL.MASK_FINER.PRE_NORM = False

    cfg.MODEL.MASK_FINER.MASK_DIM = [256, 256, 256, 256]
    cfg.MODEL.MASK_FINER.HIDDEN_DIM = [256, 256, 256, 256]
    cfg.MODEL.MASK_FINER.NUM_OBJECT_QUERIES = [100, 100, 100, 100]
    cfg.MODEL.MASK_FINER.NUM_RESOLUTION_SCALES = 4

    cfg.MODEL.MASK_FINER.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.MASK_FINER.ENFORCE_INPUT_PROJ = False

    cfg.MODEL.MASK_FINER.SHEPARD_POWER = 6.0
    cfg.MODEL.MASK_FINER.SHEPARD_POWER_LEARNABLE = True

    # MASK_FINER inference config
    cfg.MODEL.MASK_FINER.TEST = CN()
    cfg.MODEL.MASK_FINER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FINER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FINER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FINER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FINER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FINER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FINER.SIZE_DIVISIBILITY = 32

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FINER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FINER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FINER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # Oracle teacher config
    cfg.MODEL.MASK_FINER.ORACLE_TEACHER_RATIO = 0.5
    cfg.MODEL.MASK_FINER.MASK_DECODER_ALL_LEVELS = True

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FINER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskFinerTransformerDecoder"

    cfg.MODEL.MR_SEM_SEG_HEAD = CN()

    # MaskPredictor config
    cfg.MODEL.MR_SEM_SEG_HEAD.NAME = "MaskPredictor"
    cfg.MODEL.MR_SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.MR_SEM_SEG_HEAD.NUM_CLASSES = 150
    cfg.MODEL.MR_SEM_SEG_HEAD.LOSS_WEIGHT = 0.0

    # pixel decoder
    cfg.MODEL.MR_SEM_SEG_HEAD.NORM = "GN"
    cfg.MODEL.MR_SEM_SEG_HEAD.PIXEL_DECODER_NAME = ["MSDeformAttnPixelDecoderMaskFinerOracleTeacher", "MSDeformAttnPixelDecoderMaskFinerOracleTeacher", "MSDeformAttnPixelDecoderMaskFinerOracleTeacher", "MSDeformAttnPixelDecoderMaskFinerOracleTeacher"]
    cfg.MODEL.MR_SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.MR_SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.MR_SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.MR_SEM_SEG_HEAD.CONVS_DIM = [256, 256, 256, 256]
    cfg.MODEL.MR_SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = [6, 6, 6, 6]
    cfg.MODEL.MR_SEM_SEG_HEAD.MLP_RATIO = [ 4.0, 4.0, 4.0, 4.0 ]
    cfg.MODEL.MR_SEM_SEG_HEAD.NHEADS = [ 8, 8, 8, 8 ]
    cfg.MODEL.MR_SEM_SEG_HEAD.DROPOUT = 0.0

    # MixRes backbone

    cfg.MODEL.MR = CN()
    cfg.MODEL.MR.NAME = ["MixResViT","MixResNeighbour", "MixResNeighbour", "MixResNeighbour"]
    cfg.MODEL.MR.EMBED_DIM = [512,256,128,64]
    cfg.MODEL.MR.DEPTHS = [4, 4, 4, 4]
    cfg.MODEL.MR.NUM_HEADS = [ 32, 16, 8, 4 ]
    cfg.MODEL.MR.PATCH_SIZES = [32, 16, 8, 4]
    cfg.MODEL.MR.SPLIT_RATIO = [4, 4, 4, 4]
    cfg.MODEL.MR.MLP_RATIO = [4., 4., 4., 4.]
    cfg.MODEL.MR.UPSCALE_RATIO = [0.25, 0.25, 0.25, 0.25]
    cfg.MODEL.MR.DROP_RATE = [0.0, 0.0, 0.0, 0.0]
    cfg.MODEL.MR.DROP_PATH_RATE = [0.3, 0.3, 0.3, 0.3]
    cfg.MODEL.MR.ATTN_DROP_RATE = [0.0, 0.0, 0.0, 0.0]
    cfg.MODEL.MR.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.MR.CLUSTER_SIZE = [8, 8, 8, 8]
    cfg.MODEL.MR.NBHD_SIZE = [48, 48, 48, 48]
    cfg.MODEL.MR.KEEP_OLD_SCALE = False
    cfg.MODEL.MR.ADD_IMAGE_DATA_TO_ALL = False

