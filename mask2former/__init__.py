# Copyright (c) Facebook, Inc. and its affiliates.

from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.mask_former_instance_dataset_mapper import (
    MaskFormerInstanceDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

# models
from .maskformer_model import MaskFormer
from .maskformer_model_ml import MaskFormerML
from .maskfiner_model import MaskFiner
from .maskfiner_oracle_model import MaskFinerOracle
from .maskfiner_oracle_teacher_model import MaskFinerOracleTeacher
from .maskfiner_oracle_teacher_model_bb import MaskFinerOracleTeacherBB
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.meta_loss_evaluation import MetaLossSemSegEvaluator
from .evaluation.maskfiner_evaluation import MaskFinerSemSegEvaluator
from .evaluation.maskfiner_evaluation import MaskFinerCityscapesInstanceEvaluator
from .evaluation.semseg_evaluation import SemSegEvaluatorSave
