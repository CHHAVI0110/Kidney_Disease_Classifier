# src/cnnClassifier/entity/config_entity.py

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path  # Folder where downloaded data is extracted


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path           # Path to save the original Xception base model
    updated_base_model_path: Path   # Path to save the full model with classifier head
    params_image_size: List[int]    # [height, width, channels]
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_dropout_rate_head: float     # from params.yaml
    params_dense_units: int             # from params.yaml
    params_weight_decay: float          # from params.yaml
    params_optimizer: str               # from params.yaml (adam / adamw etc.)
    params_label_smoothing: float

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    
    # Core training params
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: List[int]       # matches IMAGE_SIZE in params.yaml
    params_learning_rate: float        # matches LEARNING_RATE
    params_classes: int                # matches CLASSES
    params_freeze_till: int            # matches FREEZE_TILL
    params_fine_tune_last_n: int       # matches FINE_TUNE_LAST_N
    
    # Regularization (anti-overfitting)
    params_dropout_rate_head: float    # matches DROPOUT_RATE_HEAD
    params_weight_decay: float         # matches WEIGHT_DECAY
    params_label_smoothing: float      # matches LABEL_SMOOTHING
    
    # Model architecture
    params_dense_units: int            # matches DENSE_UNITS
    params_optimizer: str              # matches OPTIMIZER
