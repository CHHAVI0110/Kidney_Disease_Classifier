from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig,TrainingConfig)
from pathlib import Path
import os


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion  # from config.yaml

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model  # from config.yaml

        # Make sure the root dir exists
        create_directories([Path(config.root_dir)])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),                  
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=list(self.params.IMAGE_SIZE),
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_dropout_rate_head=self.params.DROPOUT_RATE_HEAD,
            params_dense_units=self.params.DENSE_UNITS,
            params_weight_decay=self.params.WEIGHT_DECAY,
            params_optimizer=self.params.OPTIMIZER,
            params_label_smoothing=self.params.LABEL_SMOOTHING
        )

        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config['training']
        prepare_base_model = self.config['prepare_base_model']
        params = self.params

        training_data = os.path.join(
            self.config['data_ingestion']['unzip_dir'],
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
            "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
        )

        create_directories([Path(training['root_dir'])])

        
        return TrainingConfig(
        root_dir=Path(training['root_dir']),
        trained_model_path=Path(training['trained_model_path']),
        updated_base_model_path=Path(prepare_base_model['updated_base_model_path']),
        training_data=Path(training_data),

        # --- Core training params ---
        params_epochs=params.get('EPOCHS', 25),
        params_batch_size=params.get('BATCH_SIZE', 16),
        params_is_augmentation=params.get('AUGMENTATION', True),
        params_image_size=params.get('IMAGE_SIZE', [224, 224, 3]),
        params_learning_rate=params.get('LEARNING_RATE', 1e-4),
        params_classes=params.get('CLASSES', 4),
        params_freeze_till=params.get('FREEZE_TILL', 120),
        params_fine_tune_last_n=params.get('FINE_TUNE_LAST_N', 60),

        # --- Regularization ---
        params_dropout_rate_head=params.get('DROPOUT_RATE_HEAD', 0.4),
        params_weight_decay=params.get('WEIGHT_DECAY', 0.0002),
        params_label_smoothing=params.get('LABEL_SMOOTHING', 0.1),

        # --- Model architecture tweaks ---
        params_dense_units=params.get('DENSE_UNITS', 512),
        params_optimizer=params.get('OPTIMIZER', "adamw"),
)



