# main.py

from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline

# -------------------------------
# Stage 1: Data Ingestion
# -------------------------------
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# -------------------------------
# Stage 2: Prepare Base Model
# -------------------------------
STAGE_NAME = "Prepare Base Model Stage"

try:
    logger.info(f"********************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    
    prepare_base_model_pipeline = PrepareBaseModelTrainingPipeline()
    prepare_base_model_pipeline.main()
    
    # Log model save paths
    from cnnClassifier.config.configuration import ConfigurationManager
    config = ConfigurationManager()
    prepare_base_model_config = config.get_prepare_base_model_config()
    
    logger.info(f"Base model saved at: {prepare_base_model_config.base_model_path}")
    logger.info(f"Updated model saved at: {prepare_base_model_config.updated_base_model_path}")
    
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Training"
try:
    logger.info(f"**********************************")
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e