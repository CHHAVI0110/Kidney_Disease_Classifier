from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # Load config
        config_manager = ConfigurationManager()
        prepare_base_model_config = config_manager.get_prepare_base_model_config()

        # Initialize PrepareBaseModel
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        # Step 1: Load base Xception model
        base_model = prepare_base_model.get_base_model()

        # Step 2: Save base model
        prepare_base_model.save_model(
            path=prepare_base_model_config.base_model_path,
            model=base_model
        )

        # Step 3: Create updated model with head layers
        updated_model = prepare_base_model.update_base_model(fine_tune_at=400)  # optional fine-tuning

        # Step 4: Save updated model
        prepare_base_model.save_model(
            path=prepare_base_model_config.updated_base_model_path,
            model=updated_model
        )

        logger.info(f"✅ Base model saved at: {prepare_base_model_config.base_model_path}")
        logger.info(f"✅ Updated model saved at: {prepare_base_model_config.updated_base_model_path}")


if __name__ == "__main__":
    try:
        logger.info(f"********************************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
