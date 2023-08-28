from sms_spam_detection.config.configuration import ConfigurationManager
from sms_spam_detection.components.pre_processer import DataTransformation
from sms_spam_detection.components.trainer import ModelTrainer
from sms_spam_detection import logger

STAGE_NAME = "Data Preprocessing and Training"

class Trainer:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocessor_config = config.get_preprocesser_config()
        data_trans = DataTransformation(config=preprocessor_config)
        x_train, x_test, y_train, y_test= data_trans.initiate_data_transformation()
        training_config = config.get_training_config()
        trainer = ModelTrainer(config=training_config)
        score=trainer.initiate_model_trainer(X_train=x_train, X_test=x_test, y_train=y_train, y_test=y_test)
        logger.info(f">>>>>> best model score {score} completed <<<<<<\n\nx==========x")




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = Trainer()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
