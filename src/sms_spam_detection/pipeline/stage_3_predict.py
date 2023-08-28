from sms_spam_detection.config.configuration import ConfigurationManager
from sms_spam_detection.components.predict import PredictPipeline
from sms_spam_detection import logger
from pathlib import Path

STAGE_NAME = "Prediction Pipeline"

class Predict:
    def __init__(self):
        pass

    def main(self, message):
        config = ConfigurationManager()
        preprocessor_config = config.get_preprocesser_config()
        model_path=Path("artifacts/training/model.joblib")
        model = PredictPipeline(config=preprocessor_config, model_path=model_path)
        result=model.predict(message=message)
        logger.info(f">>>>>>  {result}  <<<<<<\n\nx==========x")




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        message="Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."
        obj = Predict()
        obj.main(message=message)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
