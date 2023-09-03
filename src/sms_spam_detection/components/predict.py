import pandas as pd
from sms_spam_detection.components.pre_processer import DataTransformation
from sms_spam_detection.entity.config_entity import DataPreprocesserConfig
from sms_spam_detection.utils.common import load_bin
from pathlib import Path

class PredictPipeline:
    def __init__(self,config: DataPreprocesserConfig, model_path: Path):
        self.data_transformation_config=config
        self.model_path=model_path
        self.preprocessor_vect=load_bin(self.data_transformation_config.vect_obj_dir)
        self.preprocessor_tfidf=load_bin(self.data_transformation_config.tfidf_obj_dir)
        self.ld=load_bin(self.data_transformation_config.label_encoder)

    def predict_spam(self,message):
        data=CustomData(message)
        df=data.get_data_as_data_frame()
        data_tranformer=DataTransformation(config=self.data_transformation_config)
        df['processed_message'] = df['sms_message'].apply(data_tranformer.preprocess_data)
        df.drop("sms_message", axis=1, inplace=True)
        df=self.preprocessor_vect.transform(df)
        df=self.preprocessor_tfidf.transform(df)
        print(type(df))
        model=load_bin(self.model_path)
        result=model.predict(df)
        result=self.ld.inverse_transform(result)
        return result[0]




class CustomData:
    def __init__(  self, sms: str):
        self.sms = sms


    def get_data_as_data_frame(self):
        custom_data_input_dict = {
            "sms_message": [self.sms]
        }

        return pd.DataFrame(custom_data_input_dict)
