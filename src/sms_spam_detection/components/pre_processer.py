from dataclasses import dataclass
import numpy as np 
import pandas as pd
import os
import re
import nltk
import string
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sms_spam_detection.utils.common import save_bin
from sms_spam_detection.entity.config_entity import DataPreprocesserConfig
from sklearn.model_selection import train_test_split


class DataTransformation:
    def __init__(self,config: DataPreprocesserConfig):
        self.data_transformation_config=config
    
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub('\[.*?\]\n', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    
    def remove_stopwords(self, text):
        stop_words = nltk.corpus.stopwords.words('english')
        text = ' '.join(word for word in text.split(' ') if word not in stop_words)
        return text
    
    def stemm_text(self, text):
        stemmer = nltk.SnowballStemmer("english")
        text = ' '.join(stemmer.stem(word) for word in text.split(' '))
        return text
    
    def preprocess_data(self, text):
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.stemm_text(text)
        
        return text


    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        return vect, tfidf_transformer
        
        
    def initiate_data_transformation(self):

        df = pd.read_csv(self.data_transformation_config.data_dir, encoding="latin-1")
        df.dropna(axis=1, inplace=True)
        df.columns = ['target', 'sms_message']
        train_df,test_df,_,_=train_test_split(df,df.target, random_state=42, test_size=0.2)
        preprocessor_vect, preprocessor_tfidf=self.get_data_transformer_object()

        train_df['processed_message'] = train_df['sms_message'].apply(self.preprocess_data)
        test_df['processed_message'] = test_df['sms_message'].apply(self.preprocess_data)

        le = LabelEncoder()
        le.fit(train_df['target'])
        train_df['target_encoded'] = le.transform(train_df['target'])
        test_df['target_encoded'] = le.transform(test_df['target'])

        train_x = train_df['processed_message']
        train_y = train_df['target_encoded']
        test_x = test_df['processed_message']
        test_y = test_df['target_encoded']

        preprocessor_vect.fit(train_x)
        train_x=preprocessor_vect.transform(train_x)
        test_x=preprocessor_vect.transform(test_x)

        preprocessor_tfidf.fit(train_x)
        train_x=preprocessor_tfidf.transform(train_x)
        test_x=preprocessor_tfidf.transform(test_x)

        sm = SMOTE(random_state = 2)
        train_x, train_y = sm.fit_resample(train_x, train_y)
        test_x, test_y = sm.fit_resample(test_x, test_y)
        save_bin(path=self.data_transformation_config.label_encoder,
                data=le
        )
        save_bin(path=self.data_transformation_config.vect_obj_dir,
                data=preprocessor_vect
        )
        save_bin(path=self.data_transformation_config.tfidf_obj_dir,
                data=preprocessor_tfidf
        )

        return train_x, test_x, train_y, test_y