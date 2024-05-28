import os , sys ,re ,string ,nltk 
import pandas as pd 
import numpy as np 
nltk.download('stopwords')
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from src.HateSpeechClassification.utils.utils import *
from src.HateSpeechClassification.constant import * 
from src.HateSpeechClassification.exception import ClassificationException
from src.HateSpeechClassification.logger import logging 
from src.HateSpeechClassification.entity.artifacts_entity import DataTransformationArtifacts ,DataIngestionArtifacts 
from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,DatatranformationConfig
class DataTrandformation:
    def __init__(self ,
                 data_transformation_config : DatatranformationConfig,
                 data_ingestion_artifact :DataIngestionArtifacts ):
        try:
            logging.info(f'\n\n{"*" * 20} Data Transformation Step Started {"*" *20}') 
            print(f"{'*'*20} Data Transformation Staerted{'*'*20}") 
            
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact  
            
            print(self.data_transformation_config)
            print(self.data_ingestion_artifact)
        except Exception as e:
            raise ClassificationException(e ,sys) from e
    #Imbalance_dat - Train Data
    #Add On Data : -Labeled Data
    def imbalance_data_cleaning(self):
        try:
            logging.info("Imbalance Data Cleaning Started") 

            imbalance_data_file = os.listdir(path= self.data_ingestion_artifact.data_folder_path)[1]
            imbalance_data_file_path = os.path.join(self.data_ingestion_artifact.data_folder_path , imbalance_data_file)
            imbalance_data = pd.read_csv(imbalance_data_file_path)  
            imbalance_data.drop(columns=IMBALANCE_DATA_DROP_COLUMNS ,axis=IMBALANCE_DATA_DROP_COLUMNS_AXIS ,inplace=True)
            logging.info('Imbalance Data Cleaned')
            return imbalance_data
        except Exception as e:
            raise ClassificationException(e ,sys) from e         
    def raw_data_cleaning(self):
        try:
            
            raw_data_file = os.listdir(path= self.data_ingestion_artifact.data_folder_path)[0]
            raw_data_file_path = os.path.join(self.data_ingestion_artifact.data_folder_path , raw_data_file)
            raw_data = pd.read_csv(raw_data_file_path)

            raw_data.drop(columns=RAW_DATA_DROP_COLUMN , axis=RAW_DATA_DROP_COLUMNS_AXIS ,inplace= True)

            raw_data.loc[raw_data[CLASS] == 0, CLASS] = 1
            raw_data[CLASS] = raw_data[CLASS].replace({0:1})
            raw_data[CLASS] = raw_data[CLASS].replace({2:0})
            raw_data.rename(columns={CLASS:LABEL},inplace =True)
            logging.info(f"Exited the raw_data_cleaning function and returned the raw_data {raw_data}")
            return raw_data

        except Exception as e:
            raise ClassificationException(e ,sys) from e 
    def concat_data(self):
        try:
            logging.info('Concatinate imbalance data and raw data')
            data_list = [self.imbalance_data_cleaning() ,self.raw_data_cleaning()]
            df = pd.concat(data_list) 
            return df 
        except Exception as e:
            raise ClassificationException(e ,sys) from e        
    def get_final_data(self ):
        try:
            
            final_data = self.concat_data()
            # final_data[TWEET] = final_data[TWEET].apply(self.concat_data_cleaning)
            final_data[TWEET]=final_data[TWEET].apply(concat_data_cleaning)
            file = os.path.join(os.getcwd() , 'final.csv')
            final_data.to_csv(file)
            return final_data 
        except Exception as e:
            raise ClassificationException(e, sys) from e       
    def split_data_train_test(self):
        try:
            final_data = self.get_final_data()
            X = final_data[TWEET]
            Y = final_data[LABEL]
            x_train,x_test,y_train,y_test = train_test_split(X,Y, random_state = 42)
            return x_train,x_test,y_train,y_test
        except Exception as e:
            raise ClassificationException(e, sys) from e  
    def initiate_data_transformation(self):
        try:
            
            final_data = self.get_final_data() 
            os.makedirs(self.data_transformation_config.transformed_data_dir ,exist_ok=True)
            transformed_file_path = os.path.join(
                self.data_transformation_config.transformed_data_dir ,
                self.data_transformation_config.transformed_file_name
            )
            final_data.to_csv(transformed_file_path ,index=False)
            data_transformation_artifacts = DataTransformationArtifacts(
                is_transformed= True ,
                message= "Data transformation Completed" ,
                transformed_file_path = transformed_file_path
            )
            print(f"{'*'*20} Data Transformation completed{'*'*20}") 
            logging.info(f"{'*'*20} Data Transformation completed{'*'*20}") 
            return data_transformation_artifacts
        except Exception as e:
            raise ClassificationException(e, sys) from e 