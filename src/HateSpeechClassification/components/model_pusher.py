import os , sys ,re ,string ,nltk 
import pandas as pd 
import numpy as np 
import shutil

from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,DatatranformationConfig ,ModelEvaluationConfig ,ModelPusherConfig
from src.HateSpeechClassification.entity.artifacts_entity import DataTransformationArtifacts ,DataIngestionArtifacts  ,ModelTrainingArtifacts ,ModelEvaluationArtifacts,ModelPusherArtifacts
from src.HateSpeechClassification.constant import * 
from src.HateSpeechClassification.exception import ClassificationException
from src.HateSpeechClassification.logger import logging 
from src.HateSpeechClassification.utils.utils import * 
from tensorflow.keras.models import load_model

class ModelPusher:
    def __init__(self ,
                 model_pusher_config :ModelPusherConfig ,
                 model_evaluation_artifacts :ModelEvaluationArtifacts) :
        try:
            logging.info(f'\n\n{">" * 10} Model Pusher Step Started {"<" *10}') 
            self.model_pusher_config = model_pusher_config 
            self.model_evaluation_artifacts = model_evaluation_artifacts
        except Exception as e:
            raise ClassificationException(e,sys) from e 
        
    def initiate_export_model(self):
        try:
            # evaluated_model_file_path = 'D:\\Data Science\\NLP\\Project\\HateSpeechClassification\\artifact\\model_traaining\\2024-05-27-23-44-41\\trained_model\\model.h5'
            # tokenizer_file_path = 'D:\\Data Science\\NLP\\Project\\HateSpeechClassification\\artifact\model_traaining\\2024-05-27-23-44-41\\Tokenizer\\tokenizer.pkl'
            evaluated_model_file_path = self.model_evaluation_artifacts.best_model_path 
            tokenizer_file_path = self.model_evaluation_artifacts.tokenizer_path 

            # exported_model_file_path = self.model_pusher_config.export_dir_path 
            # exported_tokenizer_file_path = self.model_pusher_config.export_dir_path
            # os.makedirs(self.model_pusher_config.export_dir_path ,exist_ok=True) 
            destination_path =self.model_pusher_config.export_dir_path
            os.makedirs(destination_path ,exist_ok=True)

            logging.info(f"Exporting model file from: [{evaluated_model_file_path}]")
            shutil.copy(src=evaluated_model_file_path ,dst=destination_path) 
            logging.info(f"Exporting model file at: [{destination_path}]")
            

            logging.info(f"Exporting tokenizer file from: [{tokenizer_file_path}]")
            shutil.copy(src= tokenizer_file_path,dst=destination_path)
            logging.info(f"Exporting tokenizer file at: [{destination_path}]") 

            exported_model_file_path = self.model_pusher_config.export_model_file_path
            exported_tokenizer_file_path =self.model_pusher_config.export_tokenizer_file_path
            model_pusher_artifacts = ModelPusherArtifacts(
                is_model_pushed =True ,
                export_model_file_path = exported_model_file_path,
                export_tokenizer_file_path = exported_tokenizer_file_path
            )
            print(model_pusher_artifacts)
            return model_pusher_artifacts
            
            

        except Exception as e:
            raise ClassificationException(e,sys) from e 
