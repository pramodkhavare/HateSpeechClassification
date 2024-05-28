from src.HateSpeechClassification.entity.artifacts_entity import DataIngestionArtifacts,DataTransformationArtifacts ,ModelTrainingArtifacts ,ModelEvaluationArtifacts ,ModelPusherArtifacts
from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,TrainingPipelineConfig
from src.HateSpeechClassification.utils.utils import *
from src.HateSpeechClassification.constant import * 
from src.HateSpeechClassification.exception import ClassificationException
from src.HateSpeechClassification.logger import logging 
from src.HateSpeechClassification.config.configuration import ConfigurationManager
from src.HateSpeechClassification.components.data_ingestion import DataIngestion 
from src.HateSpeechClassification.components.data_transformation import DataTrandformation
from src.HateSpeechClassification.components.model_training import MOdelTrainer
from src.HateSpeechClassification.components.model_evaluation import ModelEvaluation
from src.HateSpeechClassification.components.model_pusher import ModelPusher
import os ,sys
import pandas as pd
import uuid
from threading import Thread

class Pipeline(Thread):
    def __init__(self ,config:ConfigurationManager=ConfigurationManager()):
        try:
            os.makedirs(config.get_training_pipeline_config().artifact_dir ,exist_ok=True)
            super().__init__(daemon=False ,name='pipeline')
            self.config = config 

        except Exception as e:
            raise ClassificationException(e ,sys) from e 
        
    def start_data_ingestion(self) ->DataIngestionArtifacts:
        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
        
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_output = data_ingestion.initiate_data_ingestion()

            return data_ingestion_output
            
             
        except Exception as e:
            raise ClassificationException(e ,sys) from e
    def start_data_transformation(self) ->DataTransformationArtifacts:
        data_transformation_config = self.config.get_data_transformation_config()
        data_ingestion_artifact = self.start_data_ingestion()
        data_transformation = DataTrandformation(
            data_transformation_config= data_transformation_config ,
            data_ingestion_artifact= data_ingestion_artifact
        )
        data_transformation_artifacts = data_transformation.initiate_data_transformation()

        return data_transformation_artifacts 
    
    def start_model_training(self)->ModelTrainingArtifacts:

        model_training_config = self.config.get_model_trainer_config()
        data_transformer_artifacts = self.start_data_transformation()
        model_trainer = MOdelTrainer(
            model_training_config= model_training_config,
            data_transformer_artifacts= data_transformer_artifacts
        )
        model_training_artifacts = model_trainer.initiate_model_training()
        return model_training_artifacts
    def start_model_evaluation(self)->ModelEvaluationArtifacts:
        model_evaluation_config = self.config.get_model_evaluation_config()
        data_transformation_artifacts = self.start_data_transformation()

        model_training_artifacts = self.start_model_training()
        model_evaluation = ModelEvaluation(
            model_evaluation_config= model_evaluation_config,
            data_transformation_artifacts= data_transformation_artifacts,
            model_training_artifacts=  model_training_artifacts
        )
        model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
        return model_evaluation_artifacts
    
    def start_model_pusher(self)->ModelPusherArtifacts:
        try:
            model_pusher_config = self.config.get_model_pusher_config()
            model_evaluation_artifacts = self.start_model_evaluation()
            model_pusher = ModelPusher(
                model_pusher_config= model_pusher_config ,
                model_evaluation_artifacts= model_evaluation_artifacts
            )
            model_pusher_artifacts = model_pusher.initiate_export_model()
        except Exception as e:
            raise ClassificationException(e ,sys) from e


    def run_pipeline(self):
        try:
            # data_ingestion_artifacts = self.start_data_ingestion()
            # data_transformation_artifacts = self.start_data_transformation()
            # model_training_artifacts = self.start_model_training()
            # model_evaluation_artifacts = self.start_model_evaluation()
            model_pusher_artifacts = self.start_model_pusher()

        except Exception as e:
            raise ClassificationException(e ,sys) from e 
        

    def run(self):
        try:
            self.run_pipeline()

        except Exception as e:
            raise ClassificationException(e ,sys) from e 
    
if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline.run()

# D:\Data Science\NLP\Project\HateSpeechClassification\src\HateSpeechClassification\pipeline\training_pipeline.py