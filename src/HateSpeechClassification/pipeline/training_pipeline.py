from src.HateSpeechClassification.entity.artifacts_entity import DataIngestionArtifacts 
from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,TrainingPipelineConfig
from src.HateSpeechClassification.utils.utils import *
from src.HateSpeechClassification.constant import * 
from src.HateSpeechClassification.exception import ClassificationException
from src.HateSpeechClassification.logger import logging 
from src.HateSpeechClassification.config.configuration import ConfigurationManager
from src.HateSpeechClassification.components.data_ingestion import DataIngestion
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

            print('Data Ingestion Completed\n')

            return data_ingestion_output
            
             
        except Exception as e:
            raise ClassificationException(e ,sys) from e
        
    def run_pipeline(self):
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

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