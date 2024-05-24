from src.HateSpeechClassification.entity.artifacts_entity import DataIngestionArtifacts 
from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,TrainingPipelineConfig
from src.HateSpeechClassification.utils.utils import *
from src.HateSpeechClassification.constant import * 
from src.HateSpeechClassification.exception import ClassificationException
from src.HateSpeechClassification.logger import logging 


import os ,sys 
import shutil
from six.moves import urllib
import pandas as pd
import numpy as np
import tarfile 
import zipfile

class DataIngestion():
    def __init__(self ,data_ingestion_config :DataIngestionConfig):
        try:
            logging.info(f"{'*'*20}Data Ingestion Step Started{'*'*20}")
            self.config = data_ingestion_config  
            # print(self.config)
        except Exception as e:
            raise ClassificationException(e ,sys) from e
        
    def download_zip_file(self):
        try:          
            if os.path.exists(self.config.tgz_download_dir):
               (
                  shutil.rmtree(self.config.tgz_download_dir) 
               ) 
            os.makedirs(self.config.tgz_download_dir ,exist_ok=True)

            housing_file_name = os.path.basename(self.config.dataset_download_url)
            tgz_file_path = os.path.join(self.config.tgz_download_dir ,housing_file_name)

            logging.info(f"Downloading Data at file: [{tgz_file_path}]  from url: [{self.config.dataset_download_url}]")
            filename ,url = urllib.request.urlretrieve(
                url= self.config.dataset_download_url ,
                filename= tgz_file_path
            )
            logging.info(f"File : [{tgz_file_path}] has been downloaded successfully")

            return tgz_file_path

        except Exception as e:
            # logging.info(f'Unable to Donload file: [{tgz_file_path}]')
            raise ClassificationException(e,sys) from e
            

    def extract_tgz_file(self ,tgz_file_path :str):
        try:
            if os.path.exists(self.config.raw_data_dir):
                shutil.rmtree(self.config.raw_data_dir)

            os.makedirs(self.config.raw_data_dir ,exist_ok=True)

            logging.info(f"Extracting data into [{self.config.raw_data_dir}]")

            # with tarfile.open(tgz_file_path) as housing_tgz_file_obj:
            #     housing_tgz_file_obj.extractall(path=self.config.raw_data_dir)
            with zipfile.ZipFile(tgz_file_path, 'r') as zip_ref:
            # Extract all the contents
                 zip_ref.extractall(self.config.raw_data_dir)

            logging.info(f"Extraction is completed") 
            data_folder_path= self.config.raw_data_dir
            data_ingestion_artifacts =DataIngestionArtifacts(
                data_folder_path= data_folder_path ,
                is_ingested= True ,
                message=f"Data Ingestion is completed successfully"
            )

        except Exception as e:
            logging.info("Unable to unzip data")
            raise ClassificationException(e ,sys) from e
        
    def initiate_data_ingestion(self)->DataIngestionArtifacts:
        try:
            tgz_file_path = self.download_zip_file()

            data_ingestion_artifacts = self.extract_tgz_file(tgz_file_path=tgz_file_path)
            return data_ingestion_artifacts
        except Exception as e:
            raise ClassificationException(e ,sys) from e
    def _del_(self):
        logging.info(f"{'*'*20} Data Ingesteion Pipeline Completed {'*'*20}")


