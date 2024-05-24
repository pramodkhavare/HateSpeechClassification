import os ,sys 
from datetime import datetime


CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
ROOT_DIR = os.getcwd()  

CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR , CONFIG_DIR ,CONFIG_FILE_NAME)



#Hard Coded variable related with training pipeline
TRAINING_PIPELINE_CONFIG = 'training_pipeline_config' 
TRAINING_PIPELINE_CONFIG_PIPELINE_NAME = 'pipeline_name' 
TRAINING_PIPELINE_CONFIG_ARTIFACTS_DIR = 'artifact_dir'

#VARIABLE RELATED WITH DATA INGESTION
DATA_INGESTION_CONFIG_KEY = 'data_ingestion_config'
DATA_INGESTION_DIR_KEY = 'data_ingestion_dir'
DATASET_DOWNLOAD_URL_KEY = 'dataset_download_url'
ZIP_DOWNLOAD_DIR_KEY = 'zip_download_dir'
RAW_DATA_DIR_KEY = 'raw_data_dir'
INGESTED_DIR_KEY = 'ingested_dir'
NAME ="PRAMOD"