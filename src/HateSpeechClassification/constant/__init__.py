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

#VARIABLE RELATED WITH DATA TRANSFORMATION_CONFIG
DATA_TRANSFORMATION_CONFIG_KEY = 'data_transformation_config'
TRANSFORMED_DIR_KEY = 'transformed_dir'
TRANSFORMED_DATA_DIR_KEY = 'transformed_data_dir'
TRANSFORMED_FILE_NAME_KEY = 'transformed_file_name'


IMBALANCE_DATA_DROP_COLUMNS = ['id']
IMBALANCE_DATA_DROP_COLUMNS_AXIS =1 
RAW_DATA_DROP_COLUMN = ['Unnamed: 0','count','hate_speech','offensive_language','neither']
RAW_DATA_DROP_COLUMNS_AXIS =1 
CLASS  = "class"
LABEL = 'label'
TWEET = 'tweet'
MAX_WORDS = 50000 
MAX_LEN = 300 


#VARIABLE RELATED WITH MODEL TRAINING CONFIG 
MODEL_TRAINER_CONFIG_KEY = 'model_trainer_config' 
MODEL_TRAINING_DIR_KEY = 'model_training_dir'
TRAINED_MODEL_KEY = 'trained_model'
MODEL_FILE_NAME_KEY = 'model_file_name'
TOKENIZER_DIR_KEY = 'tokenizer_dir'
TOKENIZER_FILE_NAME_KEY = 'tokenizer_object_file_name'
TEST_DATA_FOLDER_PATH_KEY = 'test_data_folder_path'
TEST_DATA_FILE_NAME_KEY = 'test_data_file_name'
TRAIN_DATA_FOLDER_PATH_KEY = 'train_data_folder_path'
TRAIN_DATA_FILE_NAME_KEY = 'train_data_file_name'
BATCH_SIZE = 256 
EPOCHS = 1
VALIDATION_SPLIT = 0.2

#VARIABLE RELATED WITH MODEL EVALUATION
MODEL_EVALUATION_CONFIG_KEY = 'model_evaluation_config'
MODEL_EVALUATION_DIR_KEY = 'model_evaluation_dir'
MODEL_EVALUATION_FILE_NAME_KEY = 'model_evaluation_file_name'
BEST_MODEL_KEY ='Best_model'
TIME_STAMP = 'Time'
MODEL_PATH_KEY = 'model_path'
MODEL_HISTORY_KEY = 'Model_History'
TOKENIZER_PATH_KEY = 'Tokenizer_path'
TOKENIZER_HISTORY_KEY = 'Tokenizer_History'
#VARIABLE RELATED WITH MODEL PUSHER 
MODEL_PUSHER_CONFIG_KEY = 'model_pusher_config'
EXPORT_DIR_NAME_KEY = 'export_dir_name'
EXPORT_MODEL_FILE_NAME_KEY = 'export_model_file_name'
EXPORT_TOKENIZER_FILE_NAME_KEY = 'export_tokenizer_file_name'
