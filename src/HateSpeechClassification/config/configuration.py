print(123)
from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,TrainingPipelineConfig
from src.HateSpeechClassification.utils.utils import *
from src.HateSpeechClassification.constant import * 
from src.HateSpeechClassification.exception import ClassificationException
from src.HateSpeechClassification.logger import logging 

class ConfigurationManager():
    def __init__(self , config_file_path = CONFIG_FILE_PATH ,
                 current_time_stamp = CURRENT_TIME_STAMP):
        try:
            self.config_info = read_yaml(yaml_file_path= config_file_path) 
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise ClassificationException (e ,sys)
        

    def get_training_pipeline_config(self) ->TrainingPipelineConfig:
        try:
            config = self.config_info[TRAINING_PIPELINE_CONFIG]
            artifact_dir  = os.path.join(ROOT_DIR 
                                         ,config[TRAINING_PIPELINE_CONFIG_ARTIFACTS_DIR])
            
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            return training_pipeline_config

        except Exception as e:
            raise ClassificationException (e ,sys)
        
    def get_data_ingestion_config(self)->DataIngestionConfig:
        try:
            config = self.config_info[DATA_INGESTION_CONFIG_KEY]

            data_ingestion_dir_key = os.path.join(
                self.training_pipeline_config.artifact_dir ,
                config[DATA_INGESTION_DIR_KEY] ,
                self.time_stamp
            )

            dataset_download_url = config[DATASET_DOWNLOAD_URL_KEY]

            tgz_download_dir = os.path.join(
                self.training_pipeline_config.artifact_dir  ,
                data_ingestion_dir_key ,
                config[ZIP_DOWNLOAD_DIR_KEY]
            ) 

            raw_data_dir = os.path.join(
                self.training_pipeline_config.artifact_dir ,
                data_ingestion_dir_key ,
                config[RAW_DATA_DIR_KEY]
            )

            ingested_dir = os.path.join(
                self.training_pipeline_config.artifact_dir ,
                data_ingestion_dir_key ,
                config[INGESTED_DIR_KEY]
            )
        
            data_ingestion_config= DataIngestionConfig(
                dataset_download_url= dataset_download_url,
                tgz_download_dir= tgz_download_dir,
                raw_data_dir= raw_data_dir,
                ingested_dir= ingested_dir
            )
            logging.info("Data ingestion config step completed")
            print(data_ingestion_config)
            return data_ingestion_config

        except Exception as e:
            raise ClassificationException (e ,sys) from e 



# D:\Data Science\NLP\Project\HateSpeechClassification\src\HateSpeechClassification\config\configuration.py
if __name__ == "__main__":
    manager = ConfigurationManager()
    manager.get_data_ingestion_config()