
from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,TrainingPipelineConfig ,DatatranformationConfig ,ModelTrainingConfig, ModelEvaluationConfig ,ModelPusherConfig
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
            return data_ingestion_config

        except Exception as e:
            raise ClassificationException (e ,sys) from e 
        
    def get_data_transformation_config(self)->DatatranformationConfig:
        try:

            logging.info("Getting Data Transformation Config Component") 
            config = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            transformed_dir  = os.path.join(
                self.training_pipeline_config.artifact_dir ,
                config[TRANSFORMED_DIR_KEY] ,
                self.time_stamp
            )
            transformed_data_dir = os.path.join(
                transformed_dir ,
                config[TRANSFORMED_DATA_DIR_KEY]
            )
            transformed_file_name = config[TRANSFORMED_FILE_NAME_KEY]

            data_transformation_config = DatatranformationConfig(
                transformed_dir = transformed_dir,
                transformed_data_dir = transformed_data_dir,
                transformed_file_name = transformed_file_name
            )

            logging.info(f"Data Transformation Config : [{data_transformation_config}]")
            return data_transformation_config
        except Exception as e:
            raise ClassificationException (e ,sys) from e 
    def get_model_trainer_config(self) ->ModelTrainingConfig:
        try:
            config = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            model_training = os.path.join(
                self.training_pipeline_config.artifact_dir ,
                config[MODEL_TRAINING_DIR_KEY] ,
                self.time_stamp
            )
            trained_model_folder_path = os.path.join(
                model_training ,
                config[TRAINED_MODEL_KEY]
            )
            trained_model_file_name = config[MODEL_FILE_NAME_KEY]
            tokenizer_folder_path = os.path.join(
                model_training ,
                config[TOKENIZER_DIR_KEY]
            )

            tokenizer_file_name = config[TOKENIZER_FILE_NAME_KEY]

            test_data_folder_path = os.path.join(
                model_training ,
                config[TEST_DATA_FOLDER_PATH_KEY]
            )
            test_data_file_name = config[TEST_DATA_FILE_NAME_KEY]
            train_data_folder_path = os.path.join(
                model_training ,
                config[TRAIN_DATA_FOLDER_PATH_KEY]
            )
            train_data_file_name = config[TRAIN_DATA_FILE_NAME_KEY]
            model_training_config = ModelTrainingConfig(
                model_training= model_training,
                trained_model_folder_path= trained_model_folder_path,
                trained_model_file_name= trained_model_file_name,
                tokenizer_folder_path = tokenizer_folder_path,
                tokenizer_file_name=  tokenizer_file_name ,
                test_data_folder_path= test_data_folder_path,
                test_data_file_name= test_data_file_name ,
                train_data_folder_path = train_data_folder_path ,
                train_data_file_name =train_data_file_name
            ) 

            return model_training_config
        except Exception as e:
            raise ClassificationException (e ,sys) from e 
    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        try:
            config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]

            model_evalution_dir_path = os.path.join(
                self.training_pipeline_config.artifact_dir ,
                config[MODEL_EVALUATION_DIR_KEY] 
            )
            model_evaluation_file_path = os.path.join(
                model_evalution_dir_path ,
                config[MODEL_EVALUATION_FILE_NAME_KEY]
            )

            model_evaluation_config = ModelEvaluationConfig(
                model_evalution_dir_path = model_evalution_dir_path ,
                model_evaluation_file_path = model_evaluation_file_path ,
                time= self.time_stamp
            ) 

            return model_evaluation_config
        except Exception as e:
            raise ClassificationException (e ,sys) from e

    def get_model_pusher_config(self)->ModelPusherConfig:
        try:
            config = self.config_info[MODEL_PUSHER_CONFIG_KEY] 

            export_dir_path = os.path.join(
                self.training_pipeline_config.artifact_dir ,
                config[EXPORT_DIR_NAME_KEY]
            )
            export_model_file_path = os.path.join(
                export_dir_path , config[EXPORT_MODEL_FILE_NAME_KEY]
            )
            export_tokenizer_file_path = os.path.join(
                export_dir_path , config[EXPORT_TOKENIZER_FILE_NAME_KEY]
            )
            
            model_pusher_config = ModelPusherConfig(
                export_dir_path= export_dir_path,
                export_model_file_path= export_model_file_path,
                export_tokenizer_file_path =export_tokenizer_file_path
            )
            return model_pusher_config
        except Exception as e:
            raise ClassificationException (e ,sys) from e