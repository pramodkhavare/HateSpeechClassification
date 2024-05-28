import os ,sys 
from datetime import datetime
from dataclasses import dataclass 
from pathlib import Path
from time import time
def get_time_stamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


@dataclass(frozen= True)
class TrainingPipelineConfig:
    artifact_dir :str

@dataclass(frozen=True)
class DataIngestionConfig:
    dataset_download_url : str 
    tgz_download_dir :str 
    raw_data_dir :str 
    ingested_dir : str 

@dataclass(frozen=True)
class DatatranformationConfig:
    transformed_dir : Path 
    transformed_data_dir : Path 
    transformed_file_name :str

@dataclass(frozen=True)
class ModelTrainingConfig:
    model_training :Path 
    trained_model_folder_path :Path 
    trained_model_file_name :str 
    tokenizer_folder_path :Path 
    tokenizer_file_name :str 
    test_data_folder_path : Path 
    test_data_file_name :str
    train_data_folder_path:Path 
    train_data_file_name :str 

@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_evalution_dir_path : Path 
    model_evaluation_file_path :Path
    time: time 


@dataclass(frozen=True)
class ModelPusherConfig:
    export_dir_path :str 
    export_model_file_path :str 
    export_tokenizer_file_path :str 
