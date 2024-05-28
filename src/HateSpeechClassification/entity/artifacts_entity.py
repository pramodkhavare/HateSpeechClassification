import os ,sys 
from datetime import datetime
from dataclasses import dataclass 
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifacts:
    data_folder_path :str 
    is_ingested :bool 
    message :str 


@dataclass(frozen=True)
class DataTransformationArtifacts:
    is_transformed :bool 
    message :str 
    transformed_file_path :Path 

@dataclass(frozen=True)
class ModelTrainingArtifacts:
    trained_model_file_path :Path 
    tokenizer_file_path :Path
    test_file_path :Path 
    train_file_path :Path 

@dataclass(frozen=True)
class ModelEvaluationArtifacts:
    is_model_accepted :bool 
    model_evaluation_file_path : Path 
    best_model_path :Path
    tokenizer_path :Path  

@dataclass(frozen=True)
class ModelPusherArtifacts:
    is_model_pushed :bool 
    export_model_file_path :Path 
    export_tokenizer_file_path :Path