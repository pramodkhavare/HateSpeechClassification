import os ,sys 
from datetime import datetime
from dataclasses import dataclass 

@dataclass(frozen=True)
class DataIngestionArtifacts:
    data_folder_path :str 
    is_ingested :bool 
    message :str 
