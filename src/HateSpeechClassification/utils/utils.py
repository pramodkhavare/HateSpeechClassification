import os
from src.HateSpeechClassification.exception import ClassificationException 
from src.HateSpeechClassification.logger import logging 
import yaml 
import numpy as np
import pandas as pd 
import dill 
from src.HateSpeechClassification.constant import * 


class GCloudSync:

    def sync_folder_to_gcloud(self, gcp_bucket_url, filepath, filename):

        command = f"gsutil cp {filepath}/{filename} gs://{gcp_bucket_url}/"
        # command = f"gcloud storage cp {filepath}/{filename} gs://{gcp_bucket_url}/"
        os.system(command)

    def sync_folder_from_gcloud(self, gcp_bucket_url, filename, destination):

        command = f"gsutil cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        # command = f"gcloud storage cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        os.system(command)


def read_yaml(yaml_file_path:str):
    try:
        """
        Read yaml file and return content as dictionary
        yaml_file_path :str
        """
        with open(yaml_file_path , 'r') as file:
            content =  yaml.safe_load(file)
            return content

    except Exception as e:
        logging.info(f'unable to read Yaml file at {yaml_file_path}')
        raise ClassificationException(e ,sys)
    