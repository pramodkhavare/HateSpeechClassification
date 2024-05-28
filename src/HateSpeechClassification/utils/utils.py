import os
from src.HateSpeechClassification.exception import ClassificationException 
from src.HateSpeechClassification.logger import logging 
import yaml 
import numpy as np
import pandas as pd 
import dill 
from pathlib import Path
from src.HateSpeechClassification.constant import * 
import pickle
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords
import string ,re 
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
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
    
def save_tokenizer(tokenizer ,tokenizer_file_path :Path):
    try:
        """
        This Function will help you to save tokenizer at desired location
        """
        with open(tokenizer_file_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    except Exception as e:
        logging.info(f'unable to save tokenizer file at {tokenizer_file_path}')
        raise ClassificationException(e ,sys) 
    
def save_model(model , model_file_path :Path):
    try:
        """
        This Function will help you to save model at desired location
        """
        model.save(model_file_path) 
    except Exception as e:
        logging.info(f'unable to save Model file at {model_file_path}')
        raise ClassificationException(e ,sys) 
    
def concat_data_cleaning( words):    
        try:
            # Let's apply stemming and stopwords on the data
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if words not in stopword]
            words=" ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words=" ".join(words)
            return words 

        except Exception as e:
            raise ClassificationException(e, sys) from e 
        
def write_yaml(file_path:str ,data:dict):
    """
    This function willl create Yaml file and save data in that file
    """
    try:
        os.makedirs(os.path.dirname(file_path) ,exist_ok=True)
        with open(file_path ,'w') as yaml_file:
            if data is not None :
                yaml.dump(data ,yaml_file)

    except Exception as e:
        logging.info(f'unable to create Yaml file at {file_path}')
        raise ClassificationException(e ,sys)
    
def save_object(file_path:str,obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise ClassificationException(e,sys) from e
def model_evaluation(model ,tokenizer ,x_test ,y_test):
        try: 
            x_test = x_test.apply(concat_data_cleaning)
            test_sequences = tokenizer.texts_to_sequences(x_test) 
            test_sequences_matrix = pad_sequences(test_sequences,maxlen=MAX_LEN)

            accuracy = model.evaluate(test_sequences_matrix,y_test)
            print(accuracy)
            logging.info(f"the test accuracy is {accuracy}")

            lstm_prediction = model.predict(test_sequences_matrix)
            res = []
            for prediction in lstm_prediction:
                if prediction[0] < 0.5:
                    res.append(0)
                else:
                    res.append(1)
            logging.info(f"the confusion_matrix is {confusion_matrix(y_test,res)} ")
            return accuracy
        except Exception as e:
            raise ClassificationException(e,sys) from e

# def load_object(file_path:str):
#     """
#     file_path: str
#     """
#     try:
#         with open(file_path, "rb") as file_obj:
#             return dill.load(file_obj)
#     except Exception as e:
#         raise ClassificationException(e,sys) from e
