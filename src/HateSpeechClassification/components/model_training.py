import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import os , sys ,re ,string ,nltk 
import pandas as pd 
import numpy as np 
nltk.download('stopwords')
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords

from src.HateSpeechClassification.entity.artifacts_entity import DataTransformationArtifacts ,DataIngestionArtifacts ,ModelTrainingArtifacts
from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,DatatranformationConfig ,ModelTrainingConfig
from src.HateSpeechClassification.exception import ClassificationException
from src.HateSpeechClassification.logger import logging 
from src.HateSpeechClassification.utils.utils import *
from src.HateSpeechClassification.constant import * 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,SpatialDropout1D
from tensorflow.keras.utils import pad_sequences
class MOdelTrainer:
    def __init__(self ,
                 model_training_config : ModelTrainingConfig ,
                 data_transformer_artifacts : DataTransformationArtifacts):
        try:
            logging.info(f"{'*'*20} Model Training Started{'*'*20}")
            print(f"{'*'*20} Model Training Started{'*'*20}") 
            self.model_training_config = model_training_config 
            self.data_transformer_artifacts = data_transformer_artifacts
            print(self.model_training_config)
            print(self.data_transformer_artifacts)
        except Exception as e:
            raise ClassificationException(e ,sys) from e
    def split_data(self):
        try:
            data = pd.read_csv(self.data_transformer_artifacts.transformed_file_path)
            data[TWEET] = data[TWEET].apply(concat_data_cleaning)
            X = data[TWEET]
            Y = data[LABEL]
            logging.info("Applying train_test_split on the data")
            x_train,x_test,y_train,y_test = train_test_split(X ,Y ,random_state=42) 
            logging.info("Exited the spliting the data function")
            return x_train,x_test,y_train,y_test
        except Exception as e:
            raise ClassificationException(e ,sys) from e
        

    def get_tokenizer(self ,x_train):
        try:
            logging.info("Applying tokenization on the data")
            tokenizer = Tokenizer(num_words=MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            return tokenizer
        except Exception as e:
            raise ClassificationException(e ,sys) from e
        
    def get_model(self):
        try:
            model = Sequential()
            model.add(Embedding(input_dim=50000, output_dim=100, input_length=300))
            model.add(SpatialDropout1D(0.2))
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(1, activation='sigmoid'))
            model.build(input_shape=(None ,MAX_LEN))
            model.summary()
            model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
            return model
        except Exception as e:
            raise ClassificationException(e ,sys) from e
    
    def initiate_model_training(self):
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            x_train,x_test,y_train,y_test = self.split_data()
            test_datframe = pd.concat([x_test ,y_test],axis=1)
            train_dataframe = pd.concat([x_train ,y_train] ,axis=1)

            train_file_path = os.path.join(
                self.model_training_config.train_data_folder_path ,
                self.model_training_config.train_data_file_name
            )
            os.makedirs(self.model_training_config.train_data_folder_path ,exist_ok=True)
            train_dataframe.to_csv(train_file_path , header=['tweet' ,'label'] ,index=False)


            test_file_path = os.path.join(
                self.model_training_config.test_data_folder_path ,
                self.model_training_config.test_data_file_name 
            )
            os.makedirs(self.model_training_config.test_data_folder_path ,exist_ok=True)
            test_datframe.to_csv(test_file_path , header=['tweet' ,'label'] ,index=False)


            model  = self.get_model()

            tokenizer = self.get_tokenizer(x_train)


            tokenizer_file_path = os.path.join(
                self.model_training_config.tokenizer_folder_path ,
                self.model_training_config.tokenizer_file_name
            )
            
            os.makedirs(self.model_training_config.tokenizer_folder_path)
            save_tokenizer(tokenizer=tokenizer ,
                           tokenizer_file_path=tokenizer_file_path)
            
            sequences = tokenizer.texts_to_sequences(x_train)
            sequences_matrix = pad_sequences(sequences , maxlen = MAX_LEN)
            model.fit(
                sequences_matrix , y_train,
                batch_size = BATCH_SIZE ,
                epochs= EPOCHS , 
                validation_split = VALIDATION_SPLIT
            )
            logging.info("Model training finished ! saving the model")
            model_file_path = os.path.join(
                self.model_training_config.trained_model_folder_path ,
                self.model_training_config.trained_model_file_name
            )
            os.makedirs(self.model_training_config.trained_model_folder_path ,exist_ok=True)
            save_model(model= model ,model_file_path= model_file_path)

            model_training_artifacts = ModelTrainingArtifacts(
                trained_model_file_path= model_file_path,
                tokenizer_file_path= tokenizer_file_path ,
                test_file_path = test_file_path ,
                train_file_path = train_file_path
            )
            print(f"{'*'*20} Model Training Completed {'*'*20}") 
            logging.info(f"{'*'*20} Model Training Completed {'*'*20}") 

            return model_training_artifacts
        except Exception as e:
            raise ClassificationException(e ,sys) from e
