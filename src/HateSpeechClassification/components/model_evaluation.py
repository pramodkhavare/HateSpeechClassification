import os , sys ,re ,string ,nltk 
import pandas as pd 
import numpy as np 

from src.HateSpeechClassification.entity.config_entity import DataIngestionConfig ,DatatranformationConfig ,ModelEvaluationConfig
from src.HateSpeechClassification.entity.artifacts_entity import DataTransformationArtifacts ,DataIngestionArtifacts  ,ModelTrainingArtifacts ,ModelEvaluationArtifacts
from src.HateSpeechClassification.constant import * 
from src.HateSpeechClassification.exception import ClassificationException
from src.HateSpeechClassification.logger import logging 
from src.HateSpeechClassification.utils.utils import * 
from tensorflow.keras.models import load_model

class ModelEvaluation:
    def __init__(self ,
                 model_evaluation_config : ModelEvaluationConfig ,
                 data_transformation_artifacts : DataTransformationArtifacts ,
                 model_training_artifacts : ModelTrainingArtifacts):
        try:
            logging.info(f"{'*'*20} Model Evaluation Started{'*'*20}")
            print(f"{'*'*20} Model Evaluation Started{'*'*20}") 
            self.model_evaluation_config = model_evaluation_config 
            self.data_transformation_artifacts = data_transformation_artifacts
            self.model_training_artifacts = model_training_artifacts

        except Exception as e:
            raise ClassificationException(e ,sys) from e
        
    def get_best_model_path(self):
        try:
            """"
            This function will help to Get best model from path
            """
            logging.info(f"Getting Best Model From Past") 
            model = None 
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            
            if not os.path.exists(model_evaluation_file_path):
                logging.info(f"{model_evaluation_file_path} is not exist so we will create empty file at same location")
                write_yaml(
                    file_path= model_evaluation_file_path ,
                    data =None
                )
                return model #(model =None) 

            model_evaluation_file_content = read_yaml(model_evaluation_file_path)

            model_evaluation_file_content = dict() if model_evaluation_file_content is None else model_evaluation_file_content
            # model_evaluation_file_content = model_evaluation_file_content or dict()
            if BEST_MODEL_KEY not in model_evaluation_file_content.keys():
                return model 
            
            model_path = model_evaluation_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY]

            return model_path 
        
        except Exception as e:
            raise ClassificationException(e ,sys) from e 
        
    def update_evaluation_report(self ,model_evaluation_artifacts :ModelEvaluationArtifacts):

        try:
            eval_file_path     = model_evaluation_artifacts.model_evaluation_file_path 
            model_eval_content = read_yaml(yaml_file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content

            eval_result  = {
                BEST_MODEL_KEY : {
                    MODEL_PATH_KEY : model_evaluation_artifacts.best_model_path ,
                    TOKENIZER_PATH_KEY : model_evaluation_artifacts.tokenizer_path
                }
            }

            previous_best_model_path     = None 
            previous_best_tokenizer_path = None 
        
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model_path = model_eval_content[BEST_MODEL_KEY][MODEL_PATH_KEY]
                previous_best_tokenizer_path = model_eval_content[BEST_MODEL_KEY][TOKENIZER_PATH_KEY]
            logging.info(f"Previous Best Model Path is : {previous_best_model_path}")
            logging.info(f"Previous Best Tokenizer Path is : {previous_best_tokenizer_path}")

            if previous_best_model_path is not None:
                model_history = {self.model_evaluation_config.time : previous_best_model_path }

                if MODEL_HISTORY_KEY not in model_eval_content:
                    mo_history   = {MODEL_HISTORY_KEY :model_history}
                    eval_result.update(mo_history)
                else:
                    model_eval_content[MODEL_HISTORY_KEY].update(model_history)

            if previous_best_tokenizer_path is not None:
                tokenizer_history = {self.model_evaluation_config.time : previous_best_tokenizer_path }

                if TOKENIZER_HISTORY_KEY not in model_eval_content:
                    to_history = {TOKENIZER_HISTORY_KEY :tokenizer_history}
                    eval_result.update(to_history)
                else:
                    model_eval_content[TOKENIZER_HISTORY_KEY].update(tokenizer_history)
            



            model_eval_content.update(eval_result)

            logging.info(f'Updated evaluation result : {eval_result}')
            write_yaml(file_path=self.model_evaluation_config.model_evaluation_file_path ,data= model_eval_content) 
       

        except Exception as e:
            raise ClassificationException(e ,sys) from e  
        
    def initiate_model_evaluation(self)->ModelEvaluationArtifacts:
        try:
            trained_data_file_path = self.model_training_artifacts.train_file_path 
            test_data_file_path = self.model_training_artifacts.test_file_path 

            train_dataframe = pd.read_csv(trained_data_file_path)
            test_dataframe = pd.read_csv(test_data_file_path)
            x_train = train_dataframe[TWEET]
            y_train = train_dataframe[LABEL]
            x_test = test_dataframe[TWEET]
            y_test = test_dataframe[LABEL]
            
            tokenizer_path = self.model_training_artifacts.tokenizer_file_path 
            with open(tokenizer_path , 'rb') as token:
                tokenizer = pickle.load(token) 
            currrent_trained_model_file_path = self.model_training_artifacts.trained_model_file_path 
            current_trained_model = load_model(currrent_trained_model_file_path) ##This line of code will give you current trained model 

            current_trained_model_accuracy = model_evaluation(
                model=current_trained_model ,tokenizer=tokenizer ,
                x_test = x_test ,y_test = y_test
            )                                                           ##This line of code will give you current trained model accuracy

            print(f"current_trained_model_accuracy : {current_trained_model_accuracy}")


            previous_best_model_path = self.get_best_model_path()

            if previous_best_model_path is None:
                is_model_accepted = True 
                logging.info('We dont have previous best model so we are accepting current best model as best model')
                model_evaluation_artifacts = ModelEvaluationArtifacts(
                    is_model_accepted= is_model_accepted ,
                    best_model_path = currrent_trained_model_file_path,
                    model_evaluation_file_path= self.model_evaluation_config.model_evaluation_file_path,      #path of .yaml file
                    tokenizer_path = tokenizer_path
                )
                print(model_evaluation_artifacts)
                self.update_evaluation_report(
                    model_evaluation_artifacts=model_evaluation_artifacts
                )
                return model_evaluation_artifacts
            
            else:
 
                logging.info("we have previous best model so fetching it")
                previous_best_model = load_model(previous_best_model_path)
                previous_best_model_accuracy = model_evaluation(model=previous_best_model ,
                                                       tokenizer= tokenizer ,
                                                       x_test=x_test ,y_test=y_test)
                print(f"previous_best_model_accuracy : {previous_best_model_accuracy}")
                logging.info("Comparing loss between best_model_loss and trained_model_loss ")
            
            
            if previous_best_model_accuracy[1] > current_trained_model_accuracy[1]:
                model_evaluation_artifacts = ModelEvaluationArtifacts(
                    is_model_accepted= False,
                    model_evaluation_file_path= self.model_evaluation_config.model_evaluation_file_path ,
                    best_model_path=previous_best_model_path  ,
                    tokenizer_path = tokenizer_path
                )
                print(model_evaluation_artifacts)
                self.update_evaluation_report(
                    model_evaluation_artifacts =model_evaluation_artifacts
                )
                logging.info(f"{'*'*20} Model Evaluation Completed{'*'*20}")
                print(f"{'*'*20} Model Evaluation Completed{'*'*20}") 
                return model_evaluation_artifacts
            
            else:
                logging.info('Trained Model Is Performing Better Than Previous One')
                model_evaluation_artifacts = ModelEvaluationArtifacts(
                    is_model_accepted= True ,
                    best_model_path= currrent_trained_model_file_path ,
                    model_evaluation_file_path= self.model_evaluation_config.model_evaluation_file_path ,
                    tokenizer_path = tokenizer_path
                )
                print(model_evaluation_artifacts)
                self.update_evaluation_report(
                    model_evaluation_artifacts= model_evaluation_artifacts
                )
                logging.info(f"{'*'*20} Model Evaluation Completed{'*'*20}")
                print(f"{'*'*20} Model Evaluation Completed{'*'*20}") 
                return model_evaluation_artifacts


        except Exception as e:
            raise ClassificationException(e ,sys) from e   
        
    