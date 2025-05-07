import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml,load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn


logger = get_logger(__name__)

class ModelTraining:

    def __init__(self,TRAIN_PATH,TEST_PATH,model_output_path):
        self.train_path=TRAIN_PATH
        self.test_path=TEST_PATH
        self.model_output_path=model_output_path

        self.params_dist= LIGHTGBM_PARAMS
        self.random_search_params= RANDOM_SEARCH_PARAMS
        
    def load_and_split_data(self):
        try:
            logger.info(f"Load and Split data Starting...{self.train_path}")
            train_df=load_data(self.train_path)

            logger.info(f"Load and Split data Starting...{self.test_path}")
            test_df=load_data(self.test_path)

            X_train=train_df.drop(columns=["booking_status"])
            y_train=train_df["booking_status"]

            X_test=test_df.drop(columns=["booking_status"])
            y_test=test_df["booking_status"]

            logger.info("Data splitted successfully")

            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Error while loading data",e)
        
    def train_model(self,X_train,y_train):
        try: 
            logger.info("initializing model")

            lgbm_model=lgb.LGBMClassifier()

            logger.info("Tuning Hyperparameters")

            random_search=RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv= self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                scoring=self.random_search_params["scoring"]
            )

            logger.info("Starting Model TRAINING with Hyperparameters")
            random_search.fit(X_train,y_train)
            logger.info("TRAINING Completed")

            best_params= random_search.best_params_
            best_lgbm_model= random_search.best_estimator_

            logger.info(f"Best parameters are: {best_params}")

            logger.info("We have the model trained")
            
            return best_lgbm_model

        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Error while training model", e)
        
    def evaluate_model(self, model, X_test,y_test):
        try:
            logger.info("Evaluating the model")
            y_pred=model.predict(X_test)
            accuracy= accuracy_score(y_test,y_pred)
            precision=precision_score(y_test,y_pred)
            recall=recall_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred)

            logger.info(f"Accuracy Score: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall : {recall}")
            logger.info(f"F1 Score: {f1}")

            return {
                "accuracy":accuracy,
                "precision":precision,
                "recall":recall,
                "f1":f1
            }
        except Exception as e:
            logger.error(f"Error while evaluating the model {e}")
            raise CustomException("Failed to evaluate the model",e)
        
    def save_model(self,model):
        try:
            logger.info("Saving the model...")
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            joblib.dump(model,self.model_output_path)

            logger.info("MODEL SAVED")
        
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Error while saving model",e)


    def run(self):
        try:
            with mlflow.start_run():
                logger.info("STARTING MODEL TRAINING PIPELINE")

                logger.info("ML FLOW STARTED")

                logger.info("LOGGING THE TRAINING AND TESTING DATASET TO MLFLOW")

                mlflow.log_artifact(self.train_path,artifact_path="datasets")
                mlflow.log_artifact(self.test_path,artifact_path="datasets")


                X_train,y_train,X_test,y_test= self.load_and_split_data()
                best_lgbm_model = self.train_model(X_train,y_train)
                metrics=self.evaluate_model(best_lgbm_model,X_test,y_test)
                self.save_model(best_lgbm_model)

                logger.info("LOGGIN THE MODEL WITH MLFLOW AND BEST PARAMS")
                mlflow.log_artifact(self.model_output_path)
                mlflow.log_params(best_lgbm_model.get_params())

                logger.info("LOGGIN METRICS WITH MLFLOW")
                mlflow.log_metrics(metrics)


                logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        
        except Exception as e:
            logger.error(f"Error in training pipeline {e}")
            raise CustomException("Failed during model training pipeline",e)
        
        
if __name__== "__main__":
    trainer= ModelTraining(processed_train_data_path,processed_test_data_path,MODEL_OUTPUT_PATH)
    trainer.run()


        