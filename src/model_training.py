import os
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data

from scipy.stats import uniform, randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        try:
            logger.info("Loading training and testing data.")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            
            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            
            logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error(f"Error loading or splitting data: {e}")
            raise CustomException(f"Error in data loading or splitting: {e}")
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Starting LightGBM model training.")
            lgb_model = lgb.LGBMClassifier(random_state = self.random_search_params['random_state'])
            
            logger.info("Performing Randomized Search for hyperparameter tuning.")
            
            random_search = RandomizedSearchCV(
                estimator=lgb_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                scoring=self.random_search_params['scoring'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                n_jobs=self.random_search_params['n_jobs']
            )
            
            logger.info("Fitting the model with training data.")
            
            random_search.fit(X_train, y_train)
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best parameters found: {random_search.best_params_}")
            return best_lgbm_model
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException(f"Error in model training: {e}")
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the trained model.")
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            logger.info(f"Model evaluation metrics - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")
            return {
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1_score': f1
            }
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException(f"Error in model evaluation: {e}")
    
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved successfully at {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            raise CustomException(f"Error in saving the model: {e}")
    
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting model training process.")
                logger.info("Starting MLFLOW experiment.")
                logger.info("Logging training and testing dataset to MLFLOW.")

                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                evaluation_metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)
                logger.info("Logging model parameters and metrics to MLFLOW.")

                mlflow.log_artifact(self.model_output_path)
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(evaluation_metrics)

                logger.info("Model training process completed successfully.")
        except CustomException as e:
            logger.error(f"Error in model training process: {e}")
            raise CustomException(f"Error in Starting model training process: {e}")

if __name__ == "__main__":
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    
    trainer.run()