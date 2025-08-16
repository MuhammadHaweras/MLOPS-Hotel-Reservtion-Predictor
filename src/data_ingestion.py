import os
import pandas as pd

from google.cloud import storage

from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.bucket_file_name = self.config['bucket_file_name']
        self.train_ratio = self.config['train_ratio']
        self.test_ratio = self.config['test_ratio']

        os.makedirs(RAW_DIR, exist_ok=True)
        
        logger.info(f"DataIngestion initialized with {self.bucket_name} and file is {self.bucket_file_name}" )
    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"File {self.bucket_file_name} downloaded from GCP bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error downloading file from GCP: {e}")
            raise CustomException(f"Error downloading file from GCP: {e}")

    def split_data(self):
        try:
            df = pd.read_csv(RAW_FILE_PATH)
            logger.info(f"Data loaded from {RAW_FILE_PATH} with shape {df.shape}")

            train_df, test_df = train_test_split(df, train_size=self.train_ratio, test_size=self.test_ratio, random_state=42)
            logger.info(f"Data split into train and test sets with shapes {train_df.shape} and {test_df.shape}")

            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Train and test data saved to {TRAIN_FILE_PATH} and {TEST_FILE_PATH}")
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise CustomException(f"Error splitting data: {e}")
    
    def run(self):
        try:
            logger.info("Starting Data ingestion Process.")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully.")
        except CustomException as e:
            logger.error(f"Data ingestion failed: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise CustomException(f"An unexpected error occurred: {e}")
        finally:
            logger.info("Data ingestion process finished.")

if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()