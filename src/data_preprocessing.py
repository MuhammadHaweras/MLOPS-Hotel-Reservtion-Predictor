import os
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
        
        logger.info("DataProcessor initialized with paths for train and test data.")
    
    def preprocess_data(self, df):
        try:
            logger.info("Starting data preprocessing.")
            logger.info("Dropping unnecessary columns")

            df.drop(columns=['Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)
            
            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']
            
            logger.info("Applying Label Encoding to categorical columns.")
            
            label_encoders = {}
            label_mappings = {}


            # Apply label encoding to each object column
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                
                # Store the encoder and mapping
                label_encoders[col] = le
                label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            
            logger.info(f"Label mapings are: ")
            for col, mapping in label_mappings.items():
                logger.info(f"{col}: {mapping}")
            
            logger.info("Handling Skewness by log transformation")
            
            skew_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x: x.skew())
            
            for col in skewness[skewness > skew_threshold].index:
                df[col] = np.log1p(df[col])
                logger.info(f"Applied log transformation to {col} due to skewness.")
            
            return df
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException(f"Error during data preprocessing: {e}")
        
        
    def balance_data(self, df):
        try:
            logger.info("Balancing data using SMOTE.")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled
            
            logger.info("Data balancing completed.")
            return balanced_df
        except Exception as e:
            logger.error(f"Error during data balancing: {e}")
            raise CustomException(f"Error during data balancing: {e}")
    
    def feature_selection(self, df):
        try:
            logger.info("Selecting features based on importance.")
            X  = df.drop(columns=['booking_status'])
            y = df['booking_status']
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            
            feature_importance =  model.feature_importances_
            feature_importance_df = pd.DataFrame({
		            'feature': X.columns,
		            'importance': feature_importance
	        })

            top_imp_features_df = feature_importance_df.sort_values(by='importance', ascending=False)
            num_features_to_select = self.config['data_processing']['no_of_features']
            
            top_10_feature_names = top_imp_features_df.head(num_features_to_select)['feature'].tolist()
            top_10_df = df[top_10_feature_names + ["booking_status"]]
            
            logger.info(f"Selected top {num_features_to_select} features: {top_10_feature_names}")
            
            return top_10_df
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise CustomException(f"Error during feature selection: {e}")
    
    def save_data(self, file_path, df):
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise CustomException(f"Error saving processed data: {e}")
    
    def process(self):
        try:
            logger.info("Loading data from RAW directory.")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            
            # Only balance train data
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            
            # Feature selection on train data
            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]
            
            self.save_data(PROCESSED_TRAIN_DATA_PATH, train_df)
            self.save_data(PROCESSED_TEST_DATA_PATH, test_df)

            logger.info("Processed train and test data saved successfully.")
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise CustomException(f"Error in data processing: {e}")


if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()