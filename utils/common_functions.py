import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml


logger = get_logger(__name__)

def read_yaml(file_path):
  try:
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"YAML file not found at {file_path}")

    with open(file_path, 'r') as yaml_file:
      config = yaml.safe_load(yaml_file)
      logger.info(f"YAML file loaded successfully from {file_path}")
      return config
  except Exception as e:
    logger.error(f"Error reading YAML file at {file_path}: {e}")
    raise CustomException(f"Error reading YAML file: {e}")

def load_data(path):
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise CustomException(f"Error loading data: {e}")