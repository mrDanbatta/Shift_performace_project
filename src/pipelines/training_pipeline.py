import pandas as pd
import numpy as np
from src.components.data_ingestion import load_data
from src.components.data_validation import validate
from src.components.data_transformation import start_transformation
from src.components.model_training import start_model_training
from src.exception import MyException
import sys
from src.logger import configure_logger
logger = configure_logger()

def run__training_pipeline():
    try:
        data = load_data()
        logger.info("Data loaded successfully.")
        
        data = validate(data)
        logger.info("Data validation completed successfully.")

        X_train, X_test, y_train, y_test = start_transformation(data)
        logger.info("Data transformation completed successfully.")

        start_model_training(X_train, X_test, y_train, y_test)
        logger.info("Model training pipeline completed successfully.")
    except MyException as e:
        logger.error(f"Error in training pipeline: {e}")
        raise MyException(e, sys)
    
if __name__ == "__main__":
    run__training_pipeline()
