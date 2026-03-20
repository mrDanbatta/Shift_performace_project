import pandas as pd
import numpy as np
import sys
from src.logger import configure_logger
from src.exception import MyException
from src.utils.schema_loader import load_schema
import os

# CSV file path
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ShiftPerformance.csv'))

def load_data():
    """
    Loads data from the CSV file and returns it as a pandas DataFrame.
    """
    logger = configure_logger()
    try:
        logger.info("Loading data from CSV file...")
        logger.info(f"CSV file path: {data_path}")
        logger.info(f"CSV file exists: {os.path.exists(data_path)}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found at: {data_path}")
        
        logger.info("Reading CSV file...")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Validate that required columns exist
        schema = load_schema()
        required_columns = [list(col.keys())[0] for col in schema['columns']]
        logger.info(f"Required columns: {required_columns}")
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"CSV is missing required columns: {missing_columns}")
        
        logger.info("Creating artifacts directory...")
        os.makedirs("artifacts/data", exist_ok=True)
        
        logger.info("Saving backup to artifacts/data/shift_performance_data.csv...")
        df.to_csv("artifacts/data/shift_performance_data.csv", index=False)
        
        logger.info("Data loaded successfully from CSV")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}", exc_info=True)
        raise MyException(e, sys)
    except Exception as e:
        logger.error(f"Error loading data from CSV: {type(e).__name__}: {e}", exc_info=True)
        raise MyException(e, sys)