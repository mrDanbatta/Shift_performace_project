import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from src.logger import configure_logger
from src.exception import MyException
from src.utils.schema_loader import load_schema
import os

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ShiftData.db'))

def load_data():
    """
    Loads data from the SQLite database and returns it as a pandas DataFrame.
    """
    logger = configure_logger()
    try:
        logger.info("Loading data from the database...")
        logger.info(f"Database path: {data_path}")
        logger.info(f"Database exists: {os.path.exists(data_path)}")
        
        schema = load_schema()
        columns = [list(col.keys())[0] for col in schema['columns']]
        logger.info(f"Schema columns: {columns}")
        
        # Create SQLite engine with timeout and proper connection settings
        engine = create_engine(
            f'sqlite:///{data_path}',
            connect_args={'timeout': 30},  # 30 second timeout
            pool_pre_ping=True,  # Test connection before using
            echo=False
        )
        logger.info("SQLite engine created with 30s timeout")
        
        # Use context manager for safe connection handling
        with engine.connect() as conn:
            logger.info("Database connection established")
            
            logger.info("Starting SQL query execution...")
            query = f"SELECT {', '.join(columns)} FROM ShiftPerformance"
            logger.info(f"Query: {query}")
            
            df = pd.read_sql_query(query, conn)
            logger.info(f"Data fetched: {df.shape[0]} rows, {df.shape[1]} columns")
        
        logger.info("Connection closed")
        
        logger.info("Creating artifacts directory...")
        os.makedirs("artifacts/data", exist_ok=True)
        
        logger.info("Saving data to CSV...")
        df.to_csv("artifacts/data/shift_performance_data.csv", index=False)
        logger.info("Data loaded and saved to artifacts/data/shift_performance_data.csv")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {type(e).__name__}: {e}", exc_info=True)
        raise MyException(e)