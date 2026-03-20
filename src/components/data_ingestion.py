import pandas as pd
import numpy as np
import sqlite3
from src.logger import configure_logger
from src.exception import MyException
from src.utils.schema_loader import load_schema
import os

data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ShiftData.db'))

def load_data():
    """
    Loads data from the SQLite database and returns it as a pandas DataFrame.
    Uses sqlite3 directly with timeout handling.
    """
    logger = configure_logger()
    try:
        logger.info("Loading data from the database...")
        logger.info(f"Database path: {data_path}")
        logger.info(f"Database exists: {os.path.exists(data_path)}")
        
        schema = load_schema()
        columns = [list(col.keys())[0] for col in schema['columns']]
        logger.info(f"Schema columns: {columns}")
        
        logger.info("Opening SQLite connection with 10s timeout...")
        # Use sqlite3 directly with timeout
        conn = sqlite3.connect(data_path, timeout=10.0)
        conn.execute('PRAGMA query_only = ON')  # Read-only mode
        logger.info("Database connection established")
        
        logger.info("Starting SQL query execution...")
        query = f"SELECT {', '.join(columns)} FROM ShiftPerformance"
        logger.info(f"Query: {query}")
        
        # Use timeout for the query
        conn.timeout = 10  # 10 second timeout
        df = pd.read_sql_query(query, conn)
        logger.info(f"Data fetched: {df.shape[0]} rows, {df.shape[1]} columns")
        
        logger.info("Closing database connection...")
        conn.close()
        logger.info("Connection closed")
        
        logger.info("Creating artifacts directory...")
        os.makedirs("artifacts/data", exist_ok=True)
        
        logger.info("Saving data to CSV...")
        df.to_csv("artifacts/data/shift_performance_data.csv", index=False)
        logger.info("Data loaded and saved to artifacts/data/shift_performance_data.csv")
        return df
    except sqlite3.OperationalError as e:
        logger.error(f"SQLite OperationalError: {e}", exc_info=True)
        logger.error("This may indicate: database is locked, table doesn't exist, or corrupted")
        raise MyException(str(e))
    except Exception as e:
        logger.error(f"Error loading data: {type(e).__name__}: {e}", exc_info=True)
        raise MyException(e)