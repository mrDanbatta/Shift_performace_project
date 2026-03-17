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
        schema = load_schema()
        columns = [list(col.keys())[0] for col in schema['columns']]
        engine = create_engine(f'sqlite:///{data_path}')
        conn = engine.connect()

        df = pd.read_sql_query(f"SELECT {', '.join(columns)} FROM ShiftPerformance", conn)
        conn.close()
        os.makedirs("artifacts/data", exist_ok=True)
        df.to_csv("artifacts/data/shift_performance_data.csv", index=False)
        logger.info("Data loaded and saved to artifacts/data/shift_performance_data.csv")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise MyException(e)