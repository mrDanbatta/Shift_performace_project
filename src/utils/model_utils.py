import joblib
import os
import sys
from src.exception import MyException
from src.logger import configure_logger

def save_object(file_path, obj):
    """
    Saves a Python object to a file using joblib.
    Args:
        file_path (str): The path where the object will be saved.
        obj: The Python object to be saved.
    Raises:
        MyException: If there is an error during saving the object.
    """
    logger = configure_logger()
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            joblib.dump(obj, file)

        logger.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logger.error(f"Error saving object at {file_path}: {e}")
        raise MyException(e, sys)
    

def load_object(file_path: str):
    """
    Loads a Python object from a file using joblib.
    Args:
        file_path (str): The path from where the object will be loaded.
    Returns:
        The loaded Python object.
    Raises:
        MyException: If there is an error during loading the object.
    """
    logger = configure_logger()
    try:
        with open(file_path, 'rb') as file:
            obj = joblib.load(file)

        logger.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {e}")
        raise MyException(e, sys)