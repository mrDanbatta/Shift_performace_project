import pandas as pd
import numpy as np
import mlflow
from src.exception import MyException
from src.logger import configure_logger
import sys

logger = configure_logger()

MLFLOW_TRACKING_URI = "https://dagshub.com/mrDanbatta/shift-optimisation-system.mlflow/"
REGISTERED_MODEL_NAME = "ShiftPerformanceModel"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_cached_model = None

def load_model():
    global _cached_model
    if _cached_model is None:
        try:
            logger.info("Loading model from MLFlow registry.")
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
            _cached_model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.warning(f"MLFlow loading failed, attempting local load: {e}")
            # try:
            #     _cached_model = mlflow.sklearn.load_model("file:///artifacts/model/best_ridge_model.pkl")
            #     logger.info("Model loaded from local storage.")
            # except Exception as e:
            #     logger.error("Both MLFlow and local loading failed.")
            #     raise MyException(e, sys)
    return _cached_model