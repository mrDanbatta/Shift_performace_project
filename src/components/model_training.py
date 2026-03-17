import mlflow
from mlflow.metrics import mse
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from src.logger import configure_logger
from src.exception import MyException
from src.utils.schema_loader import load_schema
from src.utils.model_utils import save_object
from src.components.model_pusher import ModelPusher
import sys

logger = configure_logger()

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.squeeze()  # Ensure y_train is a 1D array
        self.y_test = y_test.squeeze()    # Ensure y_test is a 1D array
        self.logger = logger
        self.schema = load_schema()
        self.model_pipeline = None

    def create_model_pipeline(self):
        """Creates a machine learning pipeline with preprocessing and a linear regression model."""
        """Creates a data transformation pipeline for numeric and categorical features."""
        try:
            self.logger.info("Creating data transformation pipeline.")
            numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = self.df.select_dtypes(include=['object']).columns
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]).set_output(transform="pandas")
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', LinearRegression())
            ])
            self.logger.info("Model pipeline created successfully.")
            return pipeline
            # self.logger.info("Data transformation pipeline created successfully.")
            # return preprocessor.fit_transform(self.df)
        except MyException as e:
            self.logger.error(f'Error creating data transformation pipeline: {e}')
            raise MyException(e, sys)

    def train_model(self):
        try:
            self.logger.info("Creating model pipeline.")
            self.model_pipeline = self.create_model_pipeline()
            self.model_pipeline.fit(self.X_train, self.y_train)
            self.logger.info("Model training completed successfully.")
            save_object("artifacts/model/model_pipeline.pkl", self.model_pipeline)
            return self.model_pipeline
        except MyException as e:
            self.logger.error(f'Error during model training: {e}')
            raise MyException(e, sys)
        
    def evaluate_model(self):
        try:
            self.logger.info("Evaluating model performance.")
            ypred = self.model_pipeline.predict(self.X_test)
            mae = sklearn.metrics.mean_absolute_error(self.y_test, ypred)
            r2 = sklearn.metrics.r2_score(self.y_test, ypred)
            self.logger.info(f"Model evaluation completed. MAE: {mae}, R2: {r2}")
            return mae, r2
        except MyException as e:
            self.logger.error(f'Error during model evaluation: {e}')
            raise MyException(e, sys)
        
def start_model_training(X_train, X_test, y_train, y_test):
    try:
        logger.info("Initialising ModelTrainer.")
        trainer = ModelTrainer(X_train, X_test, y_train, y_test)
        pipeline = trainer.train_model()
        mae, r2 = trainer.evaluate_model()
        logger.info(f"Model training and evaluation completed. MAE: {mae}, R2: {r2}")
        
        # push model to MLFlow/Dagshub
        model_pusher = ModelPusher()
        was_registered = model_pusher.push_model(
            model=pipeline,
            r2_score=r2,
            mae_score=mae
        )

        if was_registered:
            logger.info("Model registered successfully.")
        else:
            logger.info("Exsisting model performs better; new pipeline skipped.")
        return r2, mae, trainer
    
    except MyException as e:
        logger.error(f'Error in start_model_training: {e}')
        raise MyException(e, sys)