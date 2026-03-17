from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from src.logger import configure_logger
from src.exception import MyException
from src.utils.schema_loader import load_schema
from sklearn.model_selection import train_test_split
import sys

class DataTransformation:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.logger = configure_logger()
        self.logger.info("DataTransformation class initialized.")
        self.schema = load_schema()


    def split_data(self):
        """Splits the DataFrame into training and testing sets."""
        try:
            self.logger.info("Splitting data into training and testing sets.")
            self.target_column = self.schema['target_column']
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            return X, y
        except KeyError as e:
            self.logger.error(f'Error splitting data: {e}')
            raise MyException(e, sys)
        
    def train_test_split(self):
        """
        Splits the data into training and testing sets using an 80-20 split.
        """
        try:
            self.logger.info("Performing train-test split.")
            X, y = self.split_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.logger.info("Train-test split completed successfully.")
            return X_train, X_test, y_train, y_test
        except MyException as e:
            self.logger.error(f'Error during train-test split: {e}')
            raise MyException(e, sys)
        
def start_transformation(df: pd.DataFrame):
    """
    Starts the data transformation process by creating an instance of the DataTransformation class and performing train-test split.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to be transformed.
    
    Returns:
    tuple: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test).
    """
    try:
        transformer = DataTransformation(df)
        return transformer.train_test_split()
    except MyException as e:
        transformer.logger.error(f'Error in start_transformation: {e}')
        raise MyException(e, sys)