from src.logger import configure_logger
import pandas as pd
import numpy as np
from src.exception import MyException
import os

class DataValidation:
    """
    A class to perform data validation on the input dataset.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataValidation class with the input DataFrame.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame to be validated.
        """
        self.df = df
        self.logger = configure_logger()
        self.logger.info("DataValidation class initialized.")

    def check_empty_dataframe(self):
        # check if the DataFrame is empty
        if self.df.empty:
            self.logger.error("The input DataFrame is empty.")
            raise MyException("The input DataFrame is empty.")
        else:
            self.logger.info("The input DataFrame is not empty.")

    def check_missing_values(self):
        # check for missing values in the DataFrame
        missing = self.df.isna().sum()
        if missing.sum() > 0:
            self.logger.error("The input DataFrame contains missing values.")
            raise MyException("The input DataFrame contains missing values.")
        else:
            self.logger.info("The input DataFrame does not contain any missing values.")
        return missing
    
    def fill_missing_values(self):
        """Fills missing values in the DataFrame with the mean of each column and categorical columns with the mode."""
        self.df['maintenance_downtime'] = self.df['maintenance_downtime'].fillna(0)
        self.df['issue_type'] = self.df['issue_type'].fillna('No_Issue')
        self.df['maintenance_flag'] = self.df['maintenance_flag'].fillna(0)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        print(numeric_cols.unique())
        print(categorical_cols.unique())
        for col in numeric_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mean())
                self.logger.info(f"Filled missing values in numeric column '{col}' with mean.")
        for col in categorical_cols:
            if self.df[col].isna().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                self.logger.info(f"Filled missing values in categorical column '{col}' with mode.")
        return self.df
    
    def check_duplicates(self):
        # check for duplicate rows in the DataFrame
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.logger.error(f"The input DataFrame contains {duplicates} duplicate rows.")
            self.df.drop_duplicates(inplace=True)
            self.logger.info(f"Dropped {duplicates} duplicate rows from the DataFrame.")
        else:
            self.logger.info("The input DataFrame does not contain any duplicate rows.")
        return duplicates
    
    def detect_outliers(self):
        outliers_summary = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outliers_summary[col] = outliers
        self.logger.info("Outlier detection completed.")
        return outliers_summary
    
    def cap_outliers(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        self.logger.info("Outlier capping completed.")
        return self.df
    
def validate(df: pd.DataFrame):
    """
    Validates the input DataFrame by performing various checks and transformations.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to be validated.
    
    Returns:
    pd.DataFrame: The validated and transformed DataFrame.
    """
    validator = DataValidation(df)
    validator.check_empty_dataframe()
    validator.check_missing_values()
    df_filled = validator.fill_missing_values()
    validator.check_duplicates()
    outliers_summary = validator.detect_outliers()
    df_capped = validator.cap_outliers()
    df_capped.to_csv("artefacts/data/validated_data.csv", index=False)
    return df_capped

