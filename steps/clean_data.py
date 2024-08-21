import logging
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreProcessStrategy, DataSplitStrategy, DataCleaning

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
    ]:
    """
    Step for cleaning and splitting data

    Args:
        df (pd.DataFrame): Input data

    Returns:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Testing data
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        data_cleaned = data_cleaning.handle_data()

        split_strategy = DataSplitStrategy()
        X_train, X_test, y_train, y_test = DataCleaning(data_cleaned, split_strategy).handle_data()
        logging.info("Data cleaning and splitting complete")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e