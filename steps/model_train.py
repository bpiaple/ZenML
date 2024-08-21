import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from src.model_dev import LinearRegressionModel
from steps.config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    # config: ModelNameConfig,
    config: str,
) -> RegressorMixin:
    """
    Train the model on the ingested data

    Args:
        df: the ingested data

    Returns:
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    """
    try:
        model = None
        if config == "LinearRegressionModel":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            logging.error(f"Model {config} is not supported")
            raise ValueError("Model {config} is not supported")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e