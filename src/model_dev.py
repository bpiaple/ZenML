import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model

        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels

        Returns:
            None
        """
        pass

class LinearRegressionModel(Model):
    """
    Concrete class for Linear Regression model
    """

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Train the Linear Regression model

        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels

        Returns:
            None
        """
        try:
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model trained successfully")
            return model
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e