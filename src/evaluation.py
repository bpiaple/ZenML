import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, r2_score

class EvaluationStrategy(ABC):
    """
    Abstract class defining strategy for evaluating model
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate scores for the model
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            None
        """
        pass

class MSE(EvaluationStrategy):
    """
    Concrete class implementing strategy for calculating MSE
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate MSE for the model
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            None
        """
        try:
            logging.info("Calculating Mean Squared Error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e
        
class RMSE(EvaluationStrategy):
    """
    Concrete class implementing strategy for calculating RMSE
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate RMSE for the model
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            None
        """
        try:
            logging.info("Calculating Root Mean Squared Error")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e
        
class R2(EvaluationStrategy):
    """
    Concrete class implementing strategy for calculating R2 score
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate R2 score for the model
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels

        Returns:
            None
        """
        try:
            logging.info("Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 score: {e}")
            raise e