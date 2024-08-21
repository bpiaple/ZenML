import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy():
    """
    Concrete class implementing strategy for handling data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle data by reading it from a file and returning a DataFrame
        """
        try:
            # delete the blank space in the column names
            data.columns = data.columns.str.strip()
            print(data.columns)
            data = data.drop(
                [
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date"
                ],
                axis=1
            )
            data = data.dropna()
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No comment", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["custumer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e
            

class DataSplitStrategy(DataStrategy):
    """
    Strategy for dividing data into training and testing sets
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data by splitting it into training and testing sets
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14527)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {e}")
            raise e
        
class DataCleaning:
    """
    Class for preprocessing and splitting data
    """
    
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data using the strategy
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
# if __name__ == "__main__":
#     data = pd.read_csv("/workspaces/ZenML/data/olist_customers_dataset.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaned = data_cleaning.handle_data()
#     print(data_cleaned.head())
#     X_train, X_test, y_train, y_test = DataCleaning(data_cleaned, DataSplitStrategy()).handle_data()
#     print(X_train.head())
#     print(X_test.head())
#     print(y_train.head())
#     print(y_test.head())