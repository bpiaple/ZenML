import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

from src.evaluation import MSE, RMSE, R2

@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.Series
                   ) -> Tuple[
                        Annotated[float, "MSE"],
                        Annotated[float, "RMSE"],
                        Annotated[float, "R2"]
                   ]:
    """
    Evaluate the model on the ingested data

    Args:
        model: the trained model
        X_test: pd.DataFrame
        y_test: pd.Series

    Returns:
        None
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)

        logging.info(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")
        return mse, rmse, r2
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e