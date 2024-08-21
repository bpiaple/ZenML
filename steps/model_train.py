import logging
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train the model on the ingested data

    Args:
        df: the ingested data

    Returns:
        None
    """
    pass