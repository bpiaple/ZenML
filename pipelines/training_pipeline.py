from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@pipeline(enable_cache=True)
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)
