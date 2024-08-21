from zenml.steps import BaseParameters
# from pydantic import BaseModel

class ModelNameConfig(BaseParameters):
    nom_models: str = 'LinearRegressionModel'

    def __init__(self, model_name: str):
        self.model_name = model_name