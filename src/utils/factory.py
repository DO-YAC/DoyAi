import importlib
from omegaconf import OmegaConf

def get_model(config):
    """
    Auto-discovers and instantiates models based on config file name.
    Convention: configs/models/lstm.yaml → src.models.lstm.LSTMModel
    """

    model_name = OmegaConf.select(config, "models.name", default=None)

    if model_name is None:
        raise ValueError("Model config must have a 'name' field (e.g., name: lstm)")

    module = importlib.import_module(f"models.{model_name}")

    class_name = f"{model_name.upper()}Model"
    model_class = getattr(module, class_name)

    model_params = {k: v for k, v in OmegaConf.to_container(config.models).items() if k != "name"}

    return model_class(**model_params)