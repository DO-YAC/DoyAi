import importlib
from omegaconf import OmegaConf

def get_model(cfg):
    """
    Auto-discovers and instantiates models based on config file name.
    Convention: configs/models/lstm.yaml → src.models.lstm.LSTMModel
    """
    # Get the model name from Hydra's choice (e.g., "lstm" from models/lstm.yaml)
    # Note: Hydra uses folder name as key, so it's cfg.models (plural)
    model_name = OmegaConf.select(cfg, "models.name", default=None)

    if model_name is None:
        raise ValueError("Model config must have a 'name' field (e.g., name: lstm)")

    # Dynamic import: src.models.{name}
    module = importlib.import_module(f"src.models.{model_name}")

    # Convention: class name is {Name}Model (e.g., LSTMModel)
    class_name = f"{model_name.upper()}Model"
    model_class = getattr(module, class_name)

    # Extract model params (exclude 'name' which is just for routing)
    model_params = {k: v for k, v in OmegaConf.to_container(cfg.models).items() if k != "name"}

    return model_class(**model_params)