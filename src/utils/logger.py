import os
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

load_dotenv()

def setup_wandb(cfg: DictConfig):
    """
    Handles automatic login and initializes a W&B run.
    """
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        print("⚠️ No WANDB_API_KEY found in .env. Logging in may be manual.")

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    print(f"Configuration:\n{conf_dict}")

    run = wandb.init(
        entity=os.getenv("WANDB_ENTITY"), 
        project=os.getenv("WANDB_PROJECT"),
        config=conf_dict,
        job_type="train",
        notes=f"Model: {cfg.models.name}"
    )
    
    return run