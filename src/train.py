import re
from pathlib import Path

import torch
import torch.nn as nn
import hydra_zen
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from utils.factory import get_model
from utils.logger import setup_wandb
from utils.checkpoint import CheckpointManager
from utils.exporter import ModelExporter
from data import create_dataloaders

def resolve_run_number(ticker: str, model_name: str) -> str:
    """Atomically allocate the next run number by creating a run_NNN directory."""
    model_dir = Path("outputs") / f"{ticker}_{model_name}"
    model_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        run_dir = model_dir / f"run_{n:03d}"
        try:
            run_dir.mkdir()
            return f"{n:03d}"
        except FileExistsError:
            n += 1

OmegaConf.register_new_resolver("run_number", resolve_run_number)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(config):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    with open_dict(config):
        config.checkpoint.dir = str(output_dir / config.checkpoint.dir)
        config.export.dir = str(output_dir) if config.export.dir == "." else str(output_dir / config.export.dir)

    run = setup_wandb(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading data for {config.dataset.ticker}...")
    train_loader, val_loader, test_loader, pipeline = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}\n")

    model = get_model(config).to(device)
    loss_fn = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    checkpoint_manager = CheckpointManager(config)
    start_epoch = 0

    if checkpoint_manager.should_resume():
        checkpoint = checkpoint_manager.load(model, optimizer, device=device)
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _, _ = model(inputs)
            outputs = outputs.squeeze(-1)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs, _, _ = model(inputs)
                outputs = outputs.squeeze(-1)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        run.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "epoch": epoch
        })

        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save checkpoint
        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        checkpoint_manager.save(model, optimizer, epoch, metrics, pipeline)

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _, _ = model(inputs)
            outputs = outputs.squeeze(-1)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"\nFinal Test Loss: {test_loss:.6f}")
    run.log({"test/loss": test_loss})

    # Export model after training
    exporter = ModelExporter(config)
    exported_paths = exporter.export(model, pipeline, device)

    if exported_paths:
        run.log({"exported_formats": list(exported_paths.keys())})

    run.finish()


if __name__ == "__main__":
    train()
