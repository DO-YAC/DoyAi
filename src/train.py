import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import hydra_zen
import hydra
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from utils.factory import get_model
from utils.logger import setup_wandb
from utils.checkpoint import CheckpointManager
from utils.exporter import ModelExporter
from utils.metrics import MetricsCalculator
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
def train(cfg):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    with open_dict(cfg):
        cfg.checkpoint.dir = str(output_dir / cfg.checkpoint.dir)
        cfg.export.dir = str(output_dir) if cfg.export.dir == "." else str(output_dir / cfg.export.dir)

    run_name = f"{cfg.dataset.ticker}_{cfg.models.name}_{output_dir.name}"
    run = setup_wandb(cfg, run_name=run_name)

    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")
    wandb.define_metric("regression/*", step_metric="epoch")
    wandb.define_metric("directional/*", step_metric="epoch")
    wandb.define_metric("real_scale/*", step_metric="epoch")
    wandb.define_metric("error_dist/*", step_metric="epoch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading data for {cfg.dataset.ticker}...")
    train_loader, val_loader, test_loader, pipeline = create_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}\n")

    model = get_model(cfg).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    metrics_calculator = MetricsCalculator()

    checkpoint_manager = CheckpointManager(cfg)
    start_epoch = 0

    if checkpoint_manager.should_resume():
        checkpoint = checkpoint_manager.load(model, optimizer, device=device)
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")):
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
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs, _, _ = model(inputs)
                outputs = outputs.squeeze(-1)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())

        val_loss /= len(val_loader)

        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)
        val_metrics = metrics_calculator.compute(val_predictions, val_targets, pipeline)

        log_dict = {
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/val": val_loss,
        }
        for key, value in val_metrics.items():
            if isinstance(value, float) and np.isnan(value):
                continue
            category, metric = key.split("/", 1)
            log_dict[f"{category}/val_{metric}"] = value
        run.log(log_dict)

        print(f"Epoch {epoch+1}/{cfg.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"  Val R²: {val_metrics['regression/r2']:.4f} | Val DA: {val_metrics['directional/accuracy']:.1f}% | Val MAE (pips): {val_metrics.get('real_scale/mae_pips', 0):.2f}")

        # Save checkpoint
        metrics = {"train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        checkpoint_manager.save(model, optimizer, epoch, metrics, pipeline)

    model.eval()
    test_loss = 0.0
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _, _ = model(inputs)
            outputs = outputs.squeeze(-1)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()

            test_preds.append(outputs.cpu().numpy())
            test_targets.append(targets.cpu().numpy())

    test_loss /= len(test_loader)

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_metrics = metrics_calculator.compute(test_preds, test_targets, pipeline)

    run.summary["test/loss"] = test_loss
    for key, value in test_metrics.items():
        category, metric = key.split("/", 1)
        run.summary[f"test/{category}_{metric}"] = value

    print(f"\nFinal Test Loss: {test_loss:.6f}")
    print(f"  Test R²: {test_metrics['regression/r2']:.4f} | Test DA: {test_metrics['directional/accuracy']:.1f}% | Test MAE (pips): {test_metrics.get('real_scale/mae_pips', 0):.2f}")
    print(f"  Test RMSE: {test_metrics['regression/rmse']:.6f} | Test MAE: {test_metrics['regression/mae']:.6f}")
    print(f"  Test Bias: {test_metrics['error_dist/mean_error_bias']:.6f} | Test Error Std: {test_metrics['error_dist/std']:.6f}")

    # Export model after training
    exporter = ModelExporter(cfg)
    exported_paths = exporter.export(model, pipeline, device)

    if exported_paths:
        run.log({"exported_formats": list(exported_paths.keys())})

    run.finish()


if __name__ == "__main__":
    train()
