import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from omegaconf import OmegaConf, DictConfig
from utils.serialization import serialize_scaler


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Supports:
    - Saving best model based on monitored metric
    - Saving last checkpoint
    - Resuming from checkpoint
    - Storing optimizer state and scaler parameters
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.checkpoint_cfg = cfg.checkpoint
        self.enabled = self.checkpoint_cfg.enabled

        if not self.enabled:
            return

        self.checkpoint_dir = Path(self.checkpoint_cfg.dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best_only = self.checkpoint_cfg.save_best_only
        self.save_last = self.checkpoint_cfg.save_last
        self.monitor = self.checkpoint_cfg.monitor
        self.mode = self.checkpoint_cfg.mode
        self.save_optimizer = self.checkpoint_cfg.save_optimizer
        self.save_scaler = self.checkpoint_cfg.save_scaler

        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_epoch = -1

    def format_filename(self, epoch: int, suffix: str = "") -> str:
        """Format checkpoint filename using template."""
        template = self.checkpoint_cfg.filename
        filename = template.format(
            ticker=self.cfg.dataset.ticker,
            model=self.cfg.models.name,
            epoch=epoch
        )
        if suffix:
            filename = f"{filename}_{suffix}"
        return f"{filename}.pt"

    def is_better(self, current: float) -> bool:
        """Check if current value is better than best."""
        if self.mode == "min":
            return current < self.best_value
        return current > self.best_value

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        pipeline: Any = None,
    ) -> Optional[str]:
        """
        Save checkpoint if conditions are met.

        Args:
            model: The model to save
            optimizer: The optimizer state to save
            epoch: Current epoch number
            metrics: Dictionary of metrics
            pipeline: Data pipeline with scaler to save

        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        if not self.enabled:
            return None

        current_value = metrics.get(self.monitor)
        if current_value is None:
            raise ValueError(f"Monitored metric '{self.monitor}' not found in metrics")

        saved_path = None

        is_best = self.is_better(current_value)
        if is_best:
            self.best_value = current_value
            self.best_epoch = epoch

        if is_best or not self.save_best_only:
            checkpoint = self.create_checkpoint(
                model, optimizer, epoch, metrics, pipeline
            )

            if is_best:
                best_path = self.checkpoint_dir / self.format_filename(epoch, "best")
                torch.save(checkpoint, best_path)
                saved_path = str(best_path)
                print(f"Saved best checkpoint: {best_path.name}")

                run_root_best = self.checkpoint_dir.parent / "best.pt"
                torch.save(checkpoint, run_root_best)

        if self.save_last:
            checkpoint = self.create_checkpoint(
                model, optimizer, epoch, metrics, pipeline
            )
            last_path = self.checkpoint_dir / f"{self.cfg.dataset.ticker}_{self.cfg.models.name}_last.pt"
            torch.save(checkpoint, last_path)

        return saved_path

    def create_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        pipeline: Any = None,
    ) -> Dict[str, Any]:
        """Create checkpoint dictionary."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }

        if self.save_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if self.save_scaler and pipeline is not None:
            scaler_data = serialize_scaler(pipeline.scaler)
            if scaler_data is not None:
                checkpoint["scaler"] = scaler_data

        return checkpoint

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        path: Optional[str] = None,
        device: torch.device = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint into model and optimizer.

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            path: Path to checkpoint (uses resume_from config if None)
            device: Device to map tensors to

        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint_path = path or self.checkpoint_cfg.resume_from

        if checkpoint_path is None:
            raise ValueError("No checkpoint path provided")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_value = checkpoint.get("best_value", self.best_value)
        self.best_epoch = checkpoint.get("best_epoch", -1)

        print(f"  -> Resumed from epoch {checkpoint['epoch']}")
        print(f"  -> Best {self.monitor}: {self.best_value:.6f} (epoch {self.best_epoch})")

        return checkpoint

    def should_resume(self) -> bool:
        """Check if training should resume from checkpoint."""
        return self.enabled and self.checkpoint_cfg.resume_from is not None
