from typing import Tuple, Optional
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .dataset import ForexDataPipeline, ForexDataset


def create_dataloaders(
    config: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, ForexDataPipeline]:
    """
    Create train/val/test DataLoaders from Hydra config.

    Args:
        config: Hydra config with dataset configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader, pipeline)
        Pipeline is returned for inverse transforms during inference
    """
    dataset_config = config.dataset

    pipeline = ForexDataPipeline(
        ticker=dataset_config.ticker,
        features=list(dataset_config.get("features", ["o", "h", "l", "c"])),
        sequence_length=dataset_config.sequence_length,
        prediction_horizon=dataset_config.get("prediction_horizon", 1),
        train_ratio=dataset_config.get("train_ratio", 0.7),
        val_ratio=dataset_config.get("val_ratio", 0.15),
        scaler_type=dataset_config.get("scaler_type", "minmax"),
        mongodb_uri=dataset_config.get("mongodb_uri", None),
        mongodb_database=dataset_config.get("mongodb_database", "forex"),
        mongodb_collection=dataset_config.get("mongodb_collection", "ohlc"),
        start_date=dataset_config.get("start_date", None),
        end_date=dataset_config.get("end_date", None),
    )

    train_dataset, val_dataset, test_dataset = pipeline.prepare()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    return train_loader, val_loader, test_loader, pipeline
