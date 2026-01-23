from typing import List, Optional, Tuple, Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .mongodb import MongoDBClient


class ForexDataset(Dataset):
    """
    PyTorch Dataset for forex OHLC time series data.

    Creates sequences for LSTM training with configurable:
    - Features (o, h, l, c or combinations)
    - Sequence length (lookback window)
    - Prediction horizon
    - Normalization method
    """

    def __init__(
        self,
        data: np.ndarray,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        target_column: int = -1,
    ):
        """
        Args:
            data: Normalized numpy array of shape (n_samples, n_features)
            sequence_length: Number of timesteps to look back
            prediction_horizon: Number of timesteps to predict ahead
            target_column: Column index to use as prediction target
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.sequence_length]

        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        y = self.data[target_idx, self.target_column]

        return x, y


class ForexDataPipeline:
    """
    Complete data pipeline for forex data:
    - Fetches from MongoDB
    - Normalizes data
    - Creates train/val/test splits
    - Returns PyTorch Datasets
    """

    FEATURE_COLUMNS = ["o", "h", "l", "c"]

    def __init__(
        self,
        ticker: str,
        features: List[str] = None,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scaler_type: Literal["minmax", "standard"] = "minmax",
        mongodb_uri: Optional[str] = None,
        mongodb_database: str = "forex",
        mongodb_collection: str = "ohlc",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.ticker = ticker
        self.features = features or ["o", "h", "l", "c"]
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.scaler_type = scaler_type

        self.mongodb_uri = mongodb_uri
        self.mongodb_database = mongodb_database
        self.mongodb_collection = mongodb_collection
        self.start_date = start_date
        self.end_date = end_date

        self.scaler = None
        self.raw_data = None
        self.normalized_data = None

    def prepare(self) -> Tuple[ForexDataset, ForexDataset, ForexDataset]:
        """
        Fetch data, normalize, and create train/val/test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        client = MongoDBClient(
            uri=self.mongodb_uri,
            database=self.mongodb_database,
            collection=self.mongodb_collection
        )

        records = client.fetch_ohlc(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date
        )
        client.close()

        if len(records) == 0:
            raise ValueError(f"No data found for ticker {self.ticker}")

        self.raw_data = np.array([
            [record[f] for f in self.features]
            for record in records
        ], dtype=np.float32)

        print(f"Loaded {len(self.raw_data)} records for {self.ticker}")
        print(f"Date range: {records[0]['t']} to {records[-1]['t']}")
        print(f"Features: {self.features}")

        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler = StandardScaler()

        self.normalized_data = self.scaler.fit_transform(self.raw_data)

        n = len(self.normalized_data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_data = self.normalized_data[:train_end]
        val_data = self.normalized_data[train_end:val_end]
        test_data = self.normalized_data[val_end:]

        print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        target_col = self.features.index("c") if "c" in self.features else -1

        train_dataset = ForexDataset(
            train_data,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            target_column=target_col
        )
        val_dataset = ForexDataset(
            val_data,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            target_column=target_col
        )
        test_dataset = ForexDataset(
            test_data,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            target_column=target_col
        )

        return train_dataset, val_dataset, test_dataset

    def inverse_transform(self, normalized_values: np.ndarray, column: str = "c") -> np.ndarray:
        """
        Convert normalized predictions back to original scale.

        Args:
            normalized_values: Normalized values to transform
            column: Which feature column these values represent

        Returns:
            Values in original scale
        """
        col_idx = self.features.index(column)
        n_features = len(self.features)

        dummy = np.zeros((len(normalized_values), n_features))
        dummy[:, col_idx] = normalized_values.flatten()

        original = self.scaler.inverse_transform(dummy)
        return original[:, col_idx]
