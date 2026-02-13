from typing import Dict, Optional

import numpy as np
from scipy import stats


class MetricsCalculator:
    """
    Computes metrics for forex price prediction models.

    Metric categories:
    - regression/   : Core regression metrics (RMSE, MAE, R², etc.)
    - percentage/   : Scale-independent metrics (MAPE, sMAPE)
    - directional/  : Forecasting direction metrics (DA, weighted DA)
    - real_scale/   : Inverse-transformed metrics in original price units
    - error_dist/   : Error distribution statistics (bias, std, percentiles, skew, kurtosis)
    """

    def __init__(self, pip_size: float = 0.0001):
        """
        Args:
            pip_size: Size of one pip in price units (0.0001 for major forex pairs like EURUSD).
        """
        self.pip_size = pip_size

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        pipeline=None,
    ) -> Dict[str, float]:
        """
        Compute all metrics.

        Args:
            predictions: Model predictions (normalized scale), shape (N,)
            targets: Ground truth values (normalized scale), shape (N,)
            pipeline: Optional ForexDataPipeline for inverse-transforming to real scale.

        Returns:
            Flat dictionary of metric_name -> value.
        """
        predictions = predictions.flatten()
        targets = targets.flatten()

        metrics = {}
        metrics.update(self.regression(predictions, targets))
        metrics.update(self.percentage(predictions, targets))
        metrics.update(self.directional(predictions, targets))
        metrics.update(self.error_distribution(predictions, targets))

        if pipeline is not None:
            metrics.update(self.real_scale(predictions, targets, pipeline))

        return metrics

    def regression(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        errors = predictions - targets
        abs_errors = np.abs(errors)

        mse = float(np.mean(errors ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_errors))

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "regression/mse": mse,
            "regression/rmse": rmse,
            "regression/mae": mae,
            "regression/r2": r2,
            "regression/max_absolute_error": float(np.max(abs_errors)),
            "regression/median_absolute_error": float(np.median(abs_errors)),
        }

    def percentage(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        mask = np.abs(targets) > 1e-8
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100)
        else:
            mape = float("nan")

        denom = np.abs(predictions) + np.abs(targets)
        safe_mask = denom > 1e-8
        if safe_mask.sum() > 0:
            smape = float(
                np.mean(2.0 * np.abs(predictions[safe_mask] - targets[safe_mask]) / denom[safe_mask]) * 100
            )
        else:
            smape = float("nan")

        return {
            "percentage/mape": mape,
            "percentage/smape": smape,
        }

    def directional(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Directional accuracy measures whether the model predicts the correct
        direction of change between consecutive timesteps.
        """
        if len(predictions) < 2:
            return {
                "directional/accuracy": float("nan"),
                "directional/weighted_accuracy": float("nan"),
            }

        pred_direction = np.sign(np.diff(predictions))
        target_direction = np.sign(np.diff(targets))

        correct = pred_direction == target_direction
        da = float(np.mean(correct) * 100)

        target_magnitudes = np.abs(np.diff(targets))
        total_magnitude = np.sum(target_magnitudes)
        if total_magnitude > 1e-8:
            weighted_da = float(np.sum(correct * target_magnitudes) / total_magnitude * 100)
        else:
            weighted_da = float("nan")

        return {
            "directional/accuracy": da,
            "directional/weighted_accuracy": weighted_da,
        }

    def real_scale(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        pipeline,
    ) -> Dict[str, float]:
        """Compute metrics in original (un-normalized) price scale."""
        real_preds = pipeline.inverse_transform(predictions, column="c")
        real_targets = pipeline.inverse_transform(targets, column="c")

        errors = real_preds - real_targets
        abs_errors = np.abs(errors)

        mae = float(np.mean(abs_errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        max_error = float(np.max(abs_errors))
        mae_pips = mae / self.pip_size

        return {
            "real_scale/mae": mae,
            "real_scale/rmse": rmse,
            "real_scale/max_error": max_error,
            "real_scale/mae_pips": mae_pips,
        }

    def error_distribution(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        errors = predictions - targets
        abs_errors = np.abs(errors)

        return {
            "error_dist/mean_error_bias": float(np.mean(errors)),
            "error_dist/std": float(np.std(errors)),
            "error_dist/percentile_95": float(np.percentile(abs_errors, 95)),
            "error_dist/percentile_99": float(np.percentile(abs_errors, 99)),
            "error_dist/skewness": float(stats.skew(errors)),
            "error_dist/kurtosis": float(stats.kurtosis(errors)),
        }
