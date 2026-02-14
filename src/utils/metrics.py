from typing import Dict, Optional

import numpy as np
from scipy import stats
from sklearn.pipeline import Pipeline


class MetricsCalculator:
    """
    Computes metrics for forex price prediction models.

    Metric categories:
    - regression   : Core regression metrics (RMSE, MAE, R², etc.)
    - percentage   : Scale-independent metrics (MAPE, sMAPE)
    - directional  : Forecasting direction metrics (DA, weighted DA)
    - real_scale   : Inverse-transformed metrics in original price units
    - error_dist   : Error distribution statistics (bias, std, percentiles, skew, kurtosis)
    """

    def __init__(self, pip_size: float = 0.0001):
        """
        Args:
            pip_size: Size of one pip in price units.
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
            predictions : Model predictions (normalized scale), shape (N,)
            targets     : Ground truth values (normalized scale), shape (N,)
            pipeline    : Optional ForexDataPipeline for inverse-transforming to real scale.

        Returns:
            Flat dictionary of metric_name -> value.
        """
        predictions = predictions.flatten()
        targets = targets.flatten()

        if predictions.size == 0 or targets.size == 0:
            raise ValueError("predictions and targets must be non-empty arrays")
        metrics = {}
        metrics.update(self.regression(predictions, targets))
        metrics.update(self.percentage(predictions, targets))
        metrics.update(self.directional(predictions, targets))
        metrics.update(self.error_distribution(predictions, targets))

        if pipeline is not None:
            metrics.update(self.real_scale(predictions, targets, pipeline))

        return metrics

    def regression(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Standard statistical metrics for regression performance.

        MSE                 : Mean Squared Error. Heavily penalizes large outliers. Used mostly for model training.
        RMSE                : Root Mean Squared Error. The standard 'error' metric in the same units as the target.
        MAE                 : Mean Absolute Error. The average 'real-world' distance from the target.
        R2 Score            : Percentage of the data's variance explained by the model (1.0 is a perfect fit).
        Max/Median Error    : Highlights the difference between the 'worst-case' and 'typical' mistakes.
        """

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
        """
        Evaluate model performance using percentage scales.

        MAPE    : Mean Absolute Percentage Error. Shows the average error as a percentage of the actual price.
        sMAPE   : Symmetric MAPE. A balanced percentage metric that treats overestimates and underestimates equally.
        """

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
        Directional accuracy measures whether the model predicts the correct direction of change.

        Accuracy            : The percentage of time the model correctly guessed if the price would go Up or Down.
        Weighted Accuracy   : Directional accuracy that prioritizes being correct on large price swings over small ones.
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
        pipeline: Optional[Pipeline] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics in original price scale.

        MAE         : The average 'offness' in real dollars/price.
        RMSE        : The error metric that punishes big misses more severely.
        Max Error   : The single worst prediction made in this run.
        MAE Pips    : The average error translated into trading units.
        """
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
        """
        Analyze the statistical shape and reliability of errors.

        Mean Error Bias : The average magnitude of your mistakes (on the absolute scale).
        std             : How much your error sizes fluctuate; high values mean inconsistent performance.
        95th Percentile : 95% of errors fall below this value.
        99th Percentile : 99% of errors fall below this value.
        Skewness        : Measures if errors are balanced or if you have a 'tail' of huge misses.
        Kurtosis        : Measures 'Fat Tails'—how often extreme outlier errors happen compared to normal.
        """

        errors = predictions - targets
        abs_errors = np.abs(errors)
        n = abs_errors.size

        mean_error_bias = float(np.mean(abs_errors))
        std = float(np.std(abs_errors))
        percentile_95 = float(np.percentile(abs_errors, 95))
        percentile_99 = float(np.percentile(abs_errors, 99))

        if n >= 30:
            skewness = float(stats.skew(abs_errors))
            kurtosis = float(stats.kurtosis(abs_errors))
        else:
            skewness = float("nan")
            kurtosis = float("nan")

        return {
            "error_dist/mean_error_bias": mean_error_bias,
            "error_dist/std": std,
            "error_dist/percentile_95": percentile_95,
            "error_dist/percentile_99": percentile_99,
            "error_dist/skewness": skewness,
            "error_dist/kurtosis": kurtosis,
        }
