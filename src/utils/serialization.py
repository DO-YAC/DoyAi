from typing import Any, Optional, Dict


def serialize_scaler(scaler: Any) -> Optional[Dict[str, Any]]:
    """Serialize a sklearn scaler to a dictionary."""
    if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
        return {
            "type": "minmax",
            "data_min_": scaler.data_min_.tolist(),
            "data_max_": scaler.data_max_.tolist(),
            "data_range_": scaler.data_range_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "min_": scaler.min_.tolist(),
            "feature_range": scaler.feature_range,
        }
    elif hasattr(scaler, "mean_") and hasattr(scaler, "var_"):
        return {
            "type": "standard",
            "mean_": scaler.mean_.tolist(),
            "var_": scaler.var_.tolist(),
            "scale_": scaler.scale_.tolist(),
        }
    return None
