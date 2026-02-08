import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
from omegaconf import DictConfig
from utils.serialization import serialize_scaler


class ModelExporter:
    """
    Exports trained models

    Supported formats:
    - pytorch: Standard PyTorch .pt file with state_dict
    - onnx: ONNX format for cross-platform inference
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.export_cfg = cfg.export
        self.enabled = self.export_cfg.enabled

        if not self.enabled:
            return

        self.export_dir = Path(self.export_cfg.dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.formats = list(self.export_cfg.formats)

    def format_filename(self, format_ext: str) -> str:
        """Format export filename using template."""
        template = self.export_cfg.filename
        filename = template.format(
            ticker=self.cfg.dataset.ticker,
            model=self.cfg.models.name
        )
        return f"{filename}.{format_ext}"

    def export(
        self,
        model: torch.nn.Module,
        pipeline: Any = None,
        device: torch.device = None,
    ) -> Dict[str, str]:
        """
        Export model to all configured formats.

        Args:
            model: Trained model to export
            pipeline: Data pipeline with scaler info
            device: Device the model is on

        Returns:
            Dictionary mapping format name to export path
        """
        if not self.enabled:
            return {}

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        exported_paths = {}
        model.eval()

        print(f"\nExporting model to: {self.export_dir}")

        for fmt in self.formats:
            try:
                if fmt == "pytorch":
                    path = self.export_pytorch(model, pipeline)
                elif fmt == "onnx":
                    path = self.export_onnx(model, device)
                elif fmt == "torchscript":
                    path = self.export_torchscript(model, device)
                else:
                    print(f"  -> Unknown format: {fmt}, skipping")
                    continue

                exported_paths[fmt] = path
                print(f"  -> Exported {fmt}: {Path(path).name}")

            except Exception as e:
                print(f"  -> Failed to export {fmt}: {e}")

        return exported_paths

    def export_pytorch(self, model: torch.nn.Module, pipeline: Any = None) -> str:
        """Export as PyTorch state dict with metadata."""
        path = self.export_dir / self.format_filename("pt")

        export_dict = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "name": self.cfg.models.name,
                "input_dim": self.cfg.models.input_dim,
                "hidden_dim": self.cfg.models.hidden_dim,
                "layer_dim": self.cfg.models.layer_dim,
                "output_dim": self.cfg.models.output_dim,
            },
            "dataset_config": {
                "ticker": self.cfg.dataset.ticker,
                "features": list(self.cfg.dataset.features),
                "sequence_length": self.cfg.dataset.sequence_length,
                "prediction_horizon": self.cfg.dataset.prediction_horizon,
            },
        }

        if pipeline is not None and pipeline.scaler is not None:
            scaler_data = serialize_scaler(pipeline.scaler)
            if scaler_data is not None:
                export_dict["scaler"] = scaler_data

        torch.save(export_dict, path)
        return str(path)

    def export_onnx(self, model: torch.nn.Module, device: torch.device) -> str:
        """Export to ONNX format."""
        path = self.export_dir / self.format_filename("onnx")
        onnx_cfg = self.export_cfg.onnx

        batch_size = 1
        sequence_length = self.cfg.dataset.sequence_length
        input_dim = self.cfg.models.input_dim

        dummy_input = torch.randn(
            batch_size, sequence_length, input_dim,
            device=device
        )

        dynamic_axes = None
        if onnx_cfg.dynamic_axes:
            dynamic_axes = {
                "input": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"},
                "hidden": {0: "num_layers", 1: "batch_size"},
                "cell": {0: "num_layers", 1: "batch_size"},
            }

        torch.onnx.export(
            model,
            dummy_input,
            str(path),
            export_params=True,
            opset_version=onnx_cfg.opset_version,
            do_constant_folding=True,
            input_names=list(onnx_cfg.input_names),
            output_names=list(onnx_cfg.output_names),
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

        return str(path)
