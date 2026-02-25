# DoyAi

AI model training pipeline for the [DoyVestment](https://github.com/DO-YAC/DoyVestment.public) algorithmic trading platform. DoyAi trains deep learning models on historical forex data and exports them for DoyLib.

---

## Overview

DoyAi fetches OHLC candlestick data from our MongoDB, trains time-series forecasting models, and exports them in both PyTorch and ONNX formats. The exported ONNX models are consumed by **DoyLib** for real-time trade signal generation.

The entire training process is configuration-driven using Hydra, making it easy to tweak hyperparameters, swap datasets, or change model architectures without touching code. Every run is automatically tracked in Weights & Biases for full experiment reproducibility.

## Tech Stack

| Category            | Tools                       |
|---------------------|-----------------------------|
| Deep Learning       | PyTorch                     |
| Configuration       | Hydra + OmegaConf           |
| Experiment Tracking | Weights & Biases            |
| Data Processing     | NumPy, Pandas, scikit-learn |
| Model Export        | ONNX                        |
| Statistics          | SciPy                       |

## Training Pipeline

```
MongoDB (OHLC data) → Normalize → Sliding Window Sequences
  → Train → Evaluate → Checkpoint → Export (.pt + .onnx)
```

Raw OHLC candle data is pulled directly from MongoDB and normalized using MinMax or Standard scaling. The normalized data is then sliced into overlapping sliding window sequences, giving the model a fixed lookback of recent price action to learn temporal patterns from.

The data is split into train, validation, and test sets. After each epoch, the pipeline evaluates performance across all metric categories and saves the best checkpoint based on validation loss. Once training completes, a final evaluation runs on the held-out test set, and the model is exported in both PyTorch (.pt) and ONNX (.onnx) formats.

## Roadmap

> DoyAi is under active development. The following features are planned:

- [ ] **Backtesting** — Evaluate trained models against historical market data to measure real-world trading performance
- [ ] **Feature Engineering** — Enrich input data with technical indicators
- [ ] **Walk-Forward Validation** — Rolling-window train/test splits that simulate real trading conditions
- [ ] **Multi-Model Support** — Train and compare different architectures through the existing model factory
- [ ] **AI-Assisted Training** — Connect an MCP server to Perplexity for automated analysis of training outputs, hyperparameter suggestions, and model improvement recommendations

---

Part of the [DoyVestment](https://github.com/DO-YAC/DoyVestment.public) ecosystem.
