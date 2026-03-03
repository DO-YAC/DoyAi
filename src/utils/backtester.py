from typing import Dict, List
import wandb
import numpy as np
from omegaconf import DictConfig


class Backtester:
    """
    Simulates a trading strategy on historical predictions and computes
    trading-specific performance metrics (PnL, Sharpe, drawdown, etc.).
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.strategy = cfg.strategy
        self.initial_capital = cfg.initial_capital
        self.spread_pips = cfg.spread_pips
        self.pip_size = cfg.pip_size
        self.position_size = cfg.position_size
        self.pip_value = cfg.pip_value
        self.threshold_pips = cfg.threshold_pips

    def run(self, predictions: np.ndarray, targets: np.ndarray, pipeline) -> Dict:
        """
        Run the backtest simulation.

        Args:
            predictions: Model predictions in normalized scale, shape (N,)
            targets: Ground truth values in normalized scale, shape (N,)
            pipeline: ForexDataPipeline for inverse-transforming to real prices

        Returns:
            Dictionary with all trading metrics and the equity curve.
        """
        real_preds = pipeline.inverse_transform(predictions, column="c")
        real_targets = pipeline.inverse_transform(targets, column="c")

        trades = self._generate_trades(real_preds, real_targets)
        metrics = self._compute_metrics(trades)
        return metrics

    def _generate_trades(self, real_preds: np.ndarray, real_targets: np.ndarray) -> List[Dict]:
        """
        Walk through each timestep and generate trades.

        At each step i (for i in 0..N-2):
        - Current price = real_targets[i]
        - Predicted next price = real_preds[i]
        - Actual next price = real_targets[i+1]
        - Direction: long if predicted > current, short otherwise
        - Entry at current, exit at next actual
        """
        trades = []
        spread_cost = self.spread_pips * self.pip_size

        for i in range(len(real_preds) - 1):
            current_price = real_targets[i]
            predicted_price = real_preds[i]
            next_price = real_targets[i + 1]

            predicted_move = predicted_price - current_price

            if self.strategy == "threshold":
                threshold = self.threshold_pips * self.pip_size
                if abs(predicted_move) < threshold:
                    continue

            if predicted_move >= 0:
                direction = 1
            else:
                direction = -1

            price_change = next_price - current_price
            pnl_pips = (price_change * direction) / self.pip_size - self.spread_pips
            pnl_currency = pnl_pips * self.pip_value * self.position_size

            trades.append({
                "step": i,
                "direction": direction,
                "entry_price": float(current_price),
                "exit_price": float(next_price),
                "pnl_pips": float(pnl_pips),
                "pnl_currency": float(pnl_currency),
            })

        return trades

    def _compute_metrics(self, trades: List[Dict]) -> Dict:
        """Compute aggregate trading metrics from the trade list."""
        num_trades = len(trades)

        if num_trades == 0:
            return {
                "num_trades": 0,
                "total_pnl": 0.0,
                "total_pnl_pips": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "avg_win_pips": 0.0,
                "avg_loss_pips": 0.0,
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
                "return_pct": 0.0,
                "equity_curve": [self.initial_capital],
            }

        pnl_pips = np.array([t["pnl_pips"] for t in trades])
        pnl_currency = np.array([t["pnl_currency"] for t in trades])

        total_pnl = float(np.sum(pnl_currency))
        total_pnl_pips = float(np.sum(pnl_pips))

        wins = pnl_pips > 0
        losses = pnl_pips < 0
        win_rate = float(np.sum(wins) / num_trades * 100)

        gross_profit = float(np.sum(pnl_currency[wins])) if np.any(wins) else 0.0
        gross_loss = abs(float(np.sum(pnl_currency[losses]))) if np.any(losses) else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        # Equity curve and drawdown
        equity_curve = [self.initial_capital]
        for pnl in pnl_currency:
            equity_curve.append(equity_curve[-1] + pnl)

        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_drawdown = float(np.max(drawdown))
        max_drawdown_pct = float(max_drawdown / np.max(peak) * 100) if np.max(peak) > 0 else 0.0

        # Sharpe ratio (annualized, assuming ~252 trading days)
        if np.std(pnl_currency) > 0:
            sharpe_ratio = float(np.mean(pnl_currency) / np.std(pnl_currency) * np.sqrt(252))
        else:
            sharpe_ratio = 0.0

        avg_win_pips = float(np.mean(pnl_pips[wins])) if np.any(wins) else 0.0
        avg_loss_pips = float(np.mean(pnl_pips[losses])) if np.any(losses) else 0.0

        # Streaks
        longest_win_streak = self._longest_streak(pnl_pips > 0)
        longest_loss_streak = self._longest_streak(pnl_pips < 0)

        return_pct = total_pnl / self.initial_capital * 100

        return {
            "num_trades": num_trades,
            "total_pnl": total_pnl,
            "total_pnl_pips": total_pnl_pips,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "avg_win_pips": avg_win_pips,
            "avg_loss_pips": avg_loss_pips,
            "longest_win_streak": longest_win_streak,
            "longest_loss_streak": longest_loss_streak,
            "return_pct": return_pct,
            "equity_curve": [float(e) for e in equity_curve],
        }

    @staticmethod
    def _longest_streak(condition: np.ndarray) -> int:
        """Find the longest consecutive streak of True values."""
        if len(condition) == 0:
            return 0
        max_streak = 0
        current_streak = 0
        for val in condition:
            if val:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def log_to_wandb(self, results: Dict, run: wandb.sdk.wandb_run.Run) -> None:
        """Log backtest metrics to W&B under the backtest/* prefix."""
        if not self.cfg.log_to_wandb:
            return

        metrics = {
            f"backtest/{k}": v
            for k, v in results.items()
            if k != "equity_curve"
        }
        run.summary.update(metrics)

        equity_curve = results.get("equity_curve", [])
        if equity_curve:
            table = wandb.Table(columns=["trade", "equity"], data=[
                [i, eq] for i, eq in enumerate(equity_curve)
            ])
            run.log({"backtest/equity_curve": wandb.plot.line(
                table, "trade", "equity", title="Backtest Equity Curve"
            )})