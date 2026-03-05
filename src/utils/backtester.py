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
        self.initial_capital = cfg.initial_capital
        self.spread_pips = cfg.spread_pips
        self.pip_size = cfg.pip_size
        self.position_size = cfg.position_size
        self.pip_value = cfg.pip_value
        self.threshold_pips = cfg.threshold_pips
        self.annualization_factor = getattr(cfg, 'annualization_factor', 252)

    def run(self, predictions: np.ndarray, targets: np.ndarray, pipeline) -> Dict:
        """
        Run the backtest simulation.

        Args:
            predictions: Model predictions in normalized scale
            targets: Ground truth values in normalized scale
            pipeline: ForexDataPipeline for inverse-transforming to real prices

        Returns:
            Dictionary with all trading metrics and the equity curve.
        """
        real_preds = pipeline.inverse_transform(predictions, column="c")
        real_targets = pipeline.inverse_transform(targets, column="c")

        trades = self.generate_trades(real_preds, real_targets)
        metrics = self.compute_metrics(trades)
        return metrics

    def generate_trades(self, real_preds: np.ndarray, real_targets: np.ndarray) -> List[Dict]:
        """
        Walk through each timestep and generate trades.

        Data alignment (prediction_horizon=1):
            - targets[i-1] = last known close (end of input window for sample i)
            - preds[i]     = model's predicted close for the next step
            - targets[i]   = actual close at the next step (entry→exit)

        We enter at targets[i-1], predict direction via preds[i] - targets[i-1],
        and exit at targets[i]. Trades below threshold_pips are skipped.
        """
        trades = []

        for i in range(1, len(real_preds)):
            entry_price = real_targets[i - 1]
            predicted_price = real_preds[i]
            exit_price = real_targets[i]

            predicted_move_pips = (predicted_price - entry_price) / self.pip_size

            if abs(predicted_move_pips) < self.threshold_pips:
                continue

            direction = 1 if predicted_move_pips >= 0 else -1

            price_change = exit_price - entry_price
            pnl_pips = (price_change * direction) / self.pip_size - self.spread_pips
            pnl_currency = pnl_pips * self.pip_value * self.position_size

            trades.append({
                "step": i,
                "direction": direction,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "predicted_move_pips": float(predicted_move_pips),
                "pnl_pips": float(pnl_pips),
                "pnl_currency": float(pnl_currency),
            })

        return trades

    def compute_metrics(self, trades: List[Dict]) -> Dict:
        """Compute aggregate trading metrics from the trade list."""
        num_trades = len(trades)

        if num_trades == 0:
            return {
                "num_trades": 0,
                "num_breakeven": 0,
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
        breakeven = pnl_pips == 0
        num_breakeven = int(np.sum(breakeven))
        win_rate = float(np.sum(wins) / num_trades * 100)

        gross_profit = float(np.sum(pnl_currency[wins])) if np.any(wins) else 0.0
        gross_loss = abs(float(np.sum(pnl_currency[losses]))) if np.any(losses) else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 100.0
        else:
            profit_factor = 0.0

        equity_curve = [self.initial_capital]
        for pnl in pnl_currency:
            equity_curve.append(equity_curve[-1] + pnl)

        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_drawdown = float(np.max(drawdown))
        drawdown_pct = np.where(peak > 0, drawdown / peak, 0.0)
        max_drawdown_pct = float(np.max(drawdown_pct) * 100)

        safe_equity = equity[:-1].copy()
        safe_equity[safe_equity <= 0] = np.nan
        trade_returns = pnl_currency / safe_equity
        finite_returns = trade_returns[np.isfinite(trade_returns)]
        if len(finite_returns) > 1 and np.std(finite_returns) > 0:
            sharpe_ratio = float(np.mean(finite_returns) / np.std(finite_returns) * np.sqrt(self.annualization_factor))
        else:
            sharpe_ratio = 0.0

        avg_win_pips = float(np.mean(pnl_pips[wins])) if np.any(wins) else 0.0
        avg_loss_pips = float(np.mean(pnl_pips[losses])) if np.any(losses) else 0.0

        longest_win_streak = self.longest_streak(pnl_pips > 0)
        longest_loss_streak = self.longest_streak(pnl_pips < 0)

        return_pct = total_pnl / self.initial_capital * 100 if self.initial_capital > 0 else 0.0

        return {
            "num_trades": num_trades,
            "num_breakeven": num_breakeven,
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
    def longest_streak(condition: np.ndarray) -> int:
        """Find the longest consecutive streak of True values."""
        if len(condition) == 0:
            return 0
        padded = np.concatenate(([False], condition, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        if len(starts) == 0:
            return 0
        return int(np.max(ends - starts))

    def log_to_wandb(self, results: Dict, run) -> None:
        """Log backtest metrics and charts to W&B under the backtest/* prefix."""
        if not self.cfg.log_to_wandb:
            return

        metrics = {
            f"backtest/{k}": v
            for k, v in results.items()
            if k != "equity_curve"
        }
        run.summary.update(metrics)

        equity_curve = results.get("equity_curve", [])
        if not equity_curve:
            return

        equity = np.array(equity_curve)
        initial = equity[0]

        # Cumulative return %
        cum_return_pct = (equity - initial) / initial * 100
        return_table = wandb.Table(
            columns=["trade", "cumulative_return_pct"],
            data=[[i, float(r)] for i, r in enumerate(cum_return_pct)],
        )

        # Drawdown %
        peak = np.maximum.accumulate(equity)
        dd_pct = np.where(peak > 0, (peak - equity) / peak * 100, 0.0)
        dd_table = wandb.Table(
            columns=["trade", "drawdown_pct"],
            data=[[i, float(d)] for i, d in enumerate(dd_pct)],
        )

        # Per-trade PnL distribution
        pnl = np.diff(equity)
        pnl_table = wandb.Table(columns=["pnl"], data=[[float(p)] for p in pnl])

        run.log({
            "backtest/cumulative_return": wandb.plot.line(
                return_table, "trade", "cumulative_return_pct",
                title="Cumulative Return (%)",
            ),
            "backtest/drawdown": wandb.plot.line(
                dd_table, "trade", "drawdown_pct",
                title="Drawdown (%)",
            ),
            "backtest/pnl_distribution": wandb.plot.histogram(
                pnl_table, "pnl", title="Per-Trade PnL Distribution",
            ),
        })