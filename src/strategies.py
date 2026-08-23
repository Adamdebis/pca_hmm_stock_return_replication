"""Simple forecast-based portfolio rules used in the paper's experiments."""

from __future__ import annotations

import numpy as np


def equal_weight_long_short(forecast: np.ndarray, realized: np.ndarray) -> float:
    """Long positive forecasts and short negative forecasts equally."""

    forecast = np.asarray(forecast, dtype=float)
    realized = np.asarray(realized, dtype=float)
    if forecast.shape != realized.shape:
        raise ValueError("forecast and realized returns must have the same shape")
    signs = np.where(forecast >= 0.0, 1.0, -1.0)
    return float(np.mean(signs * realized))


def buy_and_hold(realized: np.ndarray) -> float:
    """Equal-weight long-only benchmark for one period."""

    return float(np.mean(np.asarray(realized, dtype=float)))


def directional_accuracy(forecast: np.ndarray, realized: np.ndarray) -> float:
    """Fraction of assets whose return direction was predicted correctly."""

    forecast = np.asarray(forecast, dtype=float)
    realized = np.asarray(realized, dtype=float)
    predicted_sign = np.where(forecast >= 0.0, 1.0, -1.0)
    actual_sign = np.where(realized >= 0.0, 1.0, -1.0)
    return float(np.mean(predicted_sign == actual_sign))


def annualized_sharpe(
    returns: np.ndarray,
    *,
    periods_per_year: int = 52,
    risk_free_per_period: float = 0.0,
) -> float:
    """Annualized Sharpe ratio for weekly or otherwise periodic returns."""

    values = np.asarray(returns, dtype=float)
    excess = values - risk_free_per_period
    standard_deviation = float(excess.std(ddof=1)) if excess.size > 1 else 0.0
    if standard_deviation <= 1e-15:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / standard_deviation)


def cumulative_return(returns: np.ndarray) -> float:
    """Compound periodic returns into a cumulative return."""

    return float(np.prod(1.0 + np.asarray(returns, dtype=float)) - 1.0)
