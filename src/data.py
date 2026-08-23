"""Data utilities and a deterministic offline demonstration dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def prices_to_returns(prices: pd.DataFrame, *, log_returns: bool = False) -> pd.DataFrame:
    """Convert a price table into aligned simple or log returns."""

    numeric = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    if log_returns:
        returns = np.log(numeric).diff()
    else:
        returns = numeric.pct_change()
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def load_returns_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV whose first column is an optional date/index column."""

    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, errors="ignore")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError("CSV needs at least two observations and two assets")
    return frame


def make_synthetic_returns(
    periods: int = 320,
    assets: int = 24,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a reproducible regime-switching return panel for offline runs.

    This is not the paper's S&P 500 data.  It is a small test fixture with
    two common factors and three latent regimes, allowing the complete PCA +
    HMM + backtest pipeline to run without downloading proprietary or moving
    market data.
    """

    if periods < 20 or assets < 4:
        raise ValueError("synthetic panel is too small for the experiment")
    rng = np.random.default_rng(seed)

    transition = np.array(
        [
            [0.88, 0.08, 0.04],
            [0.08, 0.84, 0.08],
            [0.05, 0.10, 0.85],
        ]
    )
    state_means = np.array(
        [
            [0.0025, 0.0005],
            [0.0000, 0.0015],
            [-0.0020, -0.0010],
        ]
    )
    state_scales = np.array(
        [
            [0.009, 0.006],
            [0.013, 0.009],
            [0.022, 0.014],
        ]
    )
    states = np.zeros(periods, dtype=int)
    factors = np.zeros((periods, 2), dtype=float)
    for time in range(1, periods):
        states[time] = rng.choice(3, p=transition[states[time - 1]])
    for time, state in enumerate(states):
        factors[time] = state_means[state] + rng.normal(0.0, state_scales[state])

    loadings = rng.normal(0.0, 0.7, size=(assets, 2))
    loadings[:, 0] += rng.choice([-1.0, 1.0], size=assets) * 0.6
    idiosyncratic = rng.normal(0.0, 0.012, size=(periods, assets))
    returns = factors @ loadings.T + idiosyncratic
    dates = pd.date_range("2010-01-01", periods=periods, freq="W-FRI")
    columns = [f"Asset_{index:03d}" for index in range(1, assets + 1)]
    return pd.DataFrame(returns, index=dates, columns=columns)
