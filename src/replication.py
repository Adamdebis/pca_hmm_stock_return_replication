"""Run the paper-inspired PCA + HMM forecasting and trading audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import make_synthetic_returns
from .pca_hmm import PCAResult, fit_pca, select_hmm
from .strategies import (
    annualized_sharpe,
    buy_and_hold,
    cumulative_return,
    directional_accuracy,
    equal_weight_long_short,
)


NOISE_CHOICES = (0.45, 0.30, 0.15, 0.10)


def forecast_one_window(
    training_returns: np.ndarray,
    noise_fraction: float,
    *,
    state_candidates: tuple[int, ...] = (2, 3, 4),
    max_iter: int = 35,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit PCA + HMM on one window and forecast the next period."""

    pca: PCAResult = fit_pca(training_returns, noise_fraction=noise_fraction)
    hmm, aic_table = select_hmm(
        pca.factors,
        state_candidates=state_candidates,
        max_iter=max_iter,
        random_state=random_state,
    )
    factor_forecast = hmm.forecast_next(pca.factors)
    normalized_forecast = pca.reconstruct_normalized(factor_forecast)[0]
    raw_forecast = pca.reconstruct_returns(factor_forecast)[0]
    state_path = hmm.predict_states(pca.factors)
    return {
        "pca": pca,
        "hmm": hmm,
        "aic_table": aic_table,
        "factor_forecast": factor_forecast,
        "normalized_forecast": normalized_forecast,
        "raw_forecast": raw_forecast,
        "current_state": int(state_path[-1]),
    }


def rolling_backtest(
    returns: pd.DataFrame,
    *,
    train_periods: int = 260,
    forecast_periods: int = 20,
    noise_choices: tuple[float, ...] = NOISE_CHOICES,
    state_candidates: tuple[int, ...] = (2, 3, 4),
    max_iter: int = 35,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, PCAResult]]:
    """Run a rolling, one-period-ahead forecast and portfolio backtest."""

    values = returns.to_numpy(dtype=float)
    if values.shape[0] < train_periods + forecast_periods:
        raise ValueError("return panel is shorter than train plus forecast periods")

    rows: list[dict[str, object]] = []
    example_pca: dict[float, PCAResult] = {}
    for noise_fraction in noise_choices:
        for step in range(forecast_periods):
            endpoint = train_periods + step
            fit = forecast_one_window(
                values[endpoint - train_periods : endpoint],
                noise_fraction,
                state_candidates=state_candidates,
                max_iter=max_iter,
                random_state=random_state + step,
            )
            pca = fit["pca"]
            example_pca.setdefault(noise_fraction, pca)
            actual = values[endpoint]
            raw_forecast = np.asarray(fit["raw_forecast"])
            normalized_forecast = np.asarray(fit["normalized_forecast"])
            timestamp = returns.index[endpoint]
            rows.append(
                {
                    "date": str(timestamp),
                    "noise_fraction": noise_fraction,
                    "noise_percent": int(round(noise_fraction * 100)),
                    "step": step + 1,
                    "n_components": pca.n_components,
                    "explained_variance": float(
                        pca.explained_variance_ratio[: pca.n_components].sum()
                    ),
                    "n_states": fit["hmm"].n_states,
                    "current_state": fit["current_state"],
                    "strategy_1_return": equal_weight_long_short(raw_forecast, actual),
                    "strategy_2_return": equal_weight_long_short(
                        normalized_forecast, actual
                    ),
                    "buy_hold_return": buy_and_hold(actual),
                    "strategy_1_directional_accuracy": directional_accuracy(
                        raw_forecast, actual
                    ),
                    "strategy_2_directional_accuracy": directional_accuracy(
                        normalized_forecast, actual
                    ),
                }
            )

    results = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    strategy_columns = {
        "Strategy 1: raw-return signs": "strategy_1_return",
        "Strategy 2: normalized-return signs": "strategy_2_return",
        "Buy-and-hold": "buy_hold_return",
    }
    for noise_fraction, group in results.groupby("noise_fraction", sort=False):
        for strategy, column in strategy_columns.items():
            series = group[column].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "noise_fraction": noise_fraction,
                    "noise_percent": int(round(float(noise_fraction) * 100)),
                    "strategy": strategy,
                    "annualized_sharpe": annualized_sharpe(series),
                    "cumulative_return": cumulative_return(series),
                    "mean_period_return": float(series.mean()),
                    "mean_directional_accuracy": (
                        float(
                            group[
                                "strategy_1_directional_accuracy"
                                if column == "strategy_1_return"
                                else "strategy_2_directional_accuracy"
                            ].mean()
                        )
                        if column != "buy_hold_return"
                        else np.nan
                    ),
                    "observations": len(series),
                }
            )
    return results, pd.DataFrame(summary_rows), example_pca


def paper_reported_results() -> pd.DataFrame:
    """Transcribe the paper's reported Tables 1 and 2 for audit purposes."""

    rows: list[dict[str, object]] = []
    winning_probability = {
        45: (0.532, 0.490),
        30: (0.543, 0.504),
        15: (0.538, 0.511),
        5: (0.532, 0.490),
    }
    for noise_percent, (strategy_1, strategy_2) in winning_probability.items():
        rows.extend(
            [
                {
                    "table": "Table 1",
                    "noise_percent": noise_percent,
                    "metric": "winning_probability",
                    "strategy": "Strategy 1",
                    "value": strategy_1,
                },
                {
                    "table": "Table 1",
                    "noise_percent": noise_percent,
                    "metric": "winning_probability",
                    "strategy": "Strategy 2",
                    "value": strategy_2,
                },
            ]
        )

    sharpe = {
        45: (0.688, 0.450),
        30: (0.581, 0.581),
        15: (0.703, 1.360),
        10: (0.877, 0.726),
    }
    for noise_percent, (strategy_1, strategy_2) in sharpe.items():
        rows.extend(
            [
                {
                    "table": "Table 2",
                    "noise_percent": noise_percent,
                    "metric": "annualized_sharpe",
                    "strategy": "Strategy 1",
                    "value": strategy_1,
                },
                {
                    "table": "Table 2",
                    "noise_percent": noise_percent,
                    "metric": "annualized_sharpe",
                    "strategy": "Strategy 2",
                    "value": strategy_2,
                },
            ]
        )
    rows.append(
        {
            "table": "Table 2",
            "noise_percent": np.nan,
            "metric": "annualized_sharpe",
            "strategy": "Buy-and-hold",
            "value": 0.828,
        }
    )
    return pd.DataFrame(rows)


def save_figures(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    pca_examples: dict[float, PCAResult],
    figures_dir: Path,
) -> None:
    """Save compact figures used by the README and notebooks."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    pca = pca_examples[min(pca_examples)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        np.arange(1, len(pca.cumulative_explained_variance) + 1),
        pca.cumulative_explained_variance,
        marker="o",
    )
    ax.axhline(1.0 - pca.noise_fraction, color="tab:red", linestyle="--")
    ax.axvline(pca.n_components, color="tab:green", linestyle=":")
    ax.set(
        title=f"PCA variance retained (p={pca.noise_fraction:.0%})",
        xlabel="Number of components",
        ylabel="Cumulative explained variance",
        ylim=(0, 1.05),
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "pca_explained_variance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for noise_fraction, group in results.groupby("noise_fraction", sort=False):
        cumulative = (1.0 + group["strategy_2_return"]).cumprod() - 1.0
        ax.plot(group["step"], cumulative, label=f"p={noise_fraction:.0%}")
    ax.set(
        title="Strategy 2 cumulative return in the offline demonstration",
        xlabel="Forecast week",
        ylabel="Cumulative return",
    )
    ax.legend(title="Noise removed")
    fig.tight_layout()
    fig.savefig(figures_dir / "strategy_cumulative_returns.png", dpi=180)
    plt.close(fig)

    sharpe = summary[summary["strategy"] != "Buy-and-hold"]
    pivot = sharpe.pivot(index="noise_percent", columns="strategy", values="annualized_sharpe")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(title="Annualized Sharpe ratios", xlabel="Noise fraction p (%)", ylabel="Sharpe")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "sharpe_comparison.png", dpi=180)
    plt.close(fig)

    state_components = results.groupby("noise_percent")[["n_components", "n_states"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    state_components.plot(marker="o", ax=ax)
    ax.set(title="Average selected PCA components and HMM states", xlabel="Noise fraction p (%)", ylabel="Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "model_complexity.png", dpi=180)
    plt.close(fig)


def run_full_replication(
    output_dir: str | Path = "results",
    *,
    periods: int = 320,
    assets: int = 24,
    train_periods: int = 260,
    forecast_periods: int = 20,
    seed: int = 42,
    state_candidates: tuple[int, ...] = (2, 3, 4),
    max_iter: int = 35,
) -> dict[str, pd.DataFrame]:
    """Run a quick deterministic demonstration and write audit artefacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    returns = make_synthetic_returns(periods=periods, assets=assets, seed=seed)
    returns.to_csv(output / "synthetic_returns.csv")
    results, summary, pca_examples = rolling_backtest(
        returns,
        train_periods=train_periods,
        forecast_periods=forecast_periods,
        state_candidates=state_candidates,
        max_iter=max_iter,
        random_state=seed,
    )
    results.to_csv(output / "replication_results.csv", index=False)
    summary.to_csv(output / "strategy_summary.csv", index=False)
    paper_results = paper_reported_results()
    paper_results.to_csv(output / "paper_reported_results.csv", index=False)
    save_figures(results, summary, pca_examples, output / "figures")
    return {"results": results, "summary": summary, "paper": paper_results}
