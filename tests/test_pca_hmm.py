import numpy as np

from src.data import make_synthetic_returns
from src.pca_hmm import GaussianDiagonalHMM, fit_pca
from src.replication import rolling_backtest


def test_pca_components_retain_requested_variance():
    returns = make_synthetic_returns(periods=80, assets=8, seed=7).to_numpy()
    result = fit_pca(returns, noise_fraction=0.25)
    assert result.n_components >= 1
    assert result.factors.shape == (80, result.n_components)
    assert result.explained_variance_ratio[: result.n_components].sum() >= 0.75 - 1e-12


def test_hmm_transition_rows_and_forecast_shape():
    rng = np.random.default_rng(3)
    observations = np.vstack(
        [rng.normal(-1.0, 0.2, size=(30, 2)), rng.normal(1.0, 0.2, size=(30, 2))]
    )
    model = GaussianDiagonalHMM(2, max_iter=12, random_state=3).fit(observations)
    assert np.allclose(model.transition_matrix_.sum(axis=1), 1.0)
    assert model.forecast_next(observations).shape == (2,)
    assert np.isfinite(model.aic)


def test_rolling_backtest_has_all_strategies():
    returns = make_synthetic_returns(periods=70, assets=8, seed=9)
    results, summary, _ = rolling_backtest(
        returns,
        train_periods=50,
        forecast_periods=5,
        noise_choices=(0.30,),
        state_candidates=(2, 3),
        max_iter=8,
    )
    assert len(results) == 5
    assert set(summary["strategy"]) == {
        "Strategy 1: raw-return signs",
        "Strategy 2: normalized-return signs",
        "Buy-and-hold",
    }
    assert np.isfinite(results["strategy_1_return"]).all()
