"""Principal-component and Gaussian-HMM building blocks.

The paper uses PCA to replace a large cross-section of stock returns with a
small number of factor-return series, then fits a Gaussian hidden Markov model
to those factors.  This module keeps those steps explicit so they can be
inspected and tested without hiding the mathematics inside a black box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def _as_2d(values: Array) -> Array:
    """Return a finite floating-point two-dimensional array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("matrix must contain at least two observations")
    if not np.isfinite(array).all():
        raise ValueError("matrix contains NaN or infinite values")
    return array


def _logsumexp(values: Array, axis: int | None = None) -> Array:
    """Small NumPy-only equivalent of scipy.special.logsumexp."""

    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    result = maximum + np.log(np.sum(shifted, axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(result.squeeze())
    return np.asarray(np.squeeze(result, axis=axis))


@dataclass(frozen=True)
class PCAResult:
    """Fitted PCA representation of a return matrix."""

    mean: Array
    scale: Array
    normalized: Array
    covariance: Array
    eigenvalues: Array
    eigenvectors: Array
    explained_variance_ratio: Array
    n_components: int
    factors: Array
    noise_fraction: float

    @property
    def loadings(self) -> Array:
        """The retained eigenvectors, one loading direction per column."""

        return self.eigenvectors[:, : self.n_components]

    @property
    def cumulative_explained_variance(self) -> Array:
        return np.cumsum(self.explained_variance_ratio)

    def reconstruct_normalized(self, factor_forecast: Array) -> Array:
        """Map a retained-factor forecast back to normalized assets."""

        factors = np.asarray(factor_forecast, dtype=float)
        if factors.ndim == 1:
            factors = factors[None, :]
        if factors.shape[1] != self.n_components:
            raise ValueError("factor forecast has the wrong number of components")
        return factors @ self.loadings.T

    def reconstruct_returns(self, factor_forecast: Array) -> Array:
        """Map a retained-factor forecast back to the original return scale."""

        normalized_forecast = self.reconstruct_normalized(factor_forecast)
        return normalized_forecast * self.scale + self.mean


def fit_pca(returns: Array, noise_fraction: float = 0.15) -> PCAResult:
    """Fit the paper's standardized covariance PCA model.

    ``noise_fraction`` is the paper's ``p``.  The smallest number of
    eigenvectors explaining at least ``1 - p`` of the variance is retained.
    The paper writes ``Y.T @ Y`` as its covariance matrix; dividing by
    ``T - 1`` gives the conventional sample covariance and does not change
    the eigenvectors or the explained-variance ratios.
    """

    if not 0 <= noise_fraction < 1:
        raise ValueError("noise_fraction must be in [0, 1)")

    values = _as_2d(returns)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (values - mean) / scale

    covariance = normalized.T @ normalized / (values.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    # Eigenvectors are only defined up to a sign.  This convention makes
    # output tables and plots stable across linear-algebra backends.
    for column in range(eigenvectors.shape[1]):
        largest_loading = np.argmax(np.abs(eigenvectors[:, column]))
        if eigenvectors[largest_loading, column] < 0:
            eigenvectors[:, column] *= -1

    total_variance = float(eigenvalues.sum())
    if total_variance <= 0:
        raise ValueError("return matrix has zero total variance")
    explained = eigenvalues / total_variance
    target = 1.0 - noise_fraction
    n_components = int(np.searchsorted(np.cumsum(explained), target) + 1)
    factors = normalized @ eigenvectors[:, :n_components]

    return PCAResult(
        mean=mean,
        scale=scale,
        normalized=normalized,
        covariance=covariance,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        explained_variance_ratio=explained,
        n_components=n_components,
        factors=factors,
        noise_fraction=float(noise_fraction),
    )


class GaussianDiagonalHMM:
    """A compact diagonal-covariance Gaussian HMM fitted by EM.

    The implementation exposes the transition matrix, state means, Viterbi
    state path, and AIC.  It intentionally avoids a hidden dependency on
    ``hmmlearn`` so that the repository remains reproducible in a fresh
    Anaconda environment.
    """

    def __init__(
        self,
        n_states: int,
        *,
        max_iter: int = 60,
        tolerance: float = 1e-4,
        min_variance: float = 1e-8,
        random_state: int = 42,
    ) -> None:
        if n_states < 2:
            raise ValueError("an HMM needs at least two states")
        self.n_states = int(n_states)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.min_variance = float(min_variance)
        self.random_state = int(random_state)
        self.log_likelihood_: float | None = None
        self.n_iter_: int = 0

    def _emission_log_prob(self, observations: Array) -> Array:
        difference = observations[:, None, :] - self.means_[None, :, :]
        log_determinant = np.log(self.variances_).sum(axis=1)
        quadratic = (difference**2 / self.variances_[None, :, :]).sum(axis=2)
        dimension = observations.shape[1]
        return -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + quadratic)

    def _forward_backward(self, observations: Array) -> tuple[float, Array, Array]:
        emissions = self._emission_log_prob(observations)
        n_observations = observations.shape[0]
        alpha = np.zeros((n_observations, self.n_states), dtype=float)
        scales = np.zeros(n_observations, dtype=float)

        alpha[0] = self.initial_probabilities_ * np.exp(emissions[0])
        scales[0] = max(float(alpha[0].sum()), 1e-300)
        alpha[0] /= scales[0]

        for time in range(1, n_observations):
            alpha[time] = (
                alpha[time - 1] @ self.transition_matrix_
            ) * np.exp(emissions[time])
            scales[time] = max(float(alpha[time].sum()), 1e-300)
            alpha[time] /= scales[time]

        beta = np.ones_like(alpha)
        for time in range(n_observations - 2, -1, -1):
            beta[time] = (
                self.transition_matrix_
                @ (np.exp(emissions[time + 1]) * beta[time + 1])
            ) / scales[time + 1]

        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

        transition_counts = np.zeros_like(self.transition_matrix_)
        for time in range(n_observations - 1):
            numerator = (
                alpha[time, :, None]
                * self.transition_matrix_
                * np.exp(emissions[time + 1])[None, :]
                * beta[time + 1][None, :]
            )
            denominator = max(float(numerator.sum()), 1e-300)
            transition_counts += numerator / denominator

        log_likelihood = float(np.log(scales).sum())
        return log_likelihood, gamma, transition_counts

    def fit(self, observations: Array) -> "GaussianDiagonalHMM":
        """Estimate HMM parameters with the Baum-Welch EM algorithm."""

        values = _as_2d(observations)
        n_observations, n_features = values.shape
        if n_observations < self.n_states:
            raise ValueError("not enough observations for the requested states")

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_observations, size=self.n_states, replace=False)
        global_variance = np.maximum(values.var(axis=0), self.min_variance)
        self.initial_probabilities_ = np.full(self.n_states, 1.0 / self.n_states)
        self.transition_matrix_ = np.full(
            (self.n_states, self.n_states), 1.0 / self.n_states
        )
        self.means_ = values[indices].copy()
        self.means_ += rng.normal(0.0, 0.01, size=(self.n_states, n_features))
        self.variances_ = np.tile(global_variance, (self.n_states, 1))

        previous_likelihood = -np.inf
        for iteration in range(1, self.max_iter + 1):
            likelihood, gamma, transition_counts = self._forward_backward(values)

            state_mass = np.maximum(gamma.sum(axis=0), 1e-300)
            self.initial_probabilities_ = np.maximum(gamma[0], 1e-12)
            self.initial_probabilities_ /= self.initial_probabilities_.sum()

            if n_observations > 1:
                self.transition_matrix_ = transition_counts / np.maximum(
                    transition_counts.sum(axis=1, keepdims=True), 1e-300
                )

            self.means_ = (gamma.T @ values) / state_mass[:, None]
            centered = values[:, None, :] - self.means_[None, :, :]
            self.variances_ = (
                gamma[:, :, None] * centered**2
            ).sum(axis=0) / state_mass[:, None]
            self.variances_ = np.maximum(self.variances_, self.min_variance)

            self.n_iter_ = iteration
            if abs(likelihood - previous_likelihood) < self.tolerance:
                break
            previous_likelihood = likelihood

        # Recompute after the final parameter update so the reported score is
        # associated with the parameters actually used for forecasting.
        self.log_likelihood_, _, _ = self._forward_backward(values)
        return self

    def predict_states(self, observations: Array) -> Array:
        """Return the most probable latent state path using Viterbi."""

        if self.log_likelihood_ is None:
            raise RuntimeError("fit the HMM before predicting states")
        values = _as_2d(observations)
        emissions = self._emission_log_prob(values)
        log_transition = np.log(np.maximum(self.transition_matrix_, 1e-300))
        log_initial = np.log(np.maximum(self.initial_probabilities_, 1e-300))

        scores = np.zeros((values.shape[0], self.n_states))
        backpointers = np.zeros_like(scores, dtype=int)
        scores[0] = log_initial + emissions[0]
        for time in range(1, values.shape[0]):
            candidates = scores[time - 1][:, None] + log_transition
            backpointers[time] = np.argmax(candidates, axis=0)
            scores[time] = np.max(candidates, axis=0) + emissions[time]

        path = np.zeros(values.shape[0], dtype=int)
        path[-1] = int(np.argmax(scores[-1]))
        for time in range(values.shape[0] - 2, -1, -1):
            path[time] = backpointers[time + 1, path[time + 1]]
        return path

    def forecast_next(self, observations: Array) -> Array:
        """Forecast the next factor-return vector using equation (18)."""

        states = self.predict_states(observations)
        current_state = int(states[-1])
        return self.transition_matrix_[current_state] @ self.means_

    @property
    def aic(self) -> float:
        """AIC with the full diagonal-Gaussian parameter count."""

        if self.log_likelihood_ is None:
            raise RuntimeError("fit the HMM before requesting AIC")
        n_features = self.means_.shape[1]
        parameter_count = (
            (self.n_states - 1)
            + self.n_states * (self.n_states - 1)
            + 2 * self.n_states * n_features
        )
        return float(-2.0 * self.log_likelihood_ + 2.0 * parameter_count)


def select_hmm(
    observations: Array,
    state_candidates: tuple[int, ...] = (2, 3, 4, 5),
    *,
    max_iter: int = 60,
    random_state: int = 42,
) -> tuple[GaussianDiagonalHMM, list[dict[str, float]]]:
    """Fit candidate state counts and return the model with lowest AIC."""

    candidates: list[dict[str, float]] = []
    models: list[GaussianDiagonalHMM] = []
    for offset, n_states in enumerate(state_candidates):
        model = GaussianDiagonalHMM(
            n_states,
            max_iter=max_iter,
            random_state=random_state + offset,
        ).fit(observations)
        models.append(model)
        candidates.append(
            {
                "n_states": float(n_states),
                "log_likelihood": float(model.log_likelihood_),
                "aic": float(model.aic),
            }
        )
    best_index = int(np.argmin([item["aic"] for item in candidates]))
    return models[best_index], candidates
