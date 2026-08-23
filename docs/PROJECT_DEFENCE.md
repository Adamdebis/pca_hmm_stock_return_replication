# Project defence notes

## One-sentence explanation

I reduced a large stock-return panel to a few PCA factor series, fitted a
diagonal Gaussian HMM to infer latent market regimes, forecast the next factor
returns using the transition matrix, reconstructed stock-return forecasts, and
evaluated two equal-weight trading rules against buy-and-hold.

## Why PCA before HMM?

An HMM on hundreds of correlated assets is high-dimensional and noisy. PCA
rotates the data into orthogonal directions ordered by explained variance. The
HMM is then trained on a much smaller set of factor-return series.

## Why is the state hidden?

The data contain returns, not labels such as “bull market” or “bear market.”
The HMM estimates which latent state most plausibly generated each observation.
The labels are arbitrary; their economic interpretation comes from the
estimated conditional means and variances, not from the integer name 0, 1, or 2.

## Why use AIC?

Increasing the number of states always gives the model more flexibility. AIC
balances fit against parameter count, so we select the candidate with the
lowest penalised score rather than the highest likelihood alone.

## What would I improve?

I would archive the exact historical universe and data snapshot, use walk-forward
hyperparameter selection, add transaction costs and slippage, run multiple
random starts, report confidence intervals, and compare against simpler factor
and momentum baselines.

## What did I not claim?

I did not claim exact numerical reproduction of the author's S&P 500 results,
because the original data snapshot, code, and seeds are unavailable. The
repository separates reported paper values from independent results.
