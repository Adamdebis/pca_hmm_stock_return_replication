# Paper audit

## What can be reproduced

The paper gives enough information to implement the main mathematical
pipeline: standardise returns, eigendecompose the covariance matrix, retain
the first (k) directions, fit a Gaussian HMM, use transition probabilities
and state means for a one-step forecast, reconstruct asset returns, and
calculate directional accuracy and Sharpe ratios.

## What cannot be reproduced exactly from the PDF alone

- The original Yahoo Finance download snapshot and exact ticker universe are
  not archived in the paper.
- The S&P 500 membership changes through the sample, creating survivor bias.
- No executable code or random seeds are supplied.
- HMM numerical details such as convergence tolerances, random restarts, and
  tie-breaking are not fully specified.
- The paper assumes zero transaction costs, no slippage, exact closing-price
  execution, and exact equal-weight matching.

The repository therefore labels its default run an offline, synthetic
demonstration. Supplying a dated return CSV enables an empirical run, but the
result should still be described as an independent replication unless the
original data and implementation are recovered.

## Internal inconsistencies and audit points

1. The implementation section lists (p=45\%,30\%,15\%,10\%), while Table 1
   contains a (5\%) row. `results/paper_reported_results.csv` preserves that
   distinction instead of silently changing it.
2. The printed AIC expression appears as a logarithm of a log-likelihood. The
   code uses the standard (\mathrm{AIC}=-2\log L+2q), with (q) equal to the
   number of estimated parameters, and documents this choice.
3. The paper reports a strategy-2 Sharpe ratio of 1.36 at (p=15\%), while
   strategy-1 directional accuracy is only slightly above 50%. That is not
   impossible, but it is highly sensitive to the short backtest, data choices,
   and zero-cost assumptions.

## Interpretation

The paper's result should be read as a useful research hypothesis: PCA can
denoise a broad return panel, and HMM states may supply a risk-management
signal. It is not evidence that a live strategy will reliably beat the market.
