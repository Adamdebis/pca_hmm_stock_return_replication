# Mathematical guide

This guide is written to defend the code at a graduate quantitative-finance
level. It gives the equations first and leaves implementation details visible
in `src/pca_hmm.py`.

## 1. Return matrix and standardisation

Let (X \in \mathbb{R}^{T \times n}) contain (T) weekly observations for
(n) assets. Column (j) is one asset's return history. The paper standardises
each column:

\[
Y_{tj}=\frac{X_{tj}-\mu_j}{\sigma_j}.
\]

This prevents a high-volatility asset from dominating PCA merely because its
raw numerical scale is larger.

## 2. PCA as a factor model

The covariance matrix is proportional to (Y^\top Y). The code uses the
conventional sample-covariance scaling (1/(T-1)); this scalar does not change
the eigenvectors. We solve

\[
H=Y^\top Y=EGE^\top,
\]

where the columns of (E) are orthonormal eigenvectors and (G) contains the
eigenvalues. Large eigenvalues correspond to directions explaining more
cross-sectional variation.

The factor-return scores are

\[
F=YE_k,
\]

where (E_k) contains only the first (k) eigenvectors. If the retained
eigenvalues explain at least (1-p) of total variance, the paper interprets
the remaining approximately (p) as noise and drops it.

The reconstructed normalized asset returns are

\[
\widehat{Y}_{t+1}=\widehat{F}_{t+1}E_k^\top,
\qquad
\widehat{X}_{t+1}=\widehat{Y}_{t+1}\odot\sigma+\mu.
\]

## 3. Hidden Markov model

The observed factor vector (F_t\) is driven by an unobserved state

\[
Z_t\in\{1,\ldots,N\}.
\]

The Markov property is

\[
P(Z_{t+1}=j\mid Z_t=i, Z_{t-1},\ldots)=P(Z_{t+1}=j\mid Z_t=i)=P_{ij}.
\]

The transition matrix (P) has non-negative rows that sum to one. Conditional
on state (i), the code assumes a diagonal Gaussian emission:

\[
F_t\mid Z_t=i\sim\mathcal{N}(\mu_i,\operatorname{diag}(\sigma_i^2)).
\]

The state is “hidden” because we observe factor returns but not a label saying
“bull” or “bear.” The Baum-Welch algorithm estimates the transition matrix,
state means, and state variances. Viterbi decoding identifies the most likely
current state.

## 4. One-step forecast

If the current decoded state is (i), the paper's equation (18) gives

\[
\widehat{F}_{t+1}=\sum_{j=1}^N P_{ij}\widehat{\mu}_j.
\]

This is a probability-weighted average of the next state's conditional means.
It is not a claim that one particular regime is certain.

## 5. Backtesting

Strategy 1 takes the sign of the reconstructed raw-return forecast. Positive
forecasts receive equal long weights and negative forecasts receive equal short
weights. Strategy 2 applies the same rule to normalized forecasts. For a weekly
portfolio return (r_t), the annualized Sharpe ratio is

\[
S=\sqrt{52}\frac{\overline r-r_f}{s_r}.
\]

The paper sets (r_f=0), assumes exact closing-price execution, and ignores
transaction costs. Those assumptions are retained for comparability and
reported as limitations rather than hidden from the reader.
