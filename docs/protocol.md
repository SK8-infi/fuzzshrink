# Pre-registered evaluation protocol

**Status: fixed before any empirical result was computed.** This file was committed in stage 1, before the data pipeline (stage 4) and the backtest (stage 5) existed; the git history records the order. Any later change is listed in section 9 with its date and reason. Results that depend on a deviation are labelled as such.

## 1. Questions

1. Does routing the fuzzy membership matrix into the covariance estimator lower the *realised* risk of a minimum-variance portfolio?
2. If it does, is the gain caused by the membership information, and by the mechanism we claim?
3. Is the gain large enough to matter economically, against strong existing estimators?

## 2. Data

- **Universe.** NASDAQ-100 constituents as listed on Wikipedia on the download date. The list and the date are saved in `data/universe/`. This is a current-constituent universe, so it carries survivorship bias, as in the base paper. We state this and do not correct for it.
- **Prices.** Daily adjusted closing prices from Yahoo Finance (via `yfinance`), 2014-07-01 to 2024-12-31, so that the longest window (504 days) is available before January 2017.
- **Returns.** Simple daily returns.
- **Universe in each window.** An asset is included at a rebalance date if its returns are complete over the estimation window and the following holding period.

## 3. Backtest design

| Item | Value |
|---|---|
| Out-of-sample period | 2017-01 to 2024-12 (96 monthly rebalances) |
| Rebalance date | First trading day of each calendar month |
| Estimation window | T = 252 trading days ending the day before the rebalance (primary); T ∈ {126, 504} as sensitivity |
| Holding period | Until the day before the next rebalance; weights held fixed, so the daily return is wᵀrₜ |
| Transaction cost | 10 bps per unit of one-way turnover, charged on the first day of each holding period; turnover measured against the drifted previous weights |
| Primary test bed | Unconstrained global minimum variance (GMV) |
| Secondary test bed | Long-only GMV |
| Annualisation | 252 days |

## 4. Configurations

All FuzzShrink variants use the angle combination, the closed-form base intensity of method §4.2, m = 2, C ∈ {2..8} by Xie–Beni, co-movement features, and `n_init` = 5, unless stated otherwise.

**Core ablation ladder.** Each configuration nests the one before it.

| ID | Estimator | Role |
|---|---|---|
| A0 | Sample covariance | Baseline as published |
| A1 | Constant-correlation shrinkage (C = 1) | Global intensity, no cluster structure |
| A2 | Hard-cluster block shrinkage (argmax of U) | Hard against soft partition |
| A3 | FuzzShrink, γ = 0 | Soft target only |
| A4 | FuzzShrink, γ = +1, entropy weights | **The draft's proposed estimator** |
| A5 | A4 with the rows of U permuted across assets, re-drawn each window | Falsification control |
| A6 | FuzzShrink, γ = −1, entropy weights | Direction test (the mirror) |

**Extensions.** These are exploratory, declared here, and reported separately.

| ID | Estimator | Question |
|---|---|---|
| X1 | FuzzShrink, γ = +1, typicality weights | Does targeting only outliers beat entropy? |
| X2 | Block-pair target, γ = +1, entropy weights | Does a richer target help? |
| X3 | A4 with the draft's arithmetic combination and projection | What does PSD-by-construction cost or gain? |
| X4 | A4 with risk-return clustering features | Do the base paper's features carry correlation structure? |

**References.**

| ID | Estimator |
|---|---|
| R1 | Ledoit–Wolf (2004) constant correlation, published intensity |
| R2 | Ledoit–Wolf (2020) analytical nonlinear shrinkage |
| R3 | Equal weight, 1/N |

**Randomness.**

- Seeds 0–9 drive the clustering. Each seed gives one full backtest of every configuration that depends on U.
- A5 uses 20 permutation runs per seed, 200 in total.
- Seed-dependent results are reported as mean ± standard deviation over seeds.

## 5. Metrics

**Primary.** Annualised standard deviation (SD) of daily out-of-sample GMV returns, before transaction costs.

**Secondary.** These are computed after costs:

- annualised mean
- Sharpe ratio (mean/SD of raw returns × √252)
- Sortino ratio
- maximum drawdown
- Calmar ratio
- average monthly turnover

**Diagnostics.**

- Average condition number of Σ̂.
- Rate of Lemma 1 failures.
- X3 projection activation rate and Frobenius distance moved.
- Average C, entropy, typicality and δ₀.

**Economic value.** The change in certainty-equivalent return of switching from A1 to configuration k:

ΔCER = 252·[(μₖ − μ_A1) − (λ/2)(σ²ₖ − σ²_A1)], for daily net returns, in bps per year, with λ ∈ {1, 5, 10}.

## 6. Hypotheses and decision rules

Tests use α = 0.05. p-values for H1 and H3–H6 are reported both raw and Holm-adjusted as one family of five.

| ID | Hypothesis | Test | Supported if |
|---|---|---|---|
| H1 | A4 has lower out-of-sample SD than A1 (and than A0) | Diebold–Mariano on squared returns, HAC variance, Harvey–Leybourne–Newbold correction, one-sided. Also a 95% CI for σ²_A4/σ²_A1 from a paired stationary bootstrap (expected block 21 days, 2,000 resamples) | Holm-adjusted p < 0.05 for A4 vs A1 |
| H2 | The A3→A4 change in realised variance concentrates in the top membership-entropy quartile | Per window, realised risk contributions wᵢ(Σ_real w)ᵢ over the holding period, summed by the entropy quartile of that window, averaged over windows. CI by bootstrap over windows | Top-quartile share of the total A3→A4 change > 50%, and its 95% CI excludes 25% |
| H3 | A4 has lower SD than A3: entropy conditioning adds value beyond the soft target | As H1 | Holm-adjusted p < 0.05 |
| H4 | A4 beats its own permutation control A5 | Permutation p-value: (1 + #{A5 runs with SD ≤ mean A4 SD}) / (1 + 200) | p < 0.05 |
| H5 | The direction is right: A4 (γ = +1) has lower SD than A6 (γ = −1) | As H1 | Holm-adjusted p < 0.05 |
| H6 | The soft partition beats the hard one: A3 has lower SD than A2 | As H1 | Holm-adjusted p < 0.05 |

**Sharpe ratios.** Pairs are compared with the Ledoit–Wolf (2008) HAC test, Bartlett kernel, Newey–West lag. The best configuration's Sharpe ratio is also reported as a deflated Sharpe ratio (Bailey and López de Prado, 2014). The number of trials counts every configuration and sensitivity cell we evaluate.

**How the results will be read.** We commit to these readings in advance.

- If H4 fails, the membership information is not what drives any gain, and **the paper's central claim is rejected** whatever H1 shows.
- If H1 holds but H3 fails, the gain comes from the soft target, not entropy conditioning. The paper is then re-framed around the target.
- If H5 fails, the draft's intuition about direction is wrong, and we say so.
- If A4 loses to R2 (nonlinear shrinkage), we report it plainly. FuzzShrink can still be useful when it adds to R2, which is a question for future work.

## 7. Simulation study

The simulation is a synthetic market with a known covariance matrix, so the cost of each estimator can be measured exactly.

### Model

- N = 100 assets, C* = 5 true groups, Student-t(5) shocks scaled to unit variance.
- Each asset follows r_ti = σᵢ(aᵢ f_t + bᵢ Σ_c u*_ic g_ct + pair terms + eᵢ ε_ti).
- The market factor f, the group factors g_c and the idiosyncratic shocks ε are all independent.
- The residual loading eᵢ is set so that each asset has unit variance before scaling.

There are three asset types:

| Type | Memberships u*ᵢ | Loadings |
|---|---|---|
| Crisp | One-hot | aᵢ ~ U(0.4, 0.6), bᵢ ~ U(0.5, 0.7) |
| Bridge | Dirichlet(1, 1) over two random groups | As crisp |
| Outlier | One-hot to a random group | bᵢ ~ U(0.05, 0.15), plus one private "pair factor" shared with one random other asset, loading 0.5 |

Daily volatilities are σᵢ ~ U(0.01, 0.03).

### Scenarios

Four mixes of bridge and outlier fractions (π_b, π_o): (0, 0), (0.3, 0), (0, 0.15) and (0.3, 0.15). Each is run at T ∈ {126, 252}, with 100 replications per cell.

### Loss

Excess out-of-sample variance of the GMV portfolio under the true Σ: L = w_kᵀΣw_k / w*ᵀΣw* − 1, where w* is the GMV of the true Σ. Every FuzzShrink variant is also run with the *true* memberships (an "oracle U"). That separates clustering error from estimator design.

### Pre-registered predictions

- **S1.** With crisp groups only, A2, A3 and A4 are within 5% of each other in mean loss.
- **S2.** With bridges, A3 beats A2. Entropy conditioning (A4) does not beat A3, because bridges are well described by a soft target.
- **S3.** With outliers, A4 beats A3, and X1 (typicality) is at least as good as A4.
- **S4.** In the mixed scenario, X1 has the lowest mean loss among the U-based variants.

## 8. Reporting

- Every configuration in section 4 is reported, including the ones that lose.
- Every hypothesis is reported with its test statistic, p-value and verdict.
- Sensitivity results (T, m, γ grid, long-only) appear in an appendix table, labelled exploratory.

## 9. Deviations

None yet. Format: date, what changed, why, and which results it affects.
