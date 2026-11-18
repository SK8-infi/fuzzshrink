# Protocol: head-to-head against the base paper (stage 7)

**Status: fixed before either pipeline was run.** Committed on 2026-11-10, before the comparison code produced any result; the git history records the order. Changes go in section 7.

## 1. Why a reconstruction

The base paper (Dastkhan, Mosaffa and Fabozzi, 2026) publishes no code, and its results tables are behind a paywall we could not access. We therefore rebuild its three-module pipeline from its published description and compare our pipeline against that rebuild **on identical data, dates, costs and metrics**. This is the comparison the draft paper planned (its §7: "the comparison rests on a reconstruction from its published description"). If the paper's own numbers become available, we will report them next to ours, with the caveat that their exact setup may differ.

## 2. Data

- **Universe.** The 88 stocks listed in the base paper's Appendix 1 (`data/universe/base_paper_universe.csv`). 85 still have Yahoo Finance histories. ANSS, EA and WBD were acquired or taken private after 2024 and are unavailable, which is a stated limitation.
- **Check.** Our daily mean returns and risks match the paper's appendix: median absolute gaps of 0.00016 and 0.00046.
- **Prices.** Adjusted daily closes for 2017-01-01 to 2024-12-31, matching the paper's sample. No earlier data is used.
- **Benchmarks.** Equal weight across the 85 stocks, and QQQ (a NASDAQ-100 tracker) bought and held.

## 3. Backtest

| Item | Value |
|---|---|
| Out-of-sample period | 2018-01 to 2024-12 (84 monthly rebalances); 2017 is the first training year |
| Rebalance | First trading day of each month, long-only, fully invested, at most 30% per stock |
| Estimation window | All available history up to the rebalance, capped at 252 days (risk); LSTM trained on up to 504 days, retrained every 3 months |
| Costs | 10 bps per unit of one-way turnover |
| Seeds | 0–4 (LSTM initialisation, GA and clustering randomness) |

## 4. The two pipelines

**B, the base-paper reconstruction.**

1. **Selection.** Each stock is described by trailing mean return, volatility and average correlation to the others ("interdependencies"), z-scored. Outlier-aware fuzzy c-means follows, with C by Xie–Beni over 2..8. A stock whose largest membership is below 0.5 counts as an outlier and is dropped. The top stocks by trailing Sharpe ratio are taken from each cluster, in proportion to cluster size, 20 in total.
2. **Forecast.** A pooled LSTM (one layer, hidden size 16) maps the last 60 standardised daily returns to the next 21-day return.
3. **Allocation.** A multi-objective genetic algorithm (NSGA-II: population 100, 150 generations) maximises forecast return and minimises sample-covariance risk; the Pareto point with the highest forecast Sharpe ratio is held.

**P, our pipeline.** It makes the same three moves, with the four changes this project's evidence supports. Every change is fixed now:

1. **Co-movement features.** Selection clusters on market-neutral co-movement features, which beat risk-return features in our simulation (X4).
2. **Shrunk forecasts.** Forecasts are pulled halfway toward the cross-sectional mean: μ̃ = μ̄ + ½(μ̂ − μ̄). LSTM return forecasts are noisy, and unshrunk noise is what mean-variance optimisers exploit.
3. **Better covariance.** Ledoit–Wolf (2020) nonlinear shrinkage replaces the sample covariance; it was the best estimator in our stage 6.
4. **Exact optimiser.** The maximum-Sharpe portfolio under the same constraints (long-only, 30% cap) is solved exactly instead of approximately by a GA.

P uses the **same LSTM forecasts** as B for each seed, so the comparison isolates the changes rather than training luck.

**Ablation ladder (exploratory).** B, then B + Ledoit–Wolf covariance, then + exact optimiser, then + shrunk forecasts, then + co-movement selection, which equals P.

## 5. Metrics and hypotheses

Metrics are computed after costs:

- annualised return and volatility
- Sharpe ratio (raw returns, × √252)
- Sortino ratio
- maximum drawdown
- Calmar ratio
- cumulative return
- average turnover

| ID | Hypothesis | Test | Supported if |
|---|---|---|---|
| B1 (primary) | P has a higher Sharpe ratio than B | Mean over seeds, plus the Ledoit–Wolf (2008) HAC test per seed (median p reported) | Higher in the seed mean **and** in at least 4 of 5 seeds |
| B2 | P has lower volatility than B | Diebold–Mariano on squared returns, per seed | Median one-sided p < 0.05 |
| B3 | P has a smaller maximum drawdown than B | Seed means | Smaller in the seed mean |

We commit to reporting all three, every ablation step and both benchmarks, whatever they show. Over 7 years of monthly rebalancing a Sharpe difference needs to be large to be statistically significant; B1's decision rule is therefore a consistency rule across seeds, and the HAC p-value is reported alongside it.

## 6. What "beating the base paper" means here

P beats B on the primary metric (B1). Any claim against the paper's **published** numbers will be labelled as indirect, because the setups may differ.

## 7. Deviations

None yet.
