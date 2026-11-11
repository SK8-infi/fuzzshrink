# Protocol: replication of the base paper's own 2024 test (stage 8)

**Status: fixed before any stage 8 result was computed.** Committed 2026-11-11, after reading the full base paper and before running any stage 8 code. Changes go in section 6.

## 1. Why

Stage 7 compared pipelines over 2018–2024. The base paper's published test is narrower:

- **Period:** 12 monthly portfolios in **2024**, trained on 2017-01 to the rebalance date.
- **Portfolio:** one representative stock per fuzzy cluster.
- **Optimiser:** a Sharpe-maximising genetic algorithm.
- **Costs:** a full sell (0.2%) and re-buy (0.1%) every month.
- **Headline:** $1,000 → **$5,344**, from four clusters with maximum-return representatives (Table 7).

The paper chose the number of clusters (2–8) and the representative criterion (maximum return, maximum Sharpe, minimum risk) by the final wealth **in 2024, the test year**. This stage reruns their protocol for their pipeline and ours, under two selection rules.

- **Rule E (ex post, as in the paper).** The best of the 21 (C, criterion) combinations by 2024 final wealth. This matches how their $5,344 was produced, so it is the like-for-like comparison with Table 7. It is optimistic for both pipelines.
- **Rule V (validation, the honest rule).** The best combination by final wealth over 2023, run the same way, applied unchanged to 2024.

## 2. Data and costs

- **Stocks.** The 85 base-paper stocks still available on Yahoo Finance.
- **Prices.** Adjusted closes, 2017-01-03 to 2024-12-31.
- **Rebalancing.** On the first trading day of each month, held to the next rebalance.
- **Costs.** A month's net return is (1 − 0.001)(1 + R)(1 − 0.002) − 1, exactly as the paper assumes.
- **Risk-free rate.** Zero.

## 3. Pipelines

**B24, a reconstruction of the paper.**

1. **Clustering.** Each stock's mean daily return and daily standard deviation, over all data since 2017-01, are whitened by their 2×2 covariance. That makes Euclidean fuzzy c-means equal to Mahalanobis fuzzy c-means. FCM runs with k-means++ seeding and m = 2.
2. **Representative.** One per cluster by the criterion.
3. **Forecast.** The expected next-month return of each representative, from the pooled LSTM of stage 7, retrained every month on the most recent 504 days.

   The paper used per-stock price LSTMs; this difference is stated as a limitation.
4. **Risk.** The sample covariance of daily returns since 2017-01.
5. **Genetic algorithm (paper §3.3).** It maximises expected return ÷ risk.
   - Population 50, 100 generations, elitism, fitness-proportional selection and one-point crossover.
   - Gaussian mutation with standard deviation 0.05 and probability 0.05.
   - Weights clipped to [0, 1] and renormalised, so zero weights are allowed.
   - Best of 10 runs.

**P24, ours.** It uses the same representatives' universe logic, with three changes fixed now:

1. **Co-movement features.** Clustering uses market-neutral co-movement features (stage 2) over the trailing 252 days instead of (mean, standard deviation), and Euclidean FCM. The representative criteria are unchanged.
2. **Shrunk forecasts.** Forecasts are pulled halfway toward their cross-sectional mean: μ̃ = μ̄ + ½(μ̂ − μ̄).
3. **Exact optimiser.** The exact long-only maximum-Sharpe solution (no cap) replaces the genetic algorithm. Ledoit–Wolf (2020) covariance is computed over the trailing 252 days.

Both pipelines use **the same LSTM forecasts** for each seed. Seeds 0–4.

## 4. Metrics and hypotheses

- **Metrics.** Final wealth from $1,000 after costs (the paper's headline), the paper's Eq. 15 Sharpe ratio, and a standard annualised Sharpe from the 12 monthly net returns.
- **Benchmarks.** QQQ and equal weight across the 85 stocks, both with the same monthly cost rule.

| ID | Hypothesis | Supported if |
|---|---|---|
| R1 (primary) | Under rule V, P24 ends 2024 with more wealth than B24 | Higher in the seed mean and in at least 3 of 5 seeds |
| R2 | Under rule E, P24 ends with more wealth than B24 | Higher in the seed mean |
| R3 | Under rule V, P24 beats QQQ | Higher final wealth in the seed mean |

We will also report how B24 under rule E compares with the paper's published $5,344. A large gap means the published figure could not be reproduced from the paper's description.

## 5. Reporting

Every (C, criterion) cell for both pipelines and both years is reported, along with both selection rules, the benchmarks, and the paper's own Table 7 numbers for reference. Twelve monthly returns make every comparison here statistically weak; we say so next to the results.

## 6. Deviations

None yet.
