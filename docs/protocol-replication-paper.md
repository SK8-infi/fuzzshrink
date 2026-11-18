# Protocol: replication study of the base paper (stage 9, paper 2)

**Status: fixed before any stage 9 computation.** Committed 2026-11-18, after stage 8 and after a check of the base paper's published tables (section 2), and before any stage 9 code ran. Changes go in section 8.

The paper this supports, *"Do integrated machine-learning portfolio frameworks survive honest evaluation? A replication of Dastkhan, Mosaffa and Fabozzi (2026)"*, asks five questions. Q1 uses only the published paper. Q2–Q5 are computed here.

## 1. Data and conventions

- **Stocks.** The 85 base-paper stocks available on Yahoo Finance (`data/universe/base_paper_universe.csv`). ANSS, EA and WBD are unavailable.
- **Prices.** Adjusted daily closes for 2017-01-03 to 2024-12-31. QQQ is the NASDAQ-100 benchmark.
- **Portfolios.** Formed on the first trading day of each month and held to the next rebalance.
- **Costs.** The paper's rule: 0.1% to buy and 0.2% to sell on the full portfolio every month, so net = (1 − 0.001)(1 + R)(1 − 0.002) − 1. Benchmarks pay the same cost.
- **Risk-free rate.** Zero.
- **Standard Sharpe ratio.** Monthly mean ÷ monthly standard deviation × √12. The paper's Eq. 15 is reported alongside.
- **Seeds.** 0–4, controlling the LSTM, fuzzy c-means, the GA and random draws.

## 2. Q1: are the published figures internally consistent? (published numbers only)

Computed from the paper's Tables 7, 8 and 10 and its Eq. 14–15, plus 2024 index data. Already computed while designing this protocol:

- The Table 7 wealth of $5,343.96 implies a 434.4% return; the table reports 565.89%.
- The Nasdaq-100 row reports a 1.23% annual return, while its own wealth figure implies 26.6%.
- The paper's own Eq. 15 applied to actual 2024 Nasdaq-100 data gives 2.30, against the reported 0.31.

The paper will report these as a table, recomputable by `scripts/audit_paper_tables.py`.

## 3. Q2: forecast skill

- **F1.** The paper reports an average price MAPE of 1.67% for 2024. We compute the MAPE of a **random-walk forecast** (the last known price) for the 85 stocks in 2024:
  - at horizon 1 day;
  - over horizons 1–21 days from each month-start, which is how next-month daily prices would be forecast.

  **Reading, fixed now:** if the 1-day random-walk MAPE is at most 1.67%, the reported accuracy is not evidence of forecasting skill.
- **F2.** The monthly cross-sectional Spearman correlation (information coefficient, IC) between our LSTM's next-month return forecasts and realised returns, 2018-01 to 2024-12 (84 months), for each seed. Supported if the mean IC > 0 with one-sided HAC p < 0.05 in at least 3 of 5 seeds.

## 4. Q3: how much does picking the setting on the test year matter?

**S1, against random selection.** For 2024, each seed, and each C ∈ {2..8}, we draw 100 random C-stock paths, picking new random stocks every month as in the paper's Alternative Model 1. They use the same LSTM forecasts, sample covariance and GA as the clustered pipeline. The null for "best of 21 settings" is the best of 21 random paths (three per C), resampled 10,000 times.

The observed statistic is the clustered pipeline's best-of-21 2024 wealth (rule E, as in the paper). S1 is supported if it exceeds the 95th percentile of the null.

**S2, probability of backtest overfitting** (Bailey, Borwein, López de Prado and Zhu, 2017). This uses combinatorially symmetric cross-validation:

- **Inputs:** the 21 settings' monthly net returns, 2018-01 to 2024-12 (84 months).
- **Splits:** 12 blocks of 7 months, giving C(12, 6) = 924 splits.
- **Metric:** the standard Sharpe ratio.

We report PBO for each seed. **Reading, fixed now:** PBO > 0.5 means choosing the best setting in-sample is more likely than not to give below-median performance out of sample.

## 5. Q4: the paper's module decomposition over seven years

The paper's Section 4.2 argues each module helps, using 12 months. We rerun its three alternative models over **2019-01 to 2024-12 (72 months)**, with **annual honest selection**: at the start of each year, every pipeline uses the setting with the highest wealth over the previous calendar year (2018 selects for 2019).

| ID | Pipeline |
|---|---|
| FULL | The paper's pipeline: fuzzy c-means on whitened (mean, sd), equal to Mahalanobis FCM; one representative per cluster by the criterion; the stage 7 pooled LSTM retrained monthly on 504 days; sample covariance on all history; the paper's GA (population 50, 100 generations, best of 10 runs) |
| ALT1 | Random selection: C random stocks a month, otherwise FULL. Three independent draws per C stand in for the three criteria |
| ALT2 | FULL with historical mean returns instead of LSTM forecasts |
| ALT3 | FULL's selection with equal weights (no optimiser) |

| ID | Hypothesis (the paper's own claims) | Test |
|---|---|---|
| D1 | FULL > ALT1 (clustering adds value) | One-sided HAC t-test on the monthly net-return difference |
| D2 | FULL > ALT2 (the LSTM adds value) | As D1 |
| D3 | FULL > ALT3 (the optimiser adds value) | As D1 |
| D4 | FULL > QQQ | As D1 |

## 6. Q5: a fixed, evidence-based improvement

**IMP** keeps FULL's clustering but applies the lessons of stages 6–7:

- **Broader selection:** the top 3 stocks per cluster by the criterion, so 6–24 stocks.
- **Shrunk forecasts:** LSTM forecasts pulled halfway to their cross-sectional mean.
- **Better covariance:** Ledoit–Wolf (2020) nonlinear shrinkage on the trailing 252 days.
- **Exact optimiser:** the exact long-only maximum-Sharpe portfolio with a 30% cap per stock.

IMP uses the same annual honest selection over its own 21 settings.

| ID | Hypothesis | Test |
|---|---|---|
| I1 (primary for Q5) | IMP > FULL | As D1 |
| I2 | IMP > QQQ | As D1 |

**Decision rule for D1–D4 and I1–I2.** Supported if the median one-sided p across seeds is below 0.05 and at least 3 of 5 seeds have a positive mean difference. We report raw p-values and Holm-adjusted p-values across these six, together with the standard Sharpe ratio, final wealth and maximum drawdown for every pipeline and both benchmarks.

## 7. Reporting

Every hypothesis is reported, whatever it shows. Six years of monthly data, and especially the single 2024 year, give limited power; we state this next to the results.

## 8. Deviations

None yet.
