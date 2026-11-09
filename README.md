# FuzzShrink

**Reusing what the clustering already knows to measure portfolio risk.**

A Business Economics course project at ABV-IIITM Gwalior. Machine-learning portfolio pipelines group stocks with fuzzy clustering, use the grouping once to pick stocks, and then throw it away. FuzzShrink feeds that grouping into the risk model instead. The aim is a portfolio whose risk on paper is closer to its risk in reality.

---

## 1. The problem: a portfolio is only as safe as its risk estimate

Anyone allocating money across many assets has to answer one question first: *how do these assets move together?* That answer is a covariance matrix. Everything downstream depends on it: minimum-risk portfolios, risk budgets and hedge ratios.

The difficulty is scale. With 100 stocks there are **4,950 pairwise correlations** to estimate, and a year of daily data gives only about 250 observations for each. Many of the estimated correlations are mostly noise. An optimiser cannot tell noise from signal, so it treats noise as opportunity. It piles into stocks whose risk happened to look low in the sample. Michaud (1989) called mean-variance optimisation an "error maximiser" for exactly this reason.

The result is a portfolio that looks safe on paper and turns out riskier in reality. That gap has an economic price:

- **Higher realised risk** than the investor signed up for.
- **More trading**, because noisy estimates change from month to month and drag the portfolio weights with them. Every trade pays transaction costs.
- **Worse risk-adjusted returns** after those costs.

The standard remedy is **shrinkage** (Ledoit and Wolf, 2004). Blend the noisy sample estimate with a simple, stable "target" structure, in proportions chosen to minimise the expected error. The open question is which target to shrink toward.

## 2. The gap: information that is computed, used once, and thrown away

Recent machine-learning pipelines run in three stages. Our base paper is Dastkhan, Mosaffa and Fabozzi (2026), *Annals of Operations Research*.

1. **Clustering**: fuzzy clustering groups the stocks.
2. **Forecasting**: an LSTM predicts returns.
3. **Allocation**: a genetic algorithm splits the capital.

The fuzzy clustering stage produces a **membership matrix**: for every stock, how strongly it belongs to each group. That is a detailed map of the market. It shows which stocks are clearly "semiconductors", which sit between two groups, and which fit no group at all.

The pipeline uses this map **only to select stocks**. The allocation stage then estimates risk from raw returns as if the clustering had never happened.

In economic terms, this is an investment with an unclaimed return. The clustering has already been paid for, in data and compute, but its output is not used where it would be worth most: in the risk estimate.

## 3. The idea: shrink each stock only as far as its group describes it

FuzzShrink routes the membership matrix into the covariance estimator in two ways.

1. **A soft block target.** When a correlation is too noisy to trust, fall back on the group structure: stocks in the same group move together like their group, and stocks in different groups like the market. Because memberships are fuzzy, a stock can belong partly to two groups (a chip-design firm that is also a cloud business). Its target is then a blend of both groups.
2. **A per-stock intensity.** Shrink a stock toward the group structure *only to the degree that the structure describes it*. A stock that clearly belongs to one group borrows strength from it. A stock that no group explains keeps more of its own sample behaviour.

The estimator is a drop-in replacement for the covariance input of an existing pipeline; nothing else changes. That is what lets any improvement be credited to it alone.

It also **contains the standard estimators as exact special cases**:

- With a single group, it is constant-correlation shrinkage (Ledoit and Wolf, 2004).
- With crisp, one-group-per-stock clustering, it is hard block shrinkage.

So FuzzShrink generalises established methods rather than competing with them.

## 4. What this repository adds beyond the paper draft

The first draft of our paper described the estimator and an evaluation plan. Building it exposed five problems, which this repository fixes. [docs/method.md](docs/method.md) has the details.

| Problem in the draft | What we do now |
|---|---|
| The combined estimate could fail to be a valid covariance matrix, so the draft needed a repair step after the fact. | An **"angle" combination rule** that is a valid covariance matrix **by construction**, with a short proof. The draft's special cases still hold exactly. |
| No condition was given for the target itself to be valid. | **Lemma 1**: the soft target is valid whenever within-group correlation ≥ between-group correlation ≥ 0. |
| The base intensity rule was borrowed from a different target. | **One closed-form rule** that is optimal for each variant's own target and weights, so every comparison is fair. |
| Membership entropy mixes up two kinds of ambiguity. A *bridge* stock (split between two groups) is described well by a soft target; an *outlier* (fitting no group) is not. | A **typicality** variant that targets outliers only. A **direction test** (γ < 0) checks whether the data even agree with the draft's intuition. |
| The base paper clusters on risk-return statistics, but a correlation target needs groups of stocks that move together. | **Co-movement features** (market-neutral principal-component loadings) as the default, with the draft-faithful features kept as a comparison. |

## 5. How we try to prove ourselves wrong

A covariance estimator is easy to oversell with one favourable backtest. Our evaluation is built to falsify the claim if it is false.

- **Pre-registered protocol.** The hypotheses, configurations, metrics, tests and decision rules are in [docs/protocol.md](docs/protocol.md). It was committed *before* any result was computed, and the git history shows the order.
- **A simulation with a known answer.** In a synthetic market where we choose the true groups, bridges and outliers, we can measure exactly when each component helps.
- **A permutation control.** We shuffle which stock gets which membership vector. If the shuffled version does as well, the gain did not come from the clustering, and the claim fails.
- **A mechanism test.** If entropy conditioning works for the reason we say, the risk reduction should concentrate in the most ambiguous stocks.
- **Strong baselines, not just weak ones.** These are sample covariance, Ledoit–Wolf (2004), Ledoit–Wolf (2020) nonlinear shrinkage, the current state of the art for minimum-variance portfolios, and the naive 1/N portfolio.
- **Economic value, not just statistics.** We report how much a mean-variance investor would pay per year, in basis points, to switch from standard shrinkage to FuzzShrink.

## 6. Results

*Filled in by stage 6, once the pre-registered experiments have run.*

## 7. Repository map

| Path | What it holds |
|---|---|
| `src/fuzzshrink/` | The estimator package. It needs only numpy, scipy and pandas. |
| `scripts/` | Data download, simulation and backtest runners, and figures. |
| `tests/` | Unit tests, including the draft's special-case propositions written as tests. |
| `docs/method.md` | The exact specification of every estimator. |
| `docs/protocol.md` | The pre-registered evaluation protocol. |
| `results/` | Summary tables and figures (stage 6). |

## 8. Reproducing

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The data-download and experiment commands arrive with stages 4–6.

## 9. Team

Hemateja, Neerav Mishra, Srijan Singh, Shivansh Katiyar and Tanushri. Atal Bihari Vajpayee Indian Institute of Information Technology and Management, Gwalior.

## References

- Dastkhan, H., Mosaffa, A., Fabozzi, F. J. (2026). Integrated machine learning framework for dynamic portfolio management: clustering, forecasting, and optimization. *Annals of Operations Research*. https://doi.org/10.1007/s10479-026-07053-9
- Ledoit, O., Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*, 30(4), 110–119.
- Ledoit, O., Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics*, 48(5), 3043–3065.
- Michaud, R. O. (1989). The Markowitz optimization enigma: is "optimized" optimal? *Financial Analysts Journal*, 45(1), 31–42.
- Schäfer, J., Strimmer, K. (2005). A shrinkage approach to large-scale covariance matrix estimation. *Statistical Applications in Genetics and Molecular Biology*, 4(1).
- Bezdek, J. C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms*. Plenum.
- Krishnapuram, R., Keller, J. M. (1993). A possibilistic approach to clustering. *IEEE Transactions on Fuzzy Systems*, 1(2), 98–110.
