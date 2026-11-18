# Revising the paper draft

The draft (v0) described the estimator and an evaluation plan but had no results. This guide maps each section of the draft to what should change, now that the code and results exist. Numbers come from `results/`; re-check them if you re-run anything.

## Title

The draft's title promised an improvement the data did not deliver. Two honest options:

- *FuzzShrink: A Pre-Registered Test of Recycling Fuzzy Cluster Memberships for Portfolio Risk*
- *Does Reusing Fuzzy Cluster Memberships Improve Portfolio Risk Estimates? Evidence from a Pre-Registered Evaluation*

## Abstract (proposed text, about 200 words)

> Integrated machine-learning portfolio pipelines compute a fuzzy membership matrix to select assets and then discard it. We ask whether this by-product can be recycled to estimate portfolio risk. FuzzShrink routes the memberships into a covariance estimator through a soft block shrinkage target and per-asset intensities.
>
> We first repair two defects of the original design:
>
> - a combination rule that can produce invalid covariance matrices;
> - an intensity signal, membership entropy, that cannot tell bridge assets from outliers.
>
> We then evaluate the estimator under a protocol fixed before any result was computed. In a simulation with known structure, soft targets beat hard clusters, and a direct fit-to-target intensity beats entropy.
>
> On NASDAQ-100 constituents (2017–2024, monthly global-minimum-variance rebalancing), the estimator does not reduce realised risk relative to constant-correlation shrinkage. It is also indistinguishable from its own permutation control, so by our pre-registered rule the central claim is rejected for this market.
>
> Three results survive:
>
> - soft memberships beat hard clusters in every seed;
> - the original combination rule fails catastrophically (235% annualised volatility), while the corrected rule is safe by construction;
> - Ledoit–Wolf nonlinear shrinkage is the best estimator overall.
>
> The economic lesson is that a sunk informational by-product has value only if it beats the best available alternative, and here that alternative needs no clustering at all.

## Section by section

| Draft section | Change |
|---|---|
| **1 Introduction** | Keep the "discarded membership matrix" motivation. Add the economic framing: the matrix is a *sunk* investment, and the question is the *marginal* value of reusing it, measured against the best available alternative. State the outcome in the last paragraph of the introduction; do not hide it. |
| **1.2 Contribution** | Replace the single-estimator claim with four contributions: (i) the corrected estimator (angle rule, Lemma 1, the closed-form base intensity); (ii) a simulation-based account of when cluster information helps, showing that entropy conflates bridges and outliers; (iii) a pre-registered empirical test with a permutation control; (iv) the economic-value analysis. |
| **2 Related work** | Keep Table 1. Add Ledoit and Wolf (2020) (nonlinear shrinkage, the strongest baseline), Schäfer and Strimmer (2005), Higham (2002), Michaud (1989), DeMiguel, Garlappi and Uppal (2009) on 1/N, Diebold and Mariano (1995), Ledoit and Wolf (2008), Bailey and López de Prado (2014), and Politis and Romano (1994). |
| **3 Positioning** | Add a row to Table 2 for the angle rule and one for fit weights. Soften "what is not established": the data now say what is *not* valuable on this market. |
| **4 Baseline** | Say plainly that the LSTM and genetic-algorithm modules are not reconstructed (no public code), and that the GMV test bed isolates the covariance input, as the draft's §6.1 already argued. |
| **5.1–5.3** | Unchanged, but add the co-movement features (market-neutral PCA loadings, k set by the Marchenko–Pastur edge) and Xie–Beni selection of C. Add **Lemma 1** after Eq. 4. |
| **5.5 Intensity** | Keep entropy (Eq. 5–6) as the draft's proposal. Add the fit weight q_i as the alternative (method §4.1). Explain bridges against outliers with Fig. 1. |
| **5.6–5.7** | Replace the arithmetic rule and projection with the **angle rule** and **Proposition 3**. Keep the draft's rule as ablation X3 and report what happened to it: projection in 20% of windows, 18× leverage, 235% volatility. This is a strong, concrete result. |
| **New 5.8 Base intensity** | The closed-form δ₀ (method §4.2). Note, as a finding, that it shrinks harder than Ledoit–Wolf's intensity on real data (0.29 against 0.19), and that the post-hoc check shows the verdict does not depend on this choice. |
| **6 Experimental design** | Replace with a summary of `docs/protocol.md`:<br>- data and listing overrides;<br>- configurations A0–A6, X1–X4, R1–R3;<br>- hypotheses H1–H6 and predictions S1–S4;<br>- tests (DM with HLN, stationary bootstrap, Ledoit–Wolf Sharpe test, deflated Sharpe ratio, Holm);<br>- the statement that the protocol was committed before any result (cite the repository history). |
| **New 7 Results** | 7.1 simulation (Fig. 1–2, the prediction table); 7.2 real data (Fig. 3 and 5, the hypothesis table); 7.3 sensitivity (window length, m, the γ sweep in Fig. 4, long-only); 7.4 post-hoc intensity check; 7.5 economic value (the risk-only ΔCER table). |
| **8 Limitations** | Keep the draft's list. Add: one market and one period; the simulation's structure is our own design; a flat cost model. |
| **9 Conclusion** | Rewrite around the "what this means" points in README §6.4. Close on the draft's own principle: report the controls whatever they show. |

## Questions to prepare for

- **"Your method lost. Why is this a contribution?"** The draft committed to a falsification test and we ran it. The corrected combination rule prevents a catastrophic failure (235% volatility), soft-over-hard holds in 10 of 10 seeds, and the simulation explains why entropy is the wrong signal. A pre-registered negative result also saves others from building on a claim that does not hold.
- **"Did it lose because of your δ₀ choice?"** No. With Ledoit–Wolf's intensity the variants tie R1 at best, and entropy weighting still loses to shuffled memberships (EXPLORATORY.md, p = 0.89).
- **"Why does the simulation disagree with the data?"** The simulation has strong group structure by design. NASDAQ-100 co-movement is dominated by one market factor (mean pairwise correlation 0.22–0.47 by year). Its weak group structure is too noisy at monthly horizons to beat a constant-correlation target.
- **"Is 14.60% against 14.67% really nothing?"** The variance ratio's 95% interval is [0.965, 1.021], it is significant in 0 of 10 seeds, and shuffled memberships do the same.
- **"What would you try next?"** Universes with stronger sector structure (multi-asset or international equities), shorter rebalancing horizons, combining fuzzy targets with nonlinear shrinkage, and survivorship-free constituent histories.
