# Post-hoc analysis: the base-intensity rule (not pre-registered)

**Status: exploratory.** This analysis was designed after seeing the pre-registered results.
It cannot change any verdict in `README.md`; it asks whether those verdicts are an artefact
of the pre-registered base-intensity rule.

**Why we ran it.** On the NASDAQ-100 the pre-registered rule (Schäfer–Strimmer type, which treats
the target as fixed) shrinks harder than Ledoit and Wolf's (2004) intensity for the same
constant-correlation target: mean δ₀ 0.29 against 0.19. With an identical target, R1 beats A1,
so the intensity choice matters.

**What we did.** In each window we computed κ = δ_LW / δ_SS for the constant-correlation case and
multiplied every FuzzShrink variant's base intensity by κ. With one cluster this reproduces R1
exactly, which serves as a check. Everything else is unchanged: T = 252, unconstrained GMV,
10 seeds, 20 permutation runs per seed.

| Configuration | Ann. SD (%) | Var. ratio vs R1 | DM p (lower than R1) |
|---|---|---|---|
| A1 (pre-registered intensity) | 14.67 | 1.046 | 0.985 |
| R1 Ledoit–Wolf 2004 | 14.35 | 1.000 |  |
| cA1 calibrated constant correlation | 14.35 | 1.000 | 0.819 |
| cA3 calibrated soft, γ = 0 | 14.36 | 1.002 | 0.844 |
| cA4 calibrated soft, γ = +1 | 14.49 | 1.020 | 0.978 |
| cX2 calibrated block-pair, γ = +1 | 14.37 | 1.003 | 0.653 |
| R2 Ledoit–Wolf 2020 nonlinear | 14.13 | 0.969 | 0.052 |

Permutation test with the calibrated intensity: calibrated A4 SD 14.49% against
a permuted median of 14.41% (5th percentile 14.28%,
n = 200); p = 0.886.

Check: calibrated constant correlation (cA1) and R1 differ by at most
0.0000 percentage points of annualised SD.
