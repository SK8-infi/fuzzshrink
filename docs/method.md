# FuzzShrink: method specification

This file says exactly what the code computes. "Eq. n" refers to the paper draft (v0). New material is numbered N1, N2, and so on. Where code and this file disagree, that is a bug.

## 1. Setting and notation

- $X \in \mathbb{R}^{T\times N}$: daily simple returns of $N$ assets over an estimation window of $T$ days.
- $S$: sample covariance with divisor $T-1$.
- $\hat\sigma_i = \sqrt{S_{ii}}$, $D = \operatorname{diag}(\hat\sigma)$, and $R = D^{-1} S D^{-1}$ (sample correlation, entries $r_{ij}$).
- $J$: the $N\times N$ matrix of ones. $\circ$: the elementwise (Hadamard) product.

Every FuzzShrink variant keeps the sample variances and works on correlations. The final estimate is $\hat\Sigma = D\hat R D$. The external references R1 and R2 (section 7) work on covariances directly.

## 2. Stage 1: fuzzy memberships

### 2.1 Features

Each asset is described by a feature vector $z_i$, built from the estimation window only.

**Co-movement features (default).** Eigen-decompose $R = \sum_\ell \lambda_\ell v_\ell v_\ell^\top$ with $\lambda_1 \ge \lambda_2 \ge \dots$, then:

- Drop the market mode $\ell = 1$, on which almost every stock loads, so it carries no grouping information.
- Keep the next $k$ components, where $k$ is the number of eigenvalues above the Marchenko–Pastur noise edge $\lambda_+ = (1+\sqrt{N/T})^2$, clipped to $[2, 10]$.
- Set $z_i = (\sqrt{\lambda_2}\,v_{2,i}, \dots, \sqrt{\lambda_{k+1}}\,v_{k+1,i})$.

Rows are *not* normalised. An asset that loads weakly on every group factor sits near the origin and so is equally far from every group. That is exactly the asset that membership entropy flags.

**Risk-return features (draft-faithful, ablation X4).** These are computed per asset over the window, then z-scored across assets:

- annualised mean
- annualised volatility
- skewness
- excess kurtosis
- beta against the equal-weighted universe
- maximum drawdown

### 2.2 Fuzzy c-means

Minimise $\sum_i\sum_c u_{ic}^m \lVert z_i - v_c\rVert^2$ subject to $u_{ic}\ge0$ and $\sum_c u_{ic}=1$ (Eq. 1), with fuzzifier $m=2$ (Bezdek, 1981). Alternate two updates:

- prototypes $v_c = \sum_i u_{ic}^m z_i / \sum_i u_{ic}^m$;
- memberships $u_{ic} = 1/\sum_k (d_{ic}^2/d_{ik}^2)^{1/(m-1)}$, where $d^2_{ic} = \lVert z_i - v_c\rVert^2$.

An asset sitting exactly on a prototype gets that cluster with membership 1. Initial prototypes come from k-means++ seeding with an explicit seed. Iterations stop when $\max|\Delta U| < 10^{-6}$ or after 300 iterations. Of `n_init` = 5 restarts, the one with the lowest objective is kept.

### 2.3 Number of clusters

$C \in \{2,\dots,8\}$ is chosen by the minimum Xie–Beni index $\mathrm{XB} = \sum_{i,c} u_{ic}^m d_{ic}^2 \big/ \big(N \min_{c\ne k}\lVert v_c - v_k\rVert^2\big)$. This follows the draft, which says $C$ is "selected by a fuzzy validity index".

### 2.4 Membership summaries

- **Co-membership** (Eq. 2): $s_{ij} = \sum_c u_{ic}u_{jc}$. This is the probability that $i$ and $j$ land in the same cluster when each is assigned at random from its own membership vector.
- **Normalised entropy** (Eq. 5): $H_i = -\frac{1}{\log C}\sum_c u_{ic}\log u_{ic}$, with $0\log 0 = 0$ and $H_i = 0$ when $C = 1$.

Entropy measures how *evenly split* an asset's membership is. It does not measure whether the block structure *describes* the asset, and the two can differ:

- A bridge stock with membership $(0.5, 0.5)$ has high entropy, yet a soft target describes it well: its correlations really are a blend of two groups.
- An idiosyncratic stock can also have high entropy, and no block target describes it.

Section 4.1 therefore adds a direct measure of describability (N1) as an alternative to entropy.

## 3. Stage 2: the shrinkage target

### 3.1 Two-level soft block target (Eq. 3–4)

The within-block and between-block mean correlations are weighted by co-membership:

$$\bar\rho_w = \frac{\sum_{i\ne j}s_{ij}r_{ij}}{\sum_{i\ne j}s_{ij}},\qquad \bar\rho_b = \frac{\sum_{i\ne j}(1-s_{ij})r_{ij}}{\sum_{i\ne j}(1-s_{ij})}.$$

The target is $F_{ij} = s_{ij}\bar\rho_w + (1-s_{ij})\bar\rho_b$ for $i\ne j$, and $F_{ii}=1$. When $\sum_{i\neq j}(1-s_{ij}) = 0$ (every pair fully co-member, e.g. $C=1$), $\bar\rho_b$ is never used.

**Lemma 1 (N2).** If $0\le\bar\rho_b\le\bar\rho_w\le1$, then $F$ is positive semi-definite.

*Proof.* Write $F = \bar\rho_b J + (\bar\rho_w-\bar\rho_b)UU^\top + \operatorname{diag}\big(1-\bar\rho_b-(\bar\rho_w-\bar\rho_b)s_{ii}\big)$.

- $J = \mathbf{1}\mathbf{1}^\top$ and $UU^\top$ are PSD, and their coefficients are non-negative.
- Since $s_{ii}\le1$, every diagonal entry of the last term is at least $1-\bar\rho_w\ge0$.

A sum of PSD matrices is PSD. ∎

In equity data within-group correlation exceeds between-group correlation and both are positive, so the condition holds in practice. The code records every window where it fails.

### 3.2 Block-pair target (extension X2, N3)

This variant gives each pair of clusters its own mean correlation:

- $\bar P_{ck} = \sum_{i\ne j}u_{ic}u_{jk}r_{ij} \big/ \sum_{i\ne j}u_{ic}u_{jk}$, computed as $\bar P = \big(U^\top(R-I)U\big) \oslash \big(U^\top(J-I)U\big)$.
- $F_{ij} = u_i^\top\bar P u_j$ for $i\neq j$, and $F_{ii}=1$.
- If $\bar P$ has negative eigenvalues they are clipped to zero, which makes $F$ PSD. Clipping is recorded.
- With $C=1$ this reduces to constant correlation. With one-hot $U$ it is the hard block target with a separate mean for every block pair.

## 4. Stage 3: per-asset intensity

### 4.1 Weights

Each asset gets a weight $w_i$, which sets its intensity $\delta_i = \min(1, \delta_0 w_i)$:

- **entropy weights** (draft, Eq. 6): $w_i = (1-H_i)^{\gamma}$;
- **fit weights** (N1, N4): $w_i = q_i^{\gamma}$, where

$$q_i = \frac{\sum_{j\ne i}\widehat{\operatorname{Var}}(r_{ij})}{\sum_{j\ne i}(r_{ij}-F_{ij})^2}.$$

$q_i$ compares the sampling noise in asset $i$'s correlations with how far they sit from the target. It is the Schäfer–Strimmer intensity computed on row $i$ alone.

- If the target describes asset $i$ up to noise, $q_i$ is large and the asset is shrunk strongly.
- If the target misses it systematically, $q_i$ is small and the asset keeps more of its sample correlations.

This is the draft's own intuition, "shrink an asset toward the block structure only to the degree that the block structure actually describes it", measured directly instead of through entropy. The weights still depend on the membership matrix through $F$.

The sign of $\gamma$ sets the direction:

- $\gamma = 0$: every asset gets the same intensity (ordinary shrinkage, the draft's null).
- $\gamma > 0$: ambiguous assets are shrunk **less** (the draft's hypothesis).
- $\gamma < 0$ (the "mirror", A6): ambiguous assets are shrunk **more**.

For $\gamma<0$, $1-H_i$ is floored at $10^{-6}$.

### 4.2 Base intensity $\delta_0$ (N5)

The draft took $\delta_0$ from a formula derived for a different target. We instead minimise the expected squared error of the off-diagonal correlations of the linear rule $\hat r_{ij} = r_{ij} + \delta_0 a_{ij}(F_{ij}-r_{ij})$, with $a_{ij} = \sqrt{w_iw_j}$, treating the target as fixed:

$$\delta_0^\star = \frac{\sum_{i\ne j} a_{ij}\,\widehat{\operatorname{Var}}(r_{ij})}{\sum_{i\ne j} a_{ij}^2\,(r_{ij}-F_{ij})^2}, \qquad \text{clipped to } [0,\ 1/\min_i w_i].$$

$\widehat{\operatorname{Var}}(r_{ij})$ follows Schäfer and Strimmer (2005). With standardised returns $\tilde x_{ti}$, $w_{tij} = \tilde x_{ti}\tilde x_{tj}$ and $\bar w_{ij} = T^{-1}\sum_t w_{tij}$:

$$\widehat{\operatorname{Var}}(r_{ij}) = \frac{T}{(T-1)^3}\sum_t (w_{tij}-\bar w_{ij})^2.$$

Properties of this rule:

- With $a_{ij}\equiv1$ it is exactly the Schäfer–Strimmer intensity.
- Every configuration (A1–A6, X1–X4) uses this same rule against its own target and weights, so differences between them come only from structure, not from a tuned constant.
- The upper clip makes $\delta_i = 1$ possible only for the asset(s) with the smallest weight.

*Derivation.* Write $e_{ij} = r_{ij}-\rho_{ij}$ and $g_{ij} = F_{ij}-r_{ij}$. The loss is $L(\delta_0) = \sum_{i\ne j}\mathbb{E}\,(e_{ij}+\delta_0 a_{ij}g_{ij})^2$. Setting $L'=0$ gives $\delta_0 = -\sum a_{ij}\mathbb{E}[e_{ij}g_{ij}]\big/\sum a_{ij}^2\mathbb{E}[g_{ij}^2]$. For fixed $F$ and unbiased $r$, $\mathbb{E}[e_{ij}g_{ij}] = -\operatorname{Var}(r_{ij})$.

## 5. Stage 4: combining sample and target

### 5.1 Draft rule (Eq. 7–8), kept as ablation X3

$\hat R = (J-dd^\top)\circ R + (dd^\top)\circ F$ with $d_i=\sqrt{\delta_i}$.

- The target term is PSD (Schur product theorem).
- The sample term $(J - dd^\top)\circ R$ is not guaranteed PSD, so $\hat R$ can fail to be a valid correlation matrix (draft §5.6).
- The draft therefore projects $\hat R$ onto the nearest correlation matrix whenever its smallest eigenvalue is below $10^{-10}$. We use Higham's (2002) alternating projections. Projection activations are recorded (draft §5.7).

### 5.2 Angle rule (default, N6)

Write $\delta_i = \sin^2\theta_i$, so $s_i = \sqrt{\delta_i}$ and $c_i = \sqrt{1-\delta_i}$. Then

$$\hat R = (cc^\top)\circ R + (ss^\top)\circ F = \operatorname{diag}(c)\,R\,\operatorname{diag}(c) + \operatorname{diag}(s)\,F\,\operatorname{diag}(s).$$

**Proposition 3 (N7).** If $R\succeq0$ and $F\succeq0$, then $\hat R\succeq0$ and $\hat R_{ii} = 1$. If moreover $R\succ0$ and every $\delta_i<1$, then $\hat R\succ0$.

*Proof.* Each term is a congruence transform of a PSD matrix, so each is PSD, and so is their sum. On the diagonal, $c_i^2+s_i^2=1$ and $R_{ii}=F_{ii}=1$. Strictness follows because $\operatorname{diag}(c)R\operatorname{diag}(c)\succ0$ when every $c_i>0$. ∎

How this differs from the draft rule:

- Pair $(i,j)$ is pulled toward $F$ with weight $s_is_j = \sqrt{\delta_i\delta_j}$, the draft's geometric mean (Eq. 7).
- The weight on $R$ is $c_ic_j$ instead of $1-s_is_j$.
- The two weights sum to $\cos(\theta_i-\theta_j)\le1$. The small deficit, about $(\theta_i-\theta_j)^2/2$, shrinks pairs with *unequal* intensities slightly toward zero correlation.
- When all intensities are equal the rule is exactly the ordinary convex combination. So the special cases below are unchanged, and **no projection is ever needed** once $F\succeq0$ (Lemma 1).

## 6. Special cases (draft Propositions 1–2)

- **One cluster.** If $C=1$ then $s_{ij}\equiv1$ and $H_i\equiv0$, so $\delta_i\equiv\delta_0$. $F$ is the constant-correlation target, and both combination rules reduce to $(1-\delta_0)R+\delta_0F$.
- **Hard clusters.** If $U$ is one-hot then $s_{ij}\in\{0,1\}$ and $H_i\equiv0$. $F$ is the two-level hard block target and again $\delta_i\equiv\delta_0$.

`tests/test_estimators.py` checks both reductions numerically, along with Lemma 1 and Proposition 3.

## 7. Reference estimators

- **R1, Ledoit–Wolf (2004) constant correlation**: the published estimator and intensity formula, following the authors' `covCor` code with divisor $T$.
- **R2, Ledoit–Wolf (2020) analytical nonlinear shrinkage**: the published kernel formula (Epanechnikov kernel, bandwidth $h = n^{-1/3}$), with effective sample size $n=T-1$ after demeaning. Both $p\le n$ and $p>n$ are handled.
- **R3, equal weight**: the $1/N$ portfolio, which needs no covariance estimate.

## 8. Portfolios

- **Global minimum variance (primary test bed):** $w = \hat\Sigma^{-1}\mathbf1 / \mathbf1^\top\hat\Sigma^{-1}\mathbf1$, computed with a linear solve.
- **Long-only GMV (robustness):** minimise $w^\top\hat\Sigma w$ subject to $\mathbf1^\top w=1$ and $w\ge0$. This uses accelerated projected gradient (FISTA), with step $1/\lambda_{\max}(\hat\Sigma)$ and exact Euclidean projection onto the simplex (Duchi et al., 2008), until the change in $w$ is below $10^{-10}$.

## 9. Changes from the draft

| Item | Draft | Now | Why |
|---|---|---|---|
| Combination rule | Arithmetic, then projection | Angle rule (default); arithmetic kept as X3 | PSD by construction (Prop. 3); no projection to explain away |
| Target validity | Not addressed | Lemma 1 | States when the target term itself is PSD |
| Base intensity | "Standard analytic formula or cross-validation" | Closed form for each variant's own target and weights | The standard formula assumes a different target. One rule makes the comparisons fair |
| Ambiguity signal | Entropy only | Entropy (A4) and direct fit to the target (X1) | Entropy treats bridges and outliers alike; only outliers are poorly described by a soft target |
| Direction | Assumed (ambiguous assets shrunk less) | Tested: γ = +1 against γ = −1 (A6) | Let the data choose the direction |
| Clustering features | Risk-return statistics | Co-movement loadings (default); risk-return as X4 | A correlation target needs co-moving groups |
| Strong baseline | Constant-correlation shrinkage | Adds Ledoit–Wolf (2020) nonlinear shrinkage | The current state of the art for GMV |
| Full LSTM + GA pipeline | Secondary test bed | Out of scope; GMV and long-only GMV instead | The base paper's code is not public; GMV isolates the covariance effect |

## References (method)

- Duchi, J., Shalev-Shwartz, S., Singer, Y., Chandra, T. (2008). Efficient projections onto the ℓ1-ball for learning in high dimensions. *ICML*.
- Higham, N. J. (2002). Computing the nearest correlation matrix: a problem from finance. *IMA Journal of Numerical Analysis*, 22(3), 329–343.
- Xie, X. L., Beni, G. (1991). A validity measure for fuzzy clustering. *IEEE TPAMI*, 13(8), 841–847.
