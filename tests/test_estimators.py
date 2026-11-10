import numpy as np
import pytest

from fuzzshrink.estimators import (
    base_intensity,
    block_pair_target,
    combine_angle,
    combine_arithmetic,
    constant_correlation,
    fit_ratio,
    fuzzshrink,
    hard_block,
    ledoit_wolf_cc,
    ledoit_wolf_nonlinear,
    nearest_correlation,
    two_level_target,
)
from fuzzshrink.moments import WindowStats
from fuzzshrink.portfolio import gmv_weights
from helpers import block_returns, random_memberships


def min_eig(A):
    return float(np.linalg.eigvalsh((A + A.T) / 2).min())


@pytest.fixture(scope="module")
def stats():
    X, _ = block_returns(n_blocks=3, per_block=15, T=120, seed=11)
    return WindowStats.from_returns(X)


# ------------------------------------------------------------ draft Propositions 1-2


def test_proposition_1_single_cluster_is_constant_correlation(stats):
    Sigma, info = fuzzshrink(stats, np.ones((stats.N, 1)), gamma=1.0)
    off = ~np.eye(stats.N, dtype=bool)
    rbar = stats.R[off].mean()
    F = np.full((stats.N, stats.N), rbar)
    np.fill_diagonal(F, 1.0)
    d0 = info["delta0"]
    np.testing.assert_allclose(info["delta"], d0)
    expected = ((1 - d0) * stats.R + d0 * F) * np.outer(stats.sd, stats.sd)
    np.testing.assert_allclose(Sigma, expected, atol=1e-14)
    # identical under the draft's arithmetic rule
    Sigma_arith, _ = fuzzshrink(stats, np.ones((stats.N, 1)), gamma=1.0, combine="arithmetic")
    np.testing.assert_allclose(Sigma_arith, Sigma, atol=1e-14)


def test_proposition_2_hard_partition_is_block_shrinkage(stats):
    labels = np.random.default_rng(0).integers(0, 3, size=stats.N)
    U = np.eye(3)[labels]
    Sigma, info = fuzzshrink(stats, U, gamma=2.0)
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(stats.N, dtype=bool)
    rho_w = stats.R[same & off].mean()
    rho_b = stats.R[~same].mean()
    F = np.where(same, rho_w, rho_b)
    np.fill_diagonal(F, 1.0)
    d0 = info["delta0"]
    np.testing.assert_allclose(info["delta"], d0)  # entropy is zero, so gamma has no effect
    expected = ((1 - d0) * stats.R + d0 * F) * np.outer(stats.sd, stats.sd)
    np.testing.assert_allclose(Sigma, expected, atol=1e-14)


def test_named_special_cases_match_fuzzshrink(stats):
    U = random_memberships(stats.N, 3, seed=5)
    S1, _ = constant_correlation(stats)
    S1b, _ = fuzzshrink(stats, np.ones((stats.N, 1)), gamma=0.0)
    np.testing.assert_allclose(S1, S1b)
    S2, _ = hard_block(stats, U)
    S2b, _ = fuzzshrink(stats, np.eye(3)[U.argmax(axis=1)], gamma=0.0)
    np.testing.assert_allclose(S2, S2b)


# ---------------------------------------------------------- Lemma 1 and Proposition 3


@pytest.mark.parametrize("seed", range(20))
def test_lemma_1_soft_target_is_psd(seed):
    rng = np.random.default_rng(seed)
    U = random_memberships(30, 4, seed=seed)
    rho_b, rho_w = np.sort(rng.uniform(0, 1, size=2))
    S = U @ U.T
    F = rho_b + (rho_w - rho_b) * S
    np.fill_diagonal(F, 1.0)
    assert min_eig(F) >= -1e-10


def test_two_level_target_reports_lemma_condition(stats):
    labels = np.repeat(np.arange(3), 15)  # the fixture's true blocks
    U = 0.7 * np.eye(3)[labels] + 0.1  # soft memberships aligned with the blocks
    F, info = two_level_target(stats.R, U)
    assert info["lemma1_holds"]
    assert info["rho_within"] > info["rho_between"] > 0
    assert min_eig(F) >= -1e-10


@pytest.mark.parametrize("seed", range(20))
def test_proposition_3_angle_rule_is_psd_with_unit_diagonal(seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40, 25))
    R = np.corrcoef(X, rowvar=False)
    U = random_memberships(25, 3, seed=seed)
    F, _ = two_level_target(R, U)  # pure-noise data, so Lemma 1 may not apply
    F_psd = nearest_correlation(F) if min_eig(F) < 0 else F
    delta = rng.uniform(0, 1, size=25)
    delta[:3] = [0.0, 1.0, 0.5]
    Rhat = combine_angle(R, F_psd, delta)
    np.testing.assert_allclose(np.diag(Rhat), 1.0)
    assert min_eig(Rhat) >= -1e-10


def test_draft_rule_can_fail_and_angle_rule_cannot():
    R = 0.99 * np.ones((3, 3)) + 0.01 * np.eye(3)  # valid correlation matrix
    F = np.eye(3)  # valid target
    delta = np.array([0.0, 1.0, 1.0])
    Rhat_arith, info = combine_arithmetic(R, F, delta)
    assert info["min_eig_before"] < -0.3 and info["projected"]
    assert min_eig(Rhat_arith) >= -1e-9  # repaired by projection
    assert min_eig(combine_angle(R, F, delta)) >= -1e-12  # valid with no repair


def test_angle_rule_equals_convex_combination_for_equal_intensities(stats):
    U = random_memberships(stats.N, 3, seed=2)
    F, _ = two_level_target(stats.R, U)
    delta = np.full(stats.N, 0.37)
    expected = 0.63 * stats.R + 0.37 * F
    np.testing.assert_allclose(combine_angle(stats.R, F, delta), expected, atol=1e-14)


# ------------------------------------------------------------------ intensities


def test_base_intensity_with_unit_weights_is_schafer_strimmer(stats):
    U = random_memberships(stats.N, 3, seed=3)
    F, _ = two_level_target(stats.R, U)
    off = ~np.eye(stats.N, dtype=bool)
    ss = stats.varR[off].sum() / ((stats.R - F)[off] ** 2).sum()
    assert base_intensity(stats.R, F, stats.varR, np.ones(stats.N)) == pytest.approx(min(ss, 1.0))


def test_entropy_weights_shrink_ambiguous_assets_less_for_positive_gamma(stats):
    U = random_memberships(stats.N, 3, seed=4)
    _, info = fuzzshrink(stats, U, gamma=1.0)
    H, delta = info["entropy"], info["delta"]
    assert np.corrcoef(H, delta)[0, 1] < -0.9
    _, mirror = fuzzshrink(stats, U, gamma=-1.0)
    assert np.corrcoef(H, mirror["delta"])[0, 1] > 0.5


def test_fit_ratio_flags_an_asset_the_target_does_not_describe():
    X, labels = block_returns(n_blocks=3, per_block=15, T=400, seed=6)
    noise = 0.02 * np.random.default_rng(9).standard_normal((400, 1))
    stats = WindowStats.from_returns(np.hstack([X, noise]))
    U = np.vstack([np.eye(3)[labels], np.full((1, 3), 1 / 3)])
    F, _ = two_level_target(stats.R, U)
    q = fit_ratio(stats.R, F, stats.varR)
    assert q[-1] < 0.25 * np.median(q[:-1])


# ----------------------------------------------------------------- estimator sanity


@pytest.mark.parametrize("kw", [
    {"gamma": 1.0},
    {"gamma": -1.0},
    {"gamma": 1.0, "weights": "fit"},
    {"gamma": 1.0, "target": "block_pair"},
    {"gamma": 1.0, "combine": "arithmetic"},
])
def test_fuzzshrink_is_valid_covariance(stats, kw):
    U = random_memberships(stats.N, 3, seed=8)
    Sigma, _ = fuzzshrink(stats, U, **kw)
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-15)
    np.testing.assert_allclose(np.diag(Sigma), np.diag(stats.S), rtol=1e-12)
    assert min_eig(Sigma) > 0


def test_fuzzshrink_is_permutation_equivariant(stats):
    U = random_memberships(stats.N, 3, seed=10)
    perm = np.random.default_rng(1).permutation(stats.N)
    Sigma, _ = fuzzshrink(stats, U)
    Sigma_p, _ = fuzzshrink(WindowStats.from_returns(stats.X[:, perm]), U[perm])
    np.testing.assert_allclose(Sigma_p, Sigma[np.ix_(perm, perm)], atol=1e-14)


def test_block_pair_target_reduces_to_constant_correlation_for_one_cluster(stats):
    F, _ = block_pair_target(stats.R, np.ones((stats.N, 1)))
    F2, _ = two_level_target(stats.R, np.ones((stats.N, 1)))
    np.testing.assert_allclose(F, F2, atol=1e-12)


# -------------------------------------------------------------------- references


def test_ledoit_wolf_cc_matches_the_paper_formulas():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((50, 5)) @ rng.standard_normal((5, 5))
    T, N = X.shape
    Y = X - X.mean(0)
    S = Y.T @ Y / T
    sd = np.sqrt(np.diag(S))
    rbar = ((S / np.outer(sd, sd)).sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, np.diag(S))
    pi = np.zeros((N, N))
    rho = 0.0
    for i in range(N):
        for j in range(N):
            pi[i, j] = ((Y[:, i] * Y[:, j] - S[i, j]) ** 2).mean()
    rho = np.trace(pi)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            th_ii = ((Y[:, i] ** 2 - S[i, i]) * (Y[:, i] * Y[:, j] - S[i, j])).mean()
            th_jj = ((Y[:, j] ** 2 - S[j, j]) * (Y[:, i] * Y[:, j] - S[i, j])).mean()
            rho += rbar / 2 * (np.sqrt(S[j, j] / S[i, i]) * th_ii + np.sqrt(S[i, i] / S[j, j]) * th_jj)
    kappa = (pi.sum() - rho) / ((F - S) ** 2).sum()
    expected_delta = max(0.0, min(1.0, kappa / T))
    _, delta = ledoit_wolf_cc(X)
    assert delta == pytest.approx(expected_delta, rel=1e-10)


def _gmv_excess_variance(Sigma_hat, Sigma_true):
    w = gmv_weights(Sigma_hat)
    w_star = gmv_weights(Sigma_true)
    return w @ Sigma_true @ w / (w_star @ Sigma_true @ w_star) - 1.0


def test_nonlinear_shrinkage_beats_sample_covariance_for_gmv():
    rng = np.random.default_rng(12)
    p, T = 60, 120
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Sigma_true = (Q * np.linspace(0.5, 8.0, p)) @ Q.T
    L = np.linalg.cholesky(Sigma_true)
    loss_sample, loss_nl = [], []
    for _ in range(8):
        X = rng.standard_normal((T, p)) @ L.T
        loss_sample.append(_gmv_excess_variance(np.cov(X, rowvar=False), Sigma_true))
        loss_nl.append(_gmv_excess_variance(ledoit_wolf_nonlinear(X), Sigma_true))
    assert np.mean(loss_nl) < 0.6 * np.mean(loss_sample)


@pytest.mark.parametrize("T", [40, 200])
def test_nonlinear_shrinkage_is_positive_definite(T):
    X = np.random.default_rng(T).standard_normal((T, 50))
    assert min_eig(ledoit_wolf_nonlinear(X)) > 0
