import numpy as np
import pytest

from fuzzshrink.metrics import (
    cer_difference_bps,
    deflated_sharpe_ratio,
    diebold_mariano,
    hac_variance,
    holm,
    performance,
    permutation_pvalue,
    sharpe_difference_test,
    stationary_bootstrap_indices,
    variance_ratio_bootstrap,
)


def test_performance_on_a_known_series():
    r = np.array([0.01, -0.01] * 126)
    p = performance(r, r, turnover=np.array([5.0, 0.2, 0.4]))
    assert p["sd"] == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))
    assert p["mean"] == pytest.approx(0.0, abs=1e-12)
    assert p["turnover"] == pytest.approx(0.3)  # the set-up trade is excluded
    assert -0.03 < p["max_drawdown"] < -0.01  # 0.99 * 1.01 < 1 compounds into a small loss


def test_hac_variance_equals_sample_variance_without_lags():
    x = np.random.default_rng(0).standard_normal(500)
    v = hac_variance((x - x.mean())[:, None], lag=0)[0, 0]
    assert v == pytest.approx(x.var())


def test_diebold_mariano_detects_a_clear_difference_and_not_a_null_one():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(2000) * 0.8
    b = rng.standard_normal(2000)
    assert diebold_mariano(a**2, b**2)["p_less"] < 0.001
    c = rng.standard_normal(2000)
    assert diebold_mariano(b**2, c**2)["p_less"] > 0.01


def test_stationary_bootstrap_indices_are_valid_and_blocky():
    idx = stationary_bootstrap_indices(100, 50, block=10.0, rng=np.random.default_rng(2))
    assert idx.shape == (50, 100) and idx.min() >= 0 and idx.max() < 100
    steps = (np.diff(idx, axis=1) % 100) == 1
    assert 0.85 < steps.mean() < 0.95  # continuation probability is 1 - 1/10


def test_variance_ratio_bootstrap_covers_the_truth_about_95_percent_of_the_time():
    rng = np.random.default_rng(3)
    covered, small_p = 0, 0
    for k in range(20):
        a, b = 0.9 * rng.standard_normal(1500), rng.standard_normal(1500)
        out = variance_ratio_bootstrap(a, b, n_boot=300, seed=k)
        covered += out["ci_low"] < 0.81 < out["ci_high"]
        small_p += out["p_boot"] < 0.05
    assert covered >= 16
    assert small_p >= 15  # a 19% variance reduction is detectable at n = 1500


def test_sharpe_difference_test_is_calibrated_under_the_null():
    rng = np.random.default_rng(5)
    rejections = 0
    for _ in range(200):
        a = 0.0005 + 0.01 * rng.standard_normal(500)
        b = 0.0005 + 0.01 * rng.standard_normal(500)
        rejections += sharpe_difference_test(a, b)["p_two_sided"] < 0.05
    assert rejections / 200 < 0.10


def test_deflated_sharpe_ratio_penalises_many_trials():
    rng = np.random.default_rng(6)
    r = 0.0008 + 0.01 * rng.standard_normal(2000)
    few = deflated_sharpe_ratio(r, rng.normal(0, 0.005, 3))["dsr"]
    many = deflated_sharpe_ratio(r, rng.normal(0, 0.005, 500))["dsr"]
    assert many < few


def test_holm_adjustment():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_cer_difference_rewards_lower_variance_at_equal_mean():
    rng = np.random.default_rng(7)
    z = rng.standard_normal(1000) * 0.01
    assert cer_difference_bps(0.5 * z, z, risk_aversion=5) > 0


def test_permutation_pvalue():
    assert permutation_pvalue(0.1, np.array([0.2, 0.3, 0.4])) == pytest.approx(0.25)
    assert permutation_pvalue(0.5, np.array([0.2, 0.3, 0.4])) == pytest.approx(1.0)
