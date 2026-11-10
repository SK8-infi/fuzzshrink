import numpy as np
import pytest

from fuzzshrink.moments import WindowStats, corr_variance, cov_to_corr, sample_covariance


def test_sample_covariance_matches_numpy():
    X = np.random.default_rng(0).standard_normal((40, 5))
    np.testing.assert_allclose(sample_covariance(X), np.cov(X, rowvar=False), atol=1e-14)


def test_cov_to_corr_round_trip():
    X = np.random.default_rng(1).standard_normal((60, 4))
    S = sample_covariance(X)
    R, sd = cov_to_corr(S)
    np.testing.assert_allclose(np.diag(R), 1.0)
    np.testing.assert_allclose(R * np.outer(sd, sd), S, atol=1e-14)


def test_corr_variance_matches_brute_force():
    X = np.random.default_rng(2).standard_normal((25, 4))
    T, N = X.shape
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    expected = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            w = Z[:, i] * Z[:, j]
            expected[i, j] = T / (T - 1) ** 3 * ((w - w.mean()) ** 2).sum()
    np.testing.assert_allclose(corr_variance(X), expected, atol=1e-12)


def test_window_stats_rejects_missing_values():
    X = np.random.default_rng(3).standard_normal((30, 3))
    X[4, 1] = np.nan
    with pytest.raises(ValueError):
        WindowStats.from_returns(X)
