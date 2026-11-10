import numpy as np
import pytest

from fuzzshrink.moments import WindowStats
from fuzzshrink.registry import CONFIGS, Context, portfolio_weights
from fuzzshrink.simulate import simulate_market


def test_true_covariance_is_a_valid_correlation_structure():
    m = simulate_market(pi_bridge=0.3, pi_outlier=0.15, T=50, seed=1)
    sd = np.sqrt(np.diag(m.Sigma))
    P = m.Sigma / np.outer(sd, sd)
    assert np.linalg.eigvalsh(P).min() > 0
    np.testing.assert_allclose(m.U_true.sum(axis=1), 1.0)
    assert set(m.types) == {"crisp", "bridge", "outlier"}
    assert (m.types == "outlier").sum() % 2 == 0


def test_sample_covariance_converges_to_truth():
    m = simulate_market(N=20, n_groups=3, T=200_000, df=np.inf, seed=2)
    S = np.cov(m.X, rowvar=False)
    assert np.abs(S - m.Sigma).max() < 0.05 * np.abs(m.Sigma).max()


def test_outliers_have_lower_correlation_with_the_market():
    m = simulate_market(pi_outlier=0.2, T=10, seed=3)
    sd = np.sqrt(np.diag(m.Sigma))
    P = m.Sigma / np.outer(sd, sd)
    avg = (P.sum(axis=1) - 1) / (len(P) - 1)
    assert avg[m.types == "outlier"].mean() < avg[m.types == "crisp"].mean()


@pytest.mark.parametrize("name", list(CONFIGS))
def test_every_configuration_runs_and_gives_budget_weights(name):
    m = simulate_market(N=40, n_groups=3, pi_bridge=0.2, pi_outlier=0.1, T=120, seed=4)
    stats = WindowStats.from_returns(m.X)
    ctx = Context(stats, m.U_true, m.U_true, np.random.default_rng(0))
    Sigma, _ = CONFIGS[name].fn(ctx)
    w = portfolio_weights(Sigma, stats.N)
    assert abs(w.sum() - 1.0) < 1e-10 and np.all(np.isfinite(w))
