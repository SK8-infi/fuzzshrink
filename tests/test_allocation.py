import numpy as np
import pytest
from scipy.optimize import minimize

from fuzzshrink.allocation import max_sharpe_exact, min_variance_capped, nsga2_mean_variance, repair_weights


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((300, 8))
    Sigma = np.cov(A @ rng.standard_normal((8, 8)) * 0.01, rowvar=False)
    mu = rng.uniform(-0.0005, 0.0015, 8)
    return mu, Sigma


def sharpe(w, mu, Sigma):
    return w @ mu / np.sqrt(w @ Sigma @ w)


def test_repair_respects_constraints():
    w = repair_weights(np.array([5.0, 1.0, 0.0, -1.0]), cap=0.5)
    assert abs(w.sum() - 1) < 1e-12 and w.min() >= 0 and w.max() <= 0.5 + 1e-12


def test_max_sharpe_exact_beats_random_feasible_portfolios(problem):
    mu, Sigma = problem
    w = max_sharpe_exact(mu, Sigma, cap=0.4)
    assert abs(w.sum() - 1) < 1e-9 and w.min() >= -1e-12 and w.max() <= 0.4 + 1e-9
    rng = np.random.default_rng(1)
    best_random = max(sharpe(repair_weights(rng.dirichlet(np.ones(8)), 0.4), mu, Sigma) for _ in range(2000))
    assert sharpe(w, mu, Sigma) >= best_random - 1e-9


def test_max_sharpe_exact_matches_direct_optimisation(problem):
    mu, Sigma = problem
    w = max_sharpe_exact(mu, Sigma, cap=0.4)
    res = minimize(lambda x: -sharpe(x, mu, Sigma), np.full(8, 1 / 8), bounds=[(0, 0.4)] * 8,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], method="SLSQP")
    assert sharpe(w, mu, Sigma) >= -res.fun - 1e-6


def test_negative_forecasts_fall_back_to_min_variance(problem):
    _, Sigma = problem
    w = max_sharpe_exact(-np.ones(8), Sigma, cap=0.4)
    np.testing.assert_allclose(w, min_variance_capped(Sigma, 0.4), atol=1e-6)


def test_nsga2_is_feasible_and_reasonable(problem):
    mu, Sigma = problem
    w = nsga2_mean_variance(mu, Sigma, cap=0.4, pop=60, generations=60, seed=2)
    assert abs(w.sum() - 1) < 1e-9 and w.min() >= 0 and w.max() <= 0.4 + 1e-9
    exact = max_sharpe_exact(mu, Sigma, cap=0.4)
    assert sharpe(w, mu, Sigma) > 0.8 * sharpe(exact, mu, Sigma)
