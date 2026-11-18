import numpy as np

from fuzzshrink.portfolio import gmv_long_only, gmv_weights, project_simplex


def test_gmv_weights_for_diagonal_covariance_are_inverse_variance():
    var = np.array([1.0, 2.0, 4.0])
    w = gmv_weights(np.diag(var))
    np.testing.assert_allclose(w, (1 / var) / (1 / var).sum())


def test_gmv_is_the_minimum_over_random_budgets():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 6))
    Sigma = A.T @ A / 30
    w = gmv_weights(Sigma)
    best = w @ Sigma @ w
    for _ in range(200):
        v = rng.standard_normal(6)
        v /= v.sum()
        assert v @ Sigma @ v >= best - 1e-12


def test_project_simplex():
    np.testing.assert_allclose(project_simplex(np.array([0.2, 0.3, 0.5])), [0.2, 0.3, 0.5])
    np.testing.assert_allclose(project_simplex(np.array([2.0, 0.0, 0.0])), [1.0, 0.0, 0.0])
    p = project_simplex(np.random.default_rng(1).standard_normal(10) * 3)
    assert p.min() >= 0 and abs(p.sum() - 1) < 1e-12


def test_long_only_gmv_satisfies_kkt_conditions():
    R = 0.3 * np.ones((5, 5)) + 0.7 * np.eye(5)
    R[0, 1] = R[1, 0] = 0.95  # a close pair with unequal volatility invites a short
    sd = np.array([1.0, 2.0, 1.5, 1.2, 2.5])
    Sigma = R * np.outer(sd, sd)
    assert gmv_weights(Sigma).min() < 0  # the constraint binds in this example
    w = gmv_long_only(Sigma)
    assert w.min() >= 0 and abs(w.sum() - 1) < 1e-10
    grad = Sigma @ w
    active = w > 1e-8
    level = grad[active].mean()
    np.testing.assert_allclose(grad[active], level, rtol=1e-5)
    assert (grad[~active] >= level - 1e-7).all()
