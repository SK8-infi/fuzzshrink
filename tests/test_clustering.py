import numpy as np

from fuzzshrink.clustering import (
    co_membership,
    comovement_features,
    fit_memberships,
    fuzzy_cmeans,
    membership_entropy,
    riskreturn_features,
    xie_beni,
)
from fuzzshrink.moments import WindowStats
from helpers import block_returns


def blobs(seed=0, n_per=30, centers=((0, 0), (6, 0), (0, 6)), spread=0.5):
    rng = np.random.default_rng(seed)
    Z = np.vstack([rng.normal(c, spread, size=(n_per, 2)) for c in centers])
    return Z, np.repeat(np.arange(len(centers)), n_per)


def purity(U, labels):
    hard = U.argmax(axis=1)
    return sum(np.bincount(hard[labels == k]).max() for k in np.unique(labels)) / len(labels)


def test_fcm_memberships_are_valid_and_recover_blobs():
    Z, labels = blobs()
    part = fuzzy_cmeans(Z, 3, seed=1)
    np.testing.assert_allclose(part.U.sum(axis=1), 1.0)
    assert (part.U >= 0).all()
    assert purity(part.U, labels) == 1.0
    assert len(set(part.U.argmax(axis=1))) == 3


def test_fcm_is_deterministic_for_a_seed():
    Z, _ = blobs()
    a = fuzzy_cmeans(Z, 3, seed=7)
    b = fuzzy_cmeans(Z, 3, seed=7)
    np.testing.assert_array_equal(a.U, b.U)


def test_xie_beni_prefers_the_true_number_of_clusters():
    Z, _ = blobs()
    xb = {C: xie_beni(fuzzy_cmeans(Z, C, seed=2)) for C in range(2, 7)}
    assert min(xb, key=xb.get) == 3


def test_entropy_bounds():
    np.testing.assert_allclose(membership_entropy(np.eye(4)), 0.0)
    np.testing.assert_allclose(membership_entropy(np.full((3, 4), 0.25)), 1.0)
    np.testing.assert_allclose(membership_entropy(np.ones((5, 1))), 0.0)


def test_co_membership_is_symmetric_with_unit_bound():
    U = np.random.default_rng(0).dirichlet(np.ones(3), size=10)
    S = co_membership(U)
    np.testing.assert_allclose(S, S.T)
    assert S.max() <= 1.0 + 1e-12 and S.min() >= 0.0


def test_comovement_features_shape_and_market_mode_removed():
    X, _ = block_returns(T=400)
    stats = WindowStats.from_returns(X)
    Z = comovement_features(stats.R, stats.T)
    assert Z.shape[0] == stats.N and 2 <= Z.shape[1] <= 10
    # loadings on non-market modes sum to roughly zero across assets
    assert np.abs(Z.mean(axis=0)).max() < 0.1 * np.abs(Z).mean()


def test_fit_memberships_recovers_blocks_from_returns():
    X, labels = block_returns(T=500, seed=3)
    part = fit_memberships(WindowStats.from_returns(X), seed=0)
    assert part.n_clusters == 3
    assert purity(part.U, labels) >= 0.95
    assert set(part.xb_by_c) == set(range(2, 9))


def test_riskreturn_features_are_standardised():
    X, _ = block_returns(T=300)
    F = riskreturn_features(X)
    assert F.shape == (X.shape[1], 6)
    np.testing.assert_allclose(F.mean(axis=0), 0.0, atol=1e-10)
