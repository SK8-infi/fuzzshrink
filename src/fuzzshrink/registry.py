"""The pre-registered configurations (docs/protocol.md, section 4) as callables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .estimators import (
    constant_correlation,
    fuzzshrink,
    hard_block,
    ledoit_wolf_cc,
    ledoit_wolf_nonlinear,
    sample_estimator,
)
from .moments import WindowStats
from .portfolio import gmv_long_only, gmv_weights

__all__ = ["Context", "Config", "CONFIGS", "CORE", "EXTENSIONS", "REFERENCES", "U_BASED",
           "fuzz_variant", "portfolio_weights"]


@dataclass
class Context:
    """What one configuration may use in one estimation window."""

    stats: WindowStats
    U: np.ndarray | None = None  # memberships from co-movement features
    U_riskret: np.ndarray | None = None  # memberships from risk-return features (X4)
    rng: np.random.Generator | None = None  # for the permutation control (A5)


@dataclass
class Config:
    label: str
    fn: Callable[[Context], tuple[np.ndarray | None, dict]]
    uses_memberships: bool = False


def fuzz_variant(gamma: float = 1.0, *, weights: str = "entropy", target: str = "two_level",
                 combine: str = "angle", features: str = "comove", permute: bool = False):
    """A FuzzShrink configuration as a callable on a Context."""

    def fn(ctx: Context):
        U = ctx.U_riskret if features == "riskret" else ctx.U
        if U is None:
            raise ValueError(f"this configuration needs {features} memberships")
        if permute:
            U = U[ctx.rng.permutation(U.shape[0])]
        return fuzzshrink(ctx.stats, U, gamma=gamma, weights=weights, target=target, combine=combine)

    return fn


def _r1(ctx: Context):
    Sigma, delta = ledoit_wolf_cc(ctx.stats.X)
    return Sigma, {"delta0": delta}


CONFIGS: dict[str, Config] = {
    "A0": Config("Sample covariance", lambda c: (sample_estimator(c.stats), {})),
    "A1": Config("Constant correlation (C = 1)", lambda c: constant_correlation(c.stats)),
    "A2": Config("Hard block", lambda c: hard_block(c.stats, c.U), True),
    "A3": Config("FuzzShrink, gamma = 0", fuzz_variant(0.0), True),
    "A4": Config("FuzzShrink, gamma = +1 (entropy)", fuzz_variant(1.0), True),
    "A5": Config("A4 with permuted memberships", fuzz_variant(1.0, permute=True), True),
    "A6": Config("FuzzShrink, gamma = -1 (mirror)", fuzz_variant(-1.0), True),
    "X1": Config("FuzzShrink, fit weights", fuzz_variant(1.0, weights="fit"), True),
    "X2": Config("FuzzShrink, block-pair target", fuzz_variant(1.0, target="block_pair"), True),
    "X3": Config("A4 with the draft's arithmetic rule", fuzz_variant(1.0, combine="arithmetic"), True),
    "X4": Config("A4 with risk-return features", fuzz_variant(1.0, features="riskret"), True),
    "R1": Config("Ledoit-Wolf 2004", _r1),
    "R2": Config("Ledoit-Wolf 2020 nonlinear", lambda c: (ledoit_wolf_nonlinear(c.stats.X), {})),
    "R3": Config("Equal weight (1/N)", lambda c: (None, {})),
}

CORE = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
EXTENSIONS = ["X1", "X2", "X3", "X4"]
REFERENCES = ["R1", "R2", "R3"]
U_BASED = ["A2", "A3", "A4", "A6", "X1", "X2", "X3", "X4"]


def portfolio_weights(Sigma: np.ndarray | None, n_assets: int, long_only: bool = False) -> np.ndarray:
    """GMV weights for a covariance estimate; equal weights when Sigma is None (R3)."""
    if Sigma is None:
        return np.full(n_assets, 1.0 / n_assets)
    return gmv_long_only(Sigma) if long_only else gmv_weights(Sigma)
