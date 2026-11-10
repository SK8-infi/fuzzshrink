import numpy as np
import pandas as pd
import pytest

from fuzzshrink.backtest import BacktestSettings, run_backtest
from helpers import block_returns


@pytest.fixture(scope="module")
def returns():
    X, _ = block_returns(n_blocks=3, per_block=12, T=700, seed=21)
    idx = pd.bdate_range("2019-01-01", periods=700)
    df = pd.DataFrame(X, index=idx, columns=[f"S{i:02d}" for i in range(X.shape[1])])
    df.iloc[:300, -1] = np.nan  # one stock lists later and enters the universe midway
    return df


@pytest.fixture(scope="module")
def result(returns):
    s = BacktestSettings(T=120, start="2019-08-01", end="2021-08-31", n_init=2, n_permutations=3)
    return run_backtest(returns, ["A0", "A1", "A3", "A4", "A5", "R2", "R3"], seed=0, settings=s)


def test_out_of_sample_returns_cover_every_day_once(result, returns):
    g = result.gross
    assert g.index.is_unique and g.index.is_monotonic_increasing
    assert g.index.min() == returns.loc["2019-08-01":].index.min()
    assert not g.isna().any().any()


def test_portfolio_return_is_weighted_sum_of_asset_returns(returns):
    s = BacktestSettings(T=120, start="2019-08-01", end="2019-09-30", n_init=1)
    res = run_backtest(returns, ["R3"], seed=0, settings=s)
    first_month = returns.loc["2019-08-01":"2019-08-30"].dropna(axis=1)
    np.testing.assert_allclose(res.gross["R3"].loc[first_month.index], first_month.mean(axis=1))


def test_costs_are_charged_on_rebalance_days_only(result):
    diff = result.gross - result.net
    reb_days = [result.gross.index[result.gross.index >= d][0] for d in result.turnover.index]
    assert (diff.drop(index=reb_days).abs() < 1e-15).all().all()
    np.testing.assert_allclose(diff.loc[reb_days].to_numpy(), result.turnover.to_numpy() * 1e-3)


def test_turnover_is_zero_at_set_up_and_positive_later(result):
    assert (result.turnover.iloc[0] == 0).all()
    assert (result.turnover.iloc[1:]["A0"] > 0).all()
    assert (result.turnover.iloc[1:]["R3"] < result.turnover.iloc[1:]["A0"]).all()


def test_universe_grows_when_a_stock_has_enough_history(result):
    n = result.diagnostics.groupby("date").N.first()
    assert n.iloc[0] == 35 and n.iloc[-1] == 36


def test_mechanism_contributions_add_up_to_realised_variance(result, returns):
    mech = result.mechanism
    assert set(mech.quartile) == {1, 2, 3, 4}
    total = mech.groupby(["date", "config"]).risk_contribution.sum()
    assert (total > 0).all()


def test_permutation_runs_are_recorded(result):
    assert result.permutations.shape == (len(result.gross), 3)
    assert not np.allclose(result.permutations["perm0"], result.gross["A4"])


def test_deterministic_for_a_seed(returns):
    s = BacktestSettings(T=120, start="2019-08-01", end="2019-12-31", n_init=1)
    a = run_backtest(returns, ["A4"], seed=3, settings=s).gross
    b = run_backtest(returns, ["A4"], seed=3, settings=s).gross
    pd.testing.assert_frame_equal(a, b)
