import numpy as np
import pandas as pd
import pytest

from fuzzshrink.data import apply_listing_overrides, prices_to_returns, rebalance_dates, window_slices


@pytest.fixture
def prices():
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    rng = np.random.default_rng(0)
    p = pd.DataFrame(100 * np.exp(np.cumsum(0.01 * rng.standard_normal((len(idx), 4)), axis=0)),
                     index=idx, columns=list("ABCD"))
    p.loc[: "2020-03-15", "D"] = np.nan  # D lists in mid-March
    return p


def test_returns_are_simple_and_keep_gaps(prices):
    r = prices_to_returns(prices)
    np.testing.assert_allclose(r["A"].iloc[0], prices["A"].iloc[1] / prices["A"].iloc[0] - 1)
    assert r.loc[: "2020-03-15", "D"].isna().all()


def test_rebalance_dates_are_first_trading_day_of_each_month(prices):
    dates = rebalance_dates(prices.index, "2020-02-01", "2020-06-30")
    assert [d.strftime("%Y-%m-%d") for d in dates] == [
        "2020-02-03", "2020-03-02", "2020-04-01", "2020-05-01", "2020-06-01"]


def test_window_slices_respect_time_order_and_universe(prices):
    r = prices_to_returns(prices)
    dates = rebalance_dates(r.index, "2020-03-01", "2020-06-30")
    slices = list(window_slices(r, dates, T=20))
    for (date, est, hold), nxt in zip(slices, list(dates[1:]) + [None]):
        assert est.index.max() < date <= hold.index.min()
        assert len(est) == 20
        if nxt is not None:
            assert hold.index.max() < nxt
        assert list(est.columns) == list(hold.columns)
        assert est.notna().all().all() and hold.notna().all().all()
    assert "D" not in slices[0][1].columns  # no history yet in March
    assert "D" in slices[-1][1].columns  # enough history by June


def test_listing_overrides_drop_history_before_the_real_listing(prices):
    ov = pd.DataFrame({"ticker": ["B", "ZZZ"], "valid_from": ["2020-04-01", "2020-01-01"]})
    out = apply_listing_overrides(prices, ov)
    assert out.loc[:"2020-03-31", "B"].isna().all()
    assert out.loc["2020-04-01":, "B"].notna().all()
    pd.testing.assert_frame_equal(out.drop(columns="B"), prices.drop(columns="B"))


def test_window_slices_need_enough_history(prices):
    r = prices_to_returns(prices)
    with pytest.raises(ValueError):
        list(window_slices(r, rebalance_dates(r.index, "2020-01-01", "2020-02-28"), T=60))
