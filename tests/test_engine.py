"""Deterministic synthetic tests for oss_backtest_kit.engine."""

import csv
import io
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from oss_backtest_kit.engine import (CostModel, EngineConfig, EngineError, MarketData,  # noqa: E402
                                     load_market_data, run_backtest)

DAY = 86400000
T0 = 1_600_000_000_000 - (1_600_000_000_000 % DAY)
FREE = {"base": CostModel(0.0, 0.0), "stressed": CostModel(0.0, 0.0)}


def market(open_, close, fund_rate=None, symbols=("AAA",)):
    open_ = np.asarray(open_, float).reshape(len(open_), -1)
    close = np.asarray(close, float).reshape(len(close), -1)
    n = len(open_)
    fund_px = np.zeros_like(open_) if fund_rate is None else fund_rate * open_
    times = T0 + DAY * np.arange(n)
    return MarketData(list(symbols), times, open_, np.maximum(open_, close),
                      np.minimum(open_, close), close, fund_px)


def flat(n, price=100.0, **kw):
    return market([price] * n, [price] * n, **kw)


def cfg(symbols=("AAA",), **kw):
    kw.setdefault("costs", FREE)
    kw.setdefault("funding", False)
    return EngineConfig(symbols=list(symbols), **kw)


def const(w):
    return lambda t, h: dict(w)


def close_to(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_zero_weights_flat_equity():
    res = run_backtest(market([100, 101, 90, 120], [101, 90, 120, 80]), lambda t, h: {}, cfg(initial_equity=5.0))
    assert (res.equity == 5.0).all() and (res.returns == 0).all()
    assert res.fees == 0 and res.turnover.sum() == 0


def test_constant_long_known_path():
    # weights decided at close of bar 0 fill at the open of bar 1
    open_ = [100, 100, 110, 121]
    close = [100, 110, 121, 133.1]
    res = run_backtest(market(open_, close), const({"AAA": 1.0}), cfg(initial_equity=2.0))
    assert res.equity[0] == 2.0  # no position during bar 0
    assert close_to(res.equity[-1], 2.0 * 133.1 / 100.0)
    assert close_to(res.pnl_by_symbol["AAA"], res.equity[-1] - 2.0)
    short = run_backtest(market(open_, close), const({"AAA": -0.5}), cfg(initial_equity=2.0, max_leverage=1.0))
    assert short.equity[-1] < 2.0


def test_fill_at_next_open_not_prior_close():
    res = run_backtest(market([100, 110], [100, 121]), const({"AAA": 1.0}), cfg())
    assert close_to(res.returns[1], 121 / 110 - 1)  # not 121/100 - 1


def test_costs_charged_exactly_on_turnover():
    fee, slip, w, e0, o = 0.001, 0.0004, 0.5, 10.0, 100.0
    c = cfg(initial_equity=e0, rebalance_threshold=0.2,
            costs={"base": CostModel(fee, slip), "stressed": CostModel(0.01, 0.01)})
    res = run_backtest(flat(5, o), const({"AAA": w}), c)
    dq = w * e0 / o
    px = o * (1 + slip)
    exp_fee = dq * px * fee
    exp_slip = dq * o * slip
    assert close_to(res.fees, exp_fee) and close_to(res.slippage_cost, exp_slip)
    assert close_to(res.equity[-1], e0 - exp_fee - exp_slip)
    assert close_to(res.turnover.sum(), w) and close_to(res.turnover[1], w)
    assert close_to(res.pnl_by_symbol["AAA"], res.equity[-1] - e0)
    stressed = run_backtest(flat(5, o), const({"AAA": w}), c, cost_level="stressed")
    assert stressed.equity[-1] < res.equity[-1]
    # selling pays the same proportional costs: flip to flat later
    flip = lambda t, h: {"AAA": w} if t < 2 else {}
    out = run_backtest(flat(6, o), flip, c)
    assert out.fees > exp_fee  # the closing trade was costed too
    assert abs(out.turnover.sum() - 2 * w) < 1e-2


def test_funding_sign_long_and_short():
    rate = 0.001
    for w, sign in ((0.5, -1), (-0.5, 1)):
        c = cfg(initial_equity=10.0, funding=True, rebalance_threshold=0.2)
        res = run_backtest(flat(5, 100.0, fund_rate=rate), const({"AAA": w}), c)
        q = abs(w) * 10.0 / 100.0
        expected = sign * q * rate * 100.0 * 4  # funding on bars 1..4
        assert close_to(res.funding_net, expected), (w, res.funding_net, expected)
        assert close_to(res.equity[-1], 10.0 + expected)
        assert close_to(res.funding_by_symbol["AAA"], expected)
    off = run_backtest(flat(5, 100.0, fund_rate=rate), const({"AAA": 0.5}), cfg(initial_equity=10.0))
    assert off.funding_net == 0.0 and off.equity[-1] == 10.0


def test_leverage_cap_enforced():
    two = market(np.full((4, 2), 100.0), np.full((4, 2), 100.0), symbols=("AAA", "BBB"))
    ok = run_backtest(two, const({"AAA": 0.6, "BBB": -0.4}), cfg(("AAA", "BBB"), max_leverage=1.0))
    assert len(ok.equity) == 4
    for bad in ({"AAA": 0.6, "BBB": 0.5}, {"AAA": -0.6, "BBB": 0.6001}, {"ZZZ": 0.1}, {"AAA": float("nan")}):
        try:
            run_backtest(two, const(bad), cfg(("AAA", "BBB"), max_leverage=1.0))
        except EngineError:
            continue
        raise AssertionError("not rejected: %r" % (bad,))
    run_backtest(two, const({"AAA": 1.0, "BBB": 1.0}), cfg(("AAA", "BBB"), max_leverage=2.0))


def test_no_lookahead_history_is_truncated_and_readonly():
    seen = []

    def spy(t, h):
        seen.append(t)
        assert len(h.close) == t + 1 and len(h.times) == t + 1 and h.t == t
        try:
            h.close[t + 1]
        except IndexError:
            pass
        else:
            raise AssertionError("future bar visible")
        try:
            h.close[0, 0] = 1.0
        except ValueError:
            pass
        else:
            raise AssertionError("history is writable")
        return {}

    run_backtest(market(np.arange(100, 110.0), np.arange(101, 111.0)), spy, cfg())
    assert seen == list(range(9))  # last bar's close is never used for a decision


def test_no_lookahead_truncation_invariance():
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, 80))
    open_ = np.concatenate([[100.0], close[:-1]])
    full = market(open_, close)

    def strat(t, h):  # reacts to the latest close only through history
        return {"AAA": 1.0 if h.close[t, 0] > h.close[max(0, t - 5):t + 1, 0].mean() else -1.0}

    c = cfg(costs={"base": CostModel(0.0004, 0.0002), "stressed": CostModel(0.001, 0.0005)})
    a = run_backtest(full, strat, c)
    b = run_backtest(full.truncated(50), strat, c)
    assert np.array_equal(a.equity[:50], b.equity)


def test_lookahead_oracle_gains_nothing_from_history_interface():
    # Next-bar direction alternates; a rule using only the past cannot beat zero here
    # by sign-copying the last bar when it alternates, it loses every bar.
    n = 40
    close = np.array([100.0 * (1.01 if i % 2 else 1.0) for i in range(n)])
    open_ = np.concatenate([[100.0], close[:-1]])
    data = market(open_, close)
    follow = lambda t, h: {"AAA": 1.0 if h.close[t, 0] >= h.open[t, 0] else -1.0}
    res = run_backtest(data, follow, cfg())
    assert res.equity[-1] < 1.0


def test_rebalance_threshold_skips_small_changes():
    wob = lambda t, h: {"AAA": 0.5 + (0.01 if t % 2 else 0.0)}
    costs = {"base": CostModel(0.001, 0.0), "stressed": CostModel(0.001, 0.0)}
    loose = run_backtest(flat(20), wob, cfg(costs=costs, rebalance_threshold=0.05))
    tight = run_backtest(flat(20), wob, cfg(costs=costs, rebalance_threshold=0.0))
    assert tight.fees > 1.2 * loose.fees and loose.turnover.sum() < 0.51
    assert tight.turnover.sum() > loose.turnover.sum() + 0.1


def test_pnl_by_symbol_sums_to_equity_change_and_summary():
    rng = np.random.default_rng(4)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.02, (60, 2)), axis=0)
    open_ = np.vstack([[100.0, 100.0], close[:-1]])
    data = market(open_, close, fund_rate=0.0001, symbols=("AAA", "BBB"))
    c = EngineConfig(symbols=["AAA", "BBB"], rebalance_threshold=0.05)  # default costs, funding on
    res = run_backtest(data, const({"AAA": 0.5, "BBB": -0.5}), c)
    assert close_to(sum(res.pnl_by_symbol.values()), res.equity[-1] - 1.0, 1e-9)
    s = res.summary()
    assert s["bars"] == 60 and math.isfinite(s["sharpe_annualised"]) and 0 <= s["psr_vs_zero"] <= 1
    assert s["fees"] > 0


def test_start_end_window():
    data = flat(30)
    from datetime import datetime, timezone
    day = lambda k: datetime.fromtimestamp((T0 + k * DAY) / 1000, timezone.utc).strftime("%Y-%m-%d")
    res = run_backtest(data, const({"AAA": 0.5}), cfg(start=day(10), end=day(20)))
    assert len(res.equity) == 10 and res.times[0] == T0 + 10 * DAY
    assert res.turnover[0] > 0  # warm-up bar 9 supplies the first decision
    try:
        run_backtest(data, const({}), cfg(start=day(40)))
    except EngineError:
        pass
    else:
        raise AssertionError("empty range accepted")


def test_ruin_stops_trading():
    data = market([100, 100, 50, 50, 50], [100, 1.0, 1.0, 1.0, 1.0])
    res = run_backtest(data, const({"AAA": 3.0}), cfg(max_leverage=3.0))
    assert res.equity[-1] == 0.0 and (res.equity[2:] == 0).all()


def test_ruin_pnl_reconciles_with_equity():
    # equity 1, weight 3 long, price 100 -> 1: raw loss 2.97 exceeds equity.
    data = market([100, 100, 1, 1], [100, 1.0, 1.0, 1.0])
    res = run_backtest(data, const({"AAA": 3.0}), cfg(max_leverage=3.0))
    assert res.equity[-1] == 0.0
    assert close_to(res.pnl_by_symbol["AAA"], -2.97)
    assert close_to(res.ruin_shortfall, 1.97)
    assert close_to(sum(res.pnl_by_symbol.values()) + res.ruin_shortfall, res.equity[-1] - 1.0)
    assert close_to(res.summary()["ruin_shortfall"], 1.97)
    assert run_backtest(data, const({"AAA": 0.5}), cfg()).ruin_shortfall == 0.0


def _zip(rows):
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("x.csv", buf.getvalue())
    return out.getvalue()


def test_loader_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        man = []

        def add(kind, sym, interval, name, rows):
            (d / name).write_bytes(_zip(rows))
            man.append({"kind": kind, "symbol": sym, "interval": interval, "local_path": name})

        kl = lambda i, p: [T0 + i * DAY, p, p + 1, p - 1, p + 0.5, 1, T0 + (i + 1) * DAY - 1, 1, 1, 1, 1, 0]
        add("klines", "AAA", "1d", "a.zip", [kl(0, 100), kl(1, 101), kl(2, 102)])
        add("klines", "BBB", "1d", "b.zip", [kl(1, 50), kl(2, 51)])  # BBB misses day 0
        add("fundingRate", "AAA", "", "f.zip", [[T0 + 1 * DAY + 5, 8, "0.0002"]])
        with open(d / "MANIFEST.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, ["kind", "symbol", "interval", "local_path"])
            w.writeheader()
            w.writerows(man)
        md = load_market_data(d, ["AAA", "BBB"], "1d")
        assert md.open.shape == (3, 2) and np.isnan(md.open[0, 1]) and md.open[1, 1] == 50
        assert close_to(md.fund_px[1, 0], 0.0002 * 101) and md.fund_px[:, 1].sum() == 0
        assert md.close[2, 0] == 102.5
        res = run_backtest(md, const({"AAA": 0.5, "BBB": 0.5}), cfg(("AAA", "BBB"), funding=True))
        assert len(res.equity) == 3
        try:
            load_market_data(d, ["CCC"], "1d")
        except EngineError:
            pass
        else:
            raise AssertionError("missing symbol accepted")


def _kl(i, p):
    return [T0 + i * DAY, p, p + 1, p - 1, p + 0.5, 1, T0 + (i + 1) * DAY - 1, 1, 1, 1, 1, 0]


def _load(files, symbols=("AAA",)):
    """files: list of (kind, symbol, interval, rows); returns MarketData."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        man = []
        for k, (kind, sym, interval, rows) in enumerate(files):
            name = "f%d.zip" % k
            (d / name).write_bytes(_zip(rows))
            man.append({"kind": kind, "symbol": sym, "interval": interval, "local_path": name})
        with open(d / "MANIFEST.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, ["kind", "symbol", "interval", "local_path"])
            w.writeheader()
            w.writerows(man)
        return load_market_data(d, list(symbols), "1d")


def test_funding_after_last_bar_is_not_backfilled():
    # Bars cover day 0 and day 1. An event at day 2 00:00 happens after the last
    # bar has ended and must not change any loaded bar (truncation invariance).
    bars = ("klines", "AAA", "1d", [_kl(0, 100), _kl(1, 101)])
    inside = [T0 + DAY + 8 * 3600000, 8, "0.001"]
    after = [T0 + 2 * DAY, 8, "0.001"]
    base = _load([bars, ("fundingRate", "AAA", "", [inside])])
    more = _load([bars, ("fundingRate", "AAA", "", [inside, after])])
    assert np.array_equal(base.fund_px, more.fund_px)
    assert close_to(more.fund_px[1, 0], 0.001 * 101) and more.fund_px[0, 0] == 0.0
    res_a = run_backtest(base, const({"AAA": 1.0}), cfg(funding=True))
    res_b = run_backtest(more, const({"AAA": 1.0}), cfg(funding=True))
    assert np.array_equal(res_a.equity, res_b.equity)


def test_funding_in_gap_goes_to_its_own_bar_at_last_price():
    # Day 1 is missing for every symbol: the loader inserts a NaN bar so the
    # day-1 event is not moved onto day 0, and it is priced at the latest close.
    bars = ("klines", "AAA", "1d", [_kl(0, 100), _kl(2, 300)])
    md = _load([bars, ("fundingRate", "AAA", "", [[T0 + DAY + 5, 8, "0.001"]])])
    assert len(md.times) == 3 and np.isnan(md.open[1, 0])
    assert md.fund_px[0, 0] == 0.0 and md.fund_px[2, 0] == 0.0
    assert close_to(md.fund_px[1, 0], 0.001 * 100.5)  # day-0 close, not a max open


def test_funding_listed_twice_counts_once_and_conflicts_raise():
    bars = ("klines", "AAA", "1d", [_kl(0, 100), _kl(1, 101)])
    ev = [T0 + DAY + 5, 8, "0.001"]
    once = _load([bars, ("fundingRate", "AAA", "", [ev])])
    twice = _load([bars, ("fundingRate", "AAA", "", [ev]), ("fundingRate", "AAA", "", [ev])])
    assert np.array_equal(once.fund_px, twice.fund_px)
    for bad in ([("fundingRate", "AAA", "", [ev]), ("fundingRate", "AAA", "", [[ev[0], 8, "0.002"]])],
                [("klines", "AAA", "1d", [_kl(1, 999)])]):
        try:
            _load([bars] + bad)
        except EngineError:
            continue
        raise AssertionError("conflict accepted: %r" % (bad,))


def test_bar_spacing_must_match_config_bar():
    daily = flat(10)
    run_backtest(daily, const({"AAA": 0.5}), cfg(bar="1d"))
    gap = MarketData(["AAA"], np.delete(daily.times, 4), np.full((9, 1), 100.0),
                     np.full((9, 1), 100.0), np.full((9, 1), 100.0), np.full((9, 1), 100.0))
    for data, bar in ((daily, "4h"), (daily, "1M"), (gap, "1d")):
        try:
            run_backtest(data, const({"AAA": 0.5}), cfg(bar=bar))
        except EngineError:
            continue
        raise AssertionError("accepted bar=%s with mismatched times" % bar)


def test_loader_says_rest_tail_files_are_not_loaded():
    import warnings
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.zip").write_bytes(_zip([_kl(0, 100), _kl(1, 101)]))
        with open(d / "MANIFEST.csv", "w", newline="") as fh:
            fh.write("kind,symbol,interval,local_path\r\nklines,AAA,1d,a.zip\r\n")
        (d / "rest" / "AAA").mkdir(parents=True)
        (d / "rest" / "AAA" / "klines_1d_0_1.csv").write_text("open_time\n%d\n" % (T0 + 2 * DAY))
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter("always")
            md = load_market_data(d, ["AAA"], "1d")
        assert len(md.times) == 2  # the REST bar is not merged
        assert any("not loaded" in str(w.message) for w in got)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, repr(exc))
    print("%d passed, %d failed" % (len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
