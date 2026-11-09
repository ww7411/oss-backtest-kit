"""Tests for oss_backtest_kit.stats. Run with pytest or `python3 tests/test_stats.py`."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from oss_backtest_kit import stats  # noqa: E402


def _noise(seed, n=500, scale=0.01):
    return np.random.default_rng(seed).normal(0.0, scale, n)


def test_sharpe_basic():
    assert stats.sharpe([0.01]) == 0.0
    assert stats.sharpe([0.01, 0.01, 0.01]) == 0.0  # zero variance
    r = [0.01, -0.01, 0.02, 0.0]
    assert math.isclose(stats.sharpe(r), np.mean(r) / np.std(r, ddof=1))
    assert math.isclose(stats.sharpe_annualised(r), stats.sharpe(r) * math.sqrt(365.25))


def test_max_drawdown_known():
    assert math.isclose(stats.max_drawdown([0.1, -0.5, 0.2]), 0.5)
    assert stats.max_drawdown([0.01, 0.01]) == 0.0
    assert stats.max_drawdown([]) == 0.0


def test_psr_zero_sharpe_is_half():
    r = _noise(0)
    r = r - r.mean()  # exactly zero Sharpe
    assert abs(stats.psr(r) - 0.5) < 1e-9


def test_psr_limits_and_monotonic():
    r = _noise(1) + 0.0004
    assert stats.psr(r, 0.0) > stats.psr(r, 0.05) > stats.psr(r, 0.2)
    assert stats.psr(r + 0.02) > 0.999
    assert stats.psr(r - 0.02) < 0.001
    longer = np.tile(r, 4)
    assert stats.psr(longer) > stats.psr(r) > 0.5  # same Sharpe, more data
    assert math.isnan(stats.psr([0.25, 0.25, 0.25]))
    assert math.isnan(stats.psr([0.1, 0.2]))


def test_dsr_decreases_with_trials():
    r = _noise(2) + 0.001
    trial_srs = np.random.default_rng(3).normal(0.0, 0.05, 50)
    vals = [stats.dsr(r, n, trial_srs)[0] for n in (1, 2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals
    assert math.isclose(vals[0], stats.psr(r, 0.0))
    _, bench, var = stats.dsr(r, 100, trial_srs)
    assert bench > 0 and var > 0


def test_dsr_errors():
    for bad in ([], [1.0, float("nan")]):
        try:
            stats.dsr(bad, 3, [])
        except stats.StatsError:
            continue
        raise AssertionError("expected StatsError")
    try:
        stats.dsr([0.1, 0.2, 0.3], 0, [])
    except stats.StatsError:
        pass
    else:
        raise AssertionError("expected StatsError")


def test_expected_max_sharpe_grows():
    assert stats.expected_max_sharpe(1, 1.0) == 0.0
    a, b = stats.expected_max_sharpe(10, 1.0), stats.expected_max_sharpe(1000, 1.0)
    assert 0 < a < b
    assert math.isclose(stats.expected_max_sharpe(10, 4.0), 2 * a)


def test_n_effective():
    rng = np.random.default_rng(4)
    indep = rng.normal(size=(2000, 10))
    assert stats.n_effective(indep) > 8.5
    same = np.repeat(rng.normal(size=(500, 1)), 10, axis=1)
    assert math.isclose(stats.n_effective(same), 1.0, abs_tol=1e-6)
    assert stats.n_effective(rng.normal(size=(50, 1))) == 1.0
    with_const = np.column_stack([rng.normal(size=(200, 2)), np.zeros(200)])
    for bad in (with_const, np.array([[0.1, np.nan], [0.2, 0.3], [0.0, 0.1]]),
                rng.normal(size=(2, 3)), rng.normal(size=10)):
        try:
            stats.n_effective(bad)
        except stats.StatsError:
            continue
        raise AssertionError("degenerate input accepted")


def test_wilson_known_values():
    lo, hi = stats.wilson(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3
    lo, hi = stats.wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3
    lo, hi = stats.wilson(10, 10)
    assert abs(hi - 1.0) < 1e-9 and abs(lo - 0.7225) < 1e-3
    narrow, wide = stats.wilson(500, 1000), stats.wilson(50, 100)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_pbo_noise_near_half():
    vals = [stats.pbo_cscv(np.random.default_rng(s).normal(size=(160, 12)), 8)
            for s in range(12)]
    assert abs(np.mean(vals) - 0.5) < 0.1, vals
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_pbo_real_edge_is_low():
    rng = np.random.default_rng(5)
    m = rng.normal(0, 0.01, size=(160, 12))
    m[:, 3] += 0.01  # one configuration with a persistent edge
    assert stats.pbo_cscv(m, 8) < 0.1


def test_pbo_edge_cases():
    assert stats.pbo_cscv(np.random.default_rng(6).normal(size=(100, 1))) is None
    for kw, shape in (({"n_blocks": 7}, (100, 3)), ({"n_blocks": 8}, (10, 3))):
        try:
            stats.pbo_cscv(np.random.default_rng(7).normal(size=shape), **kw)
        except stats.StatsError:
            continue
        raise AssertionError("expected StatsError")


def test_bootstrap_reproducible_and_seed_sensitive():
    r = _noise(8, 200)
    a = stats.block_bootstrap(r, n_resamples=200, block=5, seed=42)
    b = stats.block_bootstrap(r, n_resamples=200, block=5, seed=42)
    c = stats.block_bootstrap(r, n_resamples=200, block=5, seed=43)
    assert a == b
    assert a["mdd_median"] != c["mdd_median"]
    assert a["p_loss_exceeded"] is None


def test_bootstrap_respects_block_length():
    # A ramp makes blocks identifiable: inside a block, consecutive values step
    # by +1 (mod T). With block=1 steps are essentially random.
    t = 60
    ramp = np.arange(t, dtype=float)
    rng = np.random.default_rng(9)
    block = 6
    starts = rng.integers(0, t, math.ceil(t / block))
    idx = ((starts[:, None] + np.arange(block)[None, :]) % t).ravel()[:t]
    steps = np.diff(ramp[idx]) % t
    inside = [steps[i] for i in range(len(steps)) if (i + 1) % block != 0]
    assert all(s == 1 for s in inside)
    # Public function: long blocks preserve autocorrelation better than block=1.
    base = np.random.default_rng(10).normal(0, 0.01, 400)
    trending = np.convolve(base, np.ones(20) / 20, "same") + 0.0005  # smooth, autocorrelated
    d1 = stats.block_bootstrap(trending, n_resamples=300, block=1, seed=0)["mdd_median"]
    d40 = stats.block_bootstrap(trending, n_resamples=300, block=40, seed=0)["mdd_median"]
    assert d40 > d1


def test_bootstrap_loss_threshold():
    r = _noise(11, 200)
    none_hit = stats.block_bootstrap(r, n_resamples=100, block=5, seed=0, loss_threshold=0.5)
    assert none_hit["p_loss_exceeded"] == 0.0
    all_hit = stats.block_bootstrap(r, n_resamples=100, block=5, seed=0, loss_threshold=1e-9)
    assert all_hit["p_loss_exceeded"] > 0.99
    lo, hi = all_hit["p_loss_exceeded_wilson95"]
    assert lo <= all_hit["p_loss_exceeded"] <= hi


def test_bootstrap_errors():
    for kw in ({"returns": [0.01] * 5, "block": 5}, {"returns": [0.01, float("nan")] * 10}):
        try:
            stats.block_bootstrap(**kw)
        except stats.StatsError:
            continue
        raise AssertionError("expected StatsError")


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
