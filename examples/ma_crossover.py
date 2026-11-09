"""Moving-average crossover on synthetic data: a textbook example, not a trading recommendation.

Run from the project root:  python3 examples/ma_crossover.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from oss_backtest_kit import stats  # noqa: E402
from oss_backtest_kit.engine import EngineConfig, MarketData, run_backtest  # noqa: E402

DAY = 86400000


def synthetic_market(n_bars=1000, symbols=("AAA", "BBB"), seed=0):
    """Geometric random walks with a little drift, plus a funding rate every bar."""
    rng = np.random.default_rng(seed)
    m = len(symbols)
    rets = rng.normal(0.0003, 0.02, (n_bars, m))
    close = 100.0 * np.cumprod(1.0 + rets, axis=0)
    open_ = np.vstack([np.full((1, m), 100.0), close[:-1]])
    high, low = np.maximum(open_, close) * 1.005, np.minimum(open_, close) * 0.995
    times = 1_600_000_000_000 - (1_600_000_000_000 % DAY) + DAY * np.arange(n_bars)
    fund_px = 0.0001 * open_  # constant small positive funding rate
    return MarketData(list(symbols), times, open_, high, low, close, fund_px)


def ma_crossover(fast=10, slow=40):
    """Long when the fast average is above the slow one, otherwise flat; equal split."""
    def weights(t, hist):
        if t + 1 < slow:
            return {}
        long_syms = [s for i, s in enumerate(hist.symbols)
                     if hist.close[t - fast + 1:t + 1, i].mean() > hist.close[t - slow + 1:t + 1, i].mean()]
        return {s: 1.0 / len(hist.symbols) for s in long_syms}
    return weights


def main():
    data = synthetic_market()
    cfg = EngineConfig(symbols=data.symbols, max_leverage=1.0, rebalance_threshold=0.02)
    for level in ("base", "stressed"):
        res = run_backtest(data, ma_crossover(), cfg, cost_level=level)
        s = res.summary()
        print("%-9s return=%+.3f  sharpe=%.2f  maxdd=%.3f  psr=%.3f  fees=%.4f  funding=%+.4f"
              % (level, s["total_return"], s["sharpe_annualised"], s["max_drawdown"],
                 s["psr_vs_zero"], s["fees"], s["funding_net"]))
    # Deflate against the number of variants tried (here: a small grid of windows).
    grid = [(f, sl) for f in (5, 10, 20) for sl in (30, 40, 60)]
    runs = [run_backtest(data, ma_crossover(f, sl), cfg).returns for f, sl in grid]
    srs = [stats.sharpe(r) for r in runs]
    best = int(np.argmax(srs))
    value, bench, _ = stats.dsr(runs[best], len(grid), srs)
    print("best of %d variants %s: PSR=%.3f  DSR=%.3f (benchmark SR/bar %.4f)"
          % (len(grid), grid[best], stats.psr(runs[best]), value, bench))


if __name__ == "__main__":
    main()
