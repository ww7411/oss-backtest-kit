"""Effect chart: equity curve plus PSR / DSR, all computed by the library on synthetic data.

SYNTHETIC DATA, NOT A TRADING RESULT. The prices are a seeded random walk; the chart only
shows how the statistics adjust a "best of N variants" backtest for the number of tries.

Install (matplotlib is an optional extra, only needed to draw the picture):

    pip install "oss-backtest-kit[examples]"        # from PyPI, after release
    # or, from a clone of the repository:
    pip install -e ".[examples]"

Run from the project root (writes examples/equity_psr_dsr.png):

    python3 examples/equity_psr_dsr.py
    python3 examples/equity_psr_dsr.py --social docs/social_preview.png   # also the 1280x640 card

The numbers come from `compute()`, which needs no matplotlib and is covered by the tests.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from ma_crossover import ma_crossover, synthetic_market  # noqa: E402
from oss_backtest_kit import stats  # noqa: E402
from oss_backtest_kit.engine import EngineConfig, run_backtest  # noqa: E402

GRID = [(f, s) for f in (5, 10, 20) for s in (30, 40, 60)]
LABEL = "synthetic data, not a trading result"
BLUE, GREY, RED = "#2a6fb0", "#666666", "#b03a2e"


def compute(seed=0):
    """Run the 9-variant grid, pick the best by Sharpe, and return its statistics."""
    data = synthetic_market(seed=seed)
    cfg = EngineConfig(symbols=data.symbols, max_leverage=1.0, rebalance_threshold=0.02)
    results = [run_backtest(data, ma_crossover(f, s), cfg) for f, s in GRID]
    srs = [stats.sharpe(r.returns) for r in results]
    best = int(np.argmax(srs))
    returns = results[best].returns
    dsr_value, bench, _ = stats.dsr(returns, len(GRID), srs)
    return {
        "variant": GRID[best],
        "n_trials": len(GRID),
        "equity": results[best].equity,
        "sharpe_annualised": stats.sharpe_annualised(returns, results[best].periods_per_year),
        "psr": stats.psr(returns),
        "dsr": dsr_value,
        "benchmark_sharpe_per_bar": bench,
    }


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit('matplotlib is missing. Install the extra:  pip install -e ".[examples]"')
    return plt


def _save(fig, path):
    # Software=None keeps the PNG free of version / path style text chunks.
    fig.savefig(path, dpi=100, metadata={"Software": None})


def plot_chart(res, path):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    ax.plot(res["equity"] / res["equity"][0], color=BLUE, lw=1.6)
    ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel("bar (daily, synthetic)")
    ax.set_ylabel("equity (start = 1.0)")
    ax.set_title("Best of %d moving-average variants: equity curve and overfitting statistics"
                 % res["n_trials"], fontsize=11)
    text = ("variant (fast, slow) = %s\nannualised Sharpe = %.2f\nPSR (vs 0) = %.3f\n"
            "DSR (%d trials) = %.3f" % (res["variant"], res["sharpe_annualised"], res["psr"],
                                       res["n_trials"], res["dsr"]))
    ax.text(0.02, 0.04, text, transform=ax.transAxes, va="bottom", fontsize=10, family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": GREY})
    fig.text(0.5, 0.01, LABEL, ha="center", color=RED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)
    plt.close(fig)


def plot_social(res, path):
    """1280x640 card for the repository social preview (upload is done by hand)."""
    plt = _pyplot()
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.text(0.06, 0.80, "oss-backtest-kit", fontsize=44, weight="bold")
    fig.text(0.06, 0.66, "Is your backtest real, or just the best of many tries?", fontsize=20)
    fig.text(0.06, 0.52, "PSR  |  DSR  |  PBO  |  checked data  |  no look-ahead engine",
             fontsize=16, color=GREY)
    fig.text(0.06, 0.40, "PSR %.2f -> DSR %.2f after %d trials" % (res["psr"], res["dsr"],
                                                                  res["n_trials"]),
             fontsize=16, family="monospace", color=BLUE)
    fig.text(0.06, 0.08, LABEL + ". Research only, not investment advice.", fontsize=13, color=RED)
    ax = fig.add_axes([0.58, 0.14, 0.36, 0.30])
    ax.plot(res["equity"] / res["equity"][0], color=BLUE, lw=1.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _save(fig, path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "equity_psr_dsr.png"))
    ap.add_argument("--social", help="also write the 1280x640 social preview card here")
    args = ap.parse_args(argv)
    res = compute()
    print("best variant %s  Sharpe=%.2f  PSR=%.3f  DSR=%.3f  (%d trials)" % (
        res["variant"], res["sharpe_annualised"], res["psr"], res["dsr"], res["n_trials"]))
    plot_chart(res, args.out)
    if args.social:
        plot_social(res, args.social)


if __name__ == "__main__":
    main()
