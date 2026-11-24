"""Checkup demo on synthetic data: one overfitted group of results, one robust group.

SYNTHETIC DATA, NOT A TRADING RESULT. Nothing is downloaded and no network is used.

- "overfitted": 40 variants whose daily returns are pure noise (no edge at all). The best of
  them looks good on its own, and the checkup shows that it should not be trusted.
- "robust": 6 variants over two years, one of which has a small real edge added.

Run from the project root (prints both Markdown reports):

    python3 examples/checkup_demo.py
    python3 examples/checkup_demo.py --out-dir ./checkup_out   # also writes .md and .html files

The same checks are available for your own results from the command line:

    python -m oss_backtest_kit.checkup <result file or folder> --trials N
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from oss_backtest_kit.checkup import checkup  # noqa: E402


def overfitted_group(seed=0, days=365, variants=40):
    """Pure noise, 1% daily volatility: any 'edge' in the best variant is luck."""
    return np.random.default_rng(seed).normal(0.0, 0.01, (days, variants))


def robust_group(seed=0, days=730, variants=6, edge=0.0025):
    """Noise plus a small real daily edge in one variant."""
    m = np.random.default_rng(seed).normal(0.0, 0.01, (days, variants))
    m[:, 2] += edge
    return m


def run(seed=0):
    """Return {group name: Report} for the two groups (deterministic for a seed)."""
    # every variant tried is saved here, so the number of variants is the real total of trials
    return {"overfitted": checkup(overfitted_group(seed), n_trials=40),
            "robust": checkup(robust_group(seed), n_trials=6)}


def main(argv=None):
    p = argparse.ArgumentParser(description="Checkup demo on synthetic data (offline).")
    p.add_argument("--out-dir", help="also write <group>.md and <group>.html here")
    p.add_argument("--seed", type=int, default=0, help="seed of the synthetic data (default 0)")
    args = p.parse_args(argv)
    for name, rep in run(args.seed).items():
        print("=" * 72)
        print("Group: %s (overall: %s)" % (name, rep.overall.upper()))
        print("=" * 72)
        print(rep.to_markdown())
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            for ext, text in (("md", rep.to_markdown()), ("html", rep.to_html())):
                with open(os.path.join(args.out_dir, "%s.%s" % (name, ext)), "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
