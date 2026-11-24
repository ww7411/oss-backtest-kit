"""Command line and loader options for the overfitting checkup."""

import argparse
import math
import os
import sys

from . import stats
from .loaders import LoaderError


_CSV_OPTS = ("kind", "time_col", "value_col")
_FT_OPTS = ("strategy", "starting_balance")
_FLAG = {"kind": "--kind", "time_col": "--time-col", "value_col": "--value-col",
         "strategy": "--strategy", "starting_balance": "--starting-balance"}
_APPLIES = {"directory": _CSV_OPTS + ("starting_balance",), "a CSV file": _CSV_OPTS,
            "a freqtrade .json / .zip file": _FT_OPTS}


def _source_kind(path):
    if os.path.isdir(path):
        return "directory"
    return "a CSV file" if path.lower().endswith(".csv") else "a freqtrade .json / .zip file"


def _loader_args(source, opts):
    """Keep the loader options that fit this source; one that cannot apply is an error."""
    from .checkup import CheckupError

    given = {k: v for k, v in (opts or {}).items() if v is not None}
    unknown = sorted(set(given) - set(_FLAG))
    if unknown:
        raise CheckupError("unknown load option(s): %s" % ", ".join(unknown))
    if given and not isinstance(source, (str, os.PathLike)):
        raise CheckupError("load options only apply when the source is a file or directory path")
    if not given:
        return {}
    what = _source_kind(os.fspath(source))
    bad = [k for k in given if k not in _APPLIES[what]]
    if bad:
        raise CheckupError("%s does not apply to %s (it takes %s)" % (
            ", ".join(_FLAG[k] for k in bad), what if what != "directory" else "a directory",
            ", ".join(_FLAG[k] for k in _APPLIES[what])))
    return given


# ---- command line
def _seed(text):
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("%r is not a whole number" % text)
    if v < 0:
        raise argparse.ArgumentTypeError("must be 0 or more, got %d" % v)
    return v


def _balance(text):
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("%r is not a number" % text)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError("must be a positive number, got %s" % text)
    return v


def _parser():
    p = argparse.ArgumentParser(prog="python -m oss_backtest_kit.checkup",
                                description="Overfitting checkup for a backtest result.")
    p.add_argument("source", help="result file (.csv, .json, .zip) or a directory of results")
    p.add_argument("--trials", type=int, default=None, metavar="N",
                   help="how many versions were tried in total, saved or not (default: number of results)")
    p.add_argument("--report", metavar="OUT.md", help="also write the Markdown report here")
    p.add_argument("--html", metavar="OUT.html", help="also write a single-file HTML report here")
    p.add_argument("--select", metavar="NAME", help="with several results, describe this one (default: best Sharpe)")
    p.add_argument("--strategy", metavar="NAME", help="freqtrade file with several strategies: read this one")
    p.add_argument("--starting-balance", type=_balance, metavar="X", dest="starting_balance",
                   help="freqtrade file without a starting balance: the amount the account started with")
    p.add_argument("--time-col", metavar="COL", dest="time_col", help="CSV: name of the date / time column")
    p.add_argument("--value-col", metavar="COL", dest="value_col",
                   help="CSV: name of the returns or equity column (needs --kind)")
    p.add_argument("--kind", choices=("returns", "equity"), help="CSV: whether the value column holds returns or equity")
    p.add_argument("--fill-flat", action="store_true", dest="fill_flat",
                   help="equity curve with missing days: repeat the last equity value on them "
                        "(only if the account truly did not change)")
    p.add_argument("--seed", type=_seed, default=0, help="bootstrap seed, a whole number >= 0 (default 0)")
    return p


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _refuse_overwrite(source, outputs):
    """Fail before reading anything if an output path is an input file (same path or hard link)."""
    from .checkup import CheckupError

    inputs = [source]
    if os.path.isdir(source):
        inputs = [e.path for e in os.scandir(source) if e.is_file()]
    for out in outputs:
        if out and os.path.exists(out) and any(os.path.exists(i) and os.path.samefile(i, out) for i in inputs):
            raise CheckupError("the output would overwrite an input file (%s); choose another output path" % out)


def main(argv=None):
    """Command line entry. Exit status 0 when the checkup ran, 2 on input errors."""
    from .checkup import CheckupError, checkup

    args = _parser().parse_args(argv)
    try:
        _refuse_overwrite(args.source, (args.report, args.html))
        opts = {k: getattr(args, k) for k in ("strategy", "starting_balance", "time_col", "value_col", "kind")}
        rep = checkup(args.source, n_trials=args.trials, select=args.select, seed=args.seed, load_options=opts,
                     fill_flat=args.fill_flat)
        md = rep.to_markdown()
        if args.report:
            _write(args.report, md)
        if args.html:
            _write(args.html, rep.to_html())
    except (CheckupError, LoaderError, stats.StatsError, OSError, ValueError, OverflowError) as e:
        sys.stderr.write("error: %s\n" % e)
        return 2
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
