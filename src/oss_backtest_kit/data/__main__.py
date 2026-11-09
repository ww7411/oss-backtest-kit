"""Command line: python -m oss_backtest_kit.data {archive,rest,check} ..."""

import argparse
import sys

from . import archive, check, rest_tail
from ._http import DataError


def _csv(text):
    return [x for x in text.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m oss_backtest_kit.data")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("archive", help="download archive files with checksum verification")
    a.add_argument("--kinds", required=True, help="comma list of: %s" % ",".join(archive.KINDS))
    a.add_argument("--symbols", required=True, help="comma list, e.g. BTCUSDT")
    a.add_argument("--interval", help="bar interval for kline kinds, e.g. 1h")
    a.add_argument("--from", dest="start", required=True, help="first month YYYY-MM")
    a.add_argument("--to", dest="end", required=True, help="last month YYYY-MM")
    a.add_argument("--out", required=True, help="output directory")

    r = sub.add_parser("rest", help="fetch recent bars from the public REST API")
    r.add_argument("--kinds", required=True)
    r.add_argument("--symbols", required=True)
    r.add_argument("--interval", required=True)
    r.add_argument("--from", dest="start", required=True, help="UTC day YYYY-MM-DD")
    r.add_argument("--to", dest="end", help="UTC day YYYY-MM-DD (exclusive)")
    r.add_argument("--out", required=True)

    c = sub.add_parser("check", help="verify manifest and bar continuity")
    c.add_argument("--out", required=True)
    c.add_argument("--kind", help="series kind to check for gaps")
    c.add_argument("--symbol")
    c.add_argument("--interval")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "archive":
            for (sym, kind), stat in archive.download(
                    _csv(args.kinds), _csv(args.symbols), args.start, args.end,
                    args.out, args.interval).items():
                print(sym, kind, stat)
            return 0
        if args.cmd == "rest":
            for path in rest_tail.run(_csv(args.symbols), _csv(args.kinds), args.interval,
                                      args.start, args.out, args.end):
                print(path)
            return 0
        rep = check.check_manifest(args.out)
        print("manifest", rep)
        bad = bool(rep["problems"])
        if args.kind:
            cont = check.check_continuity(args.out, args.kind, args.symbol, args.interval)
            print("continuity", cont)
            bad = bad or bool(cont["gaps"] or cont["duplicates"] or cont["bad_close_time"])
        return 1 if bad else 0
    except archive.DiskLowError as exc:
        print("STOP:", exc)
        return 4
    except (DataError, ValueError) as exc:
        print("ERROR:", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
