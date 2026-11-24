"""Reader for freqtrade ``backtest-results`` files (json or zip).

The layout is recognised by structure, not by a version number (freqtrade
writes none that this reader relies on). Supported: the nested layout with a
top-level ``"strategy"`` object mapping strategy name to a result object that
holds a ``"trades"`` list. Each trade needs ``close_date`` and ``profit_abs``;
``open_date`` is used when present. The older bare list-of-trades layout,
which freqtrade itself no longer reads, is rejected with a clear error.
"""

import json
import math
import os
import zipfile
import zlib

import numpy as np

from ._series import LoaderError, DailyReturns, to_utc_time, _day_of

MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_SPAN_DAYS = 366 * 20  # a longer span is almost certainly a wrong date


def _read_zip(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise LoaderError("%s: not a valid zip file" % path)
    with zf:
        names = zf.namelist()
        if stem + ".json" in names:
            member = stem + ".json"
        else:
            cands = sorted({n for n in names if n.endswith(".json") and not n.endswith(".meta.json") and "/" not in n})
            if len(cands) != 1:
                raise LoaderError("%s: cannot tell which member is the results json (members: %s)"
                                  % (path, ", ".join(names) or "none"))
            member = cands[0]
        if names.count(member) > 1:
            raise LoaderError("%s: the zip holds member %r more than once; remove the extra member so "
                              "only one copy remains" % (path, member))
        if zf.getinfo(member).file_size > MAX_JSON_BYTES:
            raise LoaderError("%s: results json larger than %d bytes" % (path, MAX_JSON_BYTES))
        try:
            return zf.read(member)
        except (RuntimeError, NotImplementedError, zipfile.BadZipFile, EOFError, zlib.error, OSError) as exc:
            raise LoaderError("%s: cannot read member %r (%s)" % (path, member, exc))


def _read_bytes(path):
    if not os.path.isfile(path):
        raise LoaderError("%s: not a file" % path)
    if path.lower().endswith(".zip"):
        return _read_zip(path)
    try:
        if os.path.getsize(path) > MAX_JSON_BYTES:
            raise LoaderError("%s: file larger than %d bytes" % (path, MAX_JSON_BYTES))
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise LoaderError("%s: cannot read (%s)" % (path, exc))


def _no_dup_keys(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise LoaderError("duplicate key %r in the JSON; remove the extra entry (or, for a strategy "
                              "name, keep one copy and pass --strategy)" % (key,))
        seen.add(key)
    return dict(pairs)


def _load_document(path):
    raw = _read_bytes(path)
    if not raw.strip():
        raise LoaderError("%s: empty file" % path)
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_dup_keys)
    except LoaderError as exc:
        raise LoaderError("%s: %s" % (path, exc))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LoaderError("%s: not valid JSON (%s)" % (path, exc))
    except RecursionError:
        raise LoaderError("%s: JSON is nested too deeply" % path)
    if isinstance(doc, list):
        raise LoaderError("%s: bare list-of-trades layout is not supported; "
                          "re-run the backtest with a current freqtrade" % path)
    if not isinstance(doc, dict):
        raise LoaderError("%s: unexpected top-level JSON type %s" % (path, type(doc).__name__))
    if "strategy" not in doc:
        hint = " (this looks like a .last_result.json pointer; pass the file it names)" if "latest_backtest" in doc else ""
        raise LoaderError("%s: no top-level 'strategy' object; unknown layout%s" % (path, hint))
    strat = doc["strategy"]
    if not isinstance(strat, dict) or not strat:
        raise LoaderError("%s: 'strategy' must be a non-empty object" % path)
    return strat


def _day(text, where):
    if not isinstance(text, str):
        raise LoaderError("%s: expected a date string, got %r" % (where, text))
    return to_utc_time(text.strip(), where)


def _finite(x):
    """True for a real, finite JSON number (not a bool, not too large for a float)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def _starting_balance(res, override, where):
    for key, val in (("starting_balance", override), ("starting_balance", res.get("starting_balance")),
                     ("dry_run_wallet", res.get("dry_run_wallet"))):
        if val is not None:
            if not _finite(val) or val <= 0:
                raise LoaderError("%s: %s must be a positive number, got %r" % (where, key, val))
            return float(val)
    raise LoaderError("%s: no starting_balance or dry_run_wallet in the file; pass starting_balance= (command line: --starting-balance X)" % where)


def _one_trade(t, w):
    """Validate one trade object; return ``(close_day, open_day, profit_abs)``."""
    if not isinstance(t, dict):
        raise LoaderError("%s: not an object" % w)
    for key in ("close_date", "profit_abs"):
        if key not in t:
            raise LoaderError("%s: missing required field %r (fields: %s)" % (w, key, ", ".join(sorted(t))))
    p = t["profit_abs"]
    if not _finite(p):
        raise LoaderError("%s: profit_abs is not a finite number: %r" % (w, p))
    close = _day(t["close_date"], w + " close_date")
    opened = _day(t["open_date"], w + " open_date") if t.get("open_date") is not None else close
    if opened > close:
        raise LoaderError("%s: open_date is after close_date" % w)
    return _day_of(close), _day_of(opened), float(p)


def _trade_rows(trades, where):
    if not isinstance(trades, list) or not trades:
        raise LoaderError("%s: 'trades' must be a non-empty list" % where)
    rows = [_one_trade(t, "%s trade %d" % (where, i)) for i, t in enumerate(trades)]
    close, opened, pnl = zip(*rows)
    return np.array(close, dtype="datetime64[D]"), np.array(opened, dtype="datetime64[D]"), np.array(pnl)


def _span(res, close, opened, where):
    first, last = opened.min(), close.max()
    for key, side in (("backtest_start", 0), ("backtest_end", 1)):
        if res.get(key) is not None:
            d = _day_of(_day(res[key], where + " " + key))
            first, last = (min(first, d), last) if side == 0 else (first, max(last, d))
    if int((last - first).astype("int64")) > MAX_SPAN_DAYS:
        raise LoaderError("%s: result spans %s to %s, more than 20 years; check backtest_start / backtest_end"
                          % (where, first, last))
    return first, last


def _reconcile(res, pnl, where):
    """When the file states a total profit, it must equal the sum of the trades."""
    total = res.get("profit_total_abs")
    if total is None:
        return
    if not _finite(total):
        raise LoaderError("%s: profit_total_abs is not a finite number: %r" % (where, total))
    tol = 1e-6 * abs(float(total)) + 1e-6  # relative to the stated net profit, not to gross turnover
    if abs(float(total) - float(pnl.sum())) > tol:
        raise LoaderError("%s: profit_total_abs (%r) does not match the sum of trade profit_abs (%r); "
                          "this file's profit fields are not understood" % (where, total, float(pnl.sum())))


def _daily(res, name, override, where):
    close, opened, pnl = _trade_rows(res.get("trades"), where)
    bal = _starting_balance(res, override, where)
    _reconcile(res, pnl, where)
    first, last = _span(res, close, opened, where)
    days = first + np.arange(int((last - first).astype("int64")) + 1)
    per_day = np.zeros(days.size)
    np.add.at(per_day, (close - first).astype("int64"), pnl)
    start_eq = bal + np.concatenate([[0.0], np.cumsum(per_day)[:-1]])
    if not np.isfinite(start_eq).all():
        raise LoaderError("%s: realised equity overflows; profit values are not usable" % where)
    if (start_eq <= 0).any():
        raise LoaderError("%s: equity reaches zero or below (account wiped out); no daily return exists" % where)
    ret = per_day / start_eq
    if not np.isfinite(ret).all():
        raise LoaderError("%s: profit values overflow" % where)
    if (ret <= -1.0).any():
        raise LoaderError("%s: a day loses 100%% or more of equity" % where)
    return DailyReturns(ret, days, name, basis="realised")


def load_freqtrade_all(path, starting_balance=None):
    """Return one :class:`DailyReturns` per strategy in the file, in file order.

    When the strategy object holds ``profit_total_abs`` it is reconciled with
    the sum of the trades and a mismatch is an error; without it nothing is
    reconciled. Daily return = profit_abs of trades *closed* that day / equity at the start
    of that day (starting balance plus earlier realised profit). Days without a
    closed trade are zero. Open-trade mark-to-market is not in the file, so
    drawdown is measured on realised equity.
    """
    path = os.fspath(path)
    strat = _load_document(path)
    base = os.path.splitext(os.path.basename(path))[0]
    out = []
    for key, res in strat.items():
        if not isinstance(res, dict):
            raise LoaderError("%s: strategy %r is not an object" % (path, key))
        label = base if len(strat) == 1 else "%s:%s" % (base, key)
        out.append(_daily(res, label, starting_balance, "%s [%s]" % (path, key)))
    return out


def load_freqtrade(path, strategy=None, starting_balance=None):
    """Load one strategy from a freqtrade results json / zip.

    ``strategy`` is required when the file holds several. ``starting_balance``
    overrides the value in the file (needed only when the file has none).
    """
    path = os.fspath(path)
    strat = _load_document(path)
    if strategy is None:
        if len(strat) != 1:
            raise LoaderError("%s: holds %d strategies (%s); pass strategy= (command line: --strategy NAME)"
                              % (path, len(strat), ", ".join(strat)))
        strategy = next(iter(strat))
    if strategy not in strat:
        raise LoaderError("%s: no strategy %r (found: %s)" % (path, strategy, ", ".join(strat)))
    res = strat[strategy]
    if not isinstance(res, dict):
        raise LoaderError("%s: strategy %r is not an object" % (path, strategy))
    base = os.path.splitext(os.path.basename(path))[0]
    return _daily(res, "%s:%s" % (base, strategy) if len(strat) > 1 else base, starting_balance,
                  "%s [%s]" % (path, strategy))
