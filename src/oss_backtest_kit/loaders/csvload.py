"""CSV reader: time + returns, or time + equity curve."""

import csv
import math
import os

import numpy as np

from ._series import (LoaderError, collapse_days, equity_to_returns, time_index, DailyReturns,
                      validate_equity, validate_returns)

TIME_NAMES = ("date", "time", "timestamp", "datetime", "day", "open_time", "close_time")
RETURN_NAMES = ("return", "returns", "ret", "daily_return", "pnl_pct")
EQUITY_NAMES = ("equity", "equity_curve", "balance", "nav", "portfolio_value", "account_value", "value")
MAX_ROWS = 5_000_000
_FLAGS = {"time": "command line: --time-col NAME"}


def _find(header, names, what, path):
    """Index of the single header cell matching one of ``names`` (case-insensitive)."""
    low = [h.strip().lower() for h in header]
    hits = [i for i, h in enumerate(low) if h in names]
    if not hits:
        raise LoaderError("%s: cannot find a %s column (columns: %s); pass it explicitly (%s)"
                          % (path, what, ", ".join(header), _FLAGS.get(what, "command line: --value-col NAME --kind returns|equity")))
    if len(hits) > 1:
        raise LoaderError("%s: ambiguous %s column, several match (%s); pass it explicitly (%s)"
                          % (path, what, ", ".join(header[i] for i in hits),
                             _FLAGS.get(what, "command line: --value-col NAME --kind returns|equity")))
    return hits[0]


def _pick(header, explicit, candidates, what, path):
    if explicit is None:
        return _find(header, candidates, what, path)
    want = explicit.strip().lower()
    if want not in [h.strip().lower() for h in header]:
        raise LoaderError("%s: no column %r (columns: %s)" % (path, explicit, ", ".join(header)))
    return _find(header, (want,), what, path)


def _detect_kind(header, path):
    low = {h.strip().lower() for h in header}
    has_r = any(c in low for c in RETURN_NAMES)
    has_e = any(c in low for c in EQUITY_NAMES)
    if has_r and has_e:
        raise LoaderError("%s: both a returns and an equity column found; pass kind= (command line: --kind returns|equity)" % path)
    if not (has_r or has_e):
        raise LoaderError("%s: no returns or equity column found (columns: %s); pass value_col= and kind= "
                          "(command line: --value-col NAME --kind returns|equity)" % (path, ", ".join(header)))
    return "returns" if has_r else "equity"


def _choose_value_column(header, kind, value_col, path):
    """Return ``(column index, kind)``."""
    if kind not in (None, "returns", "equity"):
        raise LoaderError("kind must be 'returns', 'equity' or None, got %r" % (kind,))
    if value_col is not None:
        if kind is None:
            raise LoaderError("%s: with value_col you must also say kind='returns' or 'equity' (command line: --kind returns|equity)" % path)
        return _pick(header, value_col, (), "value", path), kind
    if kind is None:
        kind = _detect_kind(header, path)
    return _find(header, RETURN_NAMES if kind == "returns" else EQUITY_NAMES, kind, path), kind


def _header(row, path):
    if not row or not any(h.strip() for h in row):
        raise LoaderError("%s: empty file or missing header row" % path)
    return [h.strip() for h in row]


def _read_rows(path):
    if not os.path.isfile(path):
        raise LoaderError("%s: not a file" % path)
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rd = csv.reader(fh)
            header = _header(next(rd, None), path)
            rows = [(rd.line_num, row) for row in rd if any(c.strip() for c in row)]
    except (UnicodeDecodeError, csv.Error, OSError) as exc:
        raise LoaderError("%s: cannot read as UTF-8 CSV (%s)" % (path, exc))
    if len(rows) > MAX_ROWS:
        raise LoaderError("%s: more than %d rows" % (path, MAX_ROWS))
    if not rows:
        raise LoaderError("%s: header only, no data rows" % path)
    return header, rows


def _number(text, path, line):
    try:
        x = float(text)
    except ValueError:
        raise LoaderError("%s line %d: %r is not a number" % (path, line, text))
    if not math.isfinite(x):
        raise LoaderError("%s line %d: value is not finite" % (path, line))
    return x


def load_csv(path, kind=None, time_col=None, value_col=None, name=None):
    """Read a CSV with a header into a :class:`DailyReturns`.

    ``kind`` is ``"returns"`` (simple fractions, 0.01 = 1%) or ``"equity"``;
    ``None`` detects it from the column names and fails if ambiguous. Times
    are ISO 8601 or epoch seconds / milliseconds, taken as UTC days. Rows must
    be sorted by time. Several rows on one day are compounded (returns) or
    reduced to the last value (equity).
    """
    path = os.fspath(path)
    header, rows = _read_rows(path)
    vi, kind = _choose_value_column(header, kind, value_col, path)
    ti = _pick(header, time_col, TIME_NAMES, "time", path)
    times, vals = [], []
    for n, row in rows:
        if len(row) <= max(ti, vi):
            raise LoaderError("%s line %d: row has %d fields, need %d" % (path, n, len(row), max(ti, vi) + 1))
        times.append(row[ti].strip())
        vals.append(_number(row[vi], path, n))
    label = name or os.path.splitext(os.path.basename(path))[0]
    days, secs = time_index(times, path)
    arr = np.array(vals)
    validate_equity(arr, label) if kind == "equity" else validate_returns(arr, label)
    days, v = collapse_days(days, arr, kind, label, secs)
    if kind == "equity":
        days, v, spans = equity_to_returns(days, v, label)
        return DailyReturns(v, days, label, spans=spans, basis="equity")
    return DailyReturns(v, days, label)
