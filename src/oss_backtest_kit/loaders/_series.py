"""Daily return series container and constructors (no pandas required)."""

import datetime as _dt
import re

import numpy as np

SYNTHETIC_START = np.datetime64("2000-01-01", "D")
BASES = ("returns", "equity", "realised")


class LoaderError(ValueError):
    """Raised when an input cannot be turned into a clean daily return series."""


class DailyReturns:
    """A named series of simple returns, one per calendar day (UTC).

    ``dates`` is a strictly increasing ``datetime64[D]`` array. ``synthetic``
    is True when the caller gave no dates and consecutive placeholder days
    were generated; synthetic series can only be aligned by position.
    ``spans`` (optional int array) says how many days each return covers
    (None = one period per row); ``basis`` is where the numbers came from:
    ``"returns"``, ``"equity"`` or ``"realised"``.
    """

    def __init__(self, values, dates, name="series", synthetic=False, spans=None, basis="returns"):
        v = np.asarray(values, dtype=float)
        d = np.asarray(dates, dtype="datetime64[D]")
        if v.ndim != 1 or d.ndim != 1 or v.size != d.size:
            raise LoaderError("%s: values and dates must be 1-D and the same length" % name)
        if v.size == 0:
            raise LoaderError("%s: no observations" % name)
        if not np.isfinite(v).all():
            raise LoaderError("%s: returns contain NaN or infinity" % name)
        if (v <= -1.0).any():
            raise LoaderError("%s: a return is <= -100%%; returns must be fractions (0.01 = 1%%)" % name)
        if v.size > 1 and not (np.diff(d).astype("int64") > 0).all():
            raise LoaderError("%s: dates are not strictly increasing" % name)
        if basis not in BASES:
            raise LoaderError("%s: basis must be one of %s, got %r" % (name, ", ".join(BASES), basis))
        if spans is not None:
            spans = np.asarray(spans)
            if spans.ndim != 1 or spans.size != v.size or spans.dtype.kind not in "iu" or (spans < 1).any():
                raise LoaderError("%s: spans must be whole numbers of at least 1, one per return" % name)
            spans = spans.astype("int64")
        self.values, self.dates, self.name, self.synthetic = v, d, str(name), bool(synthetic)
        self.spans, self.basis = spans, basis

    def __len__(self):
        return int(self.values.size)

    def __repr__(self):
        return "DailyReturns(%r, n=%d, %s..%s)" % (self.name, len(self), self.dates[0], self.dates[-1])


_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATETIME_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?\s*(Z|z|[+-]\d{2}(?::?\d{2})?)?$")
EPOCH_MIN, EPOCH_MAX = 1230768000, 4102444800  # 2009-01-01 .. 2100-01-01, in seconds


def to_utc_date(value, where):
    """Parse a date / datetime / epoch value to ``datetime64[D]`` (UTC day).
    See :func:`to_utc_time` for the formats and for the time of day.
    """
    return _day_of(to_utc_time(value, where))


def _day_of(t):
    return np.datetime64(int(t // 86400), "D")


def to_utc_time(value, where):
    """Parse a date / datetime / epoch value to UTC seconds since 1970 (float).

    Accepts ``YYYY-MM-DD`` and ISO 8601 ``YYYY-MM-DD[T ]HH:MM[:SS[.f]]`` text
    (``Z`` or an offset allowed; a naive time is taken as UTC), ``datetime``
    objects, and integer epoch seconds or milliseconds that fall between 2009
    and 2100. Anything else raises :class:`LoaderError`; nothing is guessed
    from the number of digits.
    """
    if isinstance(value, _dt.datetime):
        d = value
    elif isinstance(value, _dt.date):
        d = _dt.datetime(value.year, value.month, value.day)
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return float(_from_epoch(int(value), where))
    elif isinstance(value, str):
        d = _parse_text(value.strip(), where)
        if not isinstance(d, _dt.datetime):
            return float(d)
    else:
        raise LoaderError("%s: unsupported time value %r" % (where, value))
    return d.replace(tzinfo=d.tzinfo or _dt.timezone.utc).timestamp()


def _from_epoch(n, where):
    if EPOCH_MIN <= n < EPOCH_MAX:
        secs = n
    elif EPOCH_MIN * 1000 <= n < EPOCH_MAX * 1000:
        secs = n // 1000
    else:
        raise LoaderError("%s: number %d is not an epoch time between 2009 and 2100 (seconds or milliseconds)"
                          % (where, n))
    return secs


def _parse_text(text, where):
    bad = LoaderError("%s: cannot parse time %r (use YYYY-MM-DD, ISO 8601 or epoch seconds / milliseconds)"
                      % (where, text))
    if text.isascii() and text.isdigit():
        return _from_epoch(int(text), where)
    try:
        m = _DATE_RE.match(text)
        if m:
            return _dt.datetime(*[int(g) for g in m.groups()])
        m = _DATETIME_RE.match(text)
        if not m:
            raise bad
        y, mo, da, h, mi, sec, frac, tz = m.groups()
        return _dt.datetime(int(y), int(mo), int(da), int(h), int(mi), int(sec or 0), tzinfo=_tz(tz))
    except ValueError:
        raise bad


def _tz(tz):
    if tz is None or tz in ("Z", "z"):
        return _dt.timezone.utc
    sign = -1 if tz[0] == "-" else 1
    digits = tz[1:].replace(":", "")
    if int(digits[2:] or 0) > 59:
        raise ValueError("minutes of the offset must be 00-59")
    mins = int(digits[:2]) * 60 + int(digits[2:] or 0)
    return _dt.timezone(sign * _dt.timedelta(minutes=mins))


def day_index(times, where="times"):
    """Convert a sequence of time values to a ``datetime64[D]`` array."""
    return time_index(times, where)[0]


def time_index(times, where="times"):
    """Convert time values to ``(datetime64[D] days, UTC seconds)`` arrays."""
    secs = np.array([to_utc_time(t, "%s[%d]" % (where, i)) for i, t in enumerate(times)], dtype=float)
    return np.array([_day_of(t) for t in secs], dtype="datetime64[D]"), secs


def collapse_days(days, values, kind, name, secs=None):
    """Merge several observations on one day. Returns ``(unique_days, values)``.

    Rows must be sorted by time (non-decreasing days; with ``secs`` also within
    a day, otherwise the last row of a day is not the latest). ``kind`` is
    ``"returns"`` (compounded within the day) or ``"equity"`` (last value).
    """
    if days.size > 1 and (np.diff(days).astype("int64") < 0).any():
        raise LoaderError("%s: rows are not sorted by time" % name)
    if secs is not None and secs.size > 1 and ((np.diff(secs) < 0) & (np.diff(days) == 0)).any():
        raise LoaderError("%s: rows within one day are not in time order; sort the file by time" % name)
    uniq, first = np.unique(days, return_index=True)
    ends = list(first[1:]) + [days.size]
    out = []
    for a, b in zip(first, ends):
        seg = values[a:b]
        if kind == "returns" and seg.size > 1:
            out.append(float(np.prod(1.0 + seg) - 1.0))
        else:
            out.append(float(seg[-1]))
    return uniq, np.array(out)


def validate_returns(values, name, ndim=1):
    """Return ``values`` as floats after the return-domain checks.

    Rejects booleans, complex numbers, a wrong number of dimensions, NaN /
    infinity and any return <= -100% (returns are fractions, 0.01 = 1%).
    """
    try:
        raw = np.asarray(values)
        v = None if raw.dtype.kind in "bc" else raw.astype(float)
    except (TypeError, ValueError, OverflowError):
        raise LoaderError("%s: values are not numbers" % name)
    if v is None:
        raise LoaderError("%s: boolean or complex values are not returns" % name)
    if v.ndim != ndim:
        raise LoaderError("%s: expected a %d-D array of returns, got %d-D" % (name, ndim, v.ndim))
    if not np.isfinite(v).all():
        raise LoaderError("%s: returns contain NaN or infinity; remove or fix those rows" % name)
    if (v <= -1.0).any():
        raise LoaderError("%s: a return is <= -100%%; returns must be fractions (0.01 = 1%%)" % name)
    return v


def validate_equity(values, name):
    """Equity must be finite and > 0 on every row (checked before same-day merging)."""
    e = np.asarray(values, dtype=float)
    if not np.isfinite(e).all() or (e <= 0).any():
        raise LoaderError("%s: equity must be finite and > 0 on every row" % name)


def equity_to_returns(days, equity, name):
    """Turn an equity curve into simple returns dated by the later point.

    Returns ``(days, returns, spans)``; ``spans`` is the number of days each return covers.
    """
    e = np.asarray(equity, dtype=float)
    if e.size < 2:
        raise LoaderError("%s: an equity curve needs at least 2 points" % name)
    if not np.isfinite(e).all() or (e <= 0).any():
        raise LoaderError("%s: equity must be finite and > 0" % name)
    return days[1:], e[1:] / e[:-1] - 1.0, np.diff(days).astype("int64")


def fill_flat_days(s):
    """Spread an equity-based series over every calendar day: the days a return skipped get 0.

    Returns ``(series, days_added)``. Series without spans, or with all spans 1, come back unchanged.
    """
    if s.spans is None or (s.spans == 1).all():
        return s, 0
    total = int(s.spans.sum())
    out = np.zeros(total)
    out[np.cumsum(s.spans) - 1] = s.values
    days = s.dates[0] - int(s.spans[0]) + 1 + np.arange(total)
    return DailyReturns(out, days, s.name, s.synthetic, np.ones(total, dtype="int64"), s.basis), total - len(s)


def _index_to_days(index, name):
    tz = getattr(index, "tz", None)
    if tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    try:
        return np.asarray(index, dtype="datetime64[ns]").astype("datetime64[D]")
    except (TypeError, ValueError):
        return day_index(list(index), name + " index")


def _series_parts(values, dates, name):
    """Split a pandas-like Series into plain values and index days (if datetime-like)."""
    if hasattr(values, "index") and hasattr(values, "to_numpy"):
        if getattr(getattr(values, "dtype", None), "kind", "") == "b":
            raise LoaderError("%s: boolean values are not returns" % name)
        if dates is None:
            dates = _series_dates(values.index, name)
        try:
            values = values.to_numpy(dtype=float)
        except (TypeError, ValueError):
            raise LoaderError("%s: values are not numeric" % name)
    return values, dates


def _series_dates(index, name):
    """Dates from a Series index: datetime, period or date-text index; None otherwise."""
    if hasattr(index, "to_timestamp"):  # PeriodIndex
        index = index.to_timestamp()
    kind = getattr(getattr(index, "dtype", None), "kind", "")
    if kind == "M" or getattr(index, "tz", None) is not None:
        return _index_to_days(index, name)
    if len(index) and isinstance(index[0], str):
        return day_index(list(index), name + " index")
    return None


def _unpack(values, dates, name):
    """Return ``(float array, datetime64[D] array or None)``; reads a Series index."""
    values, dates = _series_parts(values, dates, name)
    try:
        raw = np.asarray(values)
        if raw.dtype.kind in "bc":
            raise LoaderError("%s: boolean or complex values are not returns or equity" % name)
        v = raw.astype(float)
    except (TypeError, ValueError, OverflowError):
        raise LoaderError("%s: values are not numeric" % name)
    if v.ndim != 1:
        raise LoaderError("%s: expected a 1-D sequence; for several results (for example one column per "
                          "parameter set) save each as a file in a folder and pass the folder, or use "
                          "df.to_numpy() with the 2-D array form of checkup (dates are lost)" % name)
    if dates is None:
        return v, None
    if isinstance(dates, np.ndarray) and dates.dtype.kind == "M":
        d = dates.astype("datetime64[D]")
    else:
        d = day_index(list(dates), name)
    if d.size != v.size:
        raise LoaderError("%s: %d values but %d dates" % (name, v.size, d.size))
    return v, d


def from_returns(values, dates=None, name="series"):
    """Build a series from per-period returns (list, array or pandas Series).

    A Series (anything with ``.index`` and ``.to_numpy()``) supplies its own
    dates when ``dates`` is None and the index is datetime-like. Without dates
    the series gets synthetic consecutive days. Several rows on one day are
    compounded.
    """
    v, d = _unpack(values, dates, name)
    validate_returns(v, name)
    if d is None:
        return DailyReturns(v, SYNTHETIC_START + np.arange(v.size), name, synthetic=True)
    d, v = collapse_days(d, v, "returns", name)
    return DailyReturns(v, d, name)


def from_equity(values, dates=None, name="series"):
    """Build a series from an equity curve (list, array or pandas Series).

    Returns are dated by the later point of each pair, so the series has one
    fewer row than the curve. Several rows on one day keep the last value.
    """
    e, d = _unpack(values, dates, name)
    validate_equity(e, name)
    synthetic = d is None
    if synthetic:
        d = SYNTHETIC_START + np.arange(e.size)
    else:
        d, e = collapse_days(d, e, "equity", name)
    dd, r, spans = equity_to_returns(d, e, name)
    if synthetic:  # undated: row k of the returns is row k, aligned by position
        dd = SYNTHETIC_START + np.arange(r.size)
    return DailyReturns(r, dd, name, synthetic=synthetic, spans=spans, basis="equity")
