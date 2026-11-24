"""Directory loading and date alignment into a (periods x trials) matrix."""

import os

import numpy as np

from ._series import DailyReturns, LoaderError, fill_flat_days, from_returns
from .csvload import load_csv
from .freqtrade_results import load_freqtrade_all

MIN_PERIODS = 16  # pbo_cscv needs periods >= 2 * n_blocks (8 blocks by default)
MIN_OVERLAP = 0.8
SIDECAR_SUFFIXES = (".meta.json",)


class ReturnMatrix:
    """Aligned returns: ``matrix`` has shape (periods, trials), ready for ``pbo_cscv``."""

    def __init__(self, dates, names, matrix, dropped=None, skipped=(), sources=None, filled=None):
        self.dates, self.names = dates, tuple(names)
        self.matrix = matrix
        self.dropped = dict(dropped or {})
        self.skipped = tuple(skipped)
        self.sources = sources  # {name: source_info(...)} or None when not recorded
        self.filled = dict(filled or {})  # {name: days added by fill_flat}

    def __repr__(self):
        return "ReturnMatrix(periods=%d, trials=%d)" % self.matrix.shape


def _check_unique(series):
    seen = set()
    for s in series:
        if s.name in seen:
            raise LoaderError("duplicate series name %r; names must be unique" % s.name)
        seen.add(s.name)


def _common_days(series):
    flags = {s.synthetic for s in series}
    if len(flags) > 1:
        raise LoaderError("cannot align undated series with dated series")
    common = series[0].dates
    if flags == {True}:
        if len({len(s) for s in series}) != 1:
            raise LoaderError("undated series must have the same length (got %s)" % sorted({len(s) for s in series}))
        return common
    for s in series[1:]:
        common = np.intersect1d(common, s.dates)
    return common


def source_info(s, common=None):
    """Where a series came from: length, first / last day, rows lost to alignment, spans, basis."""
    spans = np.ones(len(s), dtype="int64") if s.spans is None else s.spans
    before = after = inside = 0
    if common is not None:
        before = int((s.dates < common[0]).sum())
        after = int((s.dates > common[-1]).sum())
        inside = len(s) - common.size - before - after
    return {"n": len(s), "start": str(s.dates[0]), "end": str(s.dates[-1]), "before": before,
            "after": after, "inside": inside, "max_span": int(spans.max()),
            "median_span": float(np.median(spans)), "basis": s.basis}


def align_returns(series, min_periods=MIN_PERIODS, min_overlap=MIN_OVERLAP):
    """Align several :class:`DailyReturns` on the dates all of them share.

    Raises :class:`LoaderError` when fewer than two series are given, when
    synthetic (undated) and dated series are mixed, when undated series differ
    in length, when fewer than ``min_periods`` common days remain, or when a
    series would lose more than ``1 - min_overlap`` of its days.
    """
    series = list(series)
    if len(series) < 2:
        raise LoaderError("need at least two series to build a matrix, got %d" % len(series))
    _check_unique(series)
    common = _common_days(series)
    if common.size < min_periods:
        raise LoaderError("only %d common days across the series; need at least %d" % (common.size, min_periods))
    cols, dropped, sources = [], {}, {}
    for s in series:
        cov = common.size / len(s)
        if cov < min_overlap:
            raise LoaderError("series %r keeps only %.0f%% of its days after alignment (minimum %.0f%%); "
                              "date ranges do not match" % (s.name, 100 * cov, 100 * min_overlap))
        cols.append(s.values[np.searchsorted(s.dates, common)])
        dropped[s.name] = len(s) - common.size
        sources[s.name] = source_info(s, common)
    return ReturnMatrix(common, [s.name for s in series], np.column_stack(cols), dropped, sources=sources)


def _is_candidate(name):
    low = name.lower()
    return not name.startswith(".") and low.endswith((".json", ".zip", ".csv")) and not low.endswith(SIDECAR_SUFFIXES)


def load_directory(path, min_periods=MIN_PERIODS, min_overlap=MIN_OVERLAP, starting_balance=None,
                   fill_flat=False, **csv_kwargs):
    """Load every result file in a directory (not recursive) into a matrix.

    ``.json`` / ``.zip`` are read as freqtrade results (every strategy in a
    file becomes a column); ``.csv`` via :func:`load_csv` (``csv_kwargs``).
    Symbolic links are not followed. Files with other extensions are listed in ``ReturnMatrix.skipped``.
    ``fill_flat=True`` first spreads every equity-based result over all calendar days (0 on skipped
    days; ``ReturnMatrix.filled`` counts the days added per result).
    """
    path = os.fspath(path)
    if not os.path.isdir(path):
        raise LoaderError("%s: not a directory" % path)
    entries = sorted(e for e in os.listdir(path) if os.path.isfile(os.path.join(path, e)))
    series, skipped = [], []
    for e in entries:
        full = os.path.join(path, e)
        if os.path.islink(full) or not _is_candidate(e):
            skipped.append(e)
        elif e.lower().endswith(".csv"):
            series.append(load_csv(full, **csv_kwargs))
        else:
            series.extend(load_freqtrade_all(full, starting_balance=starting_balance))
    if len(series) < 2:
        raise LoaderError("%s: found %d usable result(s); need at least two" % (path, len(series)))
    filled = {}
    if fill_flat:
        pairs = [fill_flat_days(s) for s in series]
        series, filled = [p[0] for p in pairs], {p[0].name: p[1] for p in pairs if p[1]}
    m = align_returns(series, min_periods, min_overlap)
    m.skipped, m.filled = tuple(skipped), filled
    return m


def load_returns(source, **kwargs):
    """Single entry point.

    A directory gives a :class:`ReturnMatrix`; a ``.csv`` / ``.json`` / ``.zip``
    path gives a :class:`DailyReturns`; an array or Series is taken as returns.
    """
    if isinstance(source, DailyReturns):
        return source
    if isinstance(source, (str, os.PathLike)):
        p = os.fspath(source)
        if os.path.isdir(p):
            return load_directory(p, **kwargs)
        low = p.lower()
        if low.endswith(".csv"):
            return load_csv(p, **kwargs)
        if low.endswith((".json", ".zip")):
            from .freqtrade_results import load_freqtrade
            return load_freqtrade(p, **kwargs)
        raise LoaderError("%s: unknown file type (expected .csv, .json, .zip or a directory)" % p)
    return from_returns(source, **kwargs)
