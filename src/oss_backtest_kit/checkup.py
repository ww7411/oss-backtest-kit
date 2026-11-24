"""Overfitting checkup for backtest results (spec items B2.x in docs/SPEC_B.md).

``checkup(source)`` reads a backtest result (see :mod:`oss_backtest_kit.loaders`),
computes a handful of statistics from :mod:`oss_backtest_kit.stats`, and grades
each as pass / caution / fail with one plain sentence. The thresholds are
conventions, printed in every report. The checkup lowers the risk of shipping an
overfitted backtest; it cannot show that a strategy will make money.

Command line::

    python -m oss_backtest_kit.checkup <file or directory> [--trials N]
        [--report out.md] [--html out.html] [--strategy NAME]
        [--starting-balance X] [--time-col C] [--value-col C] [--kind returns|equity] [--fill-flat]
"""

import math
import os
import sys

import numpy as np

from . import stats
from ._cli import _loader_args, main
from .loaders import DailyReturns, LoaderError, ReturnMatrix, load_returns
from .loaders._series import fill_flat_days, validate_returns
from .loaders.matrix import source_info as _source_info
from . import _disclosure as _dis

__all__ = ["CheckupError", "Metric", "Report", "THRESHOLDS", "checkup", "grade", "grade_bootstrap", "main"]

THRESHOLDS = {  # key: (pass at, caution at); see docs/SPEC_B.md for the basis
    "sharpe": (1.0, 0.5),   # higher is better: value >= first = pass, >= second = caution
    "psr": (0.95, 0.90),
    "dsr": (0.95, 0.90),
    "pbo": (0.20, 0.50),    # lower is better: value <= first = pass, <= second = caution
}
MIN_OBS = 30     # fewer observations is an error
SHORT_OBS = 180  # fewer observations: no pass, at most caution
MIN_ACTIVE = 30  # fewer non-zero returns: no pass, at most caution (sparse strategies)
VERDICTS = ("pass", "caution", "fail", "info")


class CheckupError(ValueError):
    """Raised when the input cannot be checked; the message says why."""


class Metric:
    """One line of the report: a value, a verdict and one plain sentence."""

    def __init__(self, key, label, value, shown, verdict, text, detail=None):
        self.key, self.label, self.value, self.shown = key, label, value, shown
        self.verdict, self.text, self.detail = verdict, text, dict(detail or {})

    def __repr__(self):
        return "Metric(%s=%s, %s)" % (self.key, self.shown, self.verdict)


class Report:
    """Result of :func:`checkup`; render it with ``report.to_markdown()`` / ``to_html()``."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def metric(self, key):
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(key)

    @property
    def overall(self):
        """Worst verdict among the graded metrics: fail, caution or pass."""
        got = {m.verdict for m in self.metrics}
        return "fail" if "fail" in got else "caution" if "caution" in got else "pass"

    def to_markdown(self):
        from .report import to_markdown
        return to_markdown(self)

    def to_html(self):
        from .report import to_html
        return to_html(self)


# ---- verdicts
def grade(key, value):
    """Verdict for ``key`` in sharpe / psr / dsr / pbo. NaN is a fail."""
    if value is None or not math.isfinite(value):
        return "fail"
    good, ok = THRESHOLDS[key]
    if key == "pbo":
        return "pass" if value <= good else "caution" if value <= ok else "fail"
    return "pass" if value >= good else "caution" if value >= ok else "fail"


def grade_bootstrap(point, low, high):
    """Verdict for the bootstrap Sharpe interval."""
    if not all(math.isfinite(x) for x in (point, low, high)):
        return "fail"
    if high <= 0 or point <= 0:
        return "fail"
    return "pass" if low > 0 else "caution"


def _cap(verdict, short):
    return "caution" if short and verdict == "pass" else verdict


# ---- input
def _from_2d(source):
    if source.shape[1] < 2:
        raise CheckupError("a 2-D array needs at least two columns (one per result)")
    try:
        m = validate_returns(source, "the array", ndim=2)
    except LoaderError as e:
        raise CheckupError(str(e))
    return ["result_%d" % (k + 1) for k in range(m.shape[1])], m, None, True, None, {}


def _prepare(source, select, load_options=None, fill_flat=False):
    """Return (names, matrix of shape periods x results, dates or None, synthetic, sources, filled)."""
    if isinstance(source, (str, os.PathLike)) and not os.path.exists(os.fspath(source)):
        raise CheckupError("%s: no such file or directory" % os.fspath(source))
    args = _loader_args(source, load_options)
    if isinstance(source, np.ndarray) and source.ndim > 2:
        raise CheckupError("an array must be 1-D (one result) or 2-D (periods x results), got %d dimensions"
                           % source.ndim)
    if isinstance(source, np.ndarray) and source.ndim == 2:  # periods x results, no dates
        return _from_2d(source)
    if fill_flat and isinstance(source, (str, os.PathLike)) and os.path.isdir(source):
        args["fill_flat"] = True
    loaded = source if isinstance(source, ReturnMatrix) else load_returns(source, **args)
    if isinstance(loaded, ReturnMatrix):
        return (list(loaded.names), np.asarray(loaded.matrix, dtype=float), loaded.dates, False,
                loaded.sources, loaded.filled)
    if isinstance(loaded, DailyReturns):
        added = 0
        if fill_flat:
            loaded, added = fill_flat_days(loaded)
        return ([loaded.name], loaded.values.reshape(-1, 1), loaded.dates, loaded.synthetic,
                {loaded.name: _source_info(loaded)}, {loaded.name: added} if added else {})
    raise CheckupError("unsupported input of type %s" % type(loaded).__name__)


def _check_spans(sources, aligned=False):
    """Refuse a result whose returns cover unequal numbers of days (equity points unevenly spaced)."""
    for name, info in (sources or {}).items():
        if info["max_span"] > 2 * info["median_span"] + 2:
            raise CheckupError(
                "%s: one return covers %d days while a typical one covers %d (equity points are unevenly "
                "spaced). PSR, DSR, PBO and the bootstrap treat every return as one equal period, so this "
                "result cannot be checked. Add the missing dates (repeat the last equity value only if the "
                "account truly did not change), or cut the series at the gap. %s"
                % (name, info["max_span"], round(info["median_span"]), "This matrix is already aligned, so "
                   "fill_flat has no effect here: pass fill_flat=True when loading the data, if the account "
                   "truly did not change on the missing days." if aligned else "Pass fill_flat=True "
                   "(--fill-flat) if the account truly did not change on the missing days."))


def _check_trials(n_trials, n_results):
    if n_trials is None:
        return n_results, False
    if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)) or n_trials < 1:
        raise CheckupError("n_trials must be a whole number of at least 1, got %r" % (n_trials,))
    if n_trials < n_results:
        raise CheckupError("n_trials (%d) is smaller than the number of results (%d); every result "
                           "you read was one trial" % (n_trials, n_results))
    return int(n_trials), True


def _check_matrix(names, m):
    if m.shape[0] < MIN_OBS:
        raise CheckupError("%d observations is too few; at least %d are needed (%d or more for "
                           "a pass)" % (m.shape[0], MIN_OBS, SHORT_OBS))
    if not np.isfinite(m).all():
        raise CheckupError("returns contain NaN or infinity")
    flat = np.flatnonzero(np.ptp(m, axis=0) == 0)
    if flat.size:
        raise CheckupError("constant returns (zero variance) in: %s; the Sharpe ratio is undefined. "
                           "Remove or fix these results" % ", ".join(names[i] for i in flat))


def _select(names, m, select):
    if select is None:
        return int(np.argmax([stats.sharpe(m[:, j]) for j in range(m.shape[1])]))
    if select not in names:
        raise CheckupError("select=%r is not one of the results: %s" % (select, ", ".join(names)))
    return names.index(select)


# ---- metrics
def _m_sharpe(r, short, ppy):
    v = stats.sharpe_annualised(r, ppy)
    verdict = _cap(grade("sharpe", v), short)
    say = {"pass": "Return per unit of risk is solid before costs.",
           "caution": "Return per unit of risk is modest; costs could erase it.",
           "fail": "Return per unit of risk is low or negative."}[verdict]
    return Metric("sharpe", "Annualised Sharpe", v, "%.2f" % v, verdict, say)


def _m_drawdown(r, basis=None):
    v = stats.max_drawdown(r)
    if basis == "realised":
        text = ("The realised equity (closed trades only) fell %.1f%% from its peak at worst. Open positions "
                "were not marked to market, so the real drawdown while trades were open can be larger. This "
                "shows loss you must be able to bear, not overfitting, so it has no verdict." % (100 * v))
    else:
        text = ("The equity curve fell %.1f%% from its peak at worst. This shows how much loss you must "
                "be able to bear, not overfitting, so it has no verdict." % (100 * v))
    return Metric("max_drawdown", "Max drawdown", v, "%.1f%%" % (100 * v), "info", text)


def _m_psr(r, short, n_obs=None):
    v = stats.psr(r, n_obs=n_obs)
    verdict = _cap(grade("psr", v), short)
    p = "n/a" if not math.isfinite(v) else "more than 99.9%" if v > 0.999 else "%.0f%%" % (100 * v)
    say = {"pass": "Likely better than zero skill (%s confident), allowing for the sample length and shape.",
           "caution": "Only %s confident the edge is real; treat as unproven.",
           "fail": "Only %s confident the edge is real; this may be luck."}[verdict] % p
    return Metric("psr", "PSR (probabilistic Sharpe)", v, "%.3f" % v, verdict, say)


def _m_dsr(r, trials, trial_sharpes, short, unstated=False, n_obs=None):
    v, bench, _ = stats.dsr(r, trials, trial_sharpes, n_obs)
    verdict = _cap(grade("dsr", v), short or unstated)
    if unstated and len(trial_sharpes) > 1:
        say = ("Deflated for N = %d, the number of results read. That is only a lower bound: versions you tried "
               "but did not save also count, so this line cannot pass. Pass --trials N (n_trials=N), the total "
               "number of versions you tried." % trials)
    elif unstated:
        say = ("No deflation applied: the number of trials was not given, so this cannot be judged "
               "and equals PSR. Pass --trials N (n_trials=N), the total number of versions you tried; "
               "use 1 only if this was the only one.")
    elif trials == 1:
        say = "No deflation applied: only one trial is counted, so this equals PSR. Count every version you tried."
    else:
        say = {"pass": "Still convincing after allowing for %d tries.",
               "caution": "Borderline after allowing for %d tries; the best of many tries often looks good by chance.",
               "fail": "Not convincing after allowing for %d tries; the best of many tries often looks good by chance."
               }[verdict] % trials
    return Metric("dsr", "DSR (deflated Sharpe, N=%d)" % trials, v, "%.3f" % v, verdict, say,
                  {"n_trials": trials, "benchmark_sharpe": bench})


def _m_bootstrap(r, short, opts):
    if opts["tau"] >= stats.TAU_IGNORE and r.size < 4 * opts["block"]:
        why = "block length" if opts["block"] > math.ceil(opts["tau"]) else "dependence"
        return Metric("bootstrap", "Sharpe 95% interval (bootstrap)", None, "n/a", "caution", "The %s is "
                      "too long for this sample (blocks of %d over %d rows), so the range cannot be estimated."
                      % (why, opts["block"], r.size))
    b = stats.block_bootstrap_sharpe(r, opts["n_resamples"], opts["block"], opts["seed"],
                                     periods_per_year=opts["ppy"])
    verdict = _cap(grade_bootstrap(b["point"], b["low"], b["high"]), short)
    say = {"pass": "Even the pessimistic end of the range is above zero.",
           "caution": "The range includes zero, so the edge is not clearly separate from chance.",
           "fail": "The range is at or below zero, or the point estimate is: no sign of an edge."}[verdict]
    shown = "%.2f [%.2f, %.2f]" % (b["point"], b["low"], b["high"])
    return Metric("bootstrap", "Sharpe 95% interval (bootstrap)", b["point"], shown, verdict, say, b)


def _pbo_metric(m, short):
    label = "PBO (backtest overfitting)"
    if m.shape[1] < 2:
        return Metric("pbo", label, None, "n/a", "info",
                      "Needs two or more results (one per parameter set) to estimate.")
    try:
        v, left_out = stats.pbo_cscv_detail(m)
    except stats.StatsError as e:
        return Metric("pbo", label, None, "n/a", "info", "Could not be computed: %s." % e)
    if v is None:
        return Metric("pbo", label, None, "n/a", "info",
                      "Could not be computed: the results cannot be told apart in any split.")
    verdict = _cap(grade("pbo", v), short)
    say = {"pass": "The best-looking version rarely falls below the middle on unseen data.",
           "caution": "The best-looking version often did worse than average on unseen data.",
           "fail": "The best-looking version usually did worse than average on unseen data (0.5 is pure noise)."
           }[verdict]
    if left_out:
        say += " %d splits were left out because the results tied in them." % left_out
    return Metric("pbo", label, v, "%.2f" % v, verdict, say)


def _neff_metric(m, trials):
    if m.shape[1] < 2:
        return Metric("n_effective", "Effective trials", None, "n/a", "info",
                      "Needs two or more results.")
    v = stats.n_effective(m)
    text = ("Of %d results, about %.1f behave as independent trials (a lower number means near "
            "copies of each other). N used for DSR is %d." % (m.shape[1], v, trials))
    return Metric("n_effective", "Effective trials", v, "%.1f" % v, "info", text)


# ---- main API
def checkup(source, n_trials=None, select=None, n_resamples=2000, block=5, seed=0, load_options=None,
            fill_flat=False):
    """Check a backtest result (path, directory, array, Series or loader output).

    ``n_trials``: how many versions were tried in total, saved or not. Default:
    the number of results read (1 for a single result), with a note. With several
    results, the Sharpe, PSR, DSR and bootstrap lines describe the result with the
    highest Sharpe ratio, or ``select`` (a result name). ``load_options`` is an
    optional dict passed to the loaders for a path source (``strategy``,
    ``starting_balance``, ``kind``, ``time_col``, ``value_col``). Raises
    :class:`CheckupError` (or the loaders' / stats' errors) for unusable input.
    An equity curve whose points are unevenly spaced is refused; ``fill_flat=True`` instead
    repeats the last equity value on the missing days (you assert the account did not change).
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise CheckupError("seed must be a whole number of at least 0, got %r" % (seed,))
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)) or block < 1:
        raise CheckupError("block must be a whole number of at least 1, got %r" % (block,))
    names, m, dates, synthetic, sources, filled = _prepare(source, select, load_options, fill_flat)
    _check_spans(sources, isinstance(source, ReturnMatrix))
    trials, given = _check_trials(n_trials, m.shape[1])
    _check_matrix(names, m)
    j = _select(names, m, select)
    r = m[:, j]
    ppy = _periods_per_year(dates, synthetic, m.shape[0])
    active = int(np.count_nonzero(r))
    tau, n_eff, block_used = _dis.dependence(r, block)
    short, sparse = n_eff < SHORT_OBS, active * n_eff / m.shape[0] < MIN_ACTIVE  # active / tau when dependent
    n_obs = max(n_eff, 3.0) if tau >= stats.TAU_IGNORE else None
    cov = _dis.coverage(sources, names, m.shape[0])
    row = None if cov is None else cov[j]
    basis = None if sources is None else sources[names[j]]["basis"]
    capped = short or sparse or _dis.too_much_dropped(row)
    ts = [stats.sharpe(m[:, k]) for k in range(m.shape[1])] if m.shape[1] > 1 else ()
    opts = {"n_resamples": n_resamples, "block": block_used, "seed": seed, "ppy": ppy, "tau": tau}
    metrics = [_m_sharpe(r, capped, ppy), _m_drawdown(r, basis), _m_psr(r, capped, n_obs),
               _m_dsr(r, trials, ts, capped, not given, n_obs),
               _m_bootstrap(r, capped, opts), _pbo_metric(m, capped), _neff_metric(m, trials)]
    return Report(metrics=metrics, names=names, selected=names[j], n_obs=int(m.shape[0]),
                  n_results=int(m.shape[1]), n_trials=trials, trials_given=given, short=short, sparse=sparse,
                  active=active, periods_per_year=ppy,
                  start=None if synthetic else str(dates[0]), end=None if synthetic else str(dates[-1]),
                  synthetic=synthetic, seed=seed, n_resamples=n_resamples, block=block,
                  thresholds=dict(THRESHOLDS), min_obs=MIN_OBS, short_obs=SHORT_OBS,
                  filled=filled, coverage=cov, basis=basis, tau=tau, n_eff=n_eff, block_used=block_used,
                  bases=None if sources is None else {k: sources[k]["basis"] for k in names},
                  notes=_notes(m, trials, given, (short, sparse, active, ppy, n_obs and n_eff), names[j], filled)
                  + _dis.coverage_notes(row) + _dis.basis_notes(basis)
                  + _dis.dependence_notes(tau, n_eff, m.shape[0], block_used))


def _periods_per_year(dates, synthetic, n):
    """Observations per year, from the dates: n / span. Daily data (within 5%) gives the default."""
    if synthetic or dates is None or n < 2:
        return stats.PERIODS_PER_YEAR
    span = int((dates[-1] - dates[0]).astype("int64")) + 1
    ppy = n * stats.PERIODS_PER_YEAR / span
    return stats.PERIODS_PER_YEAR if abs(ppy / stats.PERIODS_PER_YEAR - 1.0) < 0.05 else ppy


def _notes(m, trials, given, flags, selected, filled=None):
    short, sparse, active, ppy, n_eff = flags
    notes = []
    if not given:
        notes.append("The number of trials was not given, so N = %d (%s). Versions you tried but did not "
                     "save also count: pass the real total with --trials (or n_trials=).%s"
                     % (trials, "the number of results" if m.shape[1] > 1 else "a single result",
                        " Until it is given, the overall verdict is at most CAUTION." if m.shape[1] > 1 else ""))
    if m.shape[1] > 1:
        notes.append("Sharpe, PSR, DSR and the bootstrap describe the result with the highest Sharpe "
                     "(%s), the one you would be tempted to use." % selected)
    if ppy != stats.PERIODS_PER_YEAR:
        notes.append("The dates are not one per calendar day: %d observations over the span give %.1f per "
                     "year, which is used to annualise the Sharpe ratio. For daily data, fill days without "
                     "trades with 0." % (m.shape[0], ppy))
    if filled:
        notes.append("Filled %d missing days by repeating the last equity value (assumes the account did not "
                     "change on those days).%s" % (sum(filled.values()), "" if len(filled) < 2 else
                     " Per result: " + ", ".join("%s %d" % kv for kv in filled.items()) + "."))
    if sparse:
        first = ("Only %d of the returns are non-zero (fewer than %d)" % (active, MIN_ACTIVE) if n_eff is None else
                 "Only about %d independent non-zero returns (%d non-zero rows, serially dependent; fewer than %d)"
                 % (_dis.floor_to(active * n_eff / m.shape[0]), active, MIN_ACTIVE))
        notes.append(first + ": a strategy with so few active periods cannot be judged reliably, so no line "
                     "can be a pass.")
    if m.shape[1] > 1:
        notes.append("PSR and the bootstrap interval are not corrected for choosing the best of several "
                     "results; DSR and PBO are. Read those two first.")
    if short and n_eff is not None:
        notes.append("Only about %d independent observations (%d rows, serially dependent; fewer than %d): the "
                     "Sharpe estimate is noisy, so no line can be a pass." % (_dis.floor_to(n_eff), m.shape[0], SHORT_OBS))
    elif short:
        notes.append("Only %d observations (fewer than %d): the Sharpe estimate is noisy (its standard "
                     "error is above 1.4 at 180 days), so no line can be a pass." % (m.shape[0], SHORT_OBS))
    return notes


if __name__ == "__main__":
    sys.exit(main())
