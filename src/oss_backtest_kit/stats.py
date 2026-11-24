"""Statistics for judging backtest results.

Implements, following Bailey and Lopez de Prado:

* Sharpe ratio (per period and annualised)
* Probabilistic Sharpe Ratio (PSR)
* Deflated Sharpe Ratio (DSR), which corrects for trying many configurations
* Probability of Backtest Overfitting via CSCV (PBO)
* effective number of independent trials
* circular block bootstrap of a return series (drawdown / growth, and Sharpe)
* Wilson score interval for a proportion

All inputs are per-period simple returns (for example daily). Any non-finite
input raises :class:`StatsError` rather than silently producing a number.
"""

import itertools
import math

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329
PERIODS_PER_YEAR = 365.25  # calendar days; crypto trades every day
TAU_IGNORE = 1.5  # autocorrelation time below this: rows treated as independent

__all__ = [
    "StatsError", "sharpe", "sharpe_annualised", "max_drawdown", "psr",
    "expected_max_sharpe", "null_sharpe_variance", "dsr", "n_effective",
    "pbo_cscv", "pbo_cscv_detail", "block_bootstrap", "block_bootstrap_sharpe", "wilson",
    "autocorr_time", "TAU_IGNORE",
]


class StatsError(ValueError):
    """Raised for empty, non-finite or otherwise unusable input."""


def _finite_array(x, name):
    a = np.asarray(x, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise StatsError("%s must be non-empty and finite" % name)
    return a


def _is_flat(r, axis=0):
    """True when the sample std is rounding noise next to the data's own scale.

    A constant series of decimals leaves a residual of about eps * |value| after
    the mean is subtracted; ``std == 0`` misses it. The test is relative to the
    largest absolute value, so rescaling the whole series does not change it.
    """
    r = np.asarray(r, dtype=float)
    flat = r.std(axis=axis, ddof=0) <= 16 * np.finfo(float).eps * np.abs(r).max(axis=axis)
    return bool(flat) if flat.ndim == 0 else flat


def _drawdown_growth(x):
    """Return ``(max drawdown, log of final equity)`` from log equity (no overflow)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logeq = np.cumsum(np.log1p(x))
    peak = np.maximum.accumulate(np.concatenate([[0.0], logeq]))[1:]
    return float((-np.expm1(logeq - peak)).max()), float(logeq[-1])


def sharpe(returns):
    """Per-period Sharpe ratio (mean / sample std, zero risk-free rate).

    Returns 0.0 for fewer than two observations or zero variance (judged by
    a tolerance relative to the data's scale, not by ``std == 0``).
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2 or _is_flat(r):
        return 0.0
    return float(r.mean() / r.std(ddof=1))


def sharpe_annualised(returns, periods_per_year=PERIODS_PER_YEAR):
    """Sharpe ratio scaled by sqrt(periods_per_year)."""
    return sharpe(returns) * math.sqrt(periods_per_year)


def max_drawdown(returns):
    """Largest peak-to-trough fall of the compounded equity curve, in [0, 1].

    Computed on log equity, so a long series whose absolute equity would
    overflow still gives the correct relative fall.
    """
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    return _drawdown_growth(r)[0]


def _check_n_obs(n_obs):
    if n_obs is not None and not (isinstance(n_obs, (int, float, np.number)) and 3 <= n_obs < math.inf):
        raise StatsError("n_obs must be a number of at least 3, got %r" % (n_obs,))


def psr(returns, benchmark_sharpe=0.0, n_obs=None):
    """Probabilistic Sharpe Ratio.

    Probability that the true per-period Sharpe ratio exceeds
    ``benchmark_sharpe``, accounting for sample length, skewness and kurtosis.
    Returns NaN when there are fewer than 3 observations or zero variance.
    ``n_obs`` (>= 3, an effective sample size) replaces the row count in sqrt(T - 1).
    """
    _check_n_obs(n_obs)
    r = np.asarray(returns, dtype=float)
    t = r.size
    if t < 3 or _is_flat(r):
        return float("nan")
    sr = sharpe(r)
    z = (r - r.mean()) / r.std(ddof=0)
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # raw (non-excess) kurtosis
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return float(norm.cdf((sr - benchmark_sharpe) * math.sqrt((t if n_obs is None else n_obs) - 1) / denom))


def expected_max_sharpe(n_trials, variance):
    """Expected maximum Sharpe among ``n_trials`` independent null trials.

    ``variance`` is the cross-trial variance of the Sharpe estimates.
    """
    if n_trials < 2 or variance <= 0:
        return 0.0
    return math.sqrt(variance) * (
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def null_sharpe_variance(n_obs):
    """Sampling variance of a per-period Sharpe estimate under a null of zero."""
    return 1.0 / max(1, n_obs - 1)


def dsr(returns, n_trials, trial_sharpes=(), n_obs=None):
    """Deflated Sharpe Ratio.

    PSR of ``returns`` measured against the expected best Sharpe obtainable by
    luck after ``n_trials`` attempts. The cross-trial variance is taken from
    ``trial_sharpes`` (per-period Sharpe of each tried configuration) when at
    least two are given; the larger of that and the null sampling variance is
    used, which is the conservative choice. ``n_obs`` goes to the null variance and :func:`psr`.

    Returns ``(dsr_value, benchmark_sharpe, variance_used)``.
    """
    _check_n_obs(n_obs)
    r = _finite_array(returns, "returns")
    ts = np.asarray(trial_sharpes, dtype=float)
    if ts.size and not np.isfinite(ts).all():
        raise StatsError("trial_sharpes must be finite")
    if n_trials < 1:
        raise StatsError("n_trials must be >= 1, got %r" % (n_trials,))
    var = float(np.var(ts, ddof=1)) if ts.size >= 2 else 0.0
    if n_trials >= 2:
        var = max(var, null_sharpe_variance(r.size if n_obs is None else n_obs))
    bench = expected_max_sharpe(n_trials, var)
    return psr(r, bench, n_obs), bench, var


def n_effective(matrix):
    """Effective number of independent trials from a (periods x trials) matrix.

    Uses the mean pairwise correlation rho: ``N / (1 + (N - 1) * rho)``,
    clipped to ``[1, N]``. Raises :class:`StatsError` for non-finite input,
    fewer than 3 periods, or a trial with zero variance (its correlation is
    undefined; drop or fix such trials explicitly before calling).
    """
    m = _finite_array(matrix, "matrix")
    if m.ndim != 2:
        raise StatsError("matrix must be 2-D (periods x trials)")
    n = m.shape[1]
    if n < 2:
        return 1.0
    if m.shape[0] < 3:
        raise StatsError("need at least 3 periods, got %d" % m.shape[0])
    flat = np.flatnonzero(np.ptp(m, axis=0) == 0)
    if flat.size:
        raise StatsError("trials with zero variance (columns %s)" % flat.tolist())
    c = np.corrcoef(m.T)
    rho = (c.sum() - n) / (n * (n - 1))
    if not math.isfinite(rho):
        raise StatsError("mean correlation is not finite")
    rho = min(max(rho, 0.0), 1.0)
    return float(min(n, max(1.0, n / (1.0 + (n - 1) * rho))))


def _col_sharpe(x):
    """Per-column Sharpe of a (rows x trials) slice; flat columns (see ``_is_flat``) give 0."""
    sd = x.std(0, ddof=1)
    flat = _is_flat(x)
    return np.where(flat, 0.0, x.mean(0) / np.where(flat, 1.0, sd))


def pbo_cscv(matrix, n_blocks=8):
    """Probability of Backtest Overfitting by combinatorially symmetric CV.

    ``matrix`` has shape (periods, trials). Rows are cut into ``n_blocks``
    contiguous blocks; for each way of choosing half of them as in-sample, the
    in-sample best trial is located and its out-of-sample relative rank
    omega in (0, 1) is computed (average rank among ties). PBO is the share of
    splits with omega <= 0.5. Pure noise gives roughly 0.5. Returns ``None``
    with fewer than two trials.

    Ties for the best in-sample Sharpe are broken by the larger in-sample
    return sum; if that also ties, the split cannot tell the trials apart and
    is left out of the share (:func:`pbo_cscv_detail` also returns how many).
    Returns ``None`` if every split is left out. Rescaling the whole matrix
    does not change the result (above a tiny scale, about 1e-11).
    """
    return pbo_cscv_detail(matrix, n_blocks)[0]


def pbo_cscv_detail(matrix, n_blocks=8):
    """Like :func:`pbo_cscv` but returns ``(pbo, indistinguishable_splits)``."""
    m = _finite_array(matrix, "matrix")
    if m.ndim != 2:
        raise StatsError("matrix must be 2-D (periods x trials)")
    t, n = m.shape
    if n < 2:
        return None, 0
    if n_blocks % 2 or n_blocks < 4 or t < 2 * n_blocks:
        raise StatsError("need even n_blocks >= 4 and periods >= 2 * n_blocks")
    blocks = np.array_split(np.arange(t), n_blocks)
    low = total = skipped = 0
    for chosen in itertools.combinations(range(n_blocks), n_blocks // 2):
        ins = np.concatenate([blocks[i] for i in chosen])
        oos = np.concatenate([blocks[i] for i in range(n_blocks) if i not in chosen])
        sr_in = _col_sharpe(m[ins])
        sr_out = _col_sharpe(m[oos])
        tol = 1e-12
        tied = np.flatnonzero(sr_in >= sr_in.max() - tol * max(1.0, abs(sr_in.max())))
        if tied.size > 1:
            sums = m[ins].sum(0)[tied]
            top = tied[sums >= sums.max() - tol * max(1.0, abs(sums.max()))]
            if top.size > 1:
                skipped += 1
                continue
            tied = top
        best = int(tied[0])
        less = int((sr_out < sr_out[best]).sum())
        equal = int((sr_out == sr_out[best]).sum())  # includes the winner itself
        omega = (less + (equal + 1) / 2.0) / (n + 1)
        total += 1
        low += omega <= 0.5
    return (low / total if total else None), skipped


def wilson(successes, n, z=1.959963984540054):
    """Wilson score interval ``[low, high]`` for a binomial proportion (95% default)."""
    if n <= 0:
        raise StatsError("n must be positive")
    p = successes / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def block_bootstrap(returns, n_resamples=2000, block=5, seed=0,
                    loss_threshold=None, periods_per_year=PERIODS_PER_YEAR):
    """Circular block bootstrap of a return series.

    Resamples blocks of ``block`` consecutive returns (wrapping around the end)
    to rebuild series of the original length, then summarises the drawdown and
    annualised growth distribution. Block resampling keeps short-range
    autocorrelation that plain resampling would destroy.

    ``loss_threshold`` is optional. When given (a positive fraction such as
    0.05), the result also reports the share of resamples containing at least
    one single-period return <= -loss_threshold, with a Wilson interval.
    """
    r = _finite_array(returns, "returns")
    t = r.size
    if block < 1 or t < 2 * block or n_resamples < 1:
        raise StatsError("series too short: periods=%d block=%d" % (t, block))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(t / block)
    offsets = np.arange(block)
    mdd, growth, hits = [], [], 0
    for _ in range(n_resamples):
        starts = rng.integers(0, t, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % t).ravel()[:t]
        x = r[idx]
        dd, log_final = _drawdown_growth(x)
        mdd.append(dd)
        with np.errstate(over="ignore"):
            growth.append(float(np.expm1(log_final * periods_per_year / t)))
        if loss_threshold is not None:
            hits += bool((x <= -float(loss_threshold)).any())
    q = lambda a, p: float(np.quantile(a, p))
    out = {
        "n_resamples": n_resamples, "block": block, "seed": seed, "periods": t,
        "mdd_median": q(mdd, 0.5), "mdd_p95": q(mdd, 0.95),
        "growth_p05": q(growth, 0.05), "growth_median": q(growth, 0.5),
        "loss_threshold": loss_threshold,
        "p_loss_exceeded": None, "p_loss_exceeded_wilson95": None,
    }
    if loss_threshold is not None:
        out["p_loss_exceeded"] = hits / n_resamples
        out["p_loss_exceeded_wilson95"] = wilson(hits, n_resamples)
    return out


def block_bootstrap_sharpe(returns, n_resamples=2000, block=5, seed=0, level=0.95,
                           periods_per_year=PERIODS_PER_YEAR):
    """Percentile interval of the annualised Sharpe ratio, by circular block bootstrap.

    Same resampling scheme as :func:`block_bootstrap`. Returns a dict with
    ``point`` (Sharpe of the original series), ``low`` and ``high`` (the
    ``level`` central interval of the resampled Sharpe ratios), plus the
    settings used.
    """
    r = _finite_array(returns, "returns")
    t = r.size
    if block < 1 or t < 2 * block or n_resamples < 1 or not 0.0 < level < 1.0:
        raise StatsError("bad bootstrap settings: periods=%d block=%d n_resamples=%d level=%r"
                         % (t, block, n_resamples, level))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(t / block)
    offsets = np.arange(block)
    out = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, t, n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % t).ravel()[:t]
        out[i] = sharpe_annualised(r[idx], periods_per_year)
    tail = (1.0 - level) / 2.0
    return {
        "point": sharpe_annualised(r, periods_per_year),
        "low": float(np.quantile(out, tail)), "high": float(np.quantile(out, 1.0 - tail)),
        "level": level, "n_resamples": n_resamples, "block": block, "seed": seed, "periods": t,
    }


def autocorr_time(returns):
    """Integrated autocorrelation time tau (rows per independent observation), a rough estimate.

    Geyer initial positive sequence: biased FFT autocorrelations rho_k (k < T / 2) summed in pairs
    rho_2m + rho_2m+1, stopping (m >= 1) at a pair <= 2 * sqrt(2 / T); tau = max(1, 2 * sum - 1).
    A flat series gives 1; non-finite input or fewer than 3 values is a StatsError.
    """
    r = _finite_array(returns, "returns")
    if r.ndim != 1 or r.size < 3:
        raise StatsError("need a 1-D series of at least 3 returns")
    if _is_flat(r):
        return 1.0
    t, n = r.size, 1 << (2 * r.size - 1).bit_length()  # zero padding: no circular wrap
    f = np.fft.rfft(r - r.mean(), n)
    acov = np.fft.irfft(f.real ** 2 + f.imag ** 2, n)[: (t + 1) // 2] / t
    rho = acov / acov[0]
    pairs = rho[: rho.size // 2 * 2].reshape(-1, 2).sum(1)  # Gamma_m = rho_2m + rho_2m+1
    low = np.flatnonzero(pairs[1:] <= 2.0 * math.sqrt(2.0 / t))  # stop at the first m >= 1 at the noise line
    return float(max(1.0, 2.0 * pairs[: 1 + low[0] if low.size else pairs.size].sum() - 1.0))
