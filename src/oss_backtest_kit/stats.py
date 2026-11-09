"""Statistics for judging backtest results.

Implements, following Bailey and Lopez de Prado:

* Sharpe ratio (per period and annualised)
* Probabilistic Sharpe Ratio (PSR)
* Deflated Sharpe Ratio (DSR), which corrects for trying many configurations
* Probability of Backtest Overfitting via CSCV (PBO)
* effective number of independent trials
* circular block bootstrap of a return series
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

__all__ = [
    "StatsError", "sharpe", "sharpe_annualised", "max_drawdown", "psr",
    "expected_max_sharpe", "null_sharpe_variance", "dsr", "n_effective",
    "pbo_cscv", "block_bootstrap", "wilson",
]


class StatsError(ValueError):
    """Raised for empty, non-finite or otherwise unusable input."""


def _finite_array(x, name):
    a = np.asarray(x, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise StatsError("%s must be non-empty and finite" % name)
    return a


def sharpe(returns):
    """Per-period Sharpe ratio (mean / sample std, zero risk-free rate).

    Returns 0.0 for fewer than two observations or zero variance.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd)


def sharpe_annualised(returns, periods_per_year=PERIODS_PER_YEAR):
    """Sharpe ratio scaled by sqrt(periods_per_year)."""
    return sharpe(returns) * math.sqrt(periods_per_year)


def max_drawdown(returns):
    """Largest peak-to-trough fall of the compounded equity curve, in [0, 1]."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float((1.0 - eq / peak).max())


def psr(returns, benchmark_sharpe=0.0):
    """Probabilistic Sharpe Ratio.

    Probability that the true per-period Sharpe ratio exceeds
    ``benchmark_sharpe``, accounting for sample length, skewness and kurtosis.
    Returns NaN when there are fewer than 3 observations or zero variance.
    """
    r = np.asarray(returns, dtype=float)
    t = r.size
    if t < 3 or r.std(ddof=0) == 0:
        return float("nan")
    sr = sharpe(r)
    z = (r - r.mean()) / r.std(ddof=0)
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # raw (non-excess) kurtosis
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    return float(norm.cdf((sr - benchmark_sharpe) * math.sqrt(t - 1) / denom))


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


def dsr(returns, n_trials, trial_sharpes=()):
    """Deflated Sharpe Ratio.

    PSR of ``returns`` measured against the expected best Sharpe obtainable by
    luck after ``n_trials`` attempts. The cross-trial variance is taken from
    ``trial_sharpes`` (per-period Sharpe of each tried configuration) when at
    least two are given; the larger of that and the null sampling variance is
    used, which is the conservative choice.

    Returns ``(dsr_value, benchmark_sharpe, variance_used)``.
    """
    r = _finite_array(returns, "returns")
    ts = np.asarray(trial_sharpes, dtype=float)
    if ts.size and not np.isfinite(ts).all():
        raise StatsError("trial_sharpes must be finite")
    if n_trials < 1:
        raise StatsError("n_trials must be >= 1, got %r" % (n_trials,))
    var = float(np.var(ts, ddof=1)) if ts.size >= 2 else 0.0
    if n_trials >= 2:
        var = max(var, null_sharpe_variance(r.size))
    bench = expected_max_sharpe(n_trials, var)
    return psr(r, bench), bench, var


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


def pbo_cscv(matrix, n_blocks=8):
    """Probability of Backtest Overfitting by combinatorially symmetric CV.

    ``matrix`` has shape (periods, trials). Rows are cut into ``n_blocks``
    contiguous blocks; for each way of choosing half of them as in-sample, the
    in-sample best trial is located and its out-of-sample relative rank
    omega in (0, 1) is computed (average rank among ties). PBO is the share of
    splits with omega <= 0.5. Pure noise gives roughly 0.5. Returns ``None``
    with fewer than two trials.
    """
    m = _finite_array(matrix, "matrix")
    if m.ndim != 2:
        raise StatsError("matrix must be 2-D (periods x trials)")
    t, n = m.shape
    if n < 2:
        return None
    if n_blocks % 2 or n_blocks < 4 or t < 2 * n_blocks:
        raise StatsError("need even n_blocks >= 4 and periods >= 2 * n_blocks")
    blocks = np.array_split(np.arange(t), n_blocks)
    low = total = 0
    for chosen in itertools.combinations(range(n_blocks), n_blocks // 2):
        ins = np.concatenate([blocks[i] for i in chosen])
        oos = np.concatenate([blocks[i] for i in range(n_blocks) if i not in chosen])
        sr_in = m[ins].mean(0) / (m[ins].std(0, ddof=1) + 1e-18)
        sr_out = m[oos].mean(0) / (m[oos].std(0, ddof=1) + 1e-18)
        best = int(np.argmax(sr_in))
        less = int((sr_out < sr_out[best]).sum())
        equal = int((sr_out == sr_out[best]).sum())  # includes the winner itself
        omega = (less + (equal + 1) / 2.0) / (n + 1)
        total += 1
        low += omega <= 0.5
    return low / total


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
        eq = np.cumprod(1.0 + x)
        peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
        mdd.append((1.0 - eq / peak).max())
        growth.append(eq[-1] ** (periods_per_year / t) - 1.0 if eq[-1] > 0 else -1.0)
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
