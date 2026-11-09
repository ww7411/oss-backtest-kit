"""A small, config-driven backtest engine for a portfolio of perpetual futures.

Model (kept deliberately simple and explicit):

* Bars have a fixed length (``config.bar``, daily by default). Bar times must
  be exactly one bar length apart; a missing bar is a NaN row, not a gap.
* A strategy is a callable ``weights(t, history) -> {symbol: weight}``. It runs
  at the **close** of bar ``t`` and sees only bars ``0..t`` (inclusive).
* The weights are executed at the **open** of bar ``t + 1``. A weight is the
  position notional divided by equity; negative means short.
* A symbol is only traded when its target differs from the current weight by
  more than ``rebalance_threshold``.
* Costs have two parts: slippage moves the fill price against the trade, and a
  proportional fee is charged on the filled notional. Two named cost levels
  (``base`` and ``stressed``) are carried in the config.
* Funding: at each funding event the position pays ``quantity * price * rate``
  (longs pay when the rate is positive). An event at time ``t`` belongs to the
  bar with ``open_time <= t < open_time + bar length``; it is charged on the
  position held after that bar's opening fill, at that bar's open price.
  Events outside every loaded bar are dropped.
* Ruin: if equity at a bar close is zero or below, all positions are closed at
  that mark, equity stays at zero and nothing more is traded. The loss beyond
  zero is reported as ``ruin_shortfall`` so that per-symbol PnL plus the
  shortfall equals the change in equity.
* Equity is marked at each bar close. Positions in symbols with a missing bar
  are marked at their last known close and cannot be traded on that bar.

Only numpy is required. Loading from files uses the layout written by
:mod:`oss_backtest_kit.data.archive`.
"""

import csv
import datetime as dt
import io
import math
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import stats
from .data.check import interval_ms

__all__ = ["EngineError", "CostModel", "EngineConfig", "MarketData", "History",
           "BacktestResult", "run_backtest", "load_market_data"]


class EngineError(ValueError):
    """Invalid configuration, data or strategy output."""


@dataclass
class CostModel:
    """Proportional fee and slippage per unit of traded notional (0.0004 = 4 bp)."""

    fee_rate: float
    slippage: float

    def __post_init__(self):
        if self.fee_rate < 0 or self.slippage < 0:
            raise EngineError("costs must be non-negative")


def _default_costs():
    # Placeholder textbook values, not a recommendation: set your own.
    return {"base": CostModel(fee_rate=0.0004, slippage=0.0002),
            "stressed": CostModel(fee_rate=0.0010, slippage=0.0005)}


@dataclass
class EngineConfig:
    """Everything that defines a run apart from data and strategy."""

    symbols: list
    bar: str = "1d"                    # bar length, e.g. 1d, 4h
    start: str = None                  # first tracked day, YYYY-MM-DD (UTC); None = first bar
    end: str = None                    # exclusive end day; None = last bar
    initial_equity: float = 1.0
    max_leverage: float = 1.0          # cap on sum(|weight|)
    rebalance_threshold: float = 0.0   # skip trades smaller than this weight change
    funding: bool = True
    costs: dict = field(default_factory=_default_costs)

    @property
    def periods_per_year(self):
        return 365.25 * 86400000.0 / interval_ms(self.bar)


@dataclass
class MarketData:
    """Aligned price panels, shape (bars, symbols); NaN marks a missing bar.

    ``fund_px[i, c]`` is the sum over funding events inside bar ``i`` of
    ``rate * price``; a position of ``q`` units pays ``q * fund_px[i, c]``.
    """

    symbols: list
    times: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    fund_px: np.ndarray = None

    def __post_init__(self):
        n, m = len(self.times), len(self.symbols)
        for name in ("open", "high", "low", "close"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (n, m):
                raise EngineError("%s must have shape (%d, %d)" % (name, n, m))
            setattr(self, name, a)
        self.times = np.asarray(self.times, dtype=np.int64)
        if n > 1 and not (np.diff(self.times) > 0).all():
            raise EngineError("times must be strictly increasing")
        if self.fund_px is None:
            self.fund_px = np.zeros((n, m))
        self.fund_px = np.nan_to_num(np.asarray(self.fund_px, dtype=float))
        if self.fund_px.shape != (n, m):
            raise EngineError("fund_px must have shape (%d, %d)" % (n, m))
        if (self.open <= 0).any() or (self.close <= 0).any():
            raise EngineError("prices must be positive (use NaN for missing bars)")

    def truncated(self, n_bars):
        """The first ``n_bars`` bars as a new MarketData."""
        s = slice(0, n_bars)
        return MarketData(self.symbols, self.times[s], self.open[s], self.high[s],
                          self.low[s], self.close[s], self.fund_px[s])


class History:
    """Read-only view of bars ``0..t`` handed to the strategy."""

    def __init__(self, data, t):
        self.symbols = list(data.symbols)
        self.t = t
        for name in ("times", "open", "high", "low", "close"):
            view = getattr(data, name)[: t + 1]
            view = view.view()
            view.flags.writeable = False
            setattr(self, name, view)


@dataclass
class BacktestResult:
    """Outputs of one run. Arrays are per tracked bar."""

    symbols: list
    times: np.ndarray
    equity: np.ndarray
    returns: np.ndarray
    turnover: np.ndarray          # traded notional at the bar open / equity at that open
    fees: float
    slippage_cost: float
    funding_net: float            # positive = received
    pnl_by_symbol: dict           # net of fees, slippage and funding
    funding_by_symbol: dict
    periods_per_year: float
    initial_equity: float
    # Loss beyond zero equity at ruin, not borne by the account (limited liability):
    # sum(pnl_by_symbol) + ruin_shortfall == equity[-1] - initial_equity.
    ruin_shortfall: float = 0.0

    def summary(self):
        """Headline numbers; ``returns`` can be passed straight to stats.psr / stats.dsr."""
        r = self.returns
        years = len(r) / self.periods_per_year
        final = float(self.equity[-1])
        return {
            "bars": int(len(r)),
            "total_return": final / self.initial_equity - 1.0,
            "sharpe_annualised": stats.sharpe_annualised(r, self.periods_per_year),
            "max_drawdown": stats.max_drawdown(r),
            "psr_vs_zero": stats.psr(r, 0.0),
            "turnover_per_year": float(self.turnover.sum() / years) if years > 0 else 0.0,
            "fees": self.fees,
            "slippage_cost": self.slippage_cost,
            "funding_net": self.funding_net,
            "pnl_by_symbol": dict(self.pnl_by_symbol),
            "ruin_shortfall": self.ruin_shortfall,
        }


def _day_ms(text):
    d = dt.datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def _check_config(cfg, data):
    if list(cfg.symbols) != list(data.symbols):
        raise EngineError("config symbols must equal data symbols (same order)")
    if cfg.initial_equity <= 0 or cfg.max_leverage <= 0 or cfg.rebalance_threshold < 0:
        raise EngineError("initial_equity and max_leverage must be positive, threshold >= 0")
    try:
        step = interval_ms(cfg.bar)
    except ValueError as exc:
        raise EngineError(str(exc)) from None
    # Annualisation counts rows, so rows must be consecutive bars of cfg.bar.
    if len(data.times) > 1 and not (np.diff(data.times) == step).all():
        raise EngineError("bar times must be spaced exactly %s apart (config.bar); "
                          "fill missing bars with NaN rows" % cfg.bar)


def _targets(strategy, t, data, cfg):
    raw = strategy(t, History(data, t)) or {}
    w = np.zeros(len(cfg.symbols))
    index = {s: i for i, s in enumerate(cfg.symbols)}
    for sym, val in raw.items():
        if sym not in index:
            raise EngineError("strategy returned unknown symbol %r" % (sym,))
        if not math.isfinite(val):
            raise EngineError("non-finite weight for %s" % sym)
        w[index[sym]] = val
    if np.abs(w).sum() > cfg.max_leverage * (1 + 1e-9):
        raise EngineError("sum(|weights|)=%.6f exceeds max_leverage=%.6f"
                          % (np.abs(w).sum(), cfg.max_leverage))
    return w


def run_backtest(data, strategy, config, cost_level="base"):
    """Run ``strategy`` over ``data`` and return a :class:`BacktestResult`.

    Bars before ``config.start`` are history only (warm-up). The first trades
    fill at the open of the first tracked bar, using weights decided at the
    close of the bar before it (so a warm-up bar must exist for that).
    """
    _check_config(config, data)
    if cost_level not in config.costs:
        raise EngineError("unknown cost level %r; have %s" % (cost_level, sorted(config.costs)))
    cost = config.costs[cost_level]
    n, m = data.open.shape
    i0 = int(np.searchsorted(data.times, _day_ms(config.start))) if config.start else 0
    i1 = int(np.searchsorted(data.times, _day_ms(config.end))) if config.end else n
    if i1 - i0 < 1:
        raise EngineError("no bars in the requested range")

    cash, q = float(config.initial_equity), np.zeros(m)
    flow = np.zeros(m)
    fund_flow = np.zeros(m)
    last_close = np.full(m, np.nan)
    for i in range(i0):  # warm-up marks
        last_close = np.where(np.isnan(data.close[i]), last_close, data.close[i])
    fees = slip_cost = 0.0
    k = i1 - i0
    eq, ret, turn = np.zeros(k), np.zeros(k), np.zeros(k)
    prev_eq, ruined, shortfall = float(config.initial_equity), False, 0.0

    for j, i in enumerate(range(i0, i1)):
        if ruined:
            eq[j] = 0.0
            continue
        o = data.open[i]
        ok = ~np.isnan(o)
        mark = np.where(ok, o, last_close)
        e_open = cash + float(np.sum(q * np.nan_to_num(mark)))
        if i >= 1 and e_open > 0:
            tgt = _targets(strategy, i - 1, data, config)
            traded = 0.0
            for c in range(m):
                if not ok[c]:
                    continue
                cur = q[c] * o[c] / e_open
                if abs(tgt[c] - cur) <= config.rebalance_threshold or tgt[c] == cur:
                    continue
                dq = tgt[c] * e_open / o[c] - q[c]
                px = o[c] * (1 + cost.slippage) if dq > 0 else o[c] * (1 - cost.slippage)
                fee = abs(dq) * px * cost.fee_rate
                cash -= dq * px + fee
                flow[c] -= dq * px + fee
                fees += fee
                slip_cost += abs(dq) * o[c] * cost.slippage
                traded += abs(dq) * o[c]
                q[c] += dq
            turn[j] = traded / e_open
        if config.funding:
            pay = q * data.fund_px[i]
            cash -= float(pay.sum())
            flow -= pay
            fund_flow -= pay
        last_close = np.where(np.isnan(data.close[i]), last_close, data.close[i])
        eq[j] = cash + float(np.sum(q * np.nan_to_num(last_close)))
        if eq[j] <= 0:
            # Limited liability: positions are closed at the mark, equity stops at
            # zero and the loss beyond it is reported as ruin_shortfall.
            flow += q * np.nan_to_num(last_close)
            shortfall, q, cash = -eq[j], np.zeros(m), 0.0
            eq[j], ruined = 0.0, True
        ret[j] = eq[j] / prev_eq - 1.0 if prev_eq > 0 else 0.0
        prev_eq = eq[j]

    pnl = flow + q * np.nan_to_num(last_close)
    return BacktestResult(
        symbols=list(config.symbols), times=data.times[i0:i1].copy(), equity=eq, returns=ret,
        turnover=turn, fees=fees, slippage_cost=slip_cost, funding_net=float(fund_flow.sum()),
        pnl_by_symbol=dict(zip(config.symbols, map(float, pnl))),
        funding_by_symbol=dict(zip(config.symbols, map(float, fund_flow))),
        periods_per_year=config.periods_per_year, initial_equity=float(config.initial_equity),
        ruin_shortfall=float(shortfall))


# ---------------------------------------------------------------- loading

def _ms(value):
    n = int(value)
    return n // 1000 if n >= 10 ** 15 else n


def _manifest(directory):
    path = Path(directory) / "MANIFEST.csv"
    if not path.exists():
        raise EngineError("no MANIFEST.csv in %s" % directory)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _zip_records(directory, rel):
    with zipfile.ZipFile(Path(directory) / rel) as zf, zf.open(zf.namelist()[0]) as fh:
        for rec in csv.reader(io.TextIOWrapper(fh, "ascii")):
            if rec and rec[0].strip()[:1].isdigit():
                yield rec


def _kline_bars(directory, rows, sym, interval):
    """``{open_ms: [open, high, low, close]}``; the same bar in two files must agree."""
    recs = {}
    for r in rows:
        if r["kind"] == "klines" and r["symbol"] == sym and r["interval"] == interval:
            for rec in _zip_records(directory, r["local_path"]):
                t, vals = _ms(rec[0]), [float(x) for x in rec[1:5]]
                if recs.setdefault(t, vals) != vals:
                    raise EngineError("conflicting %s bars for %s at %d" % (interval, sym, t))
    if not recs:
        raise EngineError("no %s klines for %s in manifest" % (interval, sym))
    return recs


def _funding_events(directory, rows, sym):
    """``{event_ms: rate}``, one entry per event even if several files list it."""
    events = {}
    for r in rows:
        if r["kind"] != "fundingRate" or r["symbol"] != sym:
            continue
        for rec in _zip_records(directory, r["local_path"]):
            # archive layout: calc_time, interval_hours, last_funding_rate
            t, rate = _ms(rec[0]), float(rec[2])
            if events.setdefault(t, rate) != rate:
                raise EngineError("conflicting funding rates for %s at %d" % (sym, t))
    return events


def _funding_panel(directory, rows, symbols, times, step, open_, close):
    """Sum of ``rate * price`` per bar, with bar ``i`` covering ``[times[i], times[i] + step)``.

    Events before the first bar or at/after the end of the last bar belong to no
    loaded bar and are dropped (never moved onto an earlier bar). The price is the
    bar's open, or the symbol's most recent earlier close when that bar is missing.
    """
    fund_px = np.zeros((len(times), len(symbols)))
    for c, sym in enumerate(symbols):
        for t, rate in _funding_events(directory, rows, sym).items():
            if t < times[0] or t >= times[-1] + step:
                continue
            i = int((t - times[0]) // step)
            px = open_[i, c]
            if np.isnan(px):
                prev = close[:i, c][np.isfinite(close[:i, c])]
                if not prev.size:
                    continue  # no earlier price, so no position can be open
                px = prev[-1]
            fund_px[i, c] += rate * px
    return fund_px


def load_market_data(directory, symbols, interval="1d", funding=True):
    """Build :class:`MarketData` from files written by ``oss_backtest_kit.data.archive``.

    Reads ``klines`` (and ``fundingRate`` when ``funding`` is true) listed in the
    directory's manifest. Bars are placed on a regular ``interval`` grid from the
    first to the last bar time, so the time axis has no gaps; a symbol without a
    bar at some grid time gets NaN there. A bar listed in several files must have
    identical values, and a funding event listed in several files counts once.

    Only manifest files are read. CSV files written by ``data.rest_tail`` (under
    ``rest/``) are an export and are never loaded; a warning says so when they exist.
    """
    step = interval_ms(interval)
    rows = _manifest(directory)
    if (Path(directory) / "rest").is_dir():
        warnings.warn("REST tail files under %s/rest are not loaded; the data ends at the "
                      "last archive file in the manifest" % directory, stacklevel=2)
    bars = {sym: _kline_bars(directory, rows, sym, interval) for sym in symbols}
    t0 = min(min(b) for b in bars.values())
    t1 = max(max(b) for b in bars.values())
    times = np.arange(t0, t1 + step, step, dtype=np.int64)
    panels = [np.full((len(times), len(symbols)), np.nan) for _ in range(4)]
    for c, sym in enumerate(symbols):
        for t, vals in bars[sym].items():
            if (t - t0) % step:
                raise EngineError("%s bar at %d is not on the %s grid" % (sym, t, interval))
            for p in range(4):
                panels[p][(t - t0) // step, c] = vals[p]
    fund_px = (_funding_panel(directory, rows, symbols, times, step, panels[0], panels[3])
               if funding else np.zeros((len(times), len(symbols))))
    return MarketData(list(symbols), times, panels[0], panels[1], panels[2], panels[3], fund_px)
