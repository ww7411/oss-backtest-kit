"""Fetch the most recent bars from the public futures REST market-data API.

Fetches the days that are not yet in the archive and exports them as
standalone csv files under ``<out>/rest/``. These files are NOT added to
``MANIFEST.csv`` and are NOT read by ``engine.load_market_data``; a backtest on
a downloaded directory ends at the last archive file. Public endpoints only (no
keys), GET only, one allowed host. Only bars that closed before the server time
at fetch are kept. Each run writes new csv files (exclusive create, never
overwrites).
"""

import csv
import datetime as dt
import io
import json
import re
import time
from pathlib import Path

from ._http import DataError, HttpClient
from .archive import atomic_create

__all__ = ["HOST", "KLINE_PATHS", "fetch_klines", "fetch_funding", "run"]

HOST = "fapi.binance.com"
BASE = "https://%s" % HOST
WEIGHT_BUDGET = 1920  # 80% of a 2400/minute limit
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
KLINE_PATHS = {
    "klines": "/fapi/v1/klines",
    "markPriceKlines": "/fapi/v1/markPriceKlines",
    "indexPriceKlines": "/fapi/v1/indexPriceKlines",
    "premiumIndexKlines": "/fapi/v1/premiumIndexKlines",
}
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")
_INTERVAL_RE = re.compile(r"^[0-9]{1,2}[mhdwM]$")


def _get_json(client, path, params, sleep=time.sleep, pause_s=0.25):
    query = "&".join("%s=%s" % (k, v) for k, v in params.items())
    got = client.get(BASE + path + ("?" + query if query else ""))
    if got is None:
        raise DataError("not found: %s" % path)
    body, headers = got
    try:
        used = int(headers.get("X-MBX-USED-WEIGHT-1M") or 0)
    except (TypeError, ValueError, AttributeError):
        used = 0
    if used > WEIGHT_BUDGET:
        sleep(61)
    sleep(pause_s)
    return json.loads(body)


def server_time_ms(client, **kw):
    """Current server time in milliseconds."""
    return int(_get_json(client, "/fapi/v1/time", {}, **kw)["serverTime"])


PAGE = 1000
MAX_PAGES = 1000  # at most 1,000,000 rows per series and run


def _paginate(get_page, time_of, start_ms, now_ms, max_pages):
    """Collect pages from ``start_ms``. Each page must start at or after the cursor
    and be strictly increasing, so the cursor always moves forward; at most
    ``max_pages`` requests are made."""
    rows, t = [], int(start_ms)
    for _ in range(max_pages):
        if t >= now_ms:
            return rows
        data = get_page(t)
        if not data:
            return rows
        times = [int(time_of(d)) for d in data]
        if times[0] < t or any(a >= b for a, b in zip(times, times[1:])):
            raise DataError("page from %d is repeated or out of order; stopping" % t)
        rows += data
        t = times[-1] + 1
        if len(data) < PAGE:
            return rows
    raise DataError("more than %d pages; use a shorter date range" % max_pages)


def fetch_klines(client, symbol, kind, interval, start_ms, now_ms, max_pages=MAX_PAGES, **kw):
    """Closed bars (close_time < now_ms) from ``start_ms``; pages of 1000."""
    if kind not in KLINE_PATHS:
        raise ValueError("unknown kind %r; choose from %s" % (kind, sorted(KLINE_PATHS)))
    if not _SYMBOL_RE.match(symbol) or not _INTERVAL_RE.match(interval):
        raise ValueError("bad symbol or interval")
    key = "pair" if kind == "indexPriceKlines" else "symbol"
    rows = _paginate(lambda t: _get_json(client, KLINE_PATHS[kind], {
        key: symbol, "interval": interval, "startTime": t, "limit": PAGE}, **kw),
        lambda d: d[0], start_ms, now_ms, max_pages)
    return [d for d in rows if d[6] < now_ms]


def fetch_funding(client, symbol, start_ms, now_ms, max_pages=MAX_PAGES, **kw):
    """Funding events from ``start_ms`` up to ``now_ms``."""
    if not _SYMBOL_RE.match(symbol):
        raise ValueError("bad symbol")
    rows = _paginate(lambda t: _get_json(client, "/fapi/v1/fundingRate", {
        "symbol": symbol, "startTime": t, "limit": PAGE}, **kw),
        lambda d: d["fundingTime"], start_ms, now_ms, max_pages)
    return [(d["fundingTime"], d["fundingRate"], d.get("markPrice", ""))
            for d in rows if d["fundingTime"] < now_ms]


def _write(out_dir, symbol, name, start, now, cols, rows):
    path = Path(out_dir) / "rest" / symbol / ("%s_%d_%d.csv" % (name, start, now))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(list(cols) + ["fetched_ms"])
    for r in rows:
        w.writerow(list(r) + [now])
    return atomic_create(path, buf.getvalue().encode("utf-8"))


def _day_ms(text):
    d = dt.datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def run(symbols, kinds, interval, start_day, out_dir, end_day=None, client=None,
        include_funding=True, **kw):
    """Fetch tails for every symbol and kind; returns the list of files written.

    ``start_day`` / ``end_day`` are ``YYYY-MM-DD`` (UTC); ``end_day`` is
    exclusive. File names carry the cut-off time, so reruns never collide
    unless the cut-off is identical (then creation fails rather than overwrite).
    """
    client = client or HttpClient(HOST)
    start = _day_ms(start_day)
    written = []
    for symbol in symbols:
        now = server_time_ms(client, **kw)
        if end_day:
            now = min(now, _day_ms(end_day))
        for kind in kinds:
            if kind == "fundingRate":
                continue
            rows = fetch_klines(client, symbol, kind, interval, start, now, **kw)
            written.append(_write(out_dir, symbol, "%s_%s" % (kind, interval), start, now,
                                  KLINE_COLS, rows))
        if include_funding and "fundingRate" in kinds:
            rows = fetch_funding(client, symbol, start, now, **kw)
            written.append(_write(out_dir, symbol, "fundingRate", start, now,
                                  ["funding_time", "funding_rate", "mark_price"], rows))
    return written
