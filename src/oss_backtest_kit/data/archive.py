"""Download public daily/monthly archive files from data.binance.vision.

Every zip is verified against the published ``.CHECKSUM`` (SHA-256) before it
is written or recorded. Files are never overwritten. Each file is written to a
temporary name, synced, and only then published under its final name, so a
final name never holds partial content. Verified downloads are listed in
``MANIFEST.csv`` inside the output directory.

Use of the data is subject to the exchange's public data terms; this module
only fetches what the caller asks for.
"""

import csv
import datetime as dt
import hashlib
import io
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from ._http import DataError, HttpClient

__all__ = ["BASE", "HOST", "KINDS", "MANIFEST_COLS", "DiskLowError", "atomic_create",
           "build_url", "download", "months", "write_new"]

HOST = "data.binance.vision"
BASE = "https://%s/data/futures/um" % HOST
MIN_FREE_BYTES = 2 * 1024 ** 3
MAX_UNZIPPED_BYTES = 1024 ** 3
MANIFEST_COLS = ["kind", "symbol", "interval", "freq", "period", "url", "sha256",
                 "download_utc", "bytes", "rows", "first_ts_ms", "last_ts_ms", "local_path"]

# kind -> (needs_interval, daily_only)
KINDS = {
    "klines": (True, False),
    "markPriceKlines": (True, False),
    "indexPriceKlines": (True, False),
    "premiumIndexKlines": (True, False),
    "fundingRate": (False, False),
    "metrics": (False, True),
}

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")
_INTERVAL_RE = re.compile(r"^[0-9]{1,2}[mhdwM]$")


class DiskLowError(DataError):
    """Free disk space would fall below the safety margin."""


def _validate(kind, symbol, interval):
    if kind not in KINDS:
        raise ValueError("unknown kind %r; choose from %s" % (kind, sorted(KINDS)))
    if not _SYMBOL_RE.match(symbol):
        raise ValueError("bad symbol %r" % symbol)
    if interval and not _INTERVAL_RE.match(interval):
        raise ValueError("bad interval %r; use a value such as 1h" % (interval,))
    if KINDS[kind][0] and not interval:
        raise ValueError("kind %s needs an interval such as 1h" % kind)


def build_url(kind, symbol, freq, period, interval=None):
    """URL of one archive zip. ``freq`` is ``monthly`` or ``daily``."""
    _validate(kind, symbol, interval)
    if freq not in ("monthly", "daily"):
        raise ValueError("freq must be monthly or daily")
    if KINDS[kind][0]:
        folder = "%s/%s/%s" % (kind, symbol, interval)
        name = "%s-%s-%s.zip" % (symbol, interval, period)
    else:
        folder = "%s/%s" % (kind, symbol)
        name = "%s-%s-%s.zip" % (symbol, kind, period)
    return "%s/%s/%s/%s" % (BASE, freq, folder, name)


def months(start, end):
    """Yield ``YYYY-MM`` strings from ``start`` to ``end`` inclusive."""
    y, m = map(int, start.split("-"))
    y2, m2 = map(int, end.split("-"))
    while (y, m) <= (y2, m2):
        yield "%04d-%02d" % (y, m)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _need_space(path, nbytes, min_free):
    p = Path(path)
    while not p.exists():
        p = p.parent
    if shutil.disk_usage(str(p)).free - nbytes < min_free:
        raise DiskLowError("free disk space too low to write %d bytes at %s" % (nbytes, p))


def atomic_create(path, data, min_free=0):
    """Publish ``data`` at ``path`` only once it is completely written.

    The bytes go to a temporary file in the same directory, are synced, and are
    then hard-linked to ``path``; the link fails with ``FileExistsError`` if
    ``path`` exists, so nothing is overwritten. The temporary name is always
    removed, so a failure leaves neither a partial final file nor a temp file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _need_space(path.parent, len(data), min_free)
    fd, tmp = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".part", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, str(path))
    finally:
        os.unlink(tmp)
    return path


def write_new(path, data, stamp, min_free=MIN_FREE_BYTES):
    """Create ``path`` exclusively and atomically; return the path that holds ``data``.

    If the file already exists with identical content it is kept. If it exists
    with different content, the data goes to a fresh sibling name. Nothing is
    ever overwritten or truncated.
    """
    path = Path(path)
    digest = hashlib.sha256(data).hexdigest()
    candidates = [path] + [
        path.with_name("%s.redl%s_%d%s" % (path.stem, stamp, i, path.suffix)) for i in range(100)
    ]
    for cand in candidates:
        try:
            return atomic_create(cand, data, min_free)
        except FileExistsError:
            if hashlib.sha256(cand.read_bytes()).hexdigest() == digest:
                return cand
    raise DataError("no free file name near %s" % path)


def _to_ms(value):
    value = value.strip()
    if value.isdigit():
        n = int(value)
        if n >= 10 ** 15:
            return n // 1000
        return n if n >= 10 ** 12 else n * 1000
    t = dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp() * 1000)


def _scan(blob):
    """Return ``(rows, first_ms, last_ms)`` of the first csv inside a zip."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        info = zf.infolist()[0]
        if info.file_size > MAX_UNZIPPED_BYTES:
            raise DataError("zip member too large: %d bytes" % info.file_size)
        rows, first, last = 0, None, None
        with zf.open(info) as fh:
            for i, rec in enumerate(csv.reader(io.TextIOWrapper(fh, "ascii"))):
                if not rec or (i == 0 and not rec[0].strip()[:1].isdigit()):
                    continue
                ms = _to_ms(rec[0])
                first = ms if first is None else first
                last = ms
                rows += 1
    return rows, first, last


def _parse_checksum(blob):
    words = blob.decode("ascii", "replace").split()
    if words and re.fullmatch(r"[0-9a-fA-F]{64}", words[0]):
        return words[0].lower()
    return None


def _done_urls(manifest):
    if not manifest.exists():
        return set()
    with open(manifest, newline="") as fh:
        return {r["url"] for r in csv.DictReader(fh)}


def _record(manifest, row, min_free):
    """Add one complete row: a new manifest is published atomically with its
    header; an existing one gets the row in a single append write."""
    _need_space(manifest.parent, 4096, min_free)
    buf = io.StringIO()
    csv.DictWriter(buf, MANIFEST_COLS).writerow(row)
    line = buf.getvalue().encode("utf-8")
    if not manifest.exists():
        head = io.StringIO()
        csv.DictWriter(head, MANIFEST_COLS).writeheader()
        try:
            atomic_create(manifest, head.getvalue().encode("utf-8") + line, min_free)
            return
        except FileExistsError:
            pass
    fd = os.open(str(manifest), os.O_WRONLY | os.O_APPEND)
    try:
        if os.write(fd, line) != len(line):
            raise DataError("short write to %s" % manifest)
        os.fsync(fd)
    finally:
        os.close(fd)


def fetch_one(client, out_dir, kind, symbol, freq, period, interval=None,
              done=None, min_free=MIN_FREE_BYTES, pause_s=0.0, sleep=time.sleep):
    """Download, verify and record one archive file.

    Returns ``skip`` (already in manifest), ``absent`` (404), ``ok``,
    ``no_checksum`` or ``mismatch``. Only ``ok`` files are saved and recorded.
    """
    out_dir = Path(out_dir)
    url = build_url(kind, symbol, freq, period, interval)
    done = _done_urls(out_dir / "MANIFEST.csv") if done is None else done
    if url in done:
        return "skip"
    got = client.get(url)
    sleep(pause_s)
    if got is None:
        return "absent"
    blob = got[0]
    ck = client.get(url + ".CHECKSUM")
    sleep(pause_s)
    want = _parse_checksum(ck[0]) if ck else None
    if want is None:
        return "no_checksum"
    digest = hashlib.sha256(blob).hexdigest()
    if digest != want:
        return "mismatch"
    rows, first, last = _scan(blob)
    if not KINDS[kind][0]:
        interval = None  # not part of this kind's URL, so not part of its path
    rel = Path(kind) / symbol / (interval or "-") / freq / url.rsplit("/", 1)[1]
    if out_dir.resolve() not in (out_dir / rel).resolve().parents:
        raise DataError("refusing to write outside %s" % out_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = write_new(out_dir / rel, blob, stamp, min_free)
    write_new(path.with_name(path.name + ".CHECKSUM"), ck[0], stamp, min_free)
    _record(out_dir / "MANIFEST.csv", {
        "kind": kind, "symbol": symbol, "interval": interval or "", "freq": freq,
        "period": period, "url": url, "sha256": digest,
        "download_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bytes": len(blob), "rows": rows, "first_ts_ms": first, "last_ts_ms": last,
        "local_path": path.relative_to(out_dir).as_posix()}, min_free)
    done.add(url)
    return "ok"


def download(kinds, symbols, start_month, end_month, out_dir, interval=None,
             client=None, today=None, min_free=MIN_FREE_BYTES, pause_s=0.12,
             sleep=time.sleep):
    """Download a date range. Monthly files first, daily files for months without one.

    ``start_month`` / ``end_month`` are ``YYYY-MM``. Daily fallback runs up to
    yesterday (UTC); ``metrics`` exists only as daily files. Returns a dict
    ``{(symbol, kind): {status: count}}``.
    """
    for kind in kinds:
        for symbol in symbols:
            _validate(kind, symbol, interval)
    client = client or HttpClient(HOST)
    out_dir = Path(out_dir)
    today = today or dt.datetime.now(dt.timezone.utc).date()
    yesterday = today - dt.timedelta(days=1)
    done = _done_urls(out_dir / "MANIFEST.csv")
    summary = {}
    for symbol in symbols:
        for kind in kinds:
            stat = summary.setdefault((symbol, kind), {})

            def bump(status):
                stat[status] = stat.get(status, 0) + 1

            for period in months(start_month, end_month):
                res = "absent" if KINDS[kind][1] else fetch_one(
                    client, out_dir, kind, symbol, "monthly", period, interval,
                    done, min_free, pause_s, sleep)
                if res != "absent":
                    bump(res)
                    continue
                day = dt.date(int(period[:4]), int(period[5:]), 1)
                while day.strftime("%Y-%m") == period and day <= yesterday:
                    bump(fetch_one(client, out_dir, kind, symbol, "daily", day.isoformat(),
                                   interval, done, min_free, pause_s, sleep))
                    day += dt.timedelta(days=1)
    return summary
