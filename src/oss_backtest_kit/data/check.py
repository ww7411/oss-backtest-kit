"""Integrity and continuity checks for a directory filled by ``archive.download``.

Read-only: nothing is written. Two checks:

* :func:`check_manifest` - every manifest file exists, its SHA-256 equals the
  manifest value and the saved ``.CHECKSUM`` sidecar.
* :func:`check_continuity` - merged bars of one series have no duplicates, no
  gaps and consistent open/close times.
"""

import csv
import hashlib
import heapq
import io
import re
import zipfile
from pathlib import Path

__all__ = ["interval_ms", "check_manifest", "check_continuity", "check_funding_spacing"]

_UNITS = {"m": 60000, "h": 3600000, "d": 86400000, "w": 7 * 86400000}


def interval_ms(interval):
    """Bar length in ms for fixed-length intervals (m, h, d, w)."""
    m = re.fullmatch(r"([0-9]{1,2})([mhdw])", interval)
    if not m:
        raise ValueError("unsupported interval %r (monthly bars have no fixed length)" % interval)
    return int(m.group(1)) * _UNITS[m.group(2)]


def _ms(value):
    n = int(value)
    return n // 1000 if n >= 10 ** 15 else n


def _manifest_rows(out_dir):
    path = Path(out_dir) / "MANIFEST.csv"
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _records(out_dir, rel):
    with zipfile.ZipFile(Path(out_dir) / rel) as zf, zf.open(zf.namelist()[0]) as fh:
        for rec in csv.reader(io.TextIOWrapper(fh, "ascii")):
            if rec and rec[0].strip()[:1].isdigit():
                yield rec


def check_manifest(out_dir):
    """Return ``{"verified": n, "problems": {name: count}}``."""
    out_dir = Path(out_dir)
    rows = _manifest_rows(out_dir)
    problems = {}

    def bad(name):
        problems[name] = problems.get(name, 0) + 1

    for r in rows:
        p = out_dir / r["local_path"]
        if not p.is_file():
            bad("missing_file")
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != r["sha256"]:
            bad("sha_vs_manifest")
        ck = p.with_name(p.name + ".CHECKSUM")
        if not ck.is_file():
            bad("missing_checksum_file")
        elif digest != (ck.read_text().split() or [""])[0].lower():
            bad("sha_vs_checksum_file")
    return {"verified": len(rows), "problems": problems}


def check_continuity(out_dir, kind, symbol, interval):
    """Gap and shape report for one kline-type series listed in the manifest."""
    out_dir = Path(out_dir)
    step = interval_ms(interval)
    files = sorted((r for r in _manifest_rows(out_dir)
                    if r["kind"] == kind and r["symbol"] == symbol and r["interval"] == interval),
                   key=lambda r: int(r["first_ts_ms"] or 0))
    streams = (_records(out_dir, r["local_path"]) for r in files)
    prev, first, bars, dup, bad_close, gaps = None, None, 0, 0, 0, []
    for rec in heapq.merge(*streams, key=lambda rec: _ms(rec[0])):
        t = _ms(rec[0])
        if len(rec) > 6 and _ms(rec[6]) != t + step - 1:
            bad_close += 1
        if prev is not None:
            if t <= prev:
                dup += 1
                continue
            if t != prev + step:
                gaps.append((prev, t))
        first = t if first is None else first
        prev, bars = t, bars + 1
    missing = sum((b - a) // step - 1 for a, b in gaps)
    return {"files": len(files), "bars": bars, "first_ms": first, "last_ms": prev,
            "duplicates": dup, "gaps": gaps, "missing_bars": missing,
            "bad_close_time": bad_close}


def check_funding_spacing(out_dir, symbol):
    """Histogram of hours between funding events: ``{hours: count}``."""
    out_dir = Path(out_dir)
    files = sorted((r for r in _manifest_rows(out_dir)
                    if r["kind"] == "fundingRate" and r["symbol"] == symbol),
                   key=lambda r: int(r["first_ts_ms"] or 0))
    ts = sorted(_ms(rec[0]) for r in files for rec in _records(out_dir, r["local_path"]))
    hist = {}
    for a, b in zip(ts, ts[1:]):
        h = round((b - a) / 3600000.0, 2)
        hist[h] = hist.get(h, 0) + 1
    return hist
