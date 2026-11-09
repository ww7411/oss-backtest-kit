"""Offline tests for oss_backtest_kit.data. No network: a fake opener serves everything."""

import csv
import datetime as dt
import hashlib
import io
import os
import sys
import tempfile
import urllib.error
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from oss_backtest_kit.data import archive, check, rest_tail  # noqa: E402
from oss_backtest_kit.data import _http  # noqa: E402

HOUR = 3600000
T0 = 1700000000000 - (1700000000000 % HOUR)  # an hour boundary


def make_zip(bars, step=HOUR, skip=()):
    buf = io.StringIO()
    w = csv.writer(buf)
    for i in range(bars):
        if i in skip:
            continue
        t = T0 + i * step
        w.writerow([t, 1, 2, 0.5, 1.5, 10, t + step - 1, 15, 3, 5, 7, 0])
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("data.csv", buf.getvalue())
    return out.getvalue()


class FakeResp:
    def __init__(self, body, url, status=200, headers=None):
        self._body, self._url, self.status = body, url, status
        self.headers = headers or {}
        self._pos = 0

    def read1(self, n):
        chunk = self._body[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def geturl(self):
        return self._url

    def close(self):
        pass


class FakeServer:
    """Maps url -> bytes | ('http', code) | ('redirect', final_url)."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.calls.append(url)
        item = self.routes.get(url)
        if item is None:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        if isinstance(item, tuple) and item[0] == "http":
            raise urllib.error.HTTPError(url, item[1], "x", {}, None)
        if isinstance(item, tuple) and item[0] == "redirect":
            return FakeResp(b"x", item[1])
        return FakeResp(item, url)


def client_for(server, host=archive.HOST, **kw):
    kw.setdefault("sleep", lambda s: None)
    return _http.HttpClient(host, opener=server, **kw)


def served(url, blob, good=True):
    sha = hashlib.sha256(blob).hexdigest() if good else "0" * 64
    return {url: blob, url + ".CHECKSUM": ("%s  %s\n" % (sha, url.rsplit("/", 1)[1])).encode()}


def tmp():
    return tempfile.TemporaryDirectory()


def run_fetch(server, out, **kw):
    return archive.fetch_one(client_for(server), out, "klines", "BTCUSDT", "monthly",
                             "2023-11", "1h", min_free=0, **kw)


URL = archive.build_url("klines", "BTCUSDT", "monthly", "2023-11", "1h")


def test_build_url_shapes():
    assert URL.endswith("/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2023-11.zip")
    assert archive.build_url("fundingRate", "BTCUSDT", "monthly", "2023-11").endswith(
        "/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2023-11.zip")
    assert "/daily/metrics/BTCUSDT/BTCUSDT-metrics-2023-11-05.zip" in archive.build_url(
        "metrics", "BTCUSDT", "daily", "2023-11-05")
    for bad in (("nope", "BTCUSDT", "1h"), ("klines", "../x", "1h"), ("klines", "BTCUSDT", None)):
        try:
            archive.build_url(bad[0], bad[1], "monthly", "2023-11", bad[2])
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_download_verifies_and_records():
    blob = make_zip(48)
    with tmp() as d:
        res = run_fetch(FakeServer(served(URL, blob)), d)
        assert res == "ok"
        rows = list(csv.DictReader(open(Path(d) / "MANIFEST.csv", newline="")))
        assert len(rows) == 1 and rows[0]["rows"] == "48"
        assert rows[0]["sha256"] == hashlib.sha256(blob).hexdigest()
        assert (Path(d) / rows[0]["local_path"]).read_bytes() == blob
        # second run: skipped, no network
        srv = FakeServer({})
        assert run_fetch(srv, d) == "skip" and srv.calls == []
        assert check.check_manifest(d) == {"verified": 1, "problems": {}}


def test_wrong_checksum_rejected_not_recorded():
    with tmp() as d:
        res = run_fetch(FakeServer(served(URL, make_zip(5), good=False)), d)
        assert res == "mismatch"
        assert not (Path(d) / "MANIFEST.csv").exists()
        assert [p for p in Path(d).rglob("*") if p.is_file()] == []


def test_missing_checksum_rejected():
    with tmp() as d:
        res = run_fetch(FakeServer({URL: make_zip(5)}), d)
        assert res == "no_checksum" and not (Path(d) / "MANIFEST.csv").exists()


def test_absent_file():
    with tmp() as d:
        assert run_fetch(FakeServer({}), d) == "absent"


def test_redirect_refused():
    for routes in ({URL: ("http", 302)},
                   {URL: ("redirect", "https://example.org/other.zip")}):
        with tmp() as d:
            try:
                run_fetch(FakeServer(routes), d)
            except _http.RefusedError:
                pass
            else:
                raise AssertionError("redirect was not refused")
            assert not (Path(d) / "MANIFEST.csv").exists()


def test_wrong_host_and_scheme_refused():
    srv = FakeServer({})
    c = client_for(srv)
    for url in ("https://example.org/a.zip", "http://%s/a.zip" % archive.HOST,
                "https://%s:8443/a.zip" % archive.HOST,
                "https://%s.evil.test/a.zip" % archive.HOST,
                "https://user@%s/a.zip" % archive.HOST):
        try:
            c.get(url)
        except _http.RefusedError:
            continue
        raise AssertionError("not refused: " + url)
    assert srv.calls == []


def test_oversize_body_stops():
    srv = FakeServer({URL: b"x" * 5000})
    c = client_for(srv, max_bytes=1000)
    try:
        c.get(URL)
    except _http.TooLargeError:
        pass
    else:
        raise AssertionError("oversize body accepted")
    assert len(srv.calls) == 1  # final: not retried


def test_deadline_and_retry():
    ticks = iter(range(0, 10000, 100))
    srv = FakeServer({URL: b"x" * 200000})
    c = client_for(srv, deadline_s=50, retries=1, clock=lambda: next(ticks))
    try:
        c.get(URL)
    except _http.DataError:
        pass
    else:
        raise AssertionError("deadline not enforced")
    assert len(srv.calls) == 2  # transient: one retry
    c2 = client_for(FakeServer({URL: ("http", 500)}), retries=2)
    try:
        c2.get(URL)
    except _http.DataError:
        pass
    else:
        raise AssertionError("500 not raised")


def test_existing_file_never_overwritten():
    with tmp() as d:
        p = Path(d) / "a.zip"
        p.write_bytes(b"old")
        got = archive.write_new(p, b"new", "S", min_free=0)
        assert p.read_bytes() == b"old" and got != p and got.read_bytes() == b"new"
        assert archive.write_new(p, b"old", "S", min_free=0) == p  # same content kept
        assert len(list(Path(d).iterdir())) == 2


def test_disk_low_stops():
    with tmp() as d:
        try:
            archive.write_new(Path(d) / "a.zip", b"x", "S", min_free=1 << 60)
        except archive.DiskLowError:
            pass
        else:
            raise AssertionError("disk guard missing")
        assert list(Path(d).iterdir()) == []


def test_download_range_with_daily_fallback():
    routes = {}
    routes.update(served(archive.build_url("klines", "BTCUSDT", "monthly", "2024-01", "1h"), make_zip(24)))
    for day in ("2024-02-01", "2024-02-02"):
        routes.update(served(archive.build_url("klines", "BTCUSDT", "daily", day, "1h"), make_zip(24)))
    with tmp() as d:
        summary = archive.download(["klines"], ["BTCUSDT"], "2024-01", "2024-02", d, "1h",
                                   client=client_for(FakeServer(routes)),
                                   today=dt.date(2024, 2, 3), min_free=0, pause_s=0,
                                   sleep=lambda s: None)
        assert summary[("BTCUSDT", "klines")] == {"ok": 3}
        assert len(list(csv.DictReader(open(Path(d) / "MANIFEST.csv", newline="")))) == 3


def _store(d, blob, period="2023-11"):
    return run_fetch(FakeServer(served(archive.build_url("klines", "BTCUSDT", "monthly", period, "1h"), blob)), d)


def test_check_continuity_clean_and_missing_bar():
    with tmp() as d:
        _store(d, make_zip(48))
        rep = check.check_continuity(d, "klines", "BTCUSDT", "1h")
        assert rep["bars"] == 48 and rep["gaps"] == [] and rep["missing_bars"] == 0
        assert rep["bad_close_time"] == 0 and rep["duplicates"] == 0
    with tmp() as d:
        _store(d, make_zip(48, skip=(10, 11, 30)))
        rep = check.check_continuity(d, "klines", "BTCUSDT", "1h")
        assert rep["bars"] == 45 and rep["missing_bars"] == 3 and len(rep["gaps"]) == 2
        assert rep["gaps"][0] == (T0 + 9 * HOUR, T0 + 12 * HOUR)


def test_check_manifest_detects_tampering():
    with tmp() as d:
        _store(d, make_zip(10))
        row = next(csv.DictReader(open(Path(d) / "MANIFEST.csv", newline="")))
        with open(Path(d) / row["local_path"], "ab") as fh:
            fh.write(b"junk")
        probs = check.check_manifest(d)["problems"]
        assert probs.get("sha_vs_manifest") == 1 and probs.get("sha_vs_checksum_file") == 1


def test_rest_tail_unknown_route_fails_loudly():
    import json
    routes = {rest_tail.BASE + "/fapi/v1/time": json.dumps({"serverTime": T0}).encode()}
    c = client_for(FakeServer(routes), host=rest_tail.HOST)
    with tmp() as d:
        try:
            rest_tail.run(["BTCUSDT"], ["klines"], "1h", "2023-11-14", d, client=c,
                          sleep=lambda s: None, pause_s=0)
        except _http.DataError:
            pass
        else:
            raise AssertionError("expected DataError")
        assert not (Path(d) / "rest").exists()


def test_rest_tail_happy_path():
    import json
    day = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    start = int(day.timestamp() * 1000)
    now = start + 5 * HOUR + 1000
    bars = [[start + i * HOUR, "1", "2", "0.5", "1.5", "1", start + (i + 1) * HOUR - 1, "1", 1, "1", "1", "0"]
            for i in range(6)]  # bar 5 closes after `now`
    base = rest_tail.BASE
    routes = {
        base + "/fapi/v1/time": json.dumps({"serverTime": now}).encode(),
        base + "/fapi/v1/klines?symbol=BTCUSDT&interval=1h&startTime=%d&limit=1000" % start:
            json.dumps(bars).encode(),
        base + "/fapi/v1/fundingRate?symbol=BTCUSDT&startTime=%d&limit=1000" % start:
            json.dumps([{"fundingTime": start + HOUR, "fundingRate": "0.0001", "markPrice": "1"}]).encode(),
    }
    srv = FakeServer(routes)
    c = client_for(srv, host=rest_tail.HOST)
    with tmp() as d:
        files = rest_tail.run(["BTCUSDT"], ["klines", "fundingRate"], "1h", "2024-01-01", d,
                              client=c, sleep=lambda s: None, pause_s=0)
        assert len(files) == 2
        rows = list(csv.reader(open(files[0], newline="")))
        assert len(rows) == 1 + 5  # header + closed bars only
        # wrong host for the REST client is refused
        try:
            client_for(FakeServer({}), host=rest_tail.HOST).get(URL)
        except _http.RefusedError:
            pass
        else:
            raise AssertionError("host not enforced")
        # same cut-off again: exclusive create refuses to overwrite
        try:
            rest_tail.run(["BTCUSDT"], ["klines"], "1h", "2024-01-01", d, client=c,
                          sleep=lambda s: None, pause_s=0)
        except FileExistsError:
            pass
        else:
            raise AssertionError("overwrote existing file")


def test_cli_check_and_errors():
    from oss_backtest_kit.data.__main__ import main
    with tmp() as d:
        assert main(["check", "--out", d]) == 0
        _store(d, make_zip(48, skip=(5,)))
        assert main(["check", "--out", d, "--kind", "klines", "--symbol", "BTCUSDT",
                     "--interval", "1h"]) == 1  # gap found
        assert main(["archive", "--kinds", "bogus", "--symbols", "BTCUSDT", "--from", "2024-01",
                     "--to", "2024-01", "--out", d]) == 2


def test_interval_cannot_escape_out_dir():
    blob = make_zip(4)
    url = archive.build_url("fundingRate", "BTCUSDT", "monthly", "2023-11")
    with tmp() as top:
        out = Path(top) / "out"
        for bad in ("../../../escaped", "1h/../../x"):
            try:
                archive.fetch_one(client_for(FakeServer(served(url, blob))), out, "fundingRate",
                                  "BTCUSDT", "monthly", "2023-11", bad, min_free=0)
            except ValueError:
                continue
            raise AssertionError("bad interval accepted: %r" % bad)
        assert list(Path(top).iterdir()) == []
        # a valid interval is ignored for kinds whose URL has none
        assert archive.fetch_one(client_for(FakeServer(served(url, blob))), out, "fundingRate",
                                 "BTCUSDT", "monthly", "2023-11", "1h", min_free=0) == "ok"
        row = next(csv.DictReader(open(out / "MANIFEST.csv", newline="")))
        assert row["interval"] == "" and row["local_path"].startswith("fundingRate/BTCUSDT/-/monthly/")


def test_failed_write_leaves_no_partial_file():
    real = archive.os.fsync

    def boom(fd):
        raise OSError("injected")

    with tmp() as d:
        target = Path(d) / "a.zip"
        archive.os.fsync = boom
        try:
            for call in (lambda: archive.write_new(target, b"complete", "S", min_free=0),
                         lambda: run_fetch(FakeServer(served(URL, make_zip(4))), d),
                         lambda: rest_tail._write(d, "BTCUSDT", "klines_1h", 0, 1, ["a"], [[1]])):
                try:
                    call()
                except OSError:
                    continue
                raise AssertionError("injected failure not raised")
        finally:
            archive.os.fsync = real
        assert [p for p in Path(d).rglob("*") if p.is_file()] == []
        assert archive.write_new(target, b"complete", "S", min_free=0) == target
        assert target.read_bytes() == b"complete"


def test_rest_pagination_must_advance_and_is_bounded():
    import json
    page = [[T0 + i * HOUR, "1", "2", "0.5", "1.5", "1", T0 + (i + 1) * HOUR - 1, "1", 1, "1", "1", "0"]
            for i in range(1000)]
    calls = []

    def same_page(req, timeout=None):  # ignores startTime: always the same full page
        calls.append(req.full_url)
        return FakeResp(json.dumps(page).encode(), req.full_url)

    c = client_for(same_page, host=rest_tail.HOST)
    try:
        rest_tail.fetch_klines(c, "BTCUSDT", "klines", "1h", T0, T0 + 10 ** 12,
                               sleep=lambda s: None, pause_s=0)
    except _http.DataError:
        pass
    else:
        raise AssertionError("repeated page accepted")
    assert len(calls) == 2

    def moving(req, timeout=None):  # full pages that do advance
        start = int(req.full_url.split("startTime=")[1].split("&")[0])
        body = [[start + i * HOUR] + r[1:] for i, r in enumerate(page)]
        return FakeResp(json.dumps(body).encode(), req.full_url)

    try:
        rest_tail.fetch_klines(client_for(moving, host=rest_tail.HOST), "BTCUSDT", "klines", "1h",
                               T0, T0 + 10 ** 12, max_pages=3, sleep=lambda s: None, pause_s=0)
    except _http.DataError:
        pass
    else:
        raise AssertionError("page cap not enforced")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, repr(exc))
    print("%d passed, %d failed" % (len(tests) - failed, failed))
    sys.exit(1 if failed else 0)
