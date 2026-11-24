"""Tests for oss_backtest_kit.loaders (spec items B1.x in docs/SPEC_B.md)."""

import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from oss_backtest_kit import loaders, stats  # noqa: E402
from oss_backtest_kit.loaders import LoaderError  # noqa: E402

DATA = os.path.join(HERE, "data")
SINGLE = os.path.join(DATA, "freqtrade_single.json")
REAL_DIR = os.path.join(DATA, "real_freqtrade_2026_10")   # members of a real freqtrade 2026.10 result zip, see B1.22
REAL_STEM = "backtest-result-2026-11-23_14-38-07"
REAL_MEMBERS = (REAL_STEM + ".json", REAL_STEM + "_config.json", REAL_STEM + "_RsiLikeStrategy.py",
                REAL_STEM + "_market_change.feather", REAL_STEM + "_RsiLikeStrategy_wallet.feather")


def real_zip(folder):
    """Rebuild the zip freqtrade 2026.10 wrote: the files of `real_freqtrade_2026_10` are its five members, byte
    for byte, in the order the real zip lists them (results json, config json, strategy .py, market-change feather,
    wallet feather); the name is the real stem. Run: freqtrade 2026.10, Binance public 1h candles, spot, BTC/USDT and
    ETH/USDT, 1000 USDT dry-run wallet, stake 100 USDT; `freqtrade backtesting -c cfg.json --userdir ud
    --strategy RsiLikeStrategy --timerange 20261011-20261101 --export trades`."""
    z = folder / (REAL_STEM + ".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in REAL_MEMBERS:
            with open(os.path.join(REAL_DIR, m), "rb") as fh:
                zf.writestr(m, fh.read())
    return str(z)
FUT_DIR = os.path.join(DATA, "real_freqtrade_2026_10_futures")   # a real freqtrade 2026.10 futures result, see B1.22
FUT_STEM = "backtest-result-2026-11-24_03-53-16"
FUT_MEMBERS = (FUT_STEM + ".json", FUT_STEM + "_config.json", FUT_STEM + "_SmaCrossStrategy.py",
               FUT_STEM + "_market_change.feather", FUT_STEM + "_SmaCrossStrategy_wallet.feather")


def futures_zip(folder):
    """Rebuild the zip of the real futures run (Binance public 4h candles, BTC/USDT:USDT, long and short) from its members."""
    z = folder / (FUT_STEM + ".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in FUT_MEMBERS:
            with open(os.path.join(FUT_DIR, m), "rb") as fh:
                zf.writestr(m, fh.read())
    return str(z)


D = np.datetime64


def _doc(path=SINGLE):
    with open(path) as fh:
        return json.load(fh)


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return str(p)


def _strat(doc):
    return doc["strategy"]["SampleStrategy"]


# ---- B1.1 freqtrade json -> daily returns
def test_b1_1_freqtrade_json_daily_returns():
    s = loaders.load_freqtrade(SINGLE)
    assert s.dates[0] == D("2024-01-01") and s.dates[-1] == D("2024-01-07") and len(s) == 7
    exp = [1 / 1000, 0, 1 / 1001, 0, 0, 1.5 / 1002, 0]
    assert np.allclose(s.values, exp)
    assert s.name == "freqtrade_single" and not s.synthetic


def test_b1_1_span_extends_to_backtest_window_and_trade_dates():
    d = _doc()
    _strat(d)["backtest_start"] = "2023-12-30 00:00:00"
    _strat(d)["backtest_end"] = "2024-01-04 00:00:00"  # earlier than last close: last close wins
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        s = loaders.load_freqtrade(_write(__import__("pathlib").Path(t), "a.json", d))
    assert s.dates[0] == D("2023-12-30") and s.dates[-1] == D("2024-01-06")


def test_b1_1_starting_balance_override_and_dry_run_wallet_fallback(tmp_path):
    d = _doc()
    del _strat(d)["starting_balance"]
    p = _write(tmp_path, "w.json", d)
    assert np.isclose(loaders.load_freqtrade(p).values[0], 1 / 1000)  # dry_run_wallet
    del _strat(d)["dry_run_wallet"]
    p = _write(tmp_path, "n.json", d)
    with pytest.raises(LoaderError, match="starting_balance"):
        loaders.load_freqtrade(p)
    assert np.isclose(loaders.load_freqtrade(p, starting_balance=500.0).values[0], 1 / 500)


# ---- B1.2 zip
def test_b1_2_zip_equals_json(tmp_path):
    z = tmp_path / "backtest-result-x.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(SINGLE, "backtest-result-x.json")
        zf.writestr("backtest-result-x_SampleStrategy.py", "# strategy source")
        zf.writestr("backtest-result-x_market_change.feather", b"\x00")
    a, b = loaders.load_freqtrade(str(z)), loaders.load_freqtrade(SINGLE)
    assert np.array_equal(a.values, b.values) and np.array_equal(a.dates, b.dates)


def test_b1_2_zip_single_json_member_with_other_name_ok_and_ambiguity_errors(tmp_path):
    z = tmp_path / "r.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(SINGLE, "other.json")
        zf.writestr("other.meta.json", "{}")
    assert len(loaders.load_freqtrade(str(z))) == 7
    z2 = tmp_path / "amb.zip"
    with zipfile.ZipFile(z2, "w") as zf:
        zf.write(SINGLE, "a.json")
        zf.write(SINGLE, "b.json")
    with pytest.raises(LoaderError, match="which member"):
        loaders.load_freqtrade(str(z2))


def test_b1_2_bad_zip_errors(tmp_path):
    p = tmp_path / "bad.zip"
    p.write_bytes(b"not a zip")
    with pytest.raises(LoaderError, match="not a valid zip"):
        loaders.load_freqtrade(str(p))


# ---- B1.3 strategy selection
def test_b1_3_multi_strategy_selection():
    multi = os.path.join(DATA, "freqtrade_multi.json")
    with pytest.raises(LoaderError, match="AlphaSample, BetaSample"):
        loaders.load_freqtrade(multi)
    with pytest.raises(LoaderError, match="no strategy 'Nope'"):
        loaders.load_freqtrade(multi, strategy="Nope")
    b = loaders.load_freqtrade(multi, strategy="BetaSample")
    assert b.name == "freqtrade_multi:BetaSample"
    assert np.isclose(b.values.sum(), 2.0 / 500) and len(b) == 3  # Jan 2..4
    alls = loaders.load_freqtrade_all(multi)
    assert [s.name for s in alls] == ["freqtrade_multi:AlphaSample", "freqtrade_multi:BetaSample"]


# ---- B1.4 freqtrade error paths
def test_b1_4_unsupported_layouts(tmp_path):
    with pytest.raises(LoaderError, match="list-of-trades"):
        loaders.load_freqtrade(os.path.join(DATA, "freqtrade_legacy_list.json"))
    with pytest.raises(LoaderError, match="pointer"):
        loaders.load_freqtrade(_write(tmp_path, "p.json", {"latest_backtest": "x.zip"}))
    with pytest.raises(LoaderError, match="unknown layout"):
        loaders.load_freqtrade(_write(tmp_path, "u.json", {"trades": []}))
    with pytest.raises(LoaderError, match="non-empty object"):
        loaders.load_freqtrade(_write(tmp_path, "e.json", {"strategy": {}}))
    with pytest.raises(LoaderError, match="top-level JSON type"):
        loaders.load_freqtrade(_write(tmp_path, "s.json", '"text"'))


def test_b1_4_bad_files(tmp_path):
    with pytest.raises(LoaderError, match="empty file"):
        loaders.load_freqtrade(_write(tmp_path, "z.json", "  \n"))
    with pytest.raises(LoaderError, match="not valid JSON"):
        loaders.load_freqtrade(_write(tmp_path, "b.json", "{oops"))
    with pytest.raises(LoaderError, match="not a file"):
        loaders.load_freqtrade(str(tmp_path / "missing.json"))


def test_b1_4_bad_trades(tmp_path):
    def run(mutate, match):
        d = _doc()
        mutate(_strat(d))
        with pytest.raises(LoaderError, match=match):
            loaders.load_freqtrade(_write(tmp_path, "t.json", d))

    run(lambda s: s.update(trades=[]), "non-empty list")
    run(lambda s: s["trades"][1].pop("profit_abs"), "trade 1: missing required field 'profit_abs'")
    run(lambda s: s["trades"][0].pop("close_date"), "missing required field 'close_date'")
    run(lambda s: s["trades"][0].update(profit_abs="1.0"), "not a finite number")
    run(lambda s: s["trades"][0].update(profit_abs=float("nan")), "not a finite number")
    run(lambda s: s["trades"][0].update(close_date="yesterday"), "cannot parse time")
    run(lambda s: s["trades"][0].update(close_date=123.5), "expected a date string")
    run(lambda s: s["trades"][0].update(open_date="2024-02-01 00:00:00+00:00"), "open_date is after close_date")
    run(lambda s: s["trades"].__setitem__(0, 5), "not an object")
    run(lambda s: s.update(starting_balance=-5), "must be a positive number")
    run(lambda s: s.update(backtest_start="garbage"), "cannot parse time")


def test_b1_4_wiped_out_account_errors(tmp_path):
    d = _doc()
    _strat(d)["trades"][0]["profit_abs"] = -1000.0
    with pytest.raises(LoaderError, match="wiped out|100%"):
        loaders.load_freqtrade(_write(tmp_path, "w.json", d))


def test_b1_4_utc_day_assignment_uses_offset(tmp_path):
    d = _doc()
    _strat(d)["trades"] = [{"open_date": "2024-01-01 20:00:00-05:00", "close_date": "2024-01-01 22:00:00-05:00",
                            "profit_abs": 10.0}]  # 03:00 UTC on Jan 2
    s = loaders.load_freqtrade(_write(tmp_path, "o.json", d))
    assert s.dates[0] == D("2024-01-01") and s.values[1] == pytest.approx(0.01)


# ---- B1.5 / B1.6 CSV
def test_b1_5_csv_returns():
    s = loaders.load_csv(os.path.join(DATA, "returns.csv"))
    assert list(s.values) == [0.01, -0.02, 0.015, 0.0] and s.dates[0] == D("2024-01-01")


def test_b1_6_csv_equity():
    s = loaders.load_csv(os.path.join(DATA, "equity.csv"))
    assert s.dates[0] == D("2024-01-02") and len(s) == 3
    assert np.allclose(s.values, [0.01, -0.02, 0.015], atol=1e-6)


def test_b1_6_csv_options_epoch_intraday_and_explicit_columns(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("\ufefft,r2,eq\n1704067200,0.01,100\n1704070800000,0.02,101\n1704153600,0.0,102\n")
    s = loaders.load_csv(str(p), time_col="t", value_col="r2", kind="returns")
    assert list(s.dates) == [D("2024-01-01"), D("2024-01-02")]
    assert s.values[0] == pytest.approx(1.01 * 1.02 - 1)  # two rows on Jan 1 compound
    e = loaders.load_csv(str(p), time_col="t", value_col="eq", kind="equity")
    assert len(e) == 1 and e.values[0] == pytest.approx(102 / 101 - 1)  # Jan 1 keeps last (101)


# ---- B1.7 CSV errors
@pytest.mark.parametrize("text,match", [
    ("", "empty file"),
    ("date,return\n", "no data rows"),
    ("date,return\n2024-01-01,abc\n", "line 2: 'abc' is not a number"),
    ("date,return\n2024-01-01,nan\n", "not finite"),
    ("date,return\n2024-01-02,0.1\n2024-01-01,0.1\n", "not sorted"),
    ("date,return\nnot-a-date,0.1\n", "cannot parse time"),
    ("date,x\n2024-01-01,0.1\n", "no returns or equity column"),
    ("when,return\n2024-01-01,0.1\n", "cannot find a time column"),
    ("date,return,equity\n2024-01-01,0.1,5\n", "both a returns and an equity"),
    ("date,return\n2024-01-01\n", "row has 1 fields"),
    ("date,return\n2024-01-01,-1.5\n", "<= -100"),
    ("date,equity\n2024-01-01,100\n", "at least 2 points"),
    ("date,equity\n2024-01-01,100\n2024-01-02,0\n", "equity must be finite and > 0"),
])
def test_b1_7_csv_errors(tmp_path, text, match):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    with pytest.raises(LoaderError, match=match):
        loaders.load_csv(str(p))


def test_b1_7_csv_argument_errors(tmp_path):
    with pytest.raises(LoaderError, match="not a file"):
        loaders.load_csv(str(tmp_path / "none.csv"))
    with pytest.raises(LoaderError, match="kind must be"):
        loaders.load_csv(os.path.join(DATA, "returns.csv"), kind="weird")
    with pytest.raises(LoaderError, match="no column 'zzz'"):
        loaders.load_csv(os.path.join(DATA, "returns.csv"), time_col="zzz")
    with pytest.raises(LoaderError, match="must also say kind"):
        loaders.load_csv(os.path.join(DATA, "returns.csv"), value_col="return")


# ---- B1.8 arrays and Series
def test_b1_8_arrays_and_equity_arrays():
    s = loaders.from_returns([0.01, -0.01, 0.02], name="a")
    assert s.synthetic and len(s) == 3 and s.dates[1] - s.dates[0] == 1
    e = loaders.from_equity([100, 110, 99], name="e")
    assert np.allclose(e.values, [0.1, -0.1]) and len(e) == 2
    t = loaders.from_returns([0.1, 0.2], dates=["2024-03-01", "2024-03-02"])
    assert not t.synthetic and t.dates[0] == D("2024-03-01")
    with pytest.raises(LoaderError, match="3 values but 2 dates"):
        loaders.from_returns([1, 2, 3], dates=["2024-03-01", "2024-03-02"])
    with pytest.raises(LoaderError, match="1-D"):
        loaders.from_returns([[0.1, 0.2]])
    with pytest.raises(LoaderError, match="NaN"):
        loaders.from_returns([0.1, float("nan")])
    with pytest.raises(LoaderError, match="no observations"):
        loaders.from_returns([])


def test_b1_8_series_without_pandas_duck_typed():
    class FakeIndex:
        dtype = np.dtype("datetime64[ns]")
        tz = None

        def __init__(self, d):
            self.d = d

        def __array__(self, dtype=None, copy=None):
            return np.array(self.d, dtype="datetime64[ns]")

    class FakeSeries:
        def __init__(self, v, d):
            self.index, self._v = FakeIndex(d), np.array(v)

        def to_numpy(self, dtype=None):
            return self._v.astype(dtype or float)

    s = loaders.from_returns(FakeSeries([0.1, 0.2], ["2024-05-01", "2024-05-02"]), name="f")
    assert list(s.dates) == [D("2024-05-01"), D("2024-05-02")] and not s.synthetic


def test_b1_8_pandas_series_optional():
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    s = loaders.from_returns(pd.Series([0.1, 0.0, 0.2, 0.1], index=idx), name="p")
    assert s.dates[0] == D("2024-01-01") and len(s) == 4
    plain = loaders.from_returns(pd.Series([0.1, 0.2, 0.3]))  # RangeIndex -> synthetic
    assert plain.synthetic
    eq = loaders.from_equity(pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3)))
    assert np.allclose(eq.values, [1.0, 0.5])
    hourly = pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2024-01-01 22:00", periods=3, freq="h"))
    assert len(loaders.from_returns(hourly)) == 2  # Jan 1 has 2 rows, compounded


def test_b1_8_load_returns_dispatch(tmp_path):
    assert isinstance(loaders.load_returns([0.1, 0.2, 0.3]), loaders.DailyReturns)
    assert len(loaders.load_returns(os.path.join(DATA, "returns.csv"))) == 4
    assert len(loaders.load_returns(SINGLE)) == 7
    s = loaders.load_returns(SINGLE)
    assert loaders.load_returns(s) is s
    (tmp_path / "x.txt").write_text("a")
    with pytest.raises(LoaderError, match="unknown file type"):
        loaders.load_returns(str(tmp_path / "x.txt"))


# ---- B1.9 directory -> matrix -> pbo_cscv
def _make_dir(tmp_path, n=3, days=40, seed=1):
    rng = np.random.default_rng(seed)
    base = np.datetime64("2024-01-01")
    for i in range(n):
        lines = ["date,return"] + ["%s,%.6f" % (base + k, rng.normal(0, 0.01)) for k in range(days)]
        (tmp_path / ("trial_%02d.csv" % i)).write_text("\n".join(lines) + "\n")
    return str(tmp_path)


def test_b1_9_directory_matrix_feeds_pbo(tmp_path):
    m = loaders.load_directory(_make_dir(tmp_path))
    assert isinstance(m, loaders.ReturnMatrix) and m.matrix.shape == (40, 3)
    assert m.names == ("trial_00", "trial_01", "trial_02") and set(m.dropped.values()) == {0}
    p = stats.pbo_cscv(m.matrix)
    assert 0.0 <= p <= 1.0
    assert stats.n_effective(m.matrix) >= 1.0


def test_b1_9_directory_mixed_formats_and_sidecars(tmp_path):
    d = _make_dir(tmp_path, n=1, days=30)
    (tmp_path / "second.csv").write_text(open(os.path.join(d, "trial_00.csv")).read())
    (tmp_path / "x.meta.json").write_text("{}")
    (tmp_path / ".last_result.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("hi")
    m = loaders.load_directory(d)
    assert m.names == ("second", "trial_00") and m.skipped == (".last_result.json", "notes.txt", "x.meta.json")


def test_b1_9_directory_with_freqtrade_files(tmp_path):
    for n in ("one.json", "two.json"):
        (tmp_path / n).write_text(open(SINGLE).read())
    m = loaders.load_directory(str(tmp_path), min_periods=5)
    assert m.matrix.shape == (7, 2)
    multi = tmp_path / "three.json"
    multi.write_text(open(os.path.join(DATA, "freqtrade_multi.json")).read())
    with pytest.raises(LoaderError, match="common days|keeps only"):
        loaders.load_directory(str(tmp_path), min_periods=5)  # Beta covers 3 of Alpha's 7 days
    m2 = loaders.load_directory(str(tmp_path), min_periods=3, min_overlap=0.4)
    assert m2.matrix.shape == (3, 4) and m2.dropped["one"] == 4


# ---- B1.10 alignment errors
def _ser(name, start, n):
    return loaders.from_returns(np.full(n, 0.001) + np.arange(n) * 1e-5,
                                dates=[str(np.datetime64(start) + k) for k in range(n)], name=name)


def test_b1_10_alignment_errors(tmp_path):
    a, b = _ser("a", "2024-01-01", 30), _ser("b", "2024-01-11", 30)
    with pytest.raises(LoaderError, match="keeps only"):
        loaders.align_returns([a, b])  # 20 of 30 days = 67%
    ok = loaders.align_returns([a, b], min_overlap=0.6)
    assert ok.matrix.shape == (20, 2) and ok.dropped == {"a": 10, "b": 10}
    with pytest.raises(LoaderError, match="only 5 common days"):
        loaders.align_returns([_ser("a", "2024-01-01", 30), _ser("c", "2024-01-26", 30)], min_overlap=0.1)
    with pytest.raises(LoaderError, match="at least two series"):
        loaders.align_returns([a])
    with pytest.raises(LoaderError, match="duplicate series name"):
        loaders.align_returns([a, a])
    u1, u2 = loaders.from_returns([0.1] * 20, name="u1"), loaders.from_returns([0.1] * 21, name="u2")
    with pytest.raises(LoaderError, match="same length"):
        loaders.align_returns([u1, u2])
    with pytest.raises(LoaderError, match="undated"):
        loaders.align_returns([u1, a])
    assert loaders.align_returns([u1, loaders.from_returns([0.2] * 20, name="u3")]).matrix.shape == (20, 2)


def test_b1_10_directory_errors(tmp_path):
    with pytest.raises(LoaderError, match="not a directory"):
        loaders.load_directory(str(tmp_path / "nope"))
    with pytest.raises(LoaderError, match="found 0 usable"):
        loaders.load_directory(str(tmp_path))
    _make_dir(tmp_path, n=1)
    with pytest.raises(LoaderError, match="found 1 usable"):
        loaders.load_directory(str(tmp_path))
    (tmp_path / "bad.csv").write_text("date,return\n2024-01-01,oops\n")
    with pytest.raises(LoaderError, match="bad.csv line 2"):
        loaders.load_directory(str(tmp_path))


# ---- B1.11 dependencies and public-package hygiene
def test_b1_11_import_does_not_pull_pandas():
    code = ("import sys; sys.path.insert(0, %r); import oss_backtest_kit.loaders; "
            "sys.exit(1 if 'pandas' in sys.modules else 0)" % os.path.join(HERE, "..", "src"))
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# sha256 of the lowercase denied words; the words themselves live only in tools/denylist_words.txt, which is not packaged
DENY_HASHES = frozenset((
    "1469c2fefa17c016931bb9bf82d867b3aaef4f1fc3f72f53deafc73769c06554",
    "0f25c1c30f06fcf3b60a85c858950b4705582c206d24bb1b8456d43896e3e7d5",
    "72d50b8c62b9329f26012dbed25f75d2af29957345f12dcaf19862a36187d421",
    "2e9bc6c94a4cbdfe2a31d2df79103a5eb3702eaf5d7018d47a774e9540a8ec29",
    "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b",
))
DENY_LENGTHS = (6, 7, 10, 11)


def denied_hits(text, hashes=DENY_HASHES, lengths=DENY_LENGTHS):
    """Positions of denied words in `text`, by comparing the hash of every window of each denied length."""
    low = text.lower()
    return [i for n in lengths for i in range(len(low) - n + 1)
            if hashlib.sha256(low[i:i + n].encode()).hexdigest() in hashes]


def test_b1_11_denied_word_scan_matches_by_hash_only():
    probe = frozenset([hashlib.sha256(b"harmless").hexdigest()])   # a made-up entry, to prove the matcher works
    assert denied_hits("x Harmless y", probe, (8,)) == [2]
    assert denied_hits("harmless text, nothing denied") == []
    for h in DENY_HASHES:
        assert len(h) == 64 and h.islower()


def test_b1_11_loader_sources_are_plain_ascii_without_internal_names():
    root = os.path.join(HERE, "..", "src", "oss_backtest_kit", "loaders")
    for fn in os.listdir(root):
        if fn.endswith(".py"):
            text = open(os.path.join(root, fn), encoding="utf-8").read()
            assert text.isascii(), fn
            assert not denied_hits(text), fn


def test_b1_11_packaged_text_files_hold_no_denied_word():
    top = os.path.join(HERE, "..")
    files = [os.path.join(top, f) for f in ("README.md", "CHANGELOG.md", "MANIFEST.in", "pyproject.toml")]
    for base in ("src", "tests", "examples"):
        for dirpath, dirs, names in os.walk(os.path.join(top, base)):
            dirs[:] = [d for d in dirs if d not in ("audit_ce", "data", "__pycache__")]
            files += [os.path.join(dirpath, n) for n in names if n.endswith((".py", ".md", ".ipynb"))]
    assert len(files) > 20
    for f in files:
        assert not denied_hits(open(f, encoding="utf-8").read()), os.path.relpath(f, top)


def test_b1_11_test_extra_installs_pandas_and_manifest_leaves_tools_out():
    cfg = open(os.path.join(HERE, "..", "pyproject.toml"), encoding="utf-8").read()
    extras = cfg.split("[project.optional-dependencies]")[1].split("\n[")[0]
    for name in ("test", "dev"):
        line = [ln for ln in extras.splitlines() if ln.startswith(name + " =")][0]
        assert '"pandas' in line, name
    manifest = open(os.path.join(HERE, "..", "MANIFEST.in"), encoding="utf-8").read().split()
    assert "tools" not in manifest and not any(w.startswith("tools") for w in manifest)


def test_b1_11_samples_are_small():
    for fn in os.listdir(DATA):
        full = os.path.join(DATA, fn)
        if os.path.isdir(full):   # the one real freqtrade run (B1.22): all its members together stay under 200 KB
            assert sum(os.path.getsize(os.path.join(full, m)) for m in os.listdir(full)) < 200_000
        else:
            assert os.path.getsize(full) < 10_000, fn


def test_b1_12_returns_feed_stats_end_to_end():
    s = loaders.load_csv(os.path.join(DATA, "returns.csv"))
    assert math.isfinite(stats.sharpe_annualised(s.values))
    assert stats.max_drawdown(s.values) >= 0


def test_b1_13_spec_and_tests_agree():
    import re
    with open(os.path.join(HERE, "..", "docs", "SPEC_B.md"), encoding="utf-8") as fh:
        named = set(re.findall(r"`(test_b1_\w+)`", fh.read()))
    with open(__file__, encoding="utf-8") as fh:
        defined = set(re.findall(r"^def (test_b1_\w+)", fh.read(), re.M))
    assert named == defined


# ---- B1.14 native exceptions become LoaderError
def test_b1_14_zip_read_failures(tmp_path, monkeypatch):
    z = tmp_path / "r.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        zf.write(SINGLE, "r.json")
    raw = bytearray(z.read_bytes())
    at = raw.index(b'"SampleStrategy"')
    raw[at + 3] ^= 0x01  # corrupt member data, CRC no longer matches
    bad = tmp_path / "crc.zip"
    bad.write_bytes(bytes(raw))
    with pytest.raises(LoaderError, match="cannot read member"):
        loaders.load_freqtrade(str(bad))
    for exc in (RuntimeError("encrypted"), NotImplementedError("compression"), EOFError("eof")):
        def boom(self, *a, _e=exc, **k):
            raise _e
        monkeypatch.setattr(zipfile.ZipFile, "read", boom)
        with pytest.raises(LoaderError, match="cannot read member"):
            loaders.load_freqtrade(str(z))


def test_b1_14_json_extremes(tmp_path):
    with pytest.raises(LoaderError, match="nested too deeply|not valid JSON"):
        loaders.load_freqtrade(_write(tmp_path, "deep.json", "[" * 200000 + "]" * 200000))
    d = _doc()
    _strat(d)["trades"][0]["profit_abs"] = 10 ** 400
    with pytest.raises(LoaderError, match="profit_abs"):
        loaders.load_freqtrade(_write(tmp_path, "big.json", d))
    d = _doc()
    _strat(d)["starting_balance"] = 10 ** 400
    with pytest.raises(LoaderError, match="starting_balance"):
        loaders.load_freqtrade(_write(tmp_path, "bal.json", d))
    with pytest.raises(LoaderError, match="cannot read|not a file"):
        loaders.load_freqtrade(str(tmp_path))


def test_b1_14_csv_unreadable(tmp_path):
    p = tmp_path / "bin.csv"
    p.write_bytes(b"\xff\xfe\x00\x81\x80date,return\n")
    with pytest.raises(LoaderError, match="cannot read as UTF-8"):
        loaders.load_csv(str(p))
    q = tmp_path / "huge.csv"
    q.write_text("date,return\n2024-01-01," + "1" * 200000 + "\n")
    with pytest.raises(LoaderError, match="cannot read as UTF-8"):
        loaders.load_csv(str(q))


def test_b1_14_array_conversion_errors():
    for bad in (["a", "b"], [None, 1.0], [True, False], [[1], [2, 3]]):
        with pytest.raises(LoaderError):
            loaders.from_returns(bad)
        with pytest.raises(LoaderError):
            loaders.from_equity(bad)


# ---- B1.15 strict time parsing
@pytest.mark.parametrize("text", ["170406720", "20240101", "99999999999999", "1230767999", "\u00b2" * 10,
                                  "2024-13-01", "2024-01-01T10:00:00+25:00", "01/02/2024", "2024-1-1", "", "-5"])
def test_b1_15_rejected_time_text(text):
    from oss_backtest_kit.loaders._series import to_utc_date
    with pytest.raises(LoaderError):
        to_utc_date(text, "t")


def test_b1_15_accepted_time_forms():
    from oss_backtest_kit.loaders._series import to_utc_date
    assert to_utc_date("1230768000", "t") == D("2009-01-01")
    assert to_utc_date(1230768000 * 1000, "t") == D("2009-01-01")
    assert to_utc_date(1704067200000, "t") == D("2024-01-01")
    assert to_utc_date("2024-01-01T23:30:00-05:00", "t") == D("2024-01-02")
    assert to_utc_date("2024-01-01 10:00:00.123456+0530", "t") == D("2024-01-01")
    assert to_utc_date("2024-01-01T10:00Z", "t") == D("2024-01-01")
    with pytest.raises(LoaderError):
        to_utc_date(4102444800, "t")  # 2100-01-01 in seconds is outside the range
    with pytest.raises(LoaderError):
        to_utc_date(1230768000 * 1000 - 1, "t")  # between the two ranges


# ---- B1.16 ambiguous / case-insensitive CSV columns
@pytest.mark.parametrize("header,kw,match", [
    ("open_time,close_time,return", {}, "ambiguous time column"),
    ("date,return,returns", {}, "ambiguous returns column"),
    ("date,return,return", {}, "ambiguous returns column"),
    ("date,equity,balance", {}, "ambiguous equity column"),
    ("date,date,return", {"time_col": "date"}, "ambiguous"),
])
def test_b1_16_ambiguous_columns(tmp_path, header, kw, match):
    p = tmp_path / "a.csv"
    p.write_text(header + "\n" + ",".join(["2024-01-01"] + ["0.1"] * (header.count(",") )) + "\n")
    with pytest.raises(LoaderError, match=match):
        loaders.load_csv(str(p), **kw)


def test_b1_16_explicit_names_are_case_insensitive_and_resolve_ambiguity(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("Open_Time,Close_Time,Return\n2024-01-01,2024-01-02,0.1\n2024-01-02,2024-01-03,0.2\n")
    s = loaders.load_csv(str(p), time_col="close_time")
    assert s.dates[0] == D("2024-01-02")
    q = tmp_path / "d.csv"
    q.write_text("Date,Return\n2024-01-01,0.1\n")
    assert len(loaders.load_csv(str(q), time_col="date", value_col="RETURN", kind="returns")) == 1


def test_b1_16_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("date,return\n2024-01-01,0.1\n\n  ,  \n2024-01-02,0.2\n\n")
    assert list(loaders.load_csv(str(p)).values) == [0.1, 0.2]


# ---- B1.17 alignment thresholds, zip member choice, symlinks
def test_b1_17_min_periods_boundary():
    a15, b15 = _ser("a", "2024-01-01", 15), _ser("b", "2024-01-01", 15)
    with pytest.raises(LoaderError, match="only 15 common days.*need at least 16"):
        loaders.align_returns([a15, b15])
    a10, b10 = _ser("a", "2024-01-01", 10), _ser("b", "2024-01-01", 10)
    with pytest.raises(LoaderError, match="only 10 common days"):
        loaders.align_returns([a10, b10])
    assert loaders.align_returns([_ser("a", "2024-01-01", 16), _ser("b", "2024-01-01", 16)]).matrix.shape == (16, 2)


def test_b1_17_zip_prefers_same_name_member_and_ignores_subdirectories(tmp_path):
    z = tmp_path / "pick.zip"
    other = _doc()
    _strat(other)["trades"] = [{"close_date": "2024-02-01 00:00:00+00:00", "profit_abs": 1.0}]
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("zzz.json", json.dumps(other))
        zf.write(SINGLE, "pick.json")
    assert loaders.load_freqtrade(str(z)).dates[0] == D("2024-01-01")
    z2 = tmp_path / "sub.zip"
    with zipfile.ZipFile(z2, "w") as zf:
        zf.write(SINGLE, "top.json")
        zf.writestr("sub/inner.json", json.dumps(other))
    assert loaders.load_freqtrade(str(z2)).dates[0] == D("2024-01-01")


def test_b1_17_directory_does_not_follow_symlinks(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    _make_dir(d, n=2)
    outside = tmp_path / "outside.csv"
    outside.write_text((d / "trial_00.csv").read_text())
    os.symlink(str(outside), str(d / "link.csv"))
    m = loaders.load_directory(str(d))
    assert "link" not in m.names and "link.csv" in m.skipped


# ---- B1.18 return-domain guards, intraday order and undated alignment
def test_b1_18_same_day_domain_checks_run_before_merging(tmp_path):
    days = ["2025-03-10", "2025-03-10", "2025-03-11"]
    bad_returns = [-1.5, -1.5, 0.02]
    with pytest.raises(LoaderError):
        loaders.from_returns(np.array(bad_returns), days)
    ret_csv = tmp_path / "bad_ret.csv"
    ret_csv.write_text("date,return\n" + "".join("%s,%s\n" % (d, v) for d, v in zip(days, bad_returns)),
                       encoding="utf-8")
    with pytest.raises(LoaderError):
        loaders.load_csv(ret_csv)
    for bad_eq in ([float("nan"), 100.0, 102.0], [0.0, 100.0, 102.0], [-10.0, 100.0, 102.0]):
        with pytest.raises(LoaderError):
            loaders.from_equity(np.array(bad_eq), days)
        eq_csv = tmp_path / "bad_eq.csv"
        eq_csv.write_text("date,equity\n" + "".join("%s,%s\n" % (d, v) for d, v in zip(days, bad_eq)),
                          encoding="utf-8")
        with pytest.raises(LoaderError):
            loaders.load_csv(eq_csv)


def test_b1_18_intraday_order_offset_minutes_and_trade_times(tmp_path):
    rev_csv = tmp_path / "rev_equity.csv"
    rev_csv.write_text(
        "time,equity\n2025-04-05T18:30:00Z,100.0\n2025-04-05T09:15:00Z,95.0\n2025-04-06T12:00:00Z,105.0\n",
        encoding="utf-8",
    )
    with pytest.raises(LoaderError, match="not in time order"):
        loaders.load_csv(rev_csv)
    for bad_ts in ("2025-04-05T10:00:00+02:75", "2025-04-05T10:00:00-00:60"):
        off_csv = tmp_path / "bad_offset.csv"
        off_csv.write_text("time,return\n%s,0.01\n" % bad_ts, encoding="utf-8")
        with pytest.raises(LoaderError):
            loaders.load_csv(off_csv)
    trade_doc = {"strategy": {"Demo": {
        "starting_balance": 100.0,
        "trades": [{"open_date": "2025-04-05T16:00:00Z", "close_date": "2025-04-05T10:00:00Z", "profit_abs": 2.5}],
    }}}
    with pytest.raises(LoaderError, match="open_date is after close_date"):
        loaders.load_freqtrade(_write(tmp_path, "open_after_close.json", trade_doc))


def test_b1_18_rejects_invalid_2d_boolean_and_complex_returns():
    from oss_backtest_kit.checkup import CheckupError, checkup
    mat = np.tile([0.01, -0.005], (60, 1))
    mat = np.column_stack([mat[:, 0], mat[:, 1] + 0.002])
    mat[3, 0] = -1.25
    with pytest.raises(CheckupError):
        checkup(mat, n_trials=2, n_resamples=100)
    flags = np.tile([[True, False], [False, True]], (30, 1))
    with pytest.raises(CheckupError):
        checkup(flags, n_trials=2, n_resamples=100)
    with pytest.raises(LoaderError):
        loaders.from_returns(np.array([0.02 + 0.1j, -0.01 + 0.05j, 0.01 - 0.02j]))
    with pytest.raises(LoaderError):
        loaders.from_equity(np.array([100.0 + 1.0j, 101.0 + 2.0j]))
    cplx = np.column_stack([np.full(60, 0.01 + 0.02j), np.full(60, -0.01 + 0.01j)])
    with pytest.raises(CheckupError):
        checkup(cplx, n_trials=2, n_resamples=100)
    cplx_var = np.column_stack([
        np.tile([0.01, -0.005], 30) + 0.02j,
        np.tile([-0.004, 0.012], 30) + 0.01j,
    ])
    with pytest.raises(CheckupError, match="complex"):
        checkup(cplx_var, n_trials=2, n_resamples=100)


def test_b1_18_undated_returns_and_equity_align_by_position():
    r_vals = np.linspace(-0.02, 0.03, 24)
    eq_rets = np.linspace(0.025, -0.015, 24)
    eq_curve = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + eq_rets)])
    s_ret = loaders.from_returns(r_vals, name="ret_s")
    s_eq = loaders.from_equity(eq_curve, name="eq_s")
    for pair in ([s_ret, s_eq], [s_eq, s_ret]):
        aligned = loaders.align_returns(pair)
        assert aligned.matrix.shape == (24, 2) and aligned.dropped == {s.name: 0 for s in pair}
        np.testing.assert_allclose(aligned.matrix, np.column_stack([s.values for s in pair]))


# ---- B1.25 span limit and reconciliation
def test_b1_25_span_limit(tmp_path):
    d = _doc()
    _strat(d)["backtest_start"] = "1970-01-01 00:00:00"
    with pytest.raises(LoaderError, match="more than 20 years"):
        loaders.load_freqtrade(_write(tmp_path, "old.json", d))
    d = _doc()
    _strat(d)["backtest_end"] = "9999-12-31 00:00:00"
    with pytest.raises(LoaderError, match="more than 20 years"):
        loaders.load_freqtrade(_write(tmp_path, "far.json", d))


def test_b1_25_profit_total_reconciliation(tmp_path):
    d = _doc()
    _strat(d)["profit_total_abs"] = 3.5
    assert len(loaders.load_freqtrade(_write(tmp_path, "ok.json", d))) == 7
    _strat(d)["profit_total_abs"] = 4.5
    with pytest.raises(LoaderError, match="does not match"):
        loaders.load_freqtrade(_write(tmp_path, "bad.json", d))
    _strat(d)["profit_total_abs"] = "3.5"
    with pytest.raises(LoaderError, match="profit_total_abs is not a finite"):
        loaders.load_freqtrade(_write(tmp_path, "str.json", d))


def test_b1_25_duplicate_zip_member_and_duplicate_json_key(tmp_path):
    import warnings
    def _one(p):
        return {"strategy": {"Alpha": {
            "starting_balance": 100.0, "profit_total_abs": p,
            "trades": [{"close_date": "2025-05-01", "profit_abs": p}],
        }}}
    zpath = tmp_path / "dup.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("dup.json", json.dumps(_one(1.5)))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
            zf.writestr("dup.json", json.dumps(_one(-1.5)))
    with pytest.raises(LoaderError, match="more than once"):
        loaders.load_freqtrade(zpath)
    s1 = json.dumps(_one(1.5)["strategy"]["Alpha"])
    s2 = json.dumps(_one(-1.5)["strategy"]["Alpha"])
    dup_strat = _write(tmp_path, "dup_strat.json", '{"strategy":{"Alpha":%s,"Alpha":%s}}' % (s1, s2))
    with pytest.raises(LoaderError, match="duplicate key"):
        loaders.load_freqtrade(dup_strat)
    dup_field = _write(
        tmp_path, "dup_field.json",
        '{"strategy":{"Alpha":{"starting_balance":100.0,"profit_total_abs":1.5,"profit_total_abs":-1.5,'
        '"trades":[{"close_date":"2025-05-01","profit_abs":1.5}]}}}',
    )
    with pytest.raises(LoaderError, match="duplicate key"):
        loaders.load_freqtrade(dup_field)


def test_b1_25_reconciliation_tolerance_and_equity_overflow(tmp_path):
    turnover = {"strategy": {"Turnover": {
        "starting_balance": 10.0, "profit_total_abs": 0.0,
        "trades": [{"close_date": "2025-05-01", "profit_abs": 5e7},
                   {"close_date": "2025-05-02", "profit_abs": -(5e7 - 2.0)}],
    }}}
    with pytest.raises(LoaderError, match="does not match"):
        loaders.load_freqtrade(_write(tmp_path, "turnover.json", turnover))
    huge = {"strategy": {"Huge": {
        "starting_balance": 1e308,
        "trades": [{"close_date": "2025-05-01", "profit_abs": 1e308},
                   {"close_date": "2025-05-02", "profit_abs": 1e308}],
    }}}
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(LoaderError, match="overflow"):
            loaders.load_freqtrade(_write(tmp_path, "huge.json", huge))
    rel_ok = {"strategy": {"RelOk": {
        "starting_balance": 1_000_000.0, "profit_total_abs": 1_000_000.0,
        "trades": [{"close_date": "2025-05-01", "profit_abs": 500_000.0},
                   {"close_date": "2025-05-02", "profit_abs": 500_000.5}],
    }}}
    assert len(loaders.load_freqtrade(_write(tmp_path, "rel_ok.json", rel_ok))) == 2
    rel_bad = {"strategy": {"RelBad": {
        "starting_balance": 1_000_000.0, "profit_total_abs": 1_000_000.0,
        "trades": [{"close_date": "2025-05-01", "profit_abs": 500_000.0},
                   {"close_date": "2025-05-02", "profit_abs": 500_005.0}],
    }}}
    with pytest.raises(LoaderError, match="does not match"):
        loaders.load_freqtrade(_write(tmp_path, "rel_bad.json", rel_bad))


# ---- B1.19 Series indexes of text or periods
def test_b1_19_text_and_period_index_series():
    pd = pytest.importorskip("pandas")
    t = loaders.from_returns(pd.Series([0.1, 0.2], index=["2024-01-01", "2024-01-02"]))
    assert not t.synthetic and t.dates[1] == D("2024-01-02")
    p = loaders.from_returns(pd.Series([0.1, 0.2], index=pd.period_range("2024-01-01", periods=2, freq="D")))
    assert not p.synthetic and p.dates[0] == D("2024-01-01")
    with pytest.raises(LoaderError, match="boolean"):
        loaders.from_returns(pd.Series([True, False]))
    with pytest.raises(LoaderError, match="cannot parse time"):
        loaders.from_returns(pd.Series([0.1], index=["not a date"]))


# ---- B1.20 freqtrade versions: fixtures built after the official source of each version
def _field_check(path, trade_keys, strat_keys):
    doc = _doc(path)
    res = next(iter(doc["strategy"].values()))
    assert set(trade_keys) <= set(res["trades"][0]) and set(strat_keys) <= set(res)
    return res


def test_b1_20_fixtures_state_their_source_and_carry_the_official_fields():
    cases = (("freqtrade_2026_10_futures.json", "2026.10",
              ("pair", "open_date", "close_date", "profit_abs", "profit_ratio", "exit_reason", "enter_tag", "leverage",
               "is_short", "open_timestamp", "close_timestamp", "orders", "funding_fees"),
              ("trades", "profit_total_abs", "starting_balance", "dry_run_wallet", "backtest_start", "backtest_end",
               "final_balance", "trading_mode")),
             ("freqtrade_2022_1_spot.json", "2022.1",
              ("pair", "open_date", "close_date", "profit_abs", "sell_reason", "buy_tag"),
              ("trades", "profit_total_abs", "starting_balance", "dry_run_wallet", "backtest_start", "backtest_end")),
             ("freqtrade_2021_1_no_balance.json", "2021.1",
              ("pair", "open_date", "close_date", "profit_abs", "sell_reason"),
              ("trades", "profit_total_abs", "backtest_start", "backtest_end")))
    for fn, tag, tkeys, skeys in cases:
        path = os.path.join(DATA, fn)
        assert tag in _doc(path)["_fixture_note"] and "not an output of freqtrade" in _doc(path)["_fixture_note"]
        _field_check(path, tkeys, skeys)
    assert "starting_balance" not in _field_check(os.path.join(DATA, cases[2][0]), (), ())


def test_b1_20_current_layout_futures_with_short_and_extra_fields():
    s = loaders.load_freqtrade(os.path.join(DATA, "freqtrade_2026_10_futures.json"))
    assert s.dates[0] == D("2024-03-01") and s.dates[-1] == D("2024-03-12") and len(s) == 12
    # day 1: +12.5 on 1000; day 4 (short closes 03-04): -8 on 1012.5; ... equity at the start of each day
    eq = 1000.0
    exp = np.zeros(12)
    for day, pnl in ((0, 12.5), (3, -8.0), (4, 6.25), (7, 15.5), (10, -4.75)):
        exp[day] = pnl / eq
        eq += pnl
    assert np.allclose(s.values, exp)
    assert s.name == "freqtrade_2026_10_futures" and not s.synthetic


def test_b1_20_older_layouts_and_missing_starting_balance():
    s = loaders.load_freqtrade(os.path.join(DATA, "freqtrade_2022_1_spot.json"))
    assert s.dates[0] == D("2022-02-01") and s.dates[-1] == D("2022-02-10")
    assert np.isclose(s.values[0], 1.2 / 500) and np.isclose(s.values[2], -2.5 / 501.2)
    old = os.path.join(DATA, "freqtrade_2021_1_no_balance.json")
    with pytest.raises(LoaderError, match="starting_balance.*--starting-balance"):
        loaders.load_freqtrade(old)
    t = loaders.load_freqtrade(old, starting_balance=500)  # backtest_start carries +00:00
    assert t.dates[0] == D("2021-05-01") and np.isclose(t.values[0], 1.0 / 500)


def _official_zip(tmp_path, strategy="FuturesSample", stem="backtest-result-2024-03-12_10-00-00"):
    """A zip with the members freqtrade's store_backtest_results writes (2026.10), fake non-json contents."""
    doc = _doc(os.path.join(DATA, "freqtrade_2026_10_futures.json"))
    z = tmp_path / (stem + ".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(stem + ".json", json.dumps({"strategy": doc["strategy"], "strategy_comparison": doc["strategy_comparison"]}))
        zf.writestr(stem + "_config.json", json.dumps({"stake_currency": "USDT"}))
        zf.writestr("%s_%s.py" % (stem, strategy), "class %s: pass\n" % strategy)
        zf.writestr("%s_%s.json" % (stem, strategy), json.dumps({"strategy_name": strategy, "ft_stratparam_v": 1}))
        zf.writestr(stem + "_market_change.feather", b"not json")
        zf.writestr("%s_%s_wallet.feather" % (stem, strategy), b"not json")
    (tmp_path / (stem + ".meta.json")).write_text(json.dumps({strategy: {"run_id": "x"}}))
    (tmp_path / ".last_result.json").write_text(json.dumps({"latest_backtest": stem + ".zip"}))
    return z


def test_b1_20_zip_with_the_members_freqtrade_writes(tmp_path):
    z = _official_zip(tmp_path)
    s = loaders.load_freqtrade(str(z))
    ref = loaders.load_freqtrade(os.path.join(DATA, "freqtrade_2026_10_futures.json"))
    assert np.array_equal(s.values, ref.values) and s.name == z.stem
    # a folder of such a zip plus its sidecars: only the result counts
    with pytest.raises(LoaderError, match="found 1 usable"):
        loaders.load_directory(str(tmp_path))
    with pytest.raises(LoaderError, match="pointer"):
        loaders.load_freqtrade(str(tmp_path / ".last_result.json"))
    # without the same-name member, the strategy-parameter json makes the choice ambiguous: an error, not a guess
    z2 = tmp_path / "renamed.zip"
    with zipfile.ZipFile(z, "r") as src, zipfile.ZipFile(z2, "w") as dst:
        for n in src.namelist():
            dst.writestr(n, src.read(n))
    with pytest.raises(LoaderError, match="cannot tell which member"):
        loaders.load_freqtrade(str(z2))


def test_b1_20_multi_strategy_zip_reads_as_several_results(tmp_path):
    doc = _doc(os.path.join(DATA, "freqtrade_2026_10_futures.json"))
    second = json.loads(json.dumps(doc["strategy"]["FuturesSample"]))
    second["trades"] = second["trades"][:3]
    second["profit_total_abs"] = sum(t["profit_abs"] for t in second["trades"])
    doc["strategy"]["OtherSample"] = second
    stem = "backtest-result-2024-03-12_11-00-00"
    z = tmp_path / (stem + ".zip")
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(stem + ".json", json.dumps(doc))
    alls = loaders.load_freqtrade_all(str(z))
    assert [x.name for x in alls] == [stem + ":FuturesSample", stem + ":OtherSample"]
    with pytest.raises(LoaderError, match="--strategy"):
        loaders.load_freqtrade(str(z))
    assert loaders.load_freqtrade(str(z), strategy="OtherSample").name == stem + ":OtherSample"


# ---- B1.21 messages that a command line user can act on; DataFrame shaped input
def test_b1_21_loader_errors_name_the_command_line_switch(tmp_path):
    multi = os.path.join(DATA, "freqtrade_multi.json")
    with pytest.raises(LoaderError, match=r"--strategy NAME"):
        loaders.load_freqtrade(multi)
    p = tmp_path / "x.csv"
    p.write_text("date,pnl\n2024-01-01,0.1\n2024-01-02,0.2\n")
    with pytest.raises(LoaderError, match=r"--value-col NAME --kind"):
        loaders.load_csv(str(p))
    q = tmp_path / "y.csv"
    q.write_text("date,return,equity\n2024-01-01,0.1,100\n2024-01-02,0.2,101\n")
    with pytest.raises(LoaderError, match=r"--kind returns\|equity"):
        loaders.load_csv(str(q))
    r = tmp_path / "z.csv"
    r.write_text("when,return\n2024-01-01,0.1\n2024-01-02,0.2\n")
    with pytest.raises(LoaderError, match=r"--time-col"):
        loaders.load_csv(str(r))


def test_b1_21_several_columns_error_says_what_to_do():
    with pytest.raises(LoaderError, match="folder.*to_numpy"):
        loaders.from_returns(np.zeros((40, 3)))


# ---- B1.22 a result written by a real freqtrade run
def test_b1_22_real_freqtrade_2026_10_zip_reads_and_reconciles(tmp_path):
    """Members of a real zip from freqtrade 2026.10 (command line in `real_zip`), 99 trades."""
    assert sorted(os.listdir(REAL_DIR)) == sorted(REAL_MEMBERS)
    # nothing of the machine that ran it, no internal names, no CJK in any member
    banned = ("/Us" + "ers/", "/ho" + "me/", "/pri" + "vate/")
    names = DENY_HASHES - {"2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b"}   # all but the generic word that freqtrade configs hold as an empty key
    for m in REAL_MEMBERS:
        raw = open(os.path.join(REAL_DIR, m), "rb").read()
        assert not any(b.encode() in raw for b in banned), m
        assert not denied_hits(raw.decode("latin-1"), names), m
        if m.endswith((".json", ".py")):
            assert raw.decode("ascii")
    path = real_zip(tmp_path)
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == list(REAL_MEMBERS)
        doc = json.loads(zf.read(REAL_STEM + ".json"))
    (name, res), = doc["strategy"].items()
    assert name == "RsiLikeStrategy" and res["trading_mode"] == "spot" and len(res["trades"]) == 99
    s = loaders.load_freqtrade(path)
    assert s.name == REAL_STEM and not s.synthetic
    assert s.dates[0] == D("2026-10-11") and s.dates[-1] == D("2026-11-01") and len(s.values) == 22
    assert np.all(np.isfinite(s.values)) and np.count_nonzero(s.values) > 10
    # the daily returns compound to the realised profit the file states (starting balance 1000)
    equity = float(res["starting_balance"]) * np.prod(1.0 + np.asarray(s.values))
    assert abs(equity - (res["starting_balance"] + res["profit_total_abs"])) < 1e-6 * res["starting_balance"]
    assert [x.name for x in loaders.load_freqtrade_all(path)] == [REAL_STEM]
    assert loaders.load_freqtrade(path, strategy=name).name == REAL_STEM


def test_b1_22_real_freqtrade_2026_10_futures_zip_with_shorts_reads_and_reconciles(tmp_path):
    """Members of a real freqtrade 2026.10 futures result (isolated margin, 2x leverage, long and short), 12 trades."""
    assert sorted(os.listdir(FUT_DIR)) == sorted(FUT_MEMBERS)
    banned = ("/Us" + "ers/", "/ho" + "me/", "/pri" + "vate/")
    names = DENY_HASHES - {"2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b"}
    for m in FUT_MEMBERS:
        raw = open(os.path.join(FUT_DIR, m), "rb").read()
        assert not any(b.encode() in raw for b in banned), m
        assert not denied_hits(raw.decode("latin-1"), names), m
    path = futures_zip(tmp_path)
    with zipfile.ZipFile(path) as zf:
        assert zf.namelist() == list(FUT_MEMBERS)
        doc = json.loads(zf.read(FUT_STEM + ".json"))
    (name, res), = doc["strategy"].items()
    assert name == "SmaCrossStrategy" and res["trading_mode"] == "futures" and len(res["trades"]) == 12
    assert sum(1 for t in res["trades"] if t["is_short"]) >= 1
    s = loaders.load_freqtrade(path)   # also checks the daily series against profit_total_abs
    assert s.name == FUT_STEM and not s.synthetic and s.basis == "realised"
    assert s.dates[0] == D("2026-10-10") and s.dates[-1] == D("2026-11-21") and len(s.values) == 43
    assert np.all(np.isfinite(s.values)) and np.count_nonzero(s.values) > 5
    equity = float(res["starting_balance"]) * np.prod(1.0 + np.asarray(s.values))
    assert abs(equity - res["final_balance"]) < 1e-6 * res["starting_balance"]
    assert abs(res["final_balance"] - (res["starting_balance"] + res["profit_total_abs"])) < 1e-6
    assert loaders.load_freqtrade(path, strategy=name).name == FUT_STEM


# ---- B1.23 / B1.24 where each series came from
def _equity_csv(path, days, equity):
    path.write_text("date,equity\n" + "".join("%s,%r\n" % (d, float(e)) for d, e in zip(days, equity)))
    return str(path)


def test_b1_23_spans_and_basis(tmp_path):
    days = D("2024-01-01") + np.array([0, 1, 2, 5, 6, 13])
    eq = [1.0, 1.01, 1.02, 1.03, 1.05, 1.06]
    s = loaders.from_equity(eq, days)
    assert s.basis == "equity" and list(s.spans) == [1, 1, 3, 1, 7]
    c = loaders.load_csv(_equity_csv(tmp_path / "e.csv", days, eq))
    assert c.basis == "equity" and list(c.spans) == [1, 1, 3, 1, 7]
    r = loaders.from_returns([0.01, -0.01, 0.02], D("2024-01-01") + np.arange(3))
    assert r.basis == "returns" and r.spans is None
    assert loaders.from_equity(eq).spans.tolist() == [1] * 5  # undated: one period per row
    ft = loaders.load_freqtrade(SINGLE)
    assert ft.basis == "realised" and ft.spans is None
    with pytest.raises(LoaderError, match="basis"):
        loaders.DailyReturns([0.01], D("2024-01-01") + np.arange(1), basis="net")
    for bad in ([1, 2], [0], [1.5], [[1]]):
        with pytest.raises(LoaderError, match="spans"):
            loaders.DailyReturns([0.01], D("2024-01-01") + np.arange(1), spans=bad)


def test_b1_24_matrix_sources(tmp_path):
    a = loaders.from_equity(1.0 + 0.01 * np.arange(41), D("2024-01-01") + np.arange(41), name="a")
    b = loaders.from_returns(np.full(40, 0.001), D("2024-01-05") + np.arange(40), name="b")
    m = loaders.align_returns([a, b])
    assert set(m.sources) == {"a", "b"} and m.dropped == {"a": 3, "b": 3}
    sa, sb = m.sources["a"], m.sources["b"]
    assert sa == {"n": 40, "start": "2024-01-02", "end": "2024-02-10", "before": 3, "after": 0, "inside": 0,
                  "max_span": 1, "median_span": 1.0, "basis": "equity"}
    assert (sb["before"], sb["after"], sb["inside"], sb["basis"]) == (0, 3, 0, "returns")
    for name, info in m.sources.items():
        assert info["before"] + info["after"] + info["inside"] == m.dropped[name]
    gap = loaders.from_returns(np.full(38, 0.001), np.delete(D("2024-01-05") + np.arange(40), [10, 20]), name="g")
    assert loaders.align_returns([a, gap]).sources["g"]["inside"] == 0
    assert loaders.align_returns([a, gap]).sources["a"]["inside"] == 2
    assert loaders.ReturnMatrix(m.dates, m.names, m.matrix).sources is None
