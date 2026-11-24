"""Tests for oss_backtest_kit.checkup (spec items B2.x in docs/SPEC_B.md)."""

import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)

from oss_backtest_kit import _disclosure  # noqa: E402
from oss_backtest_kit import checkup as ck  # noqa: E402
from oss_backtest_kit import loaders, stats  # noqa: E402
from oss_backtest_kit.checkup import CheckupError, checkup, grade, grade_bootstrap  # noqa: E402
from oss_backtest_kit.loaders import LoaderError, ReturnMatrix  # noqa: E402


def noise(seed, t=365, k=40):
    return np.random.default_rng(seed).normal(0, 0.01, (t, k))


def edge(seed, t=730, k=6, mu=0.0025):
    m = np.random.default_rng(seed).normal(0, 0.01, (t, k))
    m[:, 2] += mu
    return m


def series(seed=1, t=400, mu=0.001):
    return np.random.default_rng(seed).normal(mu, 0.01, t)


def write_csv(path, values, start="2023-01-01"):
    days = np.datetime64(start) + np.arange(len(values))
    path.write_text("date,return\n" + "".join("%s,%r\n" % (d, float(v)) for d, v in zip(days, values)))
    return str(path)


def write_dir(tmp_path, matrix, name="runs"):
    d = tmp_path / name
    d.mkdir()
    for j in range(matrix.shape[1]):
        write_csv(d / ("v%02d.csv" % j), matrix[:, j])
    return str(d)


def run_main(argv, capsys):
    rc = ck.main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


# ---- B2.1
def test_b2_1_metrics_match_stats():
    r = series(3)
    rep = checkup(r, n_trials=1)
    assert rep.metric("sharpe").value == stats.sharpe_annualised(r)
    assert rep.metric("max_drawdown").value == stats.max_drawdown(r)
    assert rep.metric("psr").value == stats.psr(r)
    assert rep.metric("dsr").value == stats.dsr(r, 1)[0]
    b = stats.block_bootstrap_sharpe(r, 2000, 5, 0)
    got = rep.metric("bootstrap")
    assert got.detail["low"] == b["low"] and got.detail["high"] == b["high"] and got.value == b["point"]
    # several results: the best-Sharpe column is described, DSR sees every trial's Sharpe
    m = edge(1)
    rep = checkup(m)
    j = int(np.argmax([stats.sharpe(m[:, k]) for k in range(m.shape[1])]))
    assert rep.selected == "result_%d" % (j + 1)
    ts = [stats.sharpe(m[:, k]) for k in range(m.shape[1])]
    assert rep.metric("dsr").value == stats.dsr(m[:, j], 6, ts)[0]
    assert rep.metric("psr").value == stats.psr(m[:, j])
    assert rep.metric("pbo").value == stats.pbo_cscv(m)
    assert rep.metric("n_effective").value == stats.n_effective(m)
    assert rep.metric("dsr").detail["n_trials"] == 6


# ---- B2.2
def test_b2_2_threshold_boundaries():
    assert ck.THRESHOLDS == {"sharpe": (1.0, 0.5), "psr": (0.95, 0.90), "dsr": (0.95, 0.90), "pbo": (0.20, 0.50)}
    eps = 1e-9
    for key in ("sharpe", "psr", "dsr"):
        good, ok = ck.THRESHOLDS[key]
        assert grade(key, good) == "pass" and grade(key, good + eps) == "pass"
        assert grade(key, good - eps) == "caution" and grade(key, ok) == "caution"
        assert grade(key, ok - eps) == "fail" and grade(key, -1.0) == "fail"
    assert grade("pbo", 0.0) == "pass" and grade("pbo", 0.20) == "pass"
    assert grade("pbo", 0.20 + 1e-9) == "caution" and grade("pbo", 0.50) == "caution"
    assert grade("pbo", 0.50 + 1e-9) == "fail" and grade("pbo", 1.0) == "fail"
    for key in ck.THRESHOLDS:
        assert grade(key, float("nan")) == "fail" and grade(key, None) == "fail"
    assert grade_bootstrap(1.0, 0.1, 2.0) == "pass"
    assert grade_bootstrap(1.0, 0.0, 2.0) == "caution"  # lower bound exactly 0 is not above 0
    assert grade_bootstrap(1.0, -0.1, 2.0) == "caution"
    assert grade_bootstrap(1.0, -2.0, 0.0) == "fail"   # upper bound at 0
    assert grade_bootstrap(-0.1, -2.0, 1.0) == "fail"  # negative point estimate
    assert grade_bootstrap(0.0, -1.0, 1.0) == "fail"
    assert grade_bootstrap(float("nan"), 0.0, 1.0) == "fail"


def test_b2_2_end_to_end_verdict_follows_grade():
    for seed in range(3):
        rep = checkup(edge(seed), n_trials=6)
        for key in ("sharpe", "psr", "dsr", "pbo"):
            m = rep.metric(key)
            assert m.verdict == grade(key, m.value)
        b = rep.metric("bootstrap")
        assert b.verdict == grade_bootstrap(b.value, b.detail["low"], b.detail["high"])
    assert checkup(series(2)).metric("max_drawdown").verdict == "info"
    assert checkup(edge(0)).metric("n_effective").verdict == "info"
    assert checkup(-np.abs(series(2))).metric("sharpe").verdict == "fail"


# ---- B2.3
def test_b2_3_sample_size_rules():
    strong = np.random.default_rng(5).normal(0.004, 0.01, 400)
    assert checkup(strong[:180]).metric("psr").verdict == "pass"
    assert not checkup(strong[:180]).short
    short = checkup(strong[:179])
    assert short.short
    assert short.metric("psr").verdict == "caution" and short.metric("sharpe").verdict == "caution"
    assert short.metric("bootstrap").verdict == "caution" and short.metric("dsr").verdict == "caution"
    assert any("179 observations" in n and "no line can be a pass" in n for n in short.notes)
    assert not any("observations (fewer" in n for n in checkup(strong[:180]).notes)
    assert checkup(strong[:30]).n_obs == 30
    for n in (29, 5, 1):
        with pytest.raises(CheckupError, match="too few"):
            checkup(strong[:n])
    bad = checkup(-strong[:100] + 0.0)  # fail stays fail when short
    assert bad.metric("psr").verdict == "fail" and bad.metric("sharpe").verdict == "fail"
    # a matrix is judged on its rows
    m = edge(0, t=100)
    assert all(x.verdict != "pass" for x in checkup(m).metrics)
    with pytest.raises(CheckupError, match="too few"):
        checkup(edge(0, t=29))


# ---- B2.4
def test_b2_4_trials_default_and_override(tmp_path):
    rep = checkup(series(), )
    assert rep.n_trials == 1 and not rep.trials_given
    assert any("did not save" in n and "--trials" in n for n in rep.notes)
    assert "No deflation applied" in rep.metric("dsr").text
    assert rep.metric("dsr").value == rep.metric("psr").value
    m = edge(0)
    rep = checkup(m)
    assert rep.n_trials == 6 and any("N = 6" in n and "did not save" in n for n in rep.notes)
    d = checkup(m, n_trials=50)
    assert d.n_trials == 50 and d.trials_given and not any("did not save" in n for n in d.notes)
    assert d.metric("dsr").value < rep.metric("dsr").value
    assert "No deflation" not in d.metric("dsr").text
    s = checkup(series(), n_trials=20)
    assert s.metric("dsr").value < s.metric("psr").value and "No deflation" not in s.metric("dsr").text
    assert checkup(np.int64(0) + m, n_trials=np.int64(7)).n_trials == 7
    again = checkup(write_dir(tmp_path, m))
    assert again.n_trials == 6 and again.n_results == 6 and again.names[0] == "v00"


def test_b2_4_bad_trials():
    r = series()
    for bad in (0, -3, 2.5, 3.0, "5", True, object):
        with pytest.raises(CheckupError, match="n_trials"):
            checkup(r, n_trials=bad)
    with pytest.raises(CheckupError, match="smaller than the number of results"):
        checkup(edge(0), n_trials=3)
    assert checkup(edge(0), n_trials=6).n_trials == 6


# ---- B2.5
def test_b2_5_pbo_needs_two_results(monkeypatch):
    rep = checkup(series())
    for key in ("pbo", "n_effective"):
        m = rep.metric(key)
        assert m.verdict == "info" and m.value is None and m.shown == "n/a" and "two or more" in m.text
    rep = checkup(edge(0))
    assert rep.metric("pbo").value is not None and rep.metric("n_effective").value is not None
    assert 1.0 <= rep.metric("n_effective").value <= 6.0
    # a one-column 2-D array is refused rather than treated as a result
    with pytest.raises(CheckupError, match="at least two columns"):
        checkup(edge(0)[:, :1])


def test_b2_5_pbo_failure_is_info(monkeypatch):
    def boom(m, n_blocks=8):
        raise stats.StatsError("need even n_blocks >= 4 and periods >= 2 * n_blocks")
    monkeypatch.setattr(stats, "pbo_cscv_detail", boom)
    m = checkup(edge(0)).metric("pbo")
    assert m.verdict == "info" and m.value is None and "Could not be computed" in m.text and "periods" in m.text


# ---- B2.6
def test_b2_6_noise_vs_edge():
    n_psr, n_dsr, n_pbo, e_psr, e_dsr, e_pbo = [], [], [], [], [], []
    for seed in range(6):
        rn, re_ = checkup(noise(seed)), checkup(edge(seed), n_trials=6)
        assert rn.metric("dsr").verdict == "fail" and rn.overall == "fail"  # best of 40 noise series
        assert re_.metric("dsr").verdict == "pass" and re_.metric("pbo").verdict == "pass"
        assert re_.overall == "pass"
        n_dsr.append(rn.metric("dsr").value)
        e_dsr.append(re_.metric("dsr").value)
        n_pbo.append(rn.metric("pbo").value)
        e_pbo.append(re_.metric("pbo").value)
        n_psr.append(rn.metric("psr").value)
        e_psr.append(re_.metric("psr").value)
        assert rn.metric("dsr").value < rn.metric("psr").value  # deflation bites
    assert max(n_dsr) < min(e_dsr)
    assert min(n_pbo) > max(e_pbo)
    assert np.mean(n_psr) < np.mean(e_psr)
    # a single noise series with no edge: PSR and Sharpe do not pass
    flat = checkup(np.random.default_rng(11).normal(0, 0.01, 365))
    assert flat.metric("psr").verdict != "pass"


# ---- B2.7
def test_b2_7_cli_runs_as_module(tmp_path):
    f = write_csv(tmp_path / "r.csv", series(4))
    env = dict(os.environ, PYTHONPATH=SRC)
    p = subprocess.run([sys.executable, "-m", "oss_backtest_kit.checkup", f, "--trials", "3"],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0 and p.stderr == ""
    assert p.stdout.startswith("# Overfitting checkup") and "Trials counted (N): 3" in p.stdout
    p = subprocess.run([sys.executable, "-m", "oss_backtest_kit.checkup", str(tmp_path / "nope.csv")],
                       capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 2 and p.stdout == "" and p.stderr.startswith("error: ") and "no such file" in p.stderr


def test_b2_7_cli_errors(tmp_path, capsys):
    good = write_csv(tmp_path / "good.csv", series(4))
    rc, out, err = run_main([str(tmp_path / "missing.csv")], capsys)
    assert (rc, out) == (2, "") and "no such file" in err
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    rc, out, err = run_main([str(empty)], capsys)
    assert rc == 2 and out == "" and err.startswith("error: ")
    (tmp_path / "emptydir").mkdir()
    rc, out, err = run_main([str(tmp_path / "emptydir")], capsys)
    assert rc == 2 and "need at least two" in err
    nan = tmp_path / "nan.csv"
    nan.write_text("date,return\n2024-01-01,0.01\n2024-01-02,nan\n")
    assert run_main([str(nan)], capsys)[0] == 2
    short = write_csv(tmp_path / "short.csv", series(4)[:20])
    rc, out, err = run_main([short], capsys)
    assert rc == 2 and "too few" in err
    flat = write_csv(tmp_path / "flat.csv", np.full(100, 0.001))
    rc, out, err = run_main([flat], capsys)
    assert rc == 2 and "constant" in err
    rc, out, err = run_main([good, "--trials", "0"], capsys)
    assert rc == 2 and "n_trials" in err
    for argv in ([good, "--trials", "abc"], [], [good, "--bogus"]):
        with pytest.raises(SystemExit) as e:
            ck.main(argv)
        assert e.value.code == 2
        capsys.readouterr()
    rc, out, err = run_main([good, "--report", str(tmp_path / "no" / "dir" / "r.md")], capsys)
    assert rc == 2 and err.startswith("error: ")
    rc, out, err = run_main([good, "--select", "x"], capsys)  # single result: not one of the results
    assert rc == 2 and "select" in err
    assert run_main([good, "--trials", "2"], capsys)[0] == 0


def test_b2_7_cli_writes_reports(tmp_path, capsys):
    d = write_dir(tmp_path, edge(1))
    md, html = tmp_path / "out.md", tmp_path / "out.html"
    rc, out, err = run_main([d, "--report", str(md), "--html", str(html), "--trials", "9", "--seed", "3"], capsys)
    assert rc == 0 and err == ""
    assert md.read_text(encoding="utf-8") == out
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert "Trials counted (N): 9" in out and "seed 3" in out and "v00" in out


# ---- B2.8
def test_b2_8_report_lists_thresholds():
    text = checkup(edge(0)).to_markdown()
    for token in ("0.95", "0.90", "0.20", "0.50", "1.00", "Thresholds used", "conventions",
                  "cannot show that a strategy will make money"):
        assert token in text
    assert text.index("cannot show") < text.index("## Results")


# ---- B2.9
def test_b2_9_input_errors(tmp_path):
    for empty in ([], np.array([]), ""):
        with pytest.raises((CheckupError, LoaderError)):
            checkup(empty)
    with pytest.raises(CheckupError, match="no such file"):
        checkup(str(tmp_path / "missing"))
    with pytest.raises(LoaderError):
        checkup(np.array([0.01] * 40 + [float("nan")]))
    with pytest.raises(LoaderError):
        checkup(np.array([0.01] * 40 + [float("inf")]))
    with pytest.raises(CheckupError, match="constant returns .*series"):
        checkup(np.full(100, 0.002))
    with pytest.raises(CheckupError, match="constant"):
        checkup(np.zeros(100))
    m = edge(0, t=100)
    m[:, 4] = 0.0
    with pytest.raises(CheckupError, match="result_5"):
        checkup(m)
    m[:, 4] = np.nan
    with pytest.raises(CheckupError, match="NaN"):
        checkup(m)
    with pytest.raises(CheckupError, match="select='zzz'"):
        checkup(edge(0), select="zzz")
    unknown = tmp_path / "x.txt"
    unknown.write_text("1,2")
    with pytest.raises(LoaderError, match="unknown file type"):
        checkup(str(unknown))


def test_b2_9_select_and_matrix_inputs(tmp_path):
    m = edge(2, t=400)
    rep = checkup(m, select="result_1")
    assert rep.selected == "result_1" and rep.metric("psr").value == stats.psr(m[:, 0])
    assert any("result_1" in n for n in rep.notes)
    dates = np.datetime64("2023-01-01") + np.arange(400)
    rm = ReturnMatrix(dates, ["a", "b", "c", "d", "e", "f"], m)
    rep = checkup(rm)
    assert rep.names == ["a", "b", "c", "d", "e", "f"] and rep.start == "2023-01-01" and rep.end == "2024-02-04"
    assert not rep.synthetic and checkup(m).synthetic and checkup(m).start is None
    one = checkup(loaders.from_returns(m[:, 2], dates, name="mine"))
    assert one.selected == "mine" and one.n_results == 1
    d = write_dir(tmp_path, m)
    assert checkup(d, select="v02").selected == "v02"
    assert math.isfinite(checkup(d).metric("psr").value)


# ---- B2.10
def test_b2_10_block_bootstrap_sharpe():
    r = series(7, t=300)
    a = stats.block_bootstrap_sharpe(r, 300, 5, 1)
    assert a == stats.block_bootstrap_sharpe(r, 300, 5, 1)
    assert a != stats.block_bootstrap_sharpe(r, 300, 5, 2)
    assert a["low"] < a["point"] < a["high"] and a["point"] == stats.sharpe_annualised(r)
    wide = stats.block_bootstrap_sharpe(r, 300, 5, 1, level=0.99)
    assert wide["low"] <= a["low"] and wide["high"] >= a["high"]
    for kw in ({"n_resamples": 0}, {"block": 0}, {"level": 1.0}, {"level": 0.0}, {"block": 200}):
        with pytest.raises(stats.StatsError):
            stats.block_bootstrap_sharpe(r, **kw)
    with pytest.raises(stats.StatsError):
        stats.block_bootstrap_sharpe([0.01, float("nan")] * 20)
    # an all-positive series with tiny noise has an interval above zero; a mirror image sits below
    pos = 0.01 + np.random.default_rng(0).normal(0, 0.001, 100)
    assert stats.block_bootstrap_sharpe(pos)["low"] > 0
    assert stats.block_bootstrap_sharpe(-pos)["high"] < 0


# ---- B2.11
def test_b2_11_spec_and_tests_agree():
    with open(os.path.join(HERE, "..", "docs", "SPEC_B.md"), encoding="utf-8") as fh:
        spec = fh.read()
    named = set(re.findall(r"`(test_b[23]_\w+)`", spec))
    defined = set()
    for name in ("test_checkup.py", "test_report.py", "test_stats.py"):
        with open(os.path.join(HERE, name), encoding="utf-8") as fh:
            defined |= set(re.findall(r"^def (test_b[23]_\w+)", fh.read(), re.M))
    assert named == defined


# ---- B2.12 bootstrap interval pinned
def _reference_interval(r, n_resamples, block, seed, level):
    rng = np.random.default_rng(seed)
    t = r.size
    n_blocks = -(-t // block)
    out = []
    for _ in range(n_resamples):
        starts = rng.integers(0, t, n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % t).ravel()[:t]
        out.append(stats.sharpe_annualised(r[idx]))
    tail = (1 - level) / 2
    return float(np.quantile(out, tail)), float(np.quantile(out, 1 - tail))


def test_b2_12_bootstrap_interval_pinned():
    r = series(9, t=250)
    for level in (0.95, 0.5):
        got = stats.block_bootstrap_sharpe(r, 200, 4, 7, level=level)
        assert (got["low"], got["high"]) == _reference_interval(r, 200, 4, 7, level)
    lo95 = stats.block_bootstrap_sharpe(r, 200, 4, 7)
    lo90 = stats.block_bootstrap_sharpe(r, 200, 4, 7, level=0.90)
    assert lo95["low"] < lo90["low"] and lo95["high"] > lo90["high"]
    # the seed given to checkup reaches the bootstrap
    a, b = checkup(r, seed=0).metric("bootstrap"), checkup(r, seed=1).metric("bootstrap")
    assert (a.detail["low"], a.detail["high"]) != (b.detail["low"], b.detail["high"])
    again = checkup(r, seed=1).metric("bootstrap")
    assert (again.detail["low"], again.detail["high"]) == (b.detail["low"], b.detail["high"])
    assert a.detail["seed"] == 0 and b.detail["seed"] == 1 and a.detail["level"] == 0.95
    assert (b.detail["low"], b.detail["high"]) == _reference_interval(r, 2000, 5, 1, 0.95)


# ---- B2.13 overall verdict and N = 2 wording
def test_b2_13_overall_caution_and_two_trials():
    strong = np.random.default_rng(5).normal(0.004, 0.01, 400)
    assert checkup(strong, n_trials=1).overall == "pass"
    rep = checkup(strong[:179])  # strong but short: every pass becomes caution
    assert {m.verdict for m in rep.metrics} - {"info"} == {"caution"}
    assert rep.overall == "caution" and "Overall: **CAUTION**" in rep.to_markdown()
    assert checkup(-strong[:100]).overall == "fail"
    assert "No deflation applied" in checkup(strong, n_trials=1).metric("dsr").text
    assert "No deflation" not in checkup(strong, n_trials=2).metric("dsr").text
    assert checkup(strong, n_trials=2).metric("dsr").value < checkup(strong, n_trials=1).metric("dsr").value


# ---- B2.14 sparse strategies
def test_b2_14_sparse_strategy_cannot_pass():
    x2 = np.zeros(200)
    x2[::50] = 0.01
    x2[1::50] = -0.001
    rep = checkup(x2)
    assert rep.sparse and rep.active == 8
    assert all(m.verdict != "pass" for m in rep.metrics) and rep.overall != "pass"
    assert any("Only 8 of the returns are non-zero" in n for n in rep.notes)
    # boundary: 30 active periods is enough, 29 is not
    base = np.random.default_rng(2).normal(0.004, 0.01, 400)
    for k, sparse in ((30, False), (29, True)):
        y = np.zeros(400)
        y[:k] = base[:k]
        assert checkup(y).sparse is sparse
    assert not checkup(base).sparse and not any("non-zero" in n for n in checkup(base).notes)


# ---- B2.15 non-daily dates
def test_b2_15_non_daily_dates_set_the_annualisation():
    r = series(4, t=200, mu=0.003)
    daily = loaders.from_returns(r, np.datetime64("2023-01-01") + np.arange(200))
    assert checkup(daily).periods_per_year == stats.PERIODS_PER_YEAR
    assert not any("not one per calendar day" in n for n in checkup(daily).notes)
    every3 = loaders.from_returns(r, np.datetime64("2023-01-01") + 3 * np.arange(200))
    rep = checkup(every3)
    span = 3 * 199 + 1
    ppy = 200 * stats.PERIODS_PER_YEAR / span
    assert abs(rep.periods_per_year - ppy) < 1e-9
    assert rep.metric("sharpe").value == stats.sharpe_annualised(r, ppy)
    assert rep.metric("sharpe").value < checkup(daily).metric("sharpe").value
    assert any("not one per calendar day" in n and "fill days" in n for n in rep.notes)
    b = rep.metric("bootstrap")
    assert b.value == stats.sharpe_annualised(r, ppy)
    weekdays = np.arange(np.datetime64("2023-01-02"), np.datetime64("2024-03-01"))
    weekdays = weekdays[(weekdays.astype("int64") + 3) % 7 < 5]  # Monday..Friday
    w = checkup(loaders.from_returns(np.random.default_rng(1).normal(0.001, 0.01, weekdays.size), weekdays))
    assert 250 < w.periods_per_year < 275
    assert "returns (2023-01-02" in w.to_markdown() and "daily returns" not in w.to_markdown()


# ---- B2.16 option validation and stray exceptions
def test_b2_16_seed_and_stray_errors(tmp_path, capsys, monkeypatch):
    good = write_csv(tmp_path / "g.csv", series(4))
    for bad in (-1, 1.5, "3", True):
        with pytest.raises(CheckupError, match="seed"):
            checkup(series(), seed=bad)
    with pytest.raises(SystemExit) as e:
        ck.main([good, "--seed", "-1"])
    assert e.value.code == 2 and "must be 0 or more" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        ck.main([good, "--seed", "x"])
    assert e.value.code == 2
    capsys.readouterr()

    def boom(*a, **k):
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "surrogates not allowed")
    monkeypatch.setattr(ck, "checkup", boom)
    rc, out, err = run_main([good], capsys)
    assert rc == 2 and out == "" and err.startswith("error: ")
    monkeypatch.undo()
    with pytest.raises(CheckupError, match="numbers"):
        checkup(np.array([["a", "b"]] * 40))


# ---- B2.17 one result without --trials cannot pass
def test_b2_17_single_result_without_trials_cannot_pass(tmp_path, capsys):
    strong = np.random.default_rng(5).normal(0.004, 0.01, 400)
    rep = checkup(strong)
    d = rep.metric("dsr")
    assert not rep.trials_given and d.verdict == "caution" and d.value == stats.psr(strong)
    assert "not given" in d.text and "cannot be judged" in d.text and "--trials" in d.text
    assert rep.overall == "caution"
    md = rep.to_markdown()
    assert "Overall: **CAUTION**" in md and "cannot be judged" in md
    assert "Overall: **PASS**" not in md
    html = rep.to_html()
    assert 'class="v caution"' in html and "cannot be judged" in html
    # the user says N = 1: now it may pass; N > 1 is deflated
    assert checkup(strong, n_trials=1).overall == "pass"
    assert checkup(strong, n_trials=1).metric("dsr").verdict == "pass"
    # a weak result keeps its fail
    assert checkup(-strong[:100]).metric("dsr").verdict == "fail"
    assert "cannot be judged" not in checkup(edge(1)).metric("dsr").text
    # command line
    f = write_csv(tmp_path / "s.csv", strong)
    rc, out, err = run_main([f], capsys)
    assert rc == 0 and "Overall: **CAUTION**" in out and "cannot be judged" in out
    rc, out, err = run_main([f, "--trials", "1"], capsys)
    assert rc == 0 and "Overall: **PASS**" in out and "cannot be judged" not in out


# ---- B2.18 command line switches for the loader options
def test_b2_18_cli_loader_switches(tmp_path, capsys):
    data = os.path.join(HERE, "data")
    multi = os.path.join(data, "freqtrade_multi.json")
    rc, out, err = run_main([multi], capsys)
    assert rc == 2 and out == "" and "--strategy NAME" in err
    rc, out, err = run_main([multi, "--strategy", "NoSuch"], capsys)
    assert rc == 2 and "no strategy" in err
    # a longer synthetic multi-strategy file: read one strategy, read the other
    import json
    doc = {"strategy": {}}
    rng = np.random.default_rng(3)
    for name in ("StratA", "StratB"):
        trades, pnl_sum = [], 0.0
        for k in range(60):
            p = float(round(rng.normal(0.3, 2.0), 4))
            day = np.datetime64("2024-01-01") + 3 * k
            trades.append({"open_date": "%s 01:00:00+00:00" % day, "close_date": "%s 20:00:00+00:00" % day, "profit_abs": p})
            pnl_sum += p
        doc["strategy"][name] = {"trades": trades, "profit_total_abs": round(pnl_sum, 10), "starting_balance": 1000.0}
    big = tmp_path / "multi.json"
    big.write_text(json.dumps(doc))
    rc, out, err = run_main([str(big), "--strategy", "StratB", "--trials", "2"], capsys)
    assert rc == 0 and "multi:StratB" in out and "Described: multi:StratB" in out
    # starting balance missing from the file
    for s in doc["strategy"].values():
        del s["starting_balance"]
    nob = tmp_path / "nob.json"
    nob.write_text(json.dumps({"strategy": {"Only": doc["strategy"]["StratA"]}}))
    rc, out, err = run_main([str(nob)], capsys)
    assert rc == 2 and "--starting-balance" in err
    rc, out, err = run_main([str(nob), "--starting-balance", "1000", "--trials", "1"], capsys)
    assert rc == 0 and "Observations" in out
    for bad in ("0", "-5", "abc", "nan"):
        with pytest.raises(SystemExit) as e:
            ck.main([str(nob), "--starting-balance", bad])
        assert e.value.code == 2
        capsys.readouterr()
    # csv switches
    c = tmp_path / "odd.csv"
    c.write_text("when,pnl\n" + "".join("%s,%r\n" % (np.datetime64("2023-01-01") + k, float(v))
                                         for k, v in enumerate(series(4))))
    rc, out, err = run_main([str(c)], capsys)
    assert rc == 2 and "--time-col" in err or "--value-col" in err
    rc, out, err = run_main([str(c), "--time-col", "when", "--value-col", "pnl", "--kind", "returns", "--trials", "1"], capsys)
    assert rc == 0 and "Observations: 400 returns" in out
    rc, out, err = run_main([str(c), "--value-col", "pnl"], capsys)
    assert rc == 2 and "kind" in err
    with pytest.raises(SystemExit) as e:
        ck.main([str(c), "--kind", "bogus"])
    assert e.value.code == 2
    capsys.readouterr()
    # a switch that does not fit the source is an error, not ignored
    rc, out, err = run_main([str(c), "--time-col", "when", "--value-col", "pnl", "--kind", "returns", "--strategy", "X"], capsys)
    assert rc == 2 and "--strategy does not apply to a CSV file" in err
    rc, out, err = run_main([multi, "--kind", "returns"], capsys)
    assert rc == 2 and "--kind does not apply to a freqtrade" in err
    d = write_dir(tmp_path, edge(1))
    rc, out, err = run_main([d, "--strategy", "X"], capsys)
    assert rc == 2 and "--strategy does not apply to a directory" in err
    rc, out, err = run_main([d, "--time-col", "date", "--trials", "6"], capsys)
    assert rc == 0
    # the Python API takes the same options
    rep = checkup(str(big), n_trials=2, load_options={"strategy": "StratA"})
    assert rep.selected == "multi:StratA"
    with pytest.raises(CheckupError, match="unknown load option"):
        checkup(str(c), load_options={"nope": 1})
    with pytest.raises(CheckupError, match="path"):
        checkup(series(), load_options={"kind": "returns"})


# ---- B2.19 wording of PSR at the top of the range
def test_b2_19_psr_text_never_claims_certainty():
    t = np.random.default_rng(1).normal(0.01, 0.005, 600)
    rep = checkup(t, n_trials=1)
    assert rep.metric("psr").value > 0.999
    assert "more than 99.9% confident" in rep.metric("psr").text and "100%" not in rep.metric("psr").text
    mid = checkup(series(3), n_trials=1).metric("psr")
    assert "99.9%" not in mid.text


# ---- B2.21 checkup on a result written by a real freqtrade run
def test_b2_21_checkup_on_real_freqtrade_zip(tmp_path, capsys):
    """The members of a real freqtrade 2026.10 run (see `real_zip` in test_loaders.py), zipped on the spot."""
    from test_loaders import real_zip
    real = real_zip(tmp_path)
    # the real run has 22 daily results (under the 30 the checkup needs): the file is read and the length is refused
    with pytest.raises(CheckupError, match="22 observations is too few"):
        checkup(real)
    rc, out, err = run_main([real], capsys)
    assert rc == 2 and "22 observations is too few" in err and out == ""
    # the same real result, read and repeated over 3 copies of the period, goes through the whole checkup
    one = loaders.load_freqtrade(real)
    rep = checkup(np.tile(np.asarray(one.values), 3))
    assert rep.n_obs == 66 and math.isfinite(rep.metric("sharpe").value)


def test_b2_21_checkup_on_real_freqtrade_futures_zip(tmp_path, capsys):
    """The real futures result (43 daily results, short trades included) passes the 30-day minimum, no refusal."""
    from test_loaders import futures_zip
    real = futures_zip(tmp_path)
    rep = checkup(real, n_trials=1)
    assert rep.n_obs == 43 and math.isfinite(rep.metric("sharpe").value)
    rc, out, err = run_main([real, "--trials", "1"], capsys)
    assert rc == 0 and out and err == ""


def test_b2_22_cli_native_overflow_is_an_error_message(tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise OverflowError("int too large to convert")
    monkeypatch.setattr(ck, "checkup", boom)
    assert ck.main([str(tmp_path / "x.csv")]) == 2
    out = capsys.readouterr()
    assert out.err.startswith("error:") and not out.out


def test_b2_22_cli_refuses_output_overwriting_input(tmp_path, capsys, monkeypatch):
    calls = 0
    real_checkup = ck.checkup

    def counting_checkup(*a, **k):
        nonlocal calls
        calls += 1
        return real_checkup(*a, **k)

    monkeypatch.setattr(ck, "checkup", counting_checkup)
    src = tmp_path / "daily.csv"
    write_csv(src, series(4, t=40))
    orig = src.read_bytes()
    rc, out, err = run_main([str(src), "--trials", "1", "--report", str(src)], capsys)
    assert rc == 2 and err.startswith("error:") and not out and src.read_bytes() == orig
    alias = tmp_path / "report_link.html"
    os.link(src, alias)
    rc, out, err = run_main([str(src), "--trials", "1", "--html", str(alias)], capsys)
    assert rc == 2 and err.startswith("error:") and not out and src.read_bytes() == orig
    folder = tmp_path / "batch"
    folder.mkdir()
    first = folder / "a.csv"
    write_csv(first, series(5, t=40))
    write_csv(folder / "b.csv", series(6, t=40))
    first_orig = first.read_bytes()
    rc, out, err = run_main([str(folder), "--trials", "2", "--report", str(first)], capsys)
    assert rc == 2 and err.startswith("error:") and not out and first.read_bytes() == first_orig
    assert calls == 0


# ---- B2.23 / B2.24 equity points that are unevenly spaced
def _equity_file(path, gaps=(), n=120, seed=3):
    """Equity curve, one point a day except that the days listed in `gaps` are missing."""
    r = np.random.default_rng(seed).normal(0.001, 0.01, n)
    eq = np.r_[1.0, np.cumprod(1 + r)]
    days = np.datetime64("2024-01-01") + np.arange(n + 1)
    keep = np.setdiff1d(np.arange(n + 1), gaps)
    path.write_text("date,equity\n" + "".join("%s,%r\n" % (days[i], float(eq[i])) for i in keep))
    return str(path)


def test_b2_23_uneven_intervals_are_refused(tmp_path, capsys):
    f = _equity_file(tmp_path / "gap.csv", gaps=range(40, 70))
    with pytest.raises(CheckupError, match="unevenly spaced") as e:
        checkup(f, n_trials=1)
    assert "gap: one return covers 31 days while a typical one covers 1" in str(e.value)
    rc, out, err = run_main([f, "--trials", "1"], capsys)
    assert rc == 2 and err.startswith("error:") and "unevenly spaced" in err and out == ""
    d = tmp_path / "runs"
    d.mkdir()
    for k in range(2):
        _equity_file(d / ("ok%d.csv" % k), seed=k)
    _equity_file(d / "uneven.csv", gaps=range(40, 52))
    with pytest.raises(CheckupError, match="uneven: one return covers 13 days"):
        checkup(str(d), n_trials=3)
    # weekend gaps (3 days), a gap exactly on the line (median 1 -> 4 days), and weekly points all go through
    weekend = [i for i in range(121) if (i % 7) in (5, 6)]
    assert checkup(_equity_file(tmp_path / "wk.csv", gaps=weekend), n_trials=1).n_obs > 30
    assert checkup(_equity_file(tmp_path / "four.csv", gaps=[40, 41, 42]), n_trials=1).n_obs > 30
    with pytest.raises(CheckupError, match="unevenly spaced"):
        checkup(_equity_file(tmp_path / "five.csv", gaps=[40, 41, 42, 43]), n_trials=1)
    days = np.datetime64("2023-01-01") + 7 * np.arange(61)
    weekly = tmp_path / "weekly.csv"
    weekly.write_text("date,equity\n" + "".join("%s,%r\n" % (d, 1.0 + 0.01 * i) for i, d in enumerate(days)))
    assert checkup(str(weekly), n_trials=1).n_obs == 60
    # returns given directly are never checked for spacing
    assert checkup(write_csv(tmp_path / "ret.csv", series(2, 120)), n_trials=1).n_obs == 120


def test_b2_24_fill_flat_repeats_the_last_equity_value(tmp_path, capsys):
    f = _equity_file(tmp_path / "gap.csv", gaps=range(40, 70))
    with pytest.raises(CheckupError, match=r"Pass fill_flat=True \(--fill-flat\)"):
        checkup(f, n_trials=1)
    rep = checkup(f, n_trials=1, fill_flat=True)
    assert rep.n_obs == 120 and rep.active == 120 - 30 and rep.filled == {"gap": 30}
    assert any(n.startswith("Filled 30 missing days by repeating the last equity value (assumes the account "
                            "did not change on those days).") for n in rep.notes)
    assert "Filled 30 missing days" in rep.to_markdown() and "Filled 30 missing days" in rep.to_html()
    rc, out, err = run_main([f, "--trials", "1", "--fill-flat"], capsys)
    assert rc == 0 and "Filled 30 missing days" in out and err == ""
    d = tmp_path / "runs"
    d.mkdir()
    _equity_file(d / "a.csv", seed=1)
    _equity_file(d / "b.csv", gaps=range(40, 52), seed=2)
    _equity_file(d / "c.csv", gaps=range(10, 18), seed=4)
    with pytest.raises(CheckupError, match="unevenly spaced"):
        checkup(str(d), n_trials=3)
    rep = checkup(str(d), n_trials=3, fill_flat=True)
    assert rep.filled == {"b": 12, "c": 8} and rep.n_obs == 120
    assert any("Filled 20 missing days" in n and "b 12, c 8" in n for n in rep.notes)
    # no effect, no error, when nothing is uneven (returns given directly, or equity with no gaps)
    ret = checkup(write_csv(tmp_path / "ret.csv", series(2, 120)), n_trials=1, fill_flat=True)
    assert ret.filled == {} and not any("Filled" in n for n in ret.notes)
    full = checkup(_equity_file(tmp_path / "full.csv"), n_trials=1, fill_flat=True)
    assert full.filled == {} and full.n_obs == 120
    # a loaded equity series (DailyReturns) with a gap is filled too
    days = np.delete(np.datetime64("2024-01-01") + np.arange(101), np.arange(50, 60))
    ser = loaders.from_equity(np.cumprod(1 + series(5, 91)) * 1.0, days)
    assert checkup(ser, n_trials=1, fill_flat=True).filled == {"series": 10}


# ---- B2.25 coverage of the aligned window; B2.26 realised-only basis
def _offset_pair(drop):
    """Two strong results; `a` (the one described) loses `drop` days before the common window, `b` as many after."""
    n = 190 + drop
    start = np.datetime64("2024-01-01")
    va = np.random.default_rng(11).normal(0.01, 0.01, n)
    vb = np.random.default_rng(12).normal(0.01, 0.01, n)
    a = loaders.from_returns(va, start + np.arange(n), name="a")
    b = loaders.from_returns(vb, start + drop + np.arange(n), name="b")
    return loaders.align_returns([a, b])


def test_b2_25_alignment_coverage_disclosed():
    # exactly 5% excluded (10 of 200): written down, not capped; 11 of 201 (5.5%): no line can pass
    five = checkup(_offset_pair(10), n_trials=2, select="a", n_resamples=250)
    assert five.coverage == [("a", 190, 200, 10, 0, 0), ("b", 190, 200, 0, 10, 0)]
    passing = lambda rep: [m.key for m in rep.metrics if m.verdict == "pass"]  # noqa: E731
    assert "sharpe" in passing(five) and not any("More than 5%" in n for n in five.notes)
    six = checkup(_offset_pair(11), n_trials=2, select="a", n_resamples=250)
    assert any("More than 5% of its returns were excluded" in n for n in six.notes) and passing(six) == []
    assert "sharpe" in passing(checkup(_offset_pair(1), n_trials=2, select="a", n_resamples=250))
    for rendered in (six.to_markdown(), six.to_html()):
        assert ("common window 2024-01-12 to 2024-07-19; the described result kept 190 of its 201 returns "
                "(94.5%): 11 excluded (11 before the window, 0 after it, 0 inside it). Other results: "
                "b 190/201.") in rendered
        assert "11 returns of a outside it were excluded and not checked" in rendered
    # a single result excludes nothing; a matrix built without align_returns says so
    single = checkup(loaders.from_returns(series(2, 200), np.datetime64("2024-01-01") + np.arange(200)),
                     n_trials=1)
    assert single.coverage == [("series", 200, 200, 0, 0, 0)]
    assert "kept 200 of its 200 returns (100.0%): 0 excluded" in single.to_markdown()
    assert not any("excluded" in n for n in single.notes)
    bare = ReturnMatrix(np.datetime64("2024-01-01") + np.arange(120), ["x", "y"], noise(4, 120, 2))
    rep = checkup(bare, n_trials=2)
    assert rep.coverage is None and "not recorded (matrix built without align_returns)" in rep.to_html()


def _freqtrade_file(path, n=200):
    import json
    from datetime import date, timedelta
    trades = []
    for i in range(n):
        closed = date(2024, 1, 1) + timedelta(days=i + 60)
        trades.append({"open_date": (closed - timedelta(days=60)).isoformat(), "close_date": closed.isoformat(),
                       "profit_abs": 0.003 if i % 2 else 0.005})
    path.write_text(json.dumps({"strategy": {"ft": {"starting_balance": 1, "trades": trades}}}), encoding="utf-8")
    return str(path)


def test_b2_26_realised_basis_disclosed(tmp_path):
    rep = checkup(_freqtrade_file(tmp_path / "t.json"), n_trials=1, n_resamples=250)
    assert rep.basis == "realised"
    for rendered in (rep.to_markdown(), rep.to_html()):
        assert "realised: built from closed trades only (freqtrade)" in rendered
        assert "The realised equity (closed trades only) fell" in rendered
        assert "Returns come from closed trades only" in rendered
    for plain in (checkup(_equity_file(tmp_path / "e.csv"), n_trials=1), checkup(series(2, 200), n_trials=1),
                  checkup(write_csv(tmp_path / "r.csv", series(2, 200)), n_trials=1)):
        for rendered in (plain.to_markdown(), plain.to_html()):
            assert "realised" not in rendered and "closed trades" not in rendered
    eq = checkup(_equity_file(tmp_path / "e2.csv"), n_trials=1)
    assert eq.basis == "equity" and "equity curve as given" in eq.to_markdown()
    assert "The equity curve fell" in eq.metric("max_drawdown").text
    d = tmp_path / "mix"
    d.mkdir()
    _freqtrade_file(d / "t.json")
    write_csv(d / "r.csv", series(2, 260), start="2024-01-01")
    mixed = checkup(str(d), n_trials=2, select="t", n_resamples=250)
    assert mixed.basis == "realised" and mixed.bases == {"t": "realised", "r": "returns"}
    assert "Results differ in basis: realised: t; returns as given: r." in mixed.to_markdown()


# ---- B2.27 a result set without --trials cannot pass
def test_b2_27_result_set_without_trials_cannot_pass(tmp_path, capsys):
    m = edge(0)
    given = checkup(m, n_trials=6)
    assert given.overall == "pass" and given.metric("dsr").verdict == "pass"
    rep = checkup(m)
    d = rep.metric("dsr")
    assert rep.n_trials == 6 and not rep.trials_given and d.detail["n_trials"] == 6
    assert d.value == given.metric("dsr").value  # still deflated for N = number of results
    assert d.verdict == "caution" and rep.overall == "caution"
    assert d.text == ("Deflated for N = 6, the number of results read. That is only a lower bound: versions you "
                      "tried but did not save also count, so this line cannot pass. Pass --trials N (n_trials=N), "
                      "the total number of versions you tried.")
    assert any("N = 6 (the number of results)" in n and n.endswith(
        "Until it is given, the overall verdict is at most CAUTION.") for n in rep.notes)
    fact = "Trials counted (N): 6 (not given; lower bound, at most CAUTION)"
    for rendered in (rep.to_markdown(), rep.to_html()):
        assert fact in rendered and "lower bound" in rendered
    assert fact not in given.to_markdown()
    assert checkup(noise(0)).metric("dsr").verdict == "fail"  # a fail is kept
    single = checkup(series())  # a single result keeps its own wording (B2.17)
    assert "cannot be judged" in single.metric("dsr").text and "(not given; default)" in single.to_markdown()
    assert not any("Until it is given" in n for n in single.notes)
    path = write_dir(tmp_path, m)
    rc, out, err = run_main([path], capsys)
    assert rc == 0 and "Overall: **CAUTION**" in out and "lower bound" in out
    rc, out, err = run_main([path, "--trials", "6"], capsys)
    assert rc == 0 and "Overall: **PASS**" in out and "lower bound" not in out


# ---- B2.30 serial dependence in the checkup
def test_b2_30_serial_dependence_in_checkup():
    r = np.repeat(np.random.default_rng(7).normal(0.004, 0.01, 100), 4)  # each day four times: tau about 4
    rep = checkup(r, n_trials=1, block=2, n_resamples=300)
    tau = stats.autocorr_time(r)
    assert rep.tau == tau >= 1.5 and rep.n_eff == r.size / tau and rep.block_used == math.ceil(tau) == 5
    assert rep.metric("psr").value == stats.psr(r, n_obs=rep.n_eff) != stats.psr(r)
    assert rep.metric("dsr").value == stats.dsr(r, 1, (), n_obs=rep.n_eff)[0]
    assert rep.metric("bootstrap").detail["block"] == 5
    assert rep.short and rep.n_obs == 400 and all(x.verdict != "pass" for x in rep.metrics)
    for rendered in (rep.to_markdown(), rep.to_html()):
        assert ("Serial dependence: strong: about %.1f rows per independent observation, so 400 rows count as "
                "about %d independent ones (rough estimate)" % (math.floor(tau * 10) / 10, math.floor(rep.n_eff))) in rendered
        assert "Bootstrap: circular blocks of 5 (2 requested), 300 resamples, seed 0" in rendered
        assert ("The returns are serially dependent (autocorrelation time %.1f, Geyer initial positive sequence). "
                "PSR, DSR and the sample-size rule use about %d effective observations instead of 400 rows, and "
                "the bootstrap uses blocks of 5." % (math.floor(tau * 10) / 10, math.floor(rep.n_eff))) in rendered
        assert "PBO is not adjusted." in rendered
        assert ("Only about %d independent observations (400 rows, serially dependent; fewer than 180): the "
                "Sharpe estimate is noisy, so no line can be a pass." % math.floor(rep.n_eff)) in rendered
    # a user block longer than tau is kept
    assert checkup(r, n_trials=1, block=8, n_resamples=50).block_used == 8
    # enough effective observations: not short, PSR still on n_eff
    long = np.repeat(np.random.default_rng(8).normal(0.004, 0.01, 300), 2)
    lr = checkup(long, n_trials=1, n_resamples=100)
    assert lr.tau >= 1.5 and lr.n_eff >= 180 and not lr.short
    assert lr.metric("psr").value == stats.psr(long, n_obs=lr.n_eff)
    # sparse is judged on effective non-zero returns
    x = np.zeros(1000)
    for s in range(0, 1000, 50):
        x[s:s + 3] = [0.01, 0.011, 0.009]
    sp = checkup(x, n_trials=1, n_resamples=100)
    assert sp.active == 60 >= 30 and sp.tau >= 1.5 and sp.sparse and not sp.short
    assert all(v.verdict != "pass" for v in sp.metrics)
    assert any(n.startswith("Only about %d independent non-zero returns (60 non-zero rows, serially dependent; fewer than 30)"
                            % math.floor(60 / sp.tau)) for n in sp.notes)
    # fewer than 4 blocks: no resampling, caution, n/a
    few = np.repeat([0.01, -0.012], 150)
    fr = checkup(few, n_trials=1, n_resamples=100)
    b = fr.metric("bootstrap")
    assert 4 * fr.block_used > 300 and b.verdict == "caution" and b.shown == "n/a" and b.value is None
    text = "The dependence is too long for this sample (blocks of %d over 300 rows)" % fr.block_used
    assert text in b.text and text in fr.to_markdown() and text in fr.to_html()
    # an effective count below 3 is used as 3 for PSR / DSR
    trend = np.arange(300) / 1e4 + 0.001
    tr = checkup(trend, n_trials=1, n_resamples=100)
    assert tr.n_eff < 3 and tr.metric("psr").value == stats.psr(trend, n_obs=3.0)
    # rows counted as one observation each (tau < 1.5): same numbers and verdicts as before
    s = series()
    plain = checkup(s, n_trials=1)
    assert plain.tau < 1.5 and plain.n_eff == 400 and plain.block_used == 5 and not plain.short
    assert plain.metric("psr").value == stats.psr(s) and plain.metric("dsr").value == stats.dsr(s, 1)[0]
    assert plain.metric("bootstrap").value == stats.block_bootstrap_sharpe(s, 2000, 5, 0)["point"]
    md = re.sub(r"\\(.)", r"\1", plain.to_markdown())  # drop the Markdown escapes ("\<")
    assert ("Serial dependence: none material (autocorrelation time %.1f < 1.5); every row counted as one "
            "observation" % (math.floor(plain.tau * 10) / 10)) in md and "Bootstrap: circular blocks of 5, 2000 resamples, seed 0" in md
    assert "serially dependent" not in md and "requested" not in md
    # tau < 1.5 and a user block so long that T < 4 * block: still resampled, as before (not the n/a line)
    big = checkup(s, n_trials=1, block=150, n_resamples=100)
    assert big.tau < 1.5 and big.metric("bootstrap").value == stats.block_bootstrap_sharpe(s, 100, 150, 0)["point"]
    # n/a: a long user block is named as such, and the Facts line does not claim resamples that did not run
    own = checkup(r, n_trials=1, block=120, n_resamples=50)
    assert "The block length is too long for this sample (blocks of 120 over 400 rows)" in own.metric("bootstrap").text
    assert "not run (too few blocks)" in own.to_markdown() and "50 resamples" not in own.to_markdown()
    assert "not run (too few blocks)" in fr.to_html() and "100 resamples" not in fr.to_html()


def test_b2_30_notes_show_effective_count_below_formula_minimum():
    trend = np.arange(300) / 1e4 + 0.001
    rep = checkup(trend, n_trials=5, n_resamples=100)
    assert 2 <= rep.n_eff < 3 and rep.short and rep.sparse
    assert rep.metric("psr").value == stats.psr(trend, n_obs=3.0)
    assert rep.metric("dsr").value == stats.dsr(trend, 5, n_obs=3.0)[0]
    for rendered in (rep.to_markdown(), rep.to_html()):
        assert "so 300 rows count as about 2 independent ones" in rendered
        assert "Only about 2 independent observations (300 rows, serially dependent" in rendered
        assert "Only about 2 independent non-zero returns (300 non-zero rows" in rendered
        assert ("PSR, DSR and the sample-size rule use about 2 effective observations instead of 300 rows "
                "(the formulas use at least 3), and the bootstrap uses blocks of 104.") in rendered


def test_b2_30_notes_formula_minimum_boundary():
    minimum = "(the formulas use at least 3)"
    at_minimum, = _disclosure.dependence_notes(tau=10.0, n_eff=3.0, rows=30, block_used=10)
    below_minimum, = _disclosure.dependence_notes(tau=10.0, n_eff=2.999999, rows=30, block_used=10)
    assert minimum not in at_minimum
    assert minimum in below_minimum


# ---- B2.31 block and array checks
def test_b2_31_block_and_array_checks():
    dep = np.repeat(np.random.default_rng(7).normal(0.004, 0.01, 100), 4)
    for bad in (0, -1, 2.5, True):
        for x in (series(), dep):  # independent and dependent alike
            with pytest.raises(CheckupError, match="block must be a whole number of at least 1"):
                checkup(x, n_trials=1, block=bad)
    assert checkup(series(), n_trials=1, block=np.int64(5), n_resamples=50).block_used == 5
    with pytest.raises(CheckupError, match="1-D .* or 2-D"):
        checkup(np.zeros((40, 2, 2)))


# ---- B2.32 shown numbers are rounded down
def test_b2_32_shown_numbers_are_rounded_down(monkeypatch):
    rep = checkup(np.random.default_rng(666).normal(0.003, 0.01, 200), n_trials=1)
    assert 1.45 <= rep.tau < 1.5
    md = re.sub(r"\\(.)", r"\1", rep.to_markdown())
    assert "autocorrelation time 1.4 < 1.5" in md and "1.5 < 1.5" not in md
    monkeypatch.setattr(stats, "autocorr_time", lambda r: 400 / 179.7)  # about 179.7 effective observations
    text = " ".join(checkup(series(), n_trials=1).notes)
    assert "Only about 179 independent observations" in text and "about 180" not in text
    x = np.zeros(400)
    x[::7] = series(3, 58)  # 58 non-zero rows
    monkeypatch.setattr(stats, "autocorr_time", lambda r: 58 / 29.7)  # about 29.7 independent non-zero returns
    notes = " ".join(checkup(x, n_trials=1).notes)
    assert "Only about 29 independent non-zero returns (58 non-zero rows" in notes and "about 30" not in notes


# ---- B2.33 PBO ties are said; an aligned matrix is not told to pass fill_flat here
def test_b2_33_pbo_ties_and_aligned_matrix_hint(tmp_path):
    m = noise(3, 240, 4)
    m[:, 1] = m[:, 0]  # two copies tie whenever they lead
    left = stats.pbo_cscv_detail(m)[1]
    assert 0 < left and ("%d splits were left out because the results tied in them." % left) in \
        checkup(m, n_trials=4).metric("pbo").text
    assert "left out" not in checkup(noise(4, 240, 4), n_trials=4).metric("pbo").text
    d = tmp_path / "runs"
    d.mkdir()
    for k in range(2):
        _equity_file(d / ("ok%d.csv" % k), seed=k)
    _equity_file(d / "uneven.csv", gaps=range(40, 52))
    with pytest.raises(CheckupError, match="already aligned.*when loading the data") as e:
        checkup(loaders.load_returns(str(d)), n_trials=3)
    assert "Pass fill_flat=True (--fill-flat)" not in str(e.value)
