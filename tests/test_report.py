"""Tests for oss_backtest_kit.report (spec items B3.x in docs/SPEC_B.md)."""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from oss_backtest_kit import loaders  # noqa: E402
from oss_backtest_kit.checkup import checkup  # noqa: E402
from oss_backtest_kit import report as rp  # noqa: E402

EVIL = '<script>alert(1)</script>|`x`*_[l](javascript:alert(1))![i](http://evil.example/a.png)\n# Heading &amp;'


def edge(seed=0, t=400, k=4):
    m = np.random.default_rng(seed).normal(0, 0.01, (t, k))
    m[:, 1] += 0.0025
    return m


def evil_report(name=EVIL):
    r = edge()[:, 1]
    return checkup(loaders.from_returns(r, np.datetime64("2023-01-01") + np.arange(r.size), name=name))


def test_b3_1_markdown_content():
    rep = checkup(edge(), n_trials=5)
    md = rep.to_markdown()
    lines = md.splitlines()
    assert lines[0] == "# Overfitting checkup"
    assert lines[2].startswith("> **Read this first.**") and "cannot show" in lines[2]
    assert md.index("cannot show") < md.index("## Summary") < md.index("## Results") < md.index("## Thresholds used")
    for m in rep.metrics:
        assert m.label.replace("_", "\\_") in md or m.label in md
        assert rp._LABELS[m.verdict] in md and m.shown in md.replace("\\", "")
        assert m.text.split(".")[0][:30].replace("_", "") in md.replace("\\", "").replace("_", "")
    assert "Overall: **%s**" % rp._LABELS[rep.overall] in md
    for row in rp._threshold_rows(rep):
        assert row[0] in md
    assert "Trials counted (N): 5" in md and "seed 0" in md
    # single result: PBO line is present and says why it is empty
    assert "Needs two or more results" in checkup(edge()[:, 1]).to_markdown()
    # a short sample adds its note
    assert "Only 100 observations" in checkup(edge(t=100)).to_markdown()


def test_b3_2_html_offline_and_escaped():
    page = checkup(edge(), n_trials=8).to_html()  # plain names: any URL here would be ours
    low = page.lower()
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert "<script" not in low and "<link" not in low and "<img" not in low and "<iframe" not in low
    assert not re.search(r"\b(src|href|action|srcset)\s*=", low)
    assert "http://" not in low and "https://" not in low and "url(" not in low and "@import" not in low
    assert "default-src 'none'" in page and "cannot show" in page and "0.95" in page
    assert page.count("<html") == 1 and page.count("</html>") == 1
    bad = evil_report().to_html()  # file-supplied text is entity-escaped
    assert "<script" not in bad.lower() and "&lt;script&gt;alert(1)&lt;/script&gt;" in bad
    assert "&amp;amp;" in bad and "\n# Heading" not in bad
    assert not re.search(r"<(b|i|a|img)\b", bad.lower())
    quote = evil_report('x" onmouseover="alert(1)').to_html()
    assert '" onmouseover=' not in quote and "&quot;" in quote


def test_b3_3_deterministic():
    m = edge(3)
    a, b = checkup(m, seed=5), checkup(m, seed=5)
    assert a.to_markdown() == b.to_markdown() and a.to_html() == b.to_html()
    assert a.to_markdown() != checkup(m, seed=6).to_markdown()
    assert not re.search(r"\b20\d\d-\d\d-\d\d \d\d:\d\d", a.to_markdown())


def test_b3_4_labels_escaped():
    md = evil_report().to_markdown()
    bare = re.sub(r"\\.", "", md)  # drop every backslash-escaped character
    head = bare.split("## Results")[0]
    assert "<" not in bare and "![" not in bare
    assert not re.search(r"\[[^\]]*\]\(", bare)  # no live link syntax
    assert "|" not in head  # a name cannot start a table cell outside the table
    assert not re.search(r"(?m)^#{1,6} Heading", md) and "\n# Heading" not in md
    rows = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert len(rows) == 2 + 7 + 2 + 6  # results table, thresholds table: header, rule, rows
    assert all(re.sub(r"\\.", "", ln).count("|") == 5 for ln in rows)
    many = checkup(np.random.default_rng(0).normal(0, 0.01, (200, 30)))  # name list is capped
    assert "and 10 more" in many.to_markdown()
    ctrl = evil_report("a\x00b\x1bc\r\nd").to_markdown()
    assert "\x00" not in ctrl and "\x1b" not in ctrl and "\r" not in ctrl


def _matrix_with_name(name, t=400):
    m = np.random.default_rng(0).normal(0, 0.01, (t, 3))
    m[:, 1] += 0.0025
    dates = np.datetime64("2023-01-01") + np.arange(t)
    return loaders.ReturnMatrix(dates, ["a", name, "c"], m)


def test_b3_5_matrix_names_and_html_words(monkeypatch):
    from oss_backtest_kit import stats
    evil = "<img src=x onerror=alert(1)>"
    rep = checkup(_matrix_with_name(evil), select=evil)
    assert any(evil in n for n in rep.notes) or any(evil in v for _, v in rp._facts(rep))
    page, md = rep.to_html(), rep.to_markdown()
    assert "<img" not in page.lower() and "&lt;img src=x onerror=alert(1)&gt;" in page
    assert "<img" not in re.sub(r"\\.", "", md) and "\\<img" in md
    notes = page.split("<h2>Notes</h2>")[1].split("</ul>")[0]
    assert "&lt;img" in notes and "<img" not in notes
    # text built from an exception message is escaped too
    def boom(m, n_blocks=8):
        raise stats.StatsError("<b>bad</b> & worse")
    monkeypatch.setattr(stats, "pbo_cscv_detail", boom)
    page2 = checkup(_matrix_with_name("b")).to_html()
    assert "&lt;b&gt;bad&lt;/b&gt; &amp; worse" in page2 and "<b>bad" not in page2
    monkeypatch.undo()
    # verdicts are words in HTML, one per row, not only a color class
    seen = set()
    for seed in range(3):
        for r in (checkup(edge(seed)), checkup(np.random.default_rng(seed).normal(0, 0.01, 365)),
                  checkup(edge(seed, t=100))):
            for m in r.metrics:
                label = rp._LABELS[m.verdict]
                assert re.search(r'<span class="v %s">. %s</span>' % (m.verdict, label), r.to_html())
                seen.add(label)
    assert seen == {"PASS", "CAUTION", "FAIL", "INFO"}
    # threshold table: each cell matches the numbers in THRESHOLDS, in the right order
    text, html_page = checkup(edge()).to_markdown(), checkup(edge()).to_html()
    for cell in ("1.00 or more", "0.50 to below 1.00", "below 0.50", "0.95 or more", "0.90 to below 0.95",
                 "below 0.90", "0.20 or less", "above 0.20 to 0.50", "above 0.50"):
        assert cell in text and cell in html_page
    assert "0.95 to below 0.90" not in text and "0.50 to 0.20" not in text
    rows = {r[0]: r for r in rp._threshold_rows(checkup(edge()))}
    assert rows["PSR"] == ("PSR", "0.95 or more", "0.90 to below 0.95", "below 0.90")
    assert rows["PBO"] == ("PBO", "0.20 or less", "above 0.20 to 0.50", "above 0.50")
    # labels, values and sentences are escaped whatever their source
    r = checkup(edge())
    r.metrics[0].label, r.metrics[0].shown = "<i>lab</i>", "<u>v</u>"
    assert "&lt;i&gt;lab&lt;/i&gt;" in r.to_html() and "<i>lab" not in r.to_html()
    assert "<u>" not in r.to_html() and "<i>" not in re.sub(r"\\.", "", r.to_markdown())


# ---- B3.6 terminal readability: no escapes where nothing could go wrong
def test_b3_6_markdown_does_not_litter_names_and_numbers_with_backslashes():
    r = edge()[:, 1]
    rep = checkup(loaders.from_returns(r, np.datetime64("2023-01-01") + np.arange(r.size), name="backtest-result-2024_p4"),
                  n_trials=1)
    md = rep.to_markdown()
    assert "backtest-result-2024_p4" in md and "\\_" not in md
    row = [x for x in md.splitlines() if x.startswith("| Sharpe 95% interval")][0]
    assert "[" in row and "\\[" not in row and "\\]" not in row
    # names still cannot start markup: leading / trailing underscores, brackets, bars are escaped
    assert rp._md("_x_") == "\\_x\\_" and rp._md("a|b") == "a\\|b" and rp._md("[l](u)") == "\\[l\\](u)"
    assert rp._md("my_strategy_p4") == "my_strategy_p4"
    # a program-made value cannot break the table either
    assert rp._md_value("a|b") == "a\\|b"


def test_b3_7_new_fact_lines():
    # Facts order is fixed: Coverage, then Equity basis (only when the basis is known), then Serial
    # dependence, then Bootstrap.
    rep = checkup(edge(), n_trials=5)
    assert [k for k, _ in rp._facts(rep)] == ["Results read", "Described", "Observations", "Trials counted (N)",
                                              "Coverage", "Serial dependence", "Bootstrap"]
    r = edge()[:, 1]
    days = np.datetime64("2023-01-01") + np.arange(r.size)
    rep = checkup(loaders.from_returns(r, days), n_trials=1)
    assert [k for k, _ in rp._facts(rep)][4:] == ["Coverage", "Equity basis", "Serial dependence", "Bootstrap"]
    # same input and seed: identical bytes
    again = checkup(loaders.from_returns(r, days), n_trials=1)
    assert rep.to_markdown() == again.to_markdown() and rep.to_html() == again.to_html()
    # names with markup are escaped in both new lines, in the described result and in the other results
    evil = "<b>&x"
    a = loaders.from_returns(r, days, name=evil)
    b = loaders.from_returns(edge(1)[:, 1], days, name="<i>&y")
    rep = checkup(loaders.align_returns([a, b]), n_trials=2, select=evil)
    page = rep.to_html()
    assert "<b>" not in page and "<i>" not in page and "&lt;i&gt;&amp;y 400/400" in page
    assert "Results differ in basis" not in page  # one basis only: no mixed-basis sentence
    md = re.sub(r"\\.", "", rep.to_markdown())
    assert "<b>" not in md and "<i>" not in md and "Results differ in basis" not in md


def test_b3_7_serial_dependence_fact_rounds_down():
    r = np.repeat(np.random.default_rng(5).normal(0.004, 0.01, 180), 2)
    rep = checkup(r, n_trials=1, n_resamples=50)
    assert 179.5 < rep.n_eff < 180
    fact = ("Serial dependence: strong: about 2.0 rows per independent observation, so 360 rows count as "
            "about 179 independent ones (rough estimate)")
    assert "- " + fact in rep.to_markdown().splitlines()
    assert "<li>" + fact + "</li>" in rep.to_html().splitlines()


def test_b3_8_coverage_fact_text():
    from types import SimpleNamespace
    rows = [("me", 90, 100, 4, 3, 3)] + [("o%d" % k, 90, 90, 0, 0, 0) for k in range(22)]
    text = rp._coverage_fact(SimpleNamespace(coverage=rows, selected="me", synthetic=True, start=None, end=None))
    assert text.startswith("common window (undated) to (undated); the described result kept 90 of its 100 returns")
    assert "10 excluded (4 before the window, 3 after it, 3 inside it)" in text and text.endswith("o19 90/90, and 2 more.")
