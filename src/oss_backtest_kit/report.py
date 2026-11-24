"""Markdown and single-file HTML rendering of a checkup Report (spec items B3.x).

Everything that came from a file (result names) is escaped. The HTML has inline
CSS only: no script, no external resource, and a content security policy that
forbids both.
"""

import html
import re

from . import _disclosure as _dis
from .stats import TAU_IGNORE

DISCLAIMER = ("This checkup can only lower the risk of shipping an overfitted backtest. It cannot show "
              "that a strategy will make money: a result can pass every line and still lose money live.")
THRESHOLD_NOTE = ("The thresholds are conventions from the literature and practice, not guarantees. "
                  "Do not loosen them to make a result look better; a stricter bar is safer.")
_MD_SPECIAL = re.compile(r"([\\`*\[\]<>|~&]|(?<![A-Za-z0-9])_|_(?![A-Za-z0-9]))")
_LABELS = {"pass": "PASS", "caution": "CAUTION", "fail": "FAIL", "info": "INFO"}
MAX_NAMES = 20


def _clean(text):
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(text))


def _md(text):
    return _MD_SPECIAL.sub(r"\\\1", _clean(text))


def _md_value(text):
    """Escape a number text made by the program (no names in it): only the table bar and control characters."""
    return _clean(text).replace("|", "\\|")


def _threshold_rows(rep):
    t = rep.thresholds
    rows = []
    for key, label in (("sharpe", "Annualised Sharpe"), ("psr", "PSR"), ("dsr", "DSR")):
        good, ok = t[key]
        rows.append((label, "%.2f or more" % good, "%.2f to below %.2f" % (ok, good), "below %.2f" % ok))
    good, ok = t["pbo"]
    rows.append(("PBO", "%.2f or less" % good, "above %.2f to %.2f" % (good, ok), "above %.2f" % ok))
    rows.append(("Sharpe 95% interval", "lower end above 0", "range includes 0, estimate above 0",
                 "upper end at or below 0, or estimate at or below 0"))
    rows.append(("Max drawdown, effective trials", "no verdict", "", ""))
    return rows


_BASIS = {"realised": "realised: built from closed trades only (freqtrade). Open positions are not marked to "
                      "market, so unrealised losses while a trade is open are invisible.",
          "returns": "returns as given", "equity": "equity curve as given"}


def _names(items):
    return ", ".join(items[:MAX_NAMES] + (["and %d more" % (len(items) - MAX_NAMES)] if len(items) > MAX_NAMES else []))


def _coverage_fact(rep):
    if rep.coverage is None:
        return "not recorded (matrix built without align_returns)"
    me = next(c for c in rep.coverage if c[0] == rep.selected)
    others = [_clean("%s %d/%d" % (c[0], c[1], c[2])) for c in rep.coverage if c is not me]
    return ("common window %s to %s; the described result kept %d of its %d returns (%.1f%%): %d excluded "
            "(%d before the window, %d after it, %d inside it). Other results: %s."
            % ("(undated)" if rep.synthetic else rep.start, "(undated)" if rep.synthetic else rep.end,
               me[1], me[2], 100.0 * me[1] / me[2], me[2] - me[1], me[3], me[4], me[5],
               _names(others) if others else "none"))


def _basis_fact(rep):
    text = _BASIS[rep.basis]
    groups = {}
    for k, b in rep.bases.items():
        groups.setdefault(b, []).append(_clean(k))
    if len(groups) > 1:
        text += " Results differ in basis: " + "; ".join(
            "%s: %s" % (_BASIS[b].split(":")[0], _names(v))
            for b, v in sorted(groups.items())) + "."
    return text


def _facts(rep):
    names = [_clean(n) for n in rep.names]
    shown = names[:MAX_NAMES] + (["and %d more" % (len(names) - MAX_NAMES)] if len(names) > MAX_NAMES else [])
    period = "no dates given" if rep.synthetic else "%s to %s" % (rep.start, rep.end)
    return [
        ("Results read", "%d (%s)" % (rep.n_results, ", ".join(shown))),
        ("Described", rep.selected),
        ("Observations", "%d returns (%s)" % (rep.n_obs, period)),
        ("Trials counted (N)", "%d%s" % (rep.n_trials, "" if rep.trials_given else " (not given; default)"
                                         if rep.n_results == 1 else " (not given; lower bound, at most CAUTION)")),
        ("Coverage", _coverage_fact(rep)),
    ] + ([("Equity basis", _basis_fact(rep))] if rep.basis else []) + [
        ("Serial dependence", "none material (autocorrelation time %.1f < %.1f); every row counted as one "
         "observation" % (_dis.floor_to(rep.tau, 1), TAU_IGNORE) if rep.tau < TAU_IGNORE else "strong: about %.1f "
         "rows per independent observation, so %d rows count as about %d independent ones (rough estimate)"
         % (_dis.floor_to(rep.tau, 1), rep.n_obs, _dis.floor_to(rep.n_eff))),
        ("Bootstrap", "circular blocks of %d%s, %s" % (rep.block_used, "" if rep.block_used == rep.block else
         " (%d requested)" % rep.block, "not run (too few blocks)" if rep.metric("bootstrap").value is None
         else "%d resamples, seed %d" % (rep.n_resamples, rep.seed))),
    ]


def to_markdown(rep):
    """Markdown report. Deterministic for the same input and seed."""
    out = ["# Overfitting checkup", "", "> **Read this first.** " + DISCLAIMER, "",
           "## Summary", "", "Overall: **%s** (the worst verdict below)." % _LABELS[rep.overall], ""]
    out += ["- %s: %s" % (k, _md(v)) for k, v in _facts(rep)]
    out += ["", "## Results", "", "| Check | Value | Verdict | What it means |", "|---|---|---|---|"]
    out += ["| %s | %s | %s | %s |" % (_md(m.label), _md_value(m.shown), _LABELS[m.verdict], _md(m.text))
            for m in rep.metrics]
    if rep.notes:
        out += ["", "## Notes", ""] + ["- " + _md(n) for n in rep.notes]
    out += ["", "## Thresholds used", "", THRESHOLD_NOTE, "",
            "| Check | Pass | Caution | Fail |", "|---|---|---|---|"]
    out += ["| %s | %s | %s | %s |" % tuple(_md(c) for c in row) for row in _threshold_rows(rep)]
    out += ["", "With fewer than %d observations no line can pass (at most caution); with fewer than %d "
            "the checkup refuses to run." % (rep.short_obs, rep.min_obs), ""]
    return "\n".join(out)


_CSS = ("body{font-family:system-ui,sans-serif;max-width:60rem;margin:1rem auto;padding:0 1rem;"
        "color:#1b1b1b;background:#fff}table{border-collapse:collapse;width:100%;margin:.5rem 0}"
        "th,td{border:1px solid #bbb;padding:.35rem .5rem;text-align:left;vertical-align:top}"
        "blockquote{border-left:4px solid #b36b00;margin:1rem 0;padding:.25rem 1rem;background:#fff4e0}"
        ".v{font-weight:bold;white-space:nowrap}.pass{color:#0a6b2d}.caution{color:#8a5a00}.fail{color:#b00020}"
        ".info{color:#444}@media(prefers-color-scheme:dark){body{color:#e8e8e8;background:#161616}"
        "th,td{border-color:#555}blockquote{background:#2b2210}.pass{color:#5fd68a}.caution{color:#f0b840}"
        ".fail{color:#ff7d8f}.info{color:#bbb}}")
_MARK = {"pass": "+", "caution": "!", "fail": "x", "info": "-"}


def _e(text):
    return html.escape(_clean(text), quote=True)


def _row(cells, tag="td"):
    return "<tr>%s</tr>" % "".join("<%s>%s</%s>" % (tag, c, tag) for c in cells)


def to_html(rep):
    """Single-file HTML report: inline CSS, no script, nothing fetched from anywhere."""
    h = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">',
         "<title>Overfitting checkup</title><style>%s</style></head><body>" % _CSS,
         "<h1>Overfitting checkup</h1>",
         "<blockquote><strong>Read this first.</strong> %s</blockquote>" % _e(DISCLAIMER),
         "<h2>Summary</h2>",
         '<p>Overall: <span class="v %s">%s</span> (the worst verdict below).</p><ul>'
         % (rep.overall, _LABELS[rep.overall])]
    h += ["<li>%s: %s</li>" % (_e(k), _e(v)) for k, v in _facts(rep)]
    h += ["</ul><h2>Results</h2><table>", _row(["Check", "Value", "Verdict", "What it means"], "th")]
    h += [_row([_e(m.label), _e(m.shown), '<span class="v %s">%s %s</span>' % (m.verdict, _MARK[m.verdict], _LABELS[m.verdict]),
                _e(m.text)]) for m in rep.metrics]
    h.append("</table>")
    if rep.notes:
        h += ["<h2>Notes</h2><ul>"] + ["<li>%s</li>" % _e(n) for n in rep.notes] + ["</ul>"]
    h += ["<h2>Thresholds used</h2><p>%s</p><table>" % _e(THRESHOLD_NOTE), _row(["Check", "Pass", "Caution", "Fail"], "th")]
    h += [_row([_e(c) for c in row]) for row in _threshold_rows(rep)]
    h += ["</table><p>With fewer than %d observations no line can pass (at most caution); with fewer than %d "
          "the checkup refuses to run.</p></body></html>" % (rep.short_obs, rep.min_obs), ""]
    return "\n".join(h)
