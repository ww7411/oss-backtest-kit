"""What the aligned window, the equity basis and serial dependence hide (spec items B2.25 / B2.26 / B2.30)."""
import math

from . import stats

DROP_CAP = 5  # percent: more of the described result excluded than this, and no line can be a pass


def coverage(sources, names, kept):
    """One row per result: (name, kept, n, before, after, inside); None when sources were not recorded."""
    if sources is None:
        return None
    return [(k, kept, sources[k]["n"], sources[k]["before"], sources[k]["after"], sources[k]["inside"])
            for k in names]


def too_much_dropped(row):
    """True when strictly more than DROP_CAP percent of the described result was excluded."""
    return row is not None and (row[2] - row[1]) * 100 > DROP_CAP * row[2]


def coverage_notes(row):
    if row is None or row[2] == row[1]:
        return []
    notes = ["Every line describes only the common window. %d returns of %s outside it were excluded and "
             "not checked; its full history may look different." % (row[2] - row[1], row[0])]
    if too_much_dropped(row):
        notes.append("More than %d%% of its returns were excluded, so no line can be a pass." % DROP_CAP)
    return notes


def basis_notes(basis):
    if basis != "realised":
        return []
    return ["Returns come from closed trades only: a losing position held open shows no loss until it closes, "
            "so the drawdown and the day-to-day risk can be understated. Export an equity curve with open "
            "positions marked to market for a fuller picture."]


def floor_to(x, digits=0):
    """Round down, so a shown number is on the same side of a limit as the number itself."""
    k = 10 ** digits
    f = math.floor(x * k)
    return f / k if digits else f


def dependence(r, block):
    """(tau, n_eff, block_used) of the described returns; below TAU_IGNORE rows count as independent."""
    tau = stats.autocorr_time(r)
    if tau < stats.TAU_IGNORE:
        return tau, float(r.size), block
    return tau, r.size / tau, max(block, math.ceil(tau))


def dependence_notes(tau, n_eff, rows, block_used):
    if tau < stats.TAU_IGNORE:
        return []
    formula_minimum = " (the formulas use at least 3)" if n_eff < 3 else ""
    return ["The returns are serially dependent (autocorrelation time %.1f, Geyer initial positive sequence). "
            "PSR, DSR and the sample-size rule use about %d effective observations instead of %d rows%s, and the "
            "bootstrap uses blocks of %d. The estimate is rough and tends to be optimistic for very persistent "
            "series. PBO is not adjusted." % (floor_to(tau, 1), floor_to(n_eff), rows, formula_minimum, block_used)]
