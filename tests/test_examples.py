"""Smoke tests for the examples. Plot and notebook checks skip when their libraries are missing."""

import io
import json
import os
import struct
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import equity_psr_dsr  # noqa: E402
import ma_crossover  # noqa: E402


def test_ma_crossover_example_runs_and_prints_both_cost_levels():
    buf = io.StringIO()
    with redirect_stdout(buf):
        ma_crossover.main()
    out = buf.getvalue()
    assert "base" in out and "stressed" in out and "DSR=" in out


def test_effect_chart_numbers_are_deterministic_and_deflated():
    a, b = equity_psr_dsr.compute(), equity_psr_dsr.compute()
    assert a["variant"] == b["variant"] and a["psr"] == b["psr"] and a["dsr"] == b["dsr"]
    assert a["n_trials"] == 9 and len(a["equity"]) == 1000
    assert 0.0 < a["dsr"] < a["psr"] < 1.0       # deflating by 9 trials must lower the score
    assert a["variant"] in equity_psr_dsr.GRID
    assert a["psr"] == pytest.approx(0.526, abs=2e-3)   # hand-copied from the shipped picture
    assert a["dsr"] == pytest.approx(0.073, abs=2e-3)


def test_effect_chart_other_seed_gives_other_numbers():
    assert equity_psr_dsr.compute(seed=1)["psr"] != equity_psr_dsr.compute(seed=0)["psr"]


def _png_size(path):
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks, i = [], 8
    while i < len(data):
        n, = struct.unpack(">I", data[i:i + 4])
        chunks.append(data[i + 4:i + 8].decode())
        if chunks[-1] == "IHDR":
            size = struct.unpack(">II", data[i + 8:i + 16])
        i += 12 + n
    return size, chunks


def test_pictures_are_written_with_expected_size_and_no_text_metadata():
    pytest.importorskip("matplotlib")
    with tempfile.TemporaryDirectory() as tmp:
        chart, card = os.path.join(tmp, "c.png"), os.path.join(tmp, "s.png")
        equity_psr_dsr.main(["--out", chart, "--social", card])
        assert _png_size(chart)[0] == (900, 500)
        size, chunks = _png_size(card)
        assert size == (1280, 640)
        assert not {"tEXt", "iTXt", "zTXt", "eXIf"} & set(chunks)


def _notebook():
    return json.loads((ROOT / "examples" / "quickstart.ipynb").read_text(encoding="utf-8"))


def test_notebook_is_valid_and_clean():
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(ROOT / "examples" / "quickstart.ipynb"), as_version=4)
    nbformat.validate(nb)
    text = json.dumps(_notebook())
    for home_dir in ("/Us" + "ers/", "/ho" + "me/", "\\\\Us" + "ers\\\\"):
        assert home_dir not in text
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def test_notebook_code_cells_run_offline_without_the_install_cell():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    sources = ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "code"]
    assert sources[0].lstrip().startswith("#") and "%pip install" in sources[0]
    scope = {"__name__": "nb"}
    buf = io.StringIO()
    with redirect_stdout(buf):
        for src in sources[1:]:
            exec(compile(src, "<cell>", "exec"), scope)
    assert "best variant (fast, slow): (10, 30)" in buf.getvalue()
    assert np.isclose(scope["dsr_value"], equity_psr_dsr.compute()["dsr"])
