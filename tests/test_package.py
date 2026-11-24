"""Release hygiene tests: spec items B4.2, B4.3, B5.2 in docs/SPEC_B.md."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import oss_backtest_kit  # noqa: E402

PUBLIC_TOP = ["README.md", "CHANGELOG.md", "LICENSE", "pyproject.toml", "MANIFEST.in", "SCAN_REPORT.md"]
TEXT_SUFFIX = {".py", ".md", ".ipynb", ".yml", ".csv", ".json", ".toml", ".in", ""}


def _public_text_files():
    files = [ROOT / n for n in PUBLIC_TOP if (ROOT / n).exists()]
    for sub in ("src", "tests", "examples", ".github", "docs"):
        for p in sorted((ROOT / sub).rglob("*")):
            if p.is_file() and p.suffix in TEXT_SUFFIX and "__pycache__" not in p.parts:
                files.append(p)
    assert len(files) > 20
    return files


def test_b4_2_readme_sections():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    heads = re.findall(r"^## (.+)$", text, flags=re.M)
    assert heads[0] == "Try it in 30 seconds"                      # first section
    assert text.index("## Try it in 30 seconds") < text.index("![Equity curve")
    assert "Checkup: is this backtest overfitted?" in heads
    section = text.split("## Checkup: is this backtest overfitted?")[1].split("\n## ")[0]
    for needle in ("python -m oss_backtest_kit.checkup", "--trials", "examples/checkup_demo.py",
                   "cannot show that a strategy will make money", "PASS, CAUTION, FAIL"):
        assert needle in section


def test_b4_3_changelog_and_version():
    version = oss_backtest_kit.__version__
    assert version == "0.2.0"   # written out on purpose: whoever changes the version must update this test too
    assert 'version = "%s"' % version in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.findall(r"^## (\S+)", log, flags=re.M)[:2] == [version, "0.1.0"]
    assert "version %s" % version in (ROOT / "README.md").read_text(encoding="utf-8")


def test_b5_2_package_ascii_and_clean():
    # the package source is plain ASCII; no public text file holds CJK ideographs or punctuation
    for p in sorted((ROOT / "src").rglob("*.py")):
        p.read_text(encoding="ascii")
    cjk = re.compile("[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef]")
    for p in _public_text_files():
        if "audit_ce" in p.parts:  # directory excluded from the package
            continue
        assert not cjk.search(p.read_text(encoding="utf-8")), p
    # no local home-directory paths in any public text file
    for p in _public_text_files():
        text = p.read_text(encoding="utf-8")
        for prefix in ("/Us" + "ers/", "/ho" + "me/", "C:\\\\Us" + "ers"):
            assert prefix not in text, (p, prefix)


def test_spec_names_for_b4_b5_exist_in_tests():
    spec = (ROOT / "docs" / "SPEC_B.md").read_text(encoding="utf-8")
    named = set(re.findall(r"`(test_b[45]_\w+)`", spec))
    defined = set()
    for p in (ROOT / "tests").glob("test_*.py"):
        defined |= set(re.findall(r"^def (test_b[45]_\w+)", p.read_text(encoding="utf-8"), re.M))
    assert named and named == defined
