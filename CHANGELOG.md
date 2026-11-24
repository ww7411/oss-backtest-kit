# Changelog

## 0.2.0

Overfitting checkup.

- `oss_backtest_kit.loaders`: read backtest results from other tools without installing them: freqtrade result files (`.json` / `.zip`), CSV (returns or equity curve), NumPy arrays and pandas Series (pandas stays optional), and a folder of results aligned into a return matrix. Unreadable fields raise a clear error; nothing is guessed.
- The freqtrade reader was checked on a real result written by freqtrade 2026.10 (its files are kept in `tests/data/real_freqtrade_2026_10/` and zipped by the test, provenance in `docs/SPEC_B.md`) and on a futures run with long and short trades (`tests/data/real_freqtrade_2026_10_futures/`); other freqtrade versions are read from their sources only.
- `oss_backtest_kit.checkup` and `python -m oss_backtest_kit.checkup`: annualised Sharpe, maximum drawdown, PSR, DSR (with the number of trials you give), effective number of trials, PBO (two or more results) and a circular block bootstrap interval, each with a verdict (pass / caution / fail) and one plain sentence. Short or sparse series are capped at caution; fewer than 30 observations are refused.
- Reports: Markdown with a disclaimer and the thresholds used, and an optional single-file HTML report (inline CSS, no scripts, no network).
- `stats.block_bootstrap_sharpe`: percentile interval of the annualised Sharpe ratio.
- Example `examples/checkup_demo.py` (synthetic data, offline: one overfitted and one robust group), README sections "Checkup" and the 30-second try-it moved to the top, and a written specification (`docs/SPEC_B.md`) with one test or more per item.
- Checkup: one result without `--trials` can no longer pass (the DSR line says it cannot be judged and the overall verdict is at most CAUTION); command-line switches `--strategy`, `--starting-balance`, `--time-col`, `--value-col`, `--kind` for the loader options, and loader errors name them; freqtrade formats documented per version (2021.1 to 2026.10) with test fixtures; PSR above 0.999 reads "more than 99.9%"; fewer backslashes in terminal output.
- Checkup grades stricter: an equity curve with unevenly spaced points (for example a 41-day jump between daily points) is now refused with an error, because PSR, DSR, PBO and the bootstrap assume equal periods. Add the missing dates, or pass `fill_flat=True` / `--fill-flat` if the account truly did not change on the missing days (the report says how many days were filled). Loaders now record where each series came from (`DailyReturns.spans`, `.basis`, `ReturnMatrix.sources`).
- Checkup grades stricter again: when more than 5% of the described result's returns were excluded by date alignment, no line can be a pass, and the report writes how many were excluded (before, after and inside the common window). A freqtrade report now says it was built from closed trades only (open positions are not marked to market).
- Checkup grades stricter: several results without `--trials` can no longer pass either (N is the number of results, only a lower bound, so the DSR line and the overall verdict are at most CAUTION); returns with persistent serial dependence are judged on the effective number of observations (`stats.autocorr_time`, new `n_obs` argument of `stats.psr` / `stats.dsr`), with longer bootstrap blocks, and too few effective observations cap every line at caution. The report adds a `Serial dependence` line.
- Checkup input and wording: `block` must be a whole number of at least 1 (a `CheckupError` otherwise), an array of more than two dimensions is a `CheckupError`, shown autocorrelation times and effective counts are rounded down so they never contradict the limit they are compared with, the PBO line says how many splits were left out because results tied, a long bootstrap block that leaves fewer than 4 blocks is named as such, and the error for unevenly spaced results in an already aligned matrix says to pass `fill_flat` when loading.
- Fix: the download manifest is now rewritten to a temporary file and replaced atomically, so a write error can no longer leave a partial last line.
- The sdist now includes the test data files.
- The `test` and `dev` extras now install pandas, so a clean `pip install .[test]` runs every test; the denied-word test compares hashes, so the packaged tests no longer carry readable internal names.
- Serial dependence notes show the rounded-down effective observation count and explain the formula minimum when the estimate is below three.

### Fixes in the 0.1.0 API

- `stats.sharpe`, `stats.psr`, `stats.dsr`, and the bootstrap judge zero variance relative to the scale of the returns, so constant returns now give a Sharpe of 0 and a PSR of `NaN`, where 0.1.0 could give a Sharpe of about 5.75e15.
- `stats.max_drawdown` now works on log equity to avoid overflow on long series.
- `stats.pbo_cscv` now breaks in-sample Sharpe ties by return sum and skips splits that still tie, so it can return `None` even with two or more columns (0.1.0 returned `None` only with fewer than two), and the new `stats.pbo_cscv_detail` also returns the count of skipped splits.
- If an interrupted 0.1.0 download left a partial manifest row, `data` downloads the corresponding file again and `engine` raises an error if it reads the unfinished row.

## 0.1.0

First packaged release.

- Multiple-testing statistics: Sharpe, PSR, DSR, PBO (CSCV), effective number of trials, circular block bootstrap, Wilson interval.
- Binance public archive download with SHA-256 verification, manifest, and gap / duplicate checks; REST helper that exports recent bars to separate CSV files.
- Config-driven perpetual-futures backtest engine: next-open fills, funding, `base` and `stressed` cost levels.
- Packaging metadata, README quick start, MIT licence.
- Examples: effect chart script with PSR / DSR computed by the library (`examples/equity_psr_dsr.py`, matplotlib via the optional `examples` extra), a Colab quick start notebook (`examples/quickstart.ipynb`), and a 1280x640 social preview card; smoke tests for the examples. Release workflow runs the offline tests first and requires the tag to equal `v` plus the pyproject version; CI runs on pushes to `main` and on pull requests. The test jobs install matplotlib and nbformat (in the `test` extra) so the example tests run instead of being skipped; the release workflow also refuses a tag whose commit is not on `main`. The full-dependency test jobs also fail if any test is skipped.
- Dependency floors (numpy 1.21.6, scipy 1.7.3), CI workflow for Python 3.9 to 3.13, tag-triggered release workflow (PyPI Trusted Publishing), sdist includes this changelog.
