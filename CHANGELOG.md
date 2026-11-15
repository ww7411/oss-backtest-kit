# Changelog

## 0.1.0

First packaged release.

- Multiple-testing statistics: Sharpe, PSR, DSR, PBO (CSCV), effective number of trials, circular block bootstrap, Wilson interval.
- Binance public archive download with SHA-256 verification, manifest, and gap / duplicate checks; REST helper that exports recent bars to separate CSV files.
- Config-driven perpetual-futures backtest engine: next-open fills, funding, `base` and `stressed` cost levels.
- Packaging metadata, README quick start, MIT licence.
- Examples: effect chart script with PSR / DSR computed by the library (`examples/equity_psr_dsr.py`, matplotlib via the optional `examples` extra), a Colab quick start notebook (`examples/quickstart.ipynb`), and a 1280x640 social preview card; smoke tests for the examples. Release workflow runs the offline tests first and requires the tag to equal `v` plus the pyproject version; CI runs on pushes to `main` and on pull requests. The test jobs install matplotlib and nbformat (in the `test` extra) so the example tests run instead of being skipped; the release workflow also refuses a tag whose commit is not on `main`. The full-dependency test jobs also fail if any test is skipped.
- Dependency floors (numpy 1.21.6, scipy 1.7.3), CI workflow for Python 3.9 to 3.13, tag-triggered release workflow (PyPI Trusted Publishing), sdist includes this changelog.
