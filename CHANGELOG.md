# Changelog

## 0.1.0

First packaged release.

- Multiple-testing statistics: Sharpe, PSR, DSR, PBO (CSCV), effective number of trials, circular block bootstrap, Wilson interval.
- Binance public archive download with SHA-256 verification, manifest, and gap / duplicate checks; REST helper that exports recent bars to separate CSV files.
- Config-driven perpetual-futures backtest engine: next-open fills, funding, `base` and `stressed` cost levels.
- Packaging metadata, README quick start, MIT licence.
- Dependency floors (numpy 1.21.6, scipy 1.7.3), CI workflow for Python 3.9 to 3.13, tag-triggered release workflow (PyPI Trusted Publishing), sdist includes this changelog.
