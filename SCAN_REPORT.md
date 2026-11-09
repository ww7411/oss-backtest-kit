# Scan report

Scope: every file in this folder (the report itself included). Case-insensitive.

## 1. Forbidden terms, absolute paths, network and sensitive-looking words

Method: a case-insensitive recursive grep with one pattern per category below. The exact pattern list is kept out of this report on purpose.

Covers: internal project names and markers, local absolute path prefixes, messaging and cloud-provider names, words for keys, passphrases and similar, email addresses and IPv4 addresses.

Result: no matches.

## 2. Non-English characters

Check: Python regex for CJK ideographs and CJK / full-width punctuation over all file contents.

Result: none found.

## 3. Data files

Result: only source code, documentation, the licence and the checksum list. No data files.

## 4. Numeric literals in src/ (other than 0, 1, 2)

Scanned with Python's standard lexer module (the lexer); every numeric literal whose value is not 0, 1 or 2 is listed with file:line. Reviewer notes by group:

- Protocol and unit constants: millisecond conversions (1000, 60000, 3600000, 86400000), 10**12 / 10**15 timestamp unit thresholds, page limit 1000, 65536 read chunk, 64 hex digits, 1024-based size limits.
- Statistical constants: 365.25 days per year, the Euler-Mascheroni constant, the 1.959963984540054 normal quantile for 95%, quantile levels 0.5 / 0.95 / 0.05, default block 5, default resamples 2000, S=8 blocks for CSCV, small epsilons.
- Safety limits: 300 s deadline, 30 s socket timeout, retry and back-off counts, 0.12 s and 0.25 s request pauses, 1920 request-weight budget (80% of a 2400 per minute limit), 61 s wait, 2 GiB free-disk margin.
- Placeholder cost defaults in `engine.py` (fee and slippage for the `base` and `stressed` levels): generic values, to be replaced by the user.
- Nothing here is a tuned trading parameter.

```
src/oss_backtest_kit/data/__main__.py:63: 4
src/oss_backtest_kit/data/_http.py:59: 64
src/oss_backtest_kit/data/_http.py:59: 1024
src/oss_backtest_kit/data/_http.py:59: 1024
src/oss_backtest_kit/data/_http.py:60: 300.0
src/oss_backtest_kit/data/_http.py:60: 30.0
src/oss_backtest_kit/data/_http.py:60: 4
src/oss_backtest_kit/data/_http.py:76: 443
src/oss_backtest_kit/data/_http.py:90: 404
src/oss_backtest_kit/data/_http.py:92: 300
src/oss_backtest_kit/data/_http.py:92: 400
src/oss_backtest_kit/data/_http.py:94: 429
src/oss_backtest_kit/data/_http.py:94: 500
src/oss_backtest_kit/data/_http.py:100: 200
src/oss_backtest_kit/data/_http.py:101: 300
src/oss_backtest_kit/data/_http.py:101: 400
src/oss_backtest_kit/data/_http.py:108: 65536
src/oss_backtest_kit/data/archive.py:32: 1024
src/oss_backtest_kit/data/archive.py:32: 3
src/oss_backtest_kit/data/archive.py:33: 1024
src/oss_backtest_kit/data/archive.py:33: 3
src/oss_backtest_kit/data/archive.py:86: 12
src/oss_backtest_kit/data/archive.py:130: 100
src/oss_backtest_kit/data/archive.py:145: 10
src/oss_backtest_kit/data/archive.py:145: 15
src/oss_backtest_kit/data/archive.py:146: 1000
src/oss_backtest_kit/data/archive.py:147: 10
src/oss_backtest_kit/data/archive.py:147: 12
src/oss_backtest_kit/data/archive.py:147: 1000
src/oss_backtest_kit/data/archive.py:149: 1000
src/oss_backtest_kit/data/archive.py:187: 4096
src/oss_backtest_kit/data/archive.py:253: 0.12
src/oss_backtest_kit/data/archive.py:284: 4
src/oss_backtest_kit/data/archive.py:284: 5
src/oss_backtest_kit/data/check.py:21: 60000
src/oss_backtest_kit/data/check.py:21: 3600000
src/oss_backtest_kit/data/check.py:21: 86400000
src/oss_backtest_kit/data/check.py:21: 7
src/oss_backtest_kit/data/check.py:21: 86400000
src/oss_backtest_kit/data/check.py:34: 1000
src/oss_backtest_kit/data/check.py:34: 10
src/oss_backtest_kit/data/check.py:34: 15
src/oss_backtest_kit/data/check.py:88: 6
src/oss_backtest_kit/data/check.py:88: 6
src/oss_backtest_kit/data/check.py:113: 3600000.0
src/oss_backtest_kit/data/rest_tail.py:27: 1920
src/oss_backtest_kit/data/rest_tail.py:40: 0.25
src/oss_backtest_kit/data/rest_tail.py:51: 61
src/oss_backtest_kit/data/rest_tail.py:61: 1000
src/oss_backtest_kit/data/rest_tail.py:62: 1000
src/oss_backtest_kit/data/rest_tail.py:96: 6
src/oss_backtest_kit/data/rest_tail.py:122: 1000
src/oss_backtest_kit/engine.py:68: 0.0004
src/oss_backtest_kit/engine.py:68: 0.0002
src/oss_backtest_kit/engine.py:69: 0.0010
src/oss_backtest_kit/engine.py:69: 0.0005
src/oss_backtest_kit/engine.py:88: 365.25
src/oss_backtest_kit/engine.py:88: 86400000.0
src/oss_backtest_kit/engine.py:187: 1000
src/oss_backtest_kit/engine.py:215: 1e-9
src/oss_backtest_kit/engine.py:306: 1000
src/oss_backtest_kit/engine.py:306: 10
src/oss_backtest_kit/engine.py:306: 15
src/oss_backtest_kit/engine.py:330: 5
src/oss_backtest_kit/engine.py:396: 4
src/oss_backtest_kit/engine.py:401: 4
src/oss_backtest_kit/engine.py:403: 3
src/oss_backtest_kit/engine.py:405: 3
src/oss_backtest_kit/stats.py:23: 0.5772156649015329
src/oss_backtest_kit/stats.py:24: 365.25
src/oss_backtest_kit/stats.py:80: 3
src/oss_backtest_kit/stats.py:84: 3
src/oss_backtest_kit/stats.py:85: 4
src/oss_backtest_kit/stats.py:86: 1e-12
src/oss_backtest_kit/stats.py:86: 4.0
src/oss_backtest_kit/stats.py:146: 3
src/oss_backtest_kit/stats.py:159: 8
src/oss_backtest_kit/stats.py:175: 4
src/oss_backtest_kit/stats.py:182: 1e-18
src/oss_backtest_kit/stats.py:183: 1e-18
src/oss_backtest_kit/stats.py:189: 0.5
src/oss_backtest_kit/stats.py:193: 1.959963984540054
src/oss_backtest_kit/stats.py:200: 4.0
src/oss_backtest_kit/stats.py:204: 2000
src/oss_backtest_kit/stats.py:204: 5
src/oss_backtest_kit/stats.py:238: 0.5
src/oss_backtest_kit/stats.py:238: 0.95
src/oss_backtest_kit/stats.py:239: 0.05
src/oss_backtest_kit/stats.py:239: 0.5
```
