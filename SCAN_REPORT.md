# Scan report

The scan covers the public files listed in `SHA256SUMS` and the checksum list itself. The scan is rerun after each fix before 0.2.0.

## 1. Forbidden terms, absolute paths, network and sensitive-looking words

Method: a case-insensitive recursive grep with one pattern per category below. The exact pattern list is kept out of this report on purpose.

Covers: internal project names and markers, local absolute path prefixes, messaging and cloud-provider names, words for keys, passphrases and similar, email addresses and IPv4 addresses.

Result: remaining matches are ordinary code and documentation terms, including `token`, publishing permissions, decorator at-signs, and the blank credential fields in the configuration fixtures.

## 2. Non-English characters

Check: Python regex for CJK ideographs and CJK / full-width punctuation over all file contents.

Result: none found in any of the 53 text files listed in `SHA256SUMS` (nor in `SHA256SUMS` itself); additionally `src/` is checked to be pure ASCII, and a test (`test_b5_2_package_ascii_and_clean`) repeats both checks on every run for `src/` and the text files listed in `SHA256SUMS` (the checksum list itself is checked by the scan above, not by the test). A search for any non-ASCII character in all 53 text files also found none (the test input that used a superscript character now writes it as an escape).

## 3. Data files

Result: source code, documentation, packaging metadata, the licence, the checksum list, one example notebook and two generated PNG pictures, plus eight tiny invented sample files in `tests/data/` (a returns CSV, an equity CSV and six freqtrade-style result files, three of them built by hand after the freqtrade 2021.1 / 2022.1 / 2026.10 source layouts and saying so in a `_fixture_note` field; round numbers on made-up dates, each under 6 KB, no real trades, accounts or balances, and no text copied from the freqtrade sources). There are also two real result sets, each holding the five unedited members (result JSON, config JSON, strategy Python file, and two Feather files) of a zip written by a freqtrade 2026.10 dry-run backtest on public Binance candles with no credentials (provenance in `docs/SPEC_B.md`, "Real result file"): a spot run in `tests/data/real_freqtrade_2026_10/` (152 KB; 99 trades of a small demonstration strategy on a 1000 USDT dry-run wallet) and a futures run in `tests/data/real_freqtrade_2026_10_futures/` (50 KB; 12 trades, 6 short, of a small moving-average strategy on a 1000 USDT dry-run wallet at 2x leverage). Both sets were checked and hold no absolute paths, no account or host names, no key values (the config JSON files have empty `key` and `secret` fields), and no candle data (only trade results, a market-change series and a wallet series). No ledger data.

Pictures (`examples/equity_psr_dsr.png`, `docs/social_preview.png`): both are drawn from a seeded synthetic series. The PNG chunk list was read byte by byte: only the header, the pixel-size chunk, the image data and the end chunk are present, so there is no text, EXIF or software chunk, and the files contain no home-directory path or user name. The picture text contains no non-English characters. The notebook has no stored outputs and no local path.

## 3b. Commit authorship

Check: every commit to be pushed has the project identity (`oss-backtest-kit`, with the GitHub no-reply address of that name) as both author and committer, verified with `git log --format='%an %ae | %cn %ce'`. No personal name or address may appear. The check is run on the public repository before any push.

Result (2026-11-15): all commits in the public repository, including the ones prepared for the next push, have that author and committer.

## 4. Numeric literals in src/ (other than 0, 1, 2)

Scanned with Python's standard lexer module (the lexer); every numeric literal whose value is not 0, 1 or 2 is listed with file:line. Reviewer notes by group:

- Protocol and unit constants: millisecond conversions (1000, 60000, 3600000, 86400000), 10**12 / 10**15 timestamp unit thresholds, page limit 1000, 65536 read chunk, 64 hex digits, 1024-based size limits.
- Statistical constants: 365.25 days per year, the Euler-Mascheroni constant, the 1.959963984540054 normal quantile for 95%, quantile levels 0.5 / 0.95 / 0.05, default block 5, default resamples 2000, S=8 blocks for CSCV, small epsilons.
- Safety limits: 300 s deadline, 30 s socket timeout, retry and back-off counts, 0.12 s and 0.25 s request pauses, 1920 request-weight budget (80% of a 2400 per minute limit), 61 s wait, 2 GiB free-disk margin.
- Placeholder cost defaults in `engine.py` (fee and slippage for the `base` and `stressed` levels): generic values, to be replaced by the user.
- Checkup and loaders (new in 0.2.0): grade thresholds (0.95 / 0.90 for PSR and DSR, 0.20 / 0.50 for PBO, Sharpe 1.0 / 0.5), observation limits (30 minimum, 180 for a pass), 100 for percent display, bootstrap defaults (2000, block 5), loader safety limits (5 million CSV rows, 128 MiB of freqtrade JSON, a 20-year span limit, 80% date overlap, 16 minimum days), timestamp range 2009 to 2100, 20 names listed in a report; serial dependence: 1.5 for the autocorrelation time below which rows count as independent, 3 as the smallest effective sample, 4 as the fewest bootstrap blocks. They are conventions from the literature or input guards.
- Nothing here is a tuned trading parameter.

```
src/oss_backtest_kit/_disclosure.py:6: 5
src/oss_backtest_kit/_disclosure.py:19: 100
src/oss_backtest_kit/_disclosure.py:42: 10
src/oss_backtest_kit/_disclosure.py:58: 3
src/oss_backtest_kit/checkup.py:32: 0.5
src/oss_backtest_kit/checkup.py:33: 0.95
src/oss_backtest_kit/checkup.py:33: 0.90
src/oss_backtest_kit/checkup.py:34: 0.95
src/oss_backtest_kit/checkup.py:34: 0.90
src/oss_backtest_kit/checkup.py:35: 0.20
src/oss_backtest_kit/checkup.py:35: 0.50
src/oss_backtest_kit/checkup.py:37: 30
src/oss_backtest_kit/checkup.py:38: 180
src/oss_backtest_kit/checkup.py:39: 30
src/oss_backtest_kit/checkup.py:206: 100
src/oss_backtest_kit/checkup.py:209: 100
src/oss_backtest_kit/checkup.py:210: 100
src/oss_backtest_kit/checkup.py:216: 0.999
src/oss_backtest_kit/checkup.py:216: 100
src/oss_backtest_kit/checkup.py:246: 4
src/oss_backtest_kit/checkup.py:294: 2000
src/oss_backtest_kit/checkup.py:294: 5
src/oss_backtest_kit/checkup.py:322: 3.0
src/oss_backtest_kit/checkup.py:351: 0.05
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
src/oss_backtest_kit/data/archive.py:34: 1024
src/oss_backtest_kit/data/archive.py:34: 3
src/oss_backtest_kit/data/archive.py:35: 1024
src/oss_backtest_kit/data/archive.py:35: 3
src/oss_backtest_kit/data/archive.py:88: 12
src/oss_backtest_kit/data/archive.py:132: 100
src/oss_backtest_kit/data/archive.py:147: 10
src/oss_backtest_kit/data/archive.py:147: 15
src/oss_backtest_kit/data/archive.py:148: 1000
src/oss_backtest_kit/data/archive.py:149: 10
src/oss_backtest_kit/data/archive.py:149: 12
src/oss_backtest_kit/data/archive.py:149: 1000
src/oss_backtest_kit/data/archive.py:151: 1000
src/oss_backtest_kit/data/archive.py:231: 0o600
src/oss_backtest_kit/data/archive.py:260: 4096
src/oss_backtest_kit/data/archive.py:321: 0.12
src/oss_backtest_kit/data/archive.py:352: 4
src/oss_backtest_kit/data/archive.py:352: 5
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
src/oss_backtest_kit/engine.py:333: 5
src/oss_backtest_kit/engine.py:399: 4
src/oss_backtest_kit/engine.py:404: 4
src/oss_backtest_kit/engine.py:406: 3
src/oss_backtest_kit/engine.py:408: 3
src/oss_backtest_kit/loaders/_series.py:60: 1230768000
src/oss_backtest_kit/loaders/_series.py:60: 4102444800
src/oss_backtest_kit/loaders/_series.py:71: 86400
src/oss_backtest_kit/loaders/_series.py:101: 1000
src/oss_backtest_kit/loaders/_series.py:101: 1000
src/oss_backtest_kit/loaders/_series.py:102: 1000
src/oss_backtest_kit/loaders/_series.py:132: 59
src/oss_backtest_kit/loaders/_series.py:134: 60
src/oss_backtest_kit/loaders/csvload.py:15: 5_000_000
src/oss_backtest_kit/loaders/freqtrade_results.py:21: 128
src/oss_backtest_kit/loaders/freqtrade_results.py:21: 1024
src/oss_backtest_kit/loaders/freqtrade_results.py:21: 1024
src/oss_backtest_kit/loaders/freqtrade_results.py:22: 366
src/oss_backtest_kit/loaders/freqtrade_results.py:22: 20
src/oss_backtest_kit/loaders/freqtrade_results.py:172: 1e-6
src/oss_backtest_kit/loaders/freqtrade_results.py:172: 1e-6
src/oss_backtest_kit/loaders/matrix.py:11: 16
src/oss_backtest_kit/loaders/matrix.py:12: 0.8
src/oss_backtest_kit/loaders/matrix.py:86: 100
src/oss_backtest_kit/loaders/matrix.py:86: 100
src/oss_backtest_kit/report.py:20: 20
src/oss_backtest_kit/report.py:67: 100.0
src/oss_backtest_kit/report.py:67: 3
src/oss_backtest_kit/report.py:67: 4
src/oss_backtest_kit/report.py:67: 5
src/oss_backtest_kit/stats.py:23: 0.5772156649015329
src/oss_backtest_kit/stats.py:24: 365.25
src/oss_backtest_kit/stats.py:25: 1.5
src/oss_backtest_kit/stats.py:54: 16
src/oss_backtest_kit/stats.py:96: 3
src/oss_backtest_kit/stats.py:111: 3
src/oss_backtest_kit/stats.py:115: 3
src/oss_backtest_kit/stats.py:116: 4
src/oss_backtest_kit/stats.py:117: 1e-12
src/oss_backtest_kit/stats.py:117: 4.0
src/oss_backtest_kit/stats.py:178: 3
src/oss_backtest_kit/stats.py:198: 8
src/oss_backtest_kit/stats.py:217: 8
src/oss_backtest_kit/stats.py:225: 4
src/oss_backtest_kit/stats.py:234: 1e-12
src/oss_backtest_kit/stats.py:248: 0.5
src/oss_backtest_kit/stats.py:252: 1.959963984540054
src/oss_backtest_kit/stats.py:259: 4.0
src/oss_backtest_kit/stats.py:263: 2000
src/oss_backtest_kit/stats.py:263: 5
src/oss_backtest_kit/stats.py:297: 0.5
src/oss_backtest_kit/stats.py:297: 0.95
src/oss_backtest_kit/stats.py:298: 0.05
src/oss_backtest_kit/stats.py:298: 0.5
src/oss_backtest_kit/stats.py:308: 2000
src/oss_backtest_kit/stats.py:308: 5
src/oss_backtest_kit/stats.py:308: 0.95
src/oss_backtest_kit/stats.py:346: 3
```
