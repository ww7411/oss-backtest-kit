# SPEC B: overfitting checkup (0.2.0)

Status: B1 to B5 implemented and tested. Each item has one identifier and one
or more regression tests. A test name listed here that is missing from the
public test suite at release time is a release blocker.

Scope reminder: the checkup lowers the risk of shipping an overfitted
backtest. It cannot show that a strategy makes money.

Conventions
- Inputs are simple per-period returns as fractions (0.01 = 1%). Periods are
  UTC calendar days.
- Runtime dependencies stay numpy and scipy. pandas is optional and never
  imported by the package.
- Errors are explicit (`LoaderError`, `StatsError`); nothing is guessed.

## B1. Reading other tools' results (`oss_backtest_kit.loaders`)

### Supported inputs and formats

freqtrade backtest results (`user_data/backtest_results/*.json` or `*.zip`).
Source of the format: the freqtrade documentation and sources read (text only, nothing installed or run) on
2026-11-16: `freqtrade/optimize/optimize_reports/bt_storage.py` (what is written into the zip),
`freqtrade/optimize/optimize_reports/optimize_reports.py` (`generate_strategy_stats`: the per-strategy
fields), `freqtrade/data/btanalysis/bt_fileutils.py` (`BT_DATA_COLUMNS`: the per-trade fields) and
`freqtrade/misc.py` (`dump_json_to_file`: how dates are written), at tag 2026.10 (the latest release on that
day; `develop` was at 7eb798c), and the same files at tags 2025.1, 2024.1, 2023.1, 2022.1 and 2021.1 (older
tags have `freqtrade/optimize/optimize_reports.py` and `freqtrade/data/btanalysis.py` as single files).
freqtrade writes no schema version this reader can rely on, so the format is recognised by structure, not by
version number.

What was checked against the source, field by field (three of the fixtures in `tests/data` were built by hand
from this, with invented numbers; a fourth set, the five files in `tests/data/real_freqtrade_2026_10/`, was written by a
real freqtrade 2026.10 run, see "Real result file" below):

| What | Source says | Reader |
|---|---|---|
| Result file | 2025.1 and later: `<stem>.zip` (2025.1 source shows the `.json` and the market-change member; the full list below is from 2026.10) holding `<stem>.json` (top-level `strategy` and `strategy_comparison`), `<stem>_config.json`, `<stem>_<strategy>.py`, `<stem>_<strategy>.json` (strategy parameters), `<stem>_market_change.feather`, `<stem>_<strategy>_wallet.feather`; next to it `<stem>.meta.json` and `.last_result.json` (`{"latest_backtest": "<zip name>"}`). 2024.1 and older: the same top-level json written directly as `backtest-result-<datetime>.json` | picks `<stem>.json`; the other members are never read; sidecars are skipped / rejected with a pointer error |
| Strategy object, 2022.1 to 2026.10 | has `trades`, `profit_total_abs`, `starting_balance`, `dry_run_wallet`, `backtest_start`, `backtest_end` (format `YYYY-MM-DD HH:MM:SS`, UTC, no offset) | all read |
| Strategy object, 2021.1 | has no `starting_balance` or `dry_run_wallet`; `backtest_start` / `backtest_end` are datetimes written by `str()` (`YYYY-MM-DD HH:MM:SS+00:00`) | works with `starting_balance=` / `--starting-balance` |
| Trade object | `open_date`, `close_date` (written by `default=str` from a pandas Timestamp: `YYYY-MM-DD HH:MM:SS+00:00`), `profit_abs`; 2022.1 has `sell_reason` / `buy_tag`, 2026.10 has `exit_reason` / `enter_tag`, `leverage`, `is_short`, `funding_fees`, `orders`, `open_timestamp`, `close_timestamp` | only the three fields are read; short trades and futures need nothing special because `profit_abs` is already the account profit |
| `profit_total_abs` | `results["profit_abs"].sum()` in every version checked | used for the reconciliation below |

Verified: the layout and field names above, by reading the sources of the tags named; the reader against the
three hand-built fixtures (2026.10 futures with a short trade, 2022.1 spot, 2021.1 without a starting balance)
and a zip built with the member list of 2026.10; and the reader against a real file written by freqtrade 2026.10
(see "Real result file" below; one spot result with long trades only, and one futures result with long and short
trades, both one strategy). Not verified: files written by other
freqtrade versions (2025.1 to 2026.9 zips are assumed to match the 2026.10 member list), futures / short trades
in a real file of any version other than 2026.10, files with several strategies from a real run, and zips that were renamed (the reader
then cannot pick the results member and says so); versions before 2021.1 and any later change of the layout are
not covered; hyperopt results and `backtest-result-*.pkl` are not read. If a file's fields do not fit, the
reader stops with an error naming the field, and the `profit_total_abs` check turns a misread profit field into
an error instead of a silently wrong series. Re-check this list when freqtrade changes its result format.

### Real result file

`tests/data/real_freqtrade_2026_10/` holds the five members of a zip written by freqtrade 2026.10 (CCXT 4.5.93,
Python 3.12, installed from PyPI), unedited and byte for byte, in the order the zip lists them: `backtest-result-2026-11-23_14-38-07.json`
(results), `..._config.json`, `..._RsiLikeStrategy.py` (the strategy, a small RSI strategy written for this check),
`..._market_change.feather`, `..._RsiLikeStrategy_wallet.feather`; together 152 KB. The zip itself is not stored; the tests rebuild it in a temporary folder with the same name, members and order.
Source of the run: exchange Binance (public candle data only, no credentials or account), pairs BTC/USDT and
ETH/USDT, spot, 1h candles, `--timerange 20261011-20261101`, 1000 USDT dry-run wallet, stake 100 USDT, 99 trades.
Commands: `freqtrade download-data -c cfg.json --userdir ud --timerange 20260801-20261101 -t 1h` and `freqtrade
backtesting -c cfg.json --userdir ud --strategy RsiLikeStrategy --timerange 20261011-20261101 --export trades`. The
strategy has no parameters, so there is no strategy-parameter json. Redacted fields: none were needed (the config has
empty credential fields and no path or host name; the files were searched for home-directory prefixes, user and
host names and time-zone names, and a test repeats the search for prefixes and internal names). The files are
trade records and two series computed from public exchange candles, not the raw candles (see the data terms in the README).
Tests: `test_b1_22_real_freqtrade_2026_10_zip_reads_and_reconciles` (loader) and
`test_b2_21_checkup_on_real_freqtrade_zip` (checkup; the 22 daily results are fewer than the 30 the checkup needs, so it
refuses them with a clear message, and a threefold repetition of the series passes through the whole checkup).

Futures run: `tests/data/real_freqtrade_2026_10_futures/` holds the five members of a second zip from freqtrade 2026.10
(CCXT 4.5.94, Python 3.14, installed from PyPI without the optional systemd notifier), unedited, 60 KB; the zip is not stored
and the tests rebuild it. Binance public candle data only (no credentials), pair BTC/USDT:USDT, perpetual futures, isolated
margin, 4h candles, `--timerange 20261010-20261121` (43 daily results), 1000 USDT dry-run wallet, stake 100 USDT, 2x leverage,
12 trades of which 6 are short. The strategy `SmaCrossStrategy` is a small long/short moving-average cross written for this
check. Commands: `freqtrade download-data -c cfg.json --userdir ud --timerange 20260915-20261123 -t 4h --trading-mode futures` and
`freqtrade backtesting -c cfg.json --userdir ud --strategy SmaCrossStrategy --timerange 20261010-20261121 -i 4h --export trades`.
The config sets `use_order_book` for pricing because freqtrade requires it for Binance futures. No redaction was needed.
The reader read it without any change and the daily series reconciled with the file's final balance. This verifies futures and
short trades on freqtrade 2026.10 only.

What the real file showed against the source-based assumptions: all of them held. Dates are
`YYYY-MM-DD HH:MM:SS+00:00` in trades and `YYYY-MM-DD HH:MM:SS` in `backtest_start` / `backtest_end`; trade
`profit_abs` values are plain JSON floats (small ones in exponent form, for example `4.41e-06`; no `NaN`);
`profit_total_abs` equals the sum of the trades' `profit_abs`; the reader gives 22 daily returns (2026-10-11 to
2026-11-01), the series name is the zip's stem (one strategy), and the reconciliation did not reject the file.
The only surprise was in how the file is used, not in its content: a zip renamed after the run is not
read (its results member no longer carries the zip's name and the config json makes the choice ambiguous);
the reader says so in an error (B1.20).

| Layout | Status |
|---|---|
| Top-level object with `"strategy"`: `{name: {"trades": [...], ...}}` (what freqtrade 2021.1 to 2026.10 write) | supported (2022.1 and later with the balance in the file) |
| Same layout inside a `.zip` (member `<zip name>.json`; if absent, the only non-`.meta.json` json member) | supported |
| Bare list of trades (old layout; freqtrade itself no longer reads it) | rejected with an error |
| `.last_result.json` pointer file | rejected with an error naming the pointer |
| Anything else | rejected: "unknown layout" |

Fields read. Per trade: `close_date` and `profit_abs` (required), `open_date`
(optional). Per strategy: `starting_balance`, else `dry_run_wallet` (one of the
two, or the `starting_balance=` argument / `--starting-balance` switch, is required), and optionally
`backtest_start` / `backtest_end`. All other fields are ignored. Because only a
few fields are used, files from nearby freqtrade versions that keep these
fields should load; a file that lacks them fails with an error naming the
field. No claim is made for versions older than 2021.1.

Daily returns from trades: realised-profit accounting. The return of a day is
`profit_abs` of the trades closed that day divided by the equity at the start
of that day (starting balance plus all earlier realised profit). Days with no
closed trade are zero. The span runs from the earliest of the first open date
and `backtest_start` to the latest of the last close date and `backtest_end`.
When the strategy object holds `profit_total_abs` it must equal the sum of the trades' `profit_abs` (tolerance 1e-6 relative); a mismatch is an error, because it means the profit fields are not understood. Without it nothing is reconciled. A span longer than 20 years is an error. Open trades are not marked to market (the file does not hold the prices), so
drawdown is measured on realised equity. Dates are converted to UTC days.

CSV: header row required. One time column (`date`, `time`, `timestamp`,
`datetime`, `day`, `open_time`, `close_time`) and one value column, either
returns (`return`, `returns`, `ret`, `daily_return`, `pnl_pct`) or equity
(`equity`, `equity_curve`, `balance`, `nav`, `portfolio_value`,
`account_value`, `value`). Names are matched ignoring case; `time_col`,
`value_col` and `kind` override detection. Times: `YYYY-MM-DD`, ISO 8601 `YYYY-MM-DD[T ]HH:MM[:SS[.f]]` (`Z` or offset
allowed; naive is UTC), or integer epoch seconds / milliseconds between 2009 and 2100. Other forms, such as `20240101`, are errors. Rows must be sorted.
Several rows on one day: returns are compounded, equity keeps the last value. An equity curve with missing days is not zero-filled: a return that spans several days counts as one day.
An equity curve of n points gives n-1 returns, dated by the later point.

Python values: lists / arrays of returns or equity (`from_returns`,
`from_equity`) with optional dates; any object with `.index` and `.to_numpy()`
(a pandas Series) is accepted without importing pandas. A datetime-like index
supplies the dates; otherwise consecutive placeholder days are generated and
the series is marked `synthetic`.

Directory: every `.json`, `.zip` (freqtrade, every strategy in a file becomes
a column) and `.csv` file directly inside the directory, sorted by name, minus
hidden files, `*.meta.json` and symbolic links (not followed). Other files are reported in
`ReturnMatrix.skipped`. The result is a `ReturnMatrix` whose `matrix`
(periods x trials) can be passed to `stats.pbo_cscv` and `stats.n_effective`.

Alignment: the matrix uses the days present in every series. It fails when
fewer than 2 series exist, names repeat, dated and undated series are mixed,
undated series differ in length, fewer than 16 common days remain
(`pbo_cscv` needs at least 2 x 8 blocks), or any series would lose more than
20% of its days. `ReturnMatrix.dropped` records how many days each series lost.
Undated returns and undated equity curves are aligned by row position: the
first return of an equity curve is row 1 of the matrix, as for a returns array.

Return domain: every reader checks the rows before several rows on one day are
merged, and a 2-D array given to `checkup` is checked the same way. Booleans,
complex numbers, NaN / infinity and any return <= -100% are errors, as is a
zero or negative equity row or a NaN equity row. Rows inside one UTC day must
be in time order (CSV), an offset needs minutes 00-59, and a freqtrade trade
must not open after it closes (compared by time, not only by day).

### Items and tests (all in `tests/test_loaders.py`)

| ID | Requirement | Tests |
|---|---|---|
| B1.1 | freqtrade json gives zero-filled daily returns with the formula above; span rules; `starting_balance` order (argument, file, `dry_run_wallet`) | `test_b1_1_freqtrade_json_daily_returns`, `test_b1_1_span_extends_to_backtest_window_and_trade_dates`, `test_b1_1_starting_balance_override_and_dry_run_wallet_fallback` |
| B1.2 | zip gives the same series as the json; member choice; bad zip is an error | `test_b1_2_zip_equals_json`, `test_b1_2_zip_single_json_member_with_other_name_ok_and_ambiguity_errors`, `test_b1_2_bad_zip_errors` |
| B1.3 | several strategies: must choose; unknown name is an error; `load_freqtrade_all` returns all | `test_b1_3_multi_strategy_selection` |
| B1.4 | freqtrade error paths: unsupported layouts, empty / invalid / missing file, missing or non-numeric trade fields, bad dates, open after close, bad balance, wiped-out account, UTC day from offset | `test_b1_4_unsupported_layouts`, `test_b1_4_bad_files`, `test_b1_4_bad_trades`, `test_b1_4_wiped_out_account_errors`, `test_b1_4_utc_day_assignment_uses_offset` |
| B1.5 | CSV returns | `test_b1_5_csv_returns` |
| B1.6 | CSV equity; epoch times; same-day rows; explicit columns; BOM | `test_b1_6_csv_equity`, `test_b1_6_csv_options_epoch_intraday_and_explicit_columns` |
| B1.7 | CSV errors: empty, header only, non-number, non-finite, unsorted, bad time, unknown / ambiguous / missing columns, short row, return <= -100%, short or non-positive equity, bad arguments | `test_b1_7_csv_errors`, `test_b1_7_csv_argument_errors` |
| B1.8 | arrays, equity arrays, dated arrays, Series with and without pandas, `load_returns` dispatch | `test_b1_8_arrays_and_equity_arrays`, `test_b1_8_series_without_pandas_duck_typed`, `test_b1_8_pandas_series_optional`, `test_b1_8_load_returns_dispatch` |
| B1.9 | directory to matrix feeds `pbo_cscv` / `n_effective`; mixed csv / json; sidecars listed as skipped; multi-strategy files | `test_b1_9_directory_matrix_feeds_pbo`, `test_b1_9_directory_mixed_formats_and_sidecars`, `test_b1_9_directory_with_freqtrade_files` |
| B1.10 | alignment errors with reasons; directory errors (missing, empty, one result, bad file named) | `test_b1_10_alignment_errors`, `test_b1_10_directory_errors` |
| B1.11 | importing the loaders does not import pandas; loader sources are ASCII and free of internal names; packaged text files hold no denied word (matched by sha256 only, the words are not in any packaged file); the `test` and `dev` extras install pandas and the sdist leaves `tools/` out; sample files stay small | `test_b1_11_import_does_not_pull_pandas`, `test_b1_11_denied_word_scan_matches_by_hash_only`, `test_b1_11_loader_sources_are_plain_ascii_without_internal_names`, `test_b1_11_packaged_text_files_hold_no_denied_word`, `test_b1_11_test_extra_installs_pandas_and_manifest_leaves_tools_out`, `test_b1_11_samples_are_small` |
| B1.12 | loader output feeds `stats` functions | `test_b1_12_returns_feed_stats_end_to_end` |
| B1.14 | native exceptions never escape: corrupt / encrypted / unsupported zip, deeply nested or huge-number JSON, binary or oversized-field CSV, non-numeric or boolean arrays all raise `LoaderError` | `test_b1_14_zip_read_failures`, `test_b1_14_json_extremes`, `test_b1_14_csv_unreadable`, `test_b1_14_array_conversion_errors` |
| B1.15 | time parsing is strict: `YYYY-MM-DD`, ISO 8601 with optional offset, epoch seconds or milliseconds between 2009 and 2100; nothing is guessed from digit counts | `test_b1_15_rejected_time_text`, `test_b1_15_accepted_time_forms` |
| B1.16 | CSV columns: several matches for one role is an error, explicit names are case-insensitive, blank lines are skipped | `test_b1_16_ambiguous_columns`, `test_b1_16_explicit_names_are_case_insensitive_and_resolve_ambiguity`, `test_b1_16_blank_lines_are_skipped` |
| B1.17 | exactly 16 common days is the minimum; zip member choice (same-name first, subdirectories ignored); directories do not follow symlinks | `test_b1_17_min_periods_boundary`, `test_b1_17_zip_prefers_same_name_member_and_ignores_subdirectories`, `test_b1_17_directory_does_not_follow_symlinks` |
| B1.18 | return-domain checks before same-day merging, time order inside a day, offset minutes, open before close by time, 2-D / boolean / complex input, row-position alignment of undated series | `test_b1_18_same_day_domain_checks_run_before_merging`, `test_b1_18_intraday_order_offset_minutes_and_trade_times`, `test_b1_18_rejects_invalid_2d_boolean_and_complex_returns`, `test_b1_18_undated_returns_and_equity_align_by_position` |
| B1.19 | Series with text or period indexes supply dates; boolean values are rejected | `test_b1_19_text_and_period_index_series` |
| B1.20 | freqtrade versions: fixtures built from the sources of 2026.10 (futures, short trade, extra fields), 2022.1 (spot, `sell_reason`) and 2021.1 (no starting balance); each fixture names its source; a zip with the members freqtrade 2026.10 writes, plus `.meta.json` and `.last_result.json`; a multi-strategy zip | `test_b1_20_fixtures_state_their_source_and_carry_the_official_fields`, `test_b1_20_current_layout_futures_with_short_and_extra_fields`, `test_b1_20_older_layouts_and_missing_starting_balance`, `test_b1_20_zip_with_the_members_freqtrade_writes`, `test_b1_20_multi_strategy_zip_reads_as_several_results` |
| B1.21 | loader errors that ask for an option also name the command-line switch; a 2-D input (one column per result) is told to use a folder or `to_numpy()` | `test_b1_21_loader_errors_name_the_command_line_switch`, `test_b1_21_several_columns_error_says_what_to_do` |
| B1.22 | a result zip written by a real freqtrade 2026.10 run (files in `tests/data/real_freqtrade_2026_10/`, zipped by the test; provenance in "Real result file") reads, reconciles and gives the expected daily series; no member holds a home-directory path or internal name; the same for a real 2026.10 futures result with short trades (files in `tests/data/real_freqtrade_2026_10_futures/`) | `test_b1_22_real_freqtrade_2026_10_zip_reads_and_reconciles`, `test_b1_22_real_freqtrade_2026_10_futures_zip_with_shorts_reads_and_reconciles` |
| B1.23 | `DailyReturns.spans` / `basis`: equity input records the days each return covers and `basis="equity"`; freqtrade results are `"realised"`; direct returns have `spans=None`; a bad basis or bad spans is a `LoaderError` | `test_b1_23_spans_and_basis` |
| B1.24 | `align_returns` fills `ReturnMatrix.sources` (original count, first / last day, rows lost before / after / inside the common window, largest and median span, basis), which add up to `dropped`; a matrix built by hand has `sources=None` | `test_b1_24_matrix_sources` |
| B1.25 | a span over 20 years is an error; `profit_total_abs`, when present, must equal the sum of trade `profit_abs` within a tolerance relative to the stated net profit (plus a tiny absolute floor); a duplicate zip member or duplicate JSON key is an error, as is overflowing realised equity | `test_b1_25_span_limit`, `test_b1_25_profit_total_reconciliation`, `test_b1_25_duplicate_zip_member_and_duplicate_json_key`, `test_b1_25_reconciliation_tolerance_and_equity_overflow` |
| B1.13 | every B1 test named here exists and every B1 test is named here | `test_b1_13_spec_and_tests_agree` |

## B2. Checkup (`oss_backtest_kit.checkup`)

API: `checkup(source, n_trials=None, select=None, n_resamples=2000, block=5, seed=0, load_options=None)`
returns a `Report` object; command line
`python -m oss_backtest_kit.checkup <file or directory> [--trials N] [--report OUT.md] [--html OUT.html] [--select NAME] [--seed S] [--strategy NAME] [--starting-balance X] [--time-col C] [--value-col C] [--kind returns|equity]`.
The last five switches are the loader options (`strategy`, `starting_balance`, `time_col`, `value_col`,
`kind`); the API takes them as `load_options={...}` for a path source. A switch that does not fit the
source (for example `--strategy` for a CSV file) is an error, not ignored. Every loader error that asks for
such an option also names the switch ("command line: --strategy NAME"). A freqtrade file with several
strategies needs `--strategy NAME` on the command line (or put each result in its own file in a folder: a
folder reads every strategy of every file).
`source` is anything `loaders.load_returns` accepts, plus a `ReturnMatrix`, or a
2-D numpy array (periods x results, undated, results named `result_1` ...).
Exit status 0 when the checkup ran (whatever the verdicts), 2 on input errors
(unusable input, bad option, unwritable output file) with `error: <reason>` on
stderr and nothing on stdout. The Markdown report is printed to stdout; the
files are written in addition.

With several results (a directory or matrix), the Sharpe, PSR, DSR and
bootstrap lines describe one result: the one with the highest per-period
Sharpe, or `select` (a result name; an unknown name is an error). This is the
result a researcher would be tempted to ship; DSR and PBO then judge the
selection. The report names it.

Metrics, each with a verdict (`pass`, `caution`, `fail`, or `info`) and one
plain sentence. Thresholds below are suggested defaults (v0.2.0) and are
printed in every report. They are conventions from the literature and
practice, not guarantees; wording in the report must say so.

| Metric | pass | caution | fail | Basis |
|---|---|---|---|---|
| PSR (benchmark 0) | >= 0.95 | 0.90 to < 0.95 | < 0.90 | 95% confidence is the usual one-sided bar; Bailey and Lopez de Prado, "The Sharpe Ratio Efficient Frontier" (2012) |
| DSR (trials N) | >= 0.95 | 0.90 to < 0.95 | < 0.90 | Same bar, applied after deflating for N trials; Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" (2014) |
| PBO (>= 2 results) | <= 0.20 | > 0.20 to <= 0.50 | > 0.50 | 0.5 is the value for pure noise; Bailey, Borwein, Lopez de Prado, Zhu, "The Probability of Backtest Overfitting" (2017) give no hard cut-off, 0.20 is a deliberately strict working choice |
| Annualised Sharpe | >= 1.0 | 0.5 to < 1.0 | < 0.5 | Practitioner rule of thumb for daily strategies before costs are doubted; weakest evidence, shown for orientation |
| Max drawdown | (info) | | | Reflects risk appetite, not overfitting evidence, so it gets no verdict; the report shows the value and says so |
| Bootstrap Sharpe 95% interval (circular block bootstrap of the Sharpe) | lower bound > 0 | interval contains 0, point estimate > 0 | upper bound <= 0 or point estimate <= 0 | Direct reading of the resampled distribution |
| Effective trials | (info) | | | Reported next to N so the reader sees how many trials were independent; no verdict |

The bootstrap uses `stats.block_bootstrap_sharpe`: circular blocks of 5, 2000
resamples, seed 0 by default, percentile interval. A value that is not finite
(NaN) is always `fail`. The PSR sentence writes "more than 99.9%" instead of a rounded 100% when PSR is above 0.999. Annualisation uses 365.25 periods per year, as
`stats.sharpe_annualised`. DSR gets the per-period Sharpe of every result as
`trial_sharpes` when there are several.

Sparse rule: when fewer than 30 of the described result's returns are
non-zero, every `pass` also becomes `caution` (a strategy with a handful of
active days has too little evidence), with a note.

Serial dependence: the checkup measures the integrated autocorrelation time tau of the described result
with `stats.autocorr_time` (Geyer initial positive sequence with a noise floor: biased FFT autocorrelations
for lags below T / 2 summed in pairs, stopping from the second pair on at the first pair at or below
2 x sqrt(2 / T); tau = max(1, 2 x sum - 1), so negative correlation never adds evidence). Below
`stats.TAU_IGNORE` (1.5) every row counts as one observation and every number is the same as without this
rule; the Facts line reads `Serial dependence: none material (...)`. From 1.5 on, the effective number of
observations is n_eff = T / tau: PSR and DSR use it (`n_obs`, at least 3), the sample-size rule uses it
(`n_eff < 180` caps every pass; the sparse rule uses non-zero returns / tau), the bootstrap uses blocks of
max(block, ceil(tau)) and, with fewer than 4 such blocks in the sample, is not run (`n/a`, `caution`). The
error under 30 rows still counts rows. PBO is not adjusted (CSCV cuts whole blocks). Facts and Notes say
so; the estimate is rough and optimistic for very persistent series.

Annualisation: when the dates are not one per calendar day (more than 5% away
from 365.25 observations per year over the span, for example every third day
or weekdays only), the number of observations per year is taken from the
dates (`n * 365.25 / span`), and a note says so and asks for zero-filled daily
data. Undated input is assumed daily.

Sample-size rules: fewer than 30 observations is an error; 30 to 179
observations turn every `pass` into `caution` (a `fail` stays `fail`; bad news is never softened) and add a sentence saying why
(the standard error of an annualised daily Sharpe is above 1.4 at 180 days).
`n_trials` defaults to the number of results when a directory is given and to
1 for a single result; the report then says that versions that were tried and
not saved must be counted. A single result with `n_trials` of 1 gets the
DSR note "No deflation applied". A single result with `n_trials` **not given** gets no deflation either,
so DSR cannot be judged: that line is capped at `caution` (a `fail` stays `fail`) and its sentence says
that the number of trials was not given and tells the reader to pass `--trials N`; `Report.overall` is
then at most `caution`. Only an explicit `n_trials=1` lets a single result pass the DSR line. With several
results and no `n_trials`, N is the number of results and DSR is deflated for it, but that N is only a lower
bound (versions tried and not saved also count): the DSR line is capped at `caution` (a `fail` stays `fail`),
its sentence says `Deflated for N = ..., the number of results read. That is only a lower bound`, the note on
the missing trials ends with `Until it is given, the overall verdict is at most CAUTION.`, the Facts line
reads `Trials counted (N): N (not given; lower bound, at most CAUTION)`, and `Report.overall` is at most
`caution`. A given `n_trials` (also on the command line, `--trials`) lifts the cap. A given `n_trials` must be a whole number of
at least 1 (not a bool, string or float) and not smaller than the number of
results read; otherwise it is an error.

Errors (all `CheckupError`, or the loaders' `LoaderError`; the command line
reports them with exit status 2): empty or missing input, a path that does not
exist, NaN or infinite returns, constant returns (zero variance: the Sharpe
ratio is undefined; the message names the results), fewer than 30
observations, a 2-D array with fewer than two columns, bad `n_trials`, unknown
`select`. If PBO cannot be computed (`StatsError`) the line is `info` with the
reason; it never aborts the checkup.

| ID | Requirement | Tests (`tests/test_checkup.py`) |
|---|---|---|
| B2.1 | metrics match `stats.*` for the same data (annualised Sharpe, drawdown, PSR, DSR, bootstrap) | `test_b2_1_metrics_match_stats` |
| B2.2 | verdict thresholds at each boundary value and just inside / outside; constants equal the table | `test_b2_2_threshold_boundaries`, `test_b2_2_end_to_end_verdict_follows_grade` |
| B2.3 | sample-size rules (error under 30, cap at caution under 180, fail stays fail) | `test_b2_3_sample_size_rules` |
| B2.4 | `n_trials` default and override; saved-versions warning text; bad values | `test_b2_4_trials_default_and_override`, `test_b2_4_bad_trials` |
| B2.5 | PBO and effective trials only for 2+ results; PBO error text when it cannot be computed | `test_b2_5_pbo_needs_two_results`, `test_b2_5_pbo_failure_is_info` |
| B2.6 | pure noise scores worse than a planted real edge (PSR, DSR, PBO) with a fixed seed | `test_b2_6_noise_vs_edge` |
| B2.7 | command line: exit codes, messages, `--trials`, bad path, output files | `test_b2_7_cli_runs_as_module`, `test_b2_7_cli_errors`, `test_b2_7_cli_writes_reports` |
| B2.8 | report prints the thresholds it used and the disclaimer pointer | `test_b2_8_report_lists_thresholds` |
| B2.9 | error paths: empty input, missing path, NaN, constant returns (single and in a matrix), 2-D array, unknown `select`, `select` picks the result | `test_b2_9_input_errors`, `test_b2_9_select_and_matrix_inputs` |
| B2.10 | `stats.block_bootstrap_sharpe`: reproducible, interval brackets the estimate, bad settings are errors | `test_b2_10_block_bootstrap_sharpe` |
| B2.12 | the bootstrap interval equals an independent re-computation (quantiles 2.5% / 97.5%, block starts uniform over all days); the seed reaches it | `test_b2_12_bootstrap_interval_pinned` |
| B2.13 | `Report.overall` is the worst graded verdict (including caution); N = 2 is deflated, only N = 1 says "No deflation applied" | `test_b2_13_overall_caution_and_two_trials` |
| B2.14 | fewer than 30 non-zero returns in the described result: no pass (capped at caution) and a note | `test_b2_14_sparse_strategy_cannot_pass` |
| B2.15 | the annualisation uses observations per year from the dates (`n * 365.25 / span`) unless within 5% of daily; a note says so | `test_b2_15_non_daily_dates_set_the_annualisation` |
| B2.16 | `--seed` / `seed` must be a whole number >= 0; stray `ValueError`s (for example an unencodable file name) end as `error:` and exit 2; a non-numeric 2-D array is a `CheckupError` | `test_b2_16_seed_and_stray_errors` |
| B2.17 | a result or a result set without --trials cannot pass (this test: a single result): DSR capped at `caution`, wording, `Overall` at most `caution`, Markdown / HTML text, command line; `n_trials=1` may pass | `test_b2_17_single_result_without_trials_cannot_pass` |
| B2.18 | command line loader switches (`--strategy`, `--starting-balance`, `--time-col`, `--value-col`, `--kind`), bad values, switches that do not fit the source, `load_options` in the API | `test_b2_18_cli_loader_switches` |
| B2.19 | PSR above 0.999 reads "more than 99.9%", never "100%" | `test_b2_19_psr_text_never_claims_certainty` |
| B2.20 | zero variance is judged against the data's own scale (not `std == 0`) and shared by Sharpe / PSR / DSR / bootstrap; maximum drawdown is computed on log equity so long series cannot overflow; PBO ties for the best in-sample Sharpe go to the larger return sum, still-tied splits are left out and counted (`pbo_cscv_detail` returns the count), and rescaling the matrix does not change PBO (above a tiny scale, about 1e-11) | `test_b2_20_scale_relative_zero_variance`, `test_b2_20_log_equity_max_drawdown_no_overflow`, `test_b2_20_pbo_scale_invariance`, `test_b2_20_pbo_tie_breaking_and_order_invariance` (in `tests/test_stats.py`) |
| B2.21 | the checkup reads that real result (function call and command line): 22 days are refused as too few, a threefold repetition passes the whole checkup; the real futures result (43 days, short trades) is checked without refusal | `test_b2_21_checkup_on_real_freqtrade_zip`, `test_b2_21_checkup_on_real_freqtrade_futures_zip` |
| B2.22 | the command line refuses, before reading, an output path that is an input file or a hard link to one; a native `OverflowError` is reported as `error:` | `test_b2_22_cli_native_overflow_is_an_error_message`, `test_b2_22_cli_refuses_output_overwriting_input` |
| B2.23 | an equity curve whose points are unevenly spaced (largest gap above 2 x the median gap + 2 days) is a `CheckupError` containing `unevenly spaced` (single file, or one file in a directory; on the command line `error:` and exit 2); weekend gaps, a gap exactly on the line and weekly points are checked as usual; returns given directly are not checked. The loaders only record the spans | `test_b2_23_uneven_intervals_are_refused` |
| B2.24 | `fill_flat=True` / `--fill-flat`: the missing days of an equity curve repeat the last equity value (0 return on those days), the report says `Filled N missing days by repeating the last equity value`, N adding up over several results; the default error ends with the hint to pass it; it has no effect and no error for series without gaps or given as returns | `test_b2_24_fill_flat_repeats_the_last_equity_value` |
| B2.25 | the described result's exclusion by alignment is written in Facts (`Coverage`: window, kept / total, excluded, before / after / inside, other results) and in Notes, in Markdown and HTML; more than 5% excluded caps every pass at caution, exactly 5% and less do not; no `sources` reads `not recorded`; a single result excludes 0 | `test_b2_25_alignment_coverage_disclosed` |
| B2.26 | a freqtrade source (basis `realised`) gets the `Equity basis` line, the max drawdown sentence and the Notes sentence naming closed trades, open positions and mark to market (Markdown and HTML), no verdict change; CSV equity / direct returns never show that text; a directory mixing bases lists them | `test_b2_26_realised_basis_disclosed` |
| B2.27 | several results without `n_trials`: N = number of results is deflated, DSR at most caution (`fail` kept), wording `lower bound`, Notes and Facts suffix, overall at most caution in Markdown / HTML; `n_trials=k` / `--trials k` may pass; a single result keeps the B2.17 wording; command line the same | `test_b2_27_result_set_without_trials_cannot_pass` |
| B2.28 | `stats.autocorr_time` (`tests/test_stats.py`): independent noise (fixed seeds) < 1.5; AR(1) phi = 0.5 in 2 to 3.2; block-repeated series >= 50; negative correlation and a flat series give 1; equals an independent direct re-computation; non-finite values, fewer than 3 values or 2-D input are a `StatsError` | `test_b2_28_autocorr_time` |
| B2.29 | `psr` / `dsr` with `n_obs` (`tests/test_stats.py`): None is bit-for-bit the old value; a given value follows the formula (independent re-computation; DSR null variance too); `n_obs` below 3, not finite or not a number is a `StatsError` | `test_b2_29_psr_dsr_effective_n` |
| B2.30 | checkup with tau >= 1.5: PSR / DSR use n_eff (at least 3), short and sparse use effective counts, block = max(block, ceil tau), fewer than 4 blocks gives a `n/a` caution bootstrap line (worded by cause: long block or long dependence; the Facts line then says `not run`), only when tau >= 1.5: an independent series with a long block is resampled as before; the Facts `Serial dependence` and `Bootstrap` lines and the Notes sentences in Markdown and HTML; with tau < 1.5 the numbers and verdicts are those without the rule | `test_b2_30_serial_dependence_in_checkup`, `test_b2_30_notes_show_effective_count_below_formula_minimum`, `test_b2_30_notes_formula_minimum_boundary` |
| B2.31 | `block` must be a whole number of at least 1 (0, negative, fractional and boolean values are a `CheckupError`, whatever tau is); an array of more than two dimensions is a `CheckupError` | `test_b2_31_block_and_array_checks` |
| B2.32 | shown tau and effective counts are rounded down (never `1.5 < 1.5`, `about 180 ... fewer than 180`), so the shown number is on the same side of the limit as the real one | `test_b2_32_shown_numbers_are_rounded_down` |
| B2.33 | splits left out of PBO because results tied are counted in the PBO sentence; a user-built aligned matrix with unevenly spaced results is told to pass `fill_flat` when loading, not to the checkup | `test_b2_33_pbo_ties_and_aligned_matrix_hint` |
| B2.11 | every B2 / B3 test named here exists and every B2 / B3 test is named here (`test_checkup.py`, `test_report.py`, `test_stats.py`) | `test_b2_11_spec_and_tests_agree` |

## B3. Report (`oss_backtest_kit.report`; `Report.to_markdown()`, `Report.to_html()`)

Markdown report from a `Report`; optional single-file HTML (`--html PATH`) with
inline CSS, no scripts, no network access, no new dependency. First lines of
every report state that the checkup can only lower overfitting risk and cannot
show that a strategy will make money. The report holds no clock time, so the
same input and seed give the same text. Strings that came from files (result
names) are escaped: in Markdown the characters that could start markup or break
a table are backslash-escaped (an underscore inside a word and the brackets of numbers the program made are left alone) and control characters become spaces; in HTML
everything is entity-escaped, and the page carries a content security policy
that forbids scripts and any fetch. Verdicts are written as words (PASS,
CAUTION, FAIL, INFO), never by color alone. At most 20 result names are listed.

| ID | Requirement | Tests (`tests/test_report.py`) |
|---|---|---|
| B3.1 | Markdown starts with the disclaimer; contains every metric, verdict, sentence and threshold | `test_b3_1_markdown_content` |
| B3.2 | HTML is a single file with no `src=` / `href=` to other hosts, no `<script>`; text is escaped | `test_b3_2_html_offline_and_escaped` |
| B3.3 | output is deterministic for a fixed seed | `test_b3_3_deterministic` |
| B3.4 | file names or labels with markup characters do not break the report | `test_b3_4_labels_escaped` |
| B3.5 | names in several-result notes and error text are escaped in Markdown and HTML; HTML writes each verdict as a word; the threshold table matches the numbers | `test_b3_5_matrix_names_and_html_words` |
| B3.6 | Markdown does not backslash-escape underscores inside words or the brackets of program-made numbers; names still cannot start markup | `test_b3_6_markdown_does_not_litter_names_and_numbers_with_backslashes` |
| B3.7 | Facts lines come in a fixed order (`Coverage`, `Equity basis` when known, `Serial dependence`, then `Bootstrap`), the same input and seed give identical bytes, names in them are escaped in HTML and Markdown | `test_b3_7_new_fact_lines`, `test_b3_7_serial_dependence_fact_rounds_down` |
| B3.8 | the Coverage line of Facts: undated window, kept / total, before / after / inside counts, and `and N more` past 20 other results | `test_b3_8_coverage_fact_text` |

## B4. Example and documentation

| ID | Requirement | Tests |
|---|---|---|
| B4.1 | `examples/` has a checkup example on synthetic data that runs offline and prints a report | `test_b4_1_example_runs_offline` (in `tests/test_examples.py`) |
| B4.2 | README has a checkup section; the 30-second quick start is at the top | `test_b4_2_readme_sections` (in `tests/test_package.py`) |
| B4.3 | CHANGELOG has a 0.2.0 entry; the version number changes in the working tree only, nothing is published | `test_b4_3_changelog_and_version` (in `tests/test_package.py`) |

## B5. Housekeeping

| ID | Requirement | Tests |
|---|---|---|
| B5.1 | MANIFEST is written to a temporary file and replaced atomically; a failure mid-write leaves the old file intact | `test_b5_1_manifest_atomic_replace`, `test_b5_1_manifest_keeps_permissions`, `test_b5_1_manifest_concurrent_writers_lose_no_rows` (in `tests/test_data.py`); the rewrite runs under an exclusive lock (where `fcntl` exists) so concurrent writers lose no rows, and keeps the old file's permission bits |
| B5.2 | the public-release scan is re-run and recorded in SCAN_REPORT; no new internal names, no non-ASCII text in the package | `test_b5_2_package_ascii_and_clean` (in `tests/test_package.py`) |
| B5.3 | a manifest row cut short by an interrupted 0.1.0 download (missing digest or local path) (or a last line without a line break) is not a finished download: it is downloaded again and replaced in the new manifest, and the loader refuses such a row instead of reading an empty path | `test_b5_3_unfinished_manifest_rows_are_not_done_and_are_replaced`, `test_b5_3_loader_refuses_an_unfinished_manifest_row`, `test_b5_3_row_cut_inside_local_path_is_downloaded_again` |
