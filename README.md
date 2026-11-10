# oss-backtest-kit

**Is your backtest real, or did you just try enough ideas until one looked good?** A small, honest toolkit for crypto futures research: verified public data, a no-look-ahead backtest engine, and the statistics that tell you how much of your Sharpe ratio is luck.

## Highlights

- **Overfitting checks built in.** Deflated Sharpe Ratio, Probabilistic Sharpe Ratio and Probability of Backtest Overfitting (CSCV) adjust your result for how many variants you tried, so a lucky curve is easier to spot before it costs money.
- **Data you can trust.** Every Binance public archive file is checked against its published SHA-256 before it is saved; a manifest records what you have; gaps and duplicate bars are reported, not silently filled.
- **No look-ahead by design.** Your strategy is called at a bar's close and sees only bars up to that point; orders fill at the next bar's open with slippage and fees.
- **Costs that bite.** Perpetual-futures funding is included, and a `stressed` cost level lets you check whether an edge survives worse fills.
- **Small and readable.** Pure Python with only numpy and scipy; 56 offline tests; no account, API key or paid data needed.
- **MIT licensed** code (the market data has its own terms; see below).

## Who it is for

Individual researchers and students who write their own strategies in Python and want a quick, sober second opinion on a backtest. It is not a trading bot, does not place orders and does not connect to any exchange account.

## What is inside

A small research toolkit for crypto strategies, in three parts:

1. **Multiple-testing statistics** (`oss_backtest_kit.stats`): Sharpe, Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (CSCV), effective number of trials, circular block bootstrap, Wilson intervals.
2. **Public data download and verification** (`oss_backtest_kit.data`): fetches Binance public archive files, verifies each against its published SHA-256 checksum, records a manifest, and checks bar continuity. A REST helper exports the most recent days as separate CSV files (not loaded by the engine).
3. **A simple, config-driven backtest engine** (`oss_backtest_kit.engine`) for a portfolio of perpetual futures, with funding and a two-level cost model.

## DISCLAIMER

This project is for **research and education only**. It is **not investment advice**. It comes with **no warranty** of any kind; you use it **at your own risk**. **Backtest results are not live results**: past or simulated performance says little about future outcomes, and you can lose money trading.

## Status

Archived-style: published as is, with no maintenance guarantee.

## Requirements and install

- Python 3.9 or newer, with numpy and scipy (installed automatically below).
- Internet access only if you download data; everything else, tests included, runs offline.
- Some Python knowledge: you write your strategy as a small Python function.

```
git clone https://github.com/ww7411/oss-backtest-kit.git
cd oss-backtest-kit
pip install -e .
```

Tests also use pytest (optional; each test file can be run directly with `python3`).

## Quick start

Download public data (nothing is downloaded unless you run this; the output folder is yours to choose):

```
python -m oss_backtest_kit.data archive --kinds klines,fundingRate --symbols BTCUSDT \
    --interval 1d --from 2024-01 --to 2024-06 --out ./market_data
python -m oss_backtest_kit.data check --out ./market_data \
    --kind klines --symbol BTCUSDT --interval 1d
```

Kinds: `klines` (needs `--interval`), `markPriceKlines`, `indexPriceKlines`, `premiumIndexKlines`, `fundingRate`, `metrics`. Files that fail the checksum are neither saved nor recorded; existing files are never overwritten. Recent bars not yet in the archive can be exported with the `rest` subcommand; it writes standalone CSV files under `<out>/rest/` for inspection or your own merging. They are not added to the manifest and `load_market_data` does not read them, so a backtest ends at the last archive file.

Run the moving-average crossover example (synthetic data, nothing downloaded):

```
python3 examples/ma_crossover.py
```

Use the statistics:

```python
import numpy as np
from oss_backtest_kit import stats

returns = np.random.default_rng(0).normal(0.0005, 0.01, 500)
print(stats.psr(returns))                                # P(true Sharpe > 0)
value, benchmark, var = stats.dsr(returns, n_trials=20,  # penalise 20 tries
                                  trial_sharpes=[0.01, 0.03, 0.02])
print(stats.wilson(7, 50))                               # 95% interval for 7/50
```

Run a backtest on your own downloaded data:

```python
from oss_backtest_kit.engine import EngineConfig, load_market_data, run_backtest

data = load_market_data("./market_data", ["BTCUSDT"], interval="1d")
cfg = EngineConfig(symbols=["BTCUSDT"], max_leverage=1.0, rebalance_threshold=0.02)

def strategy(t, history):          # called at the close of bar t, sees bars 0..t only
    return {"BTCUSDT": 1.0}        # sum(|weights|) must not exceed max_leverage

result = run_backtest(data, strategy, cfg, cost_level="stressed")
print(result.summary())            # result.returns feeds stats.psr / stats.dsr
```

## Modules

| Module | Purpose |
| --- | --- |
| `stats` | PSR, DSR, PBO (CSCV), effective trials, block bootstrap, Wilson interval |
| `data.archive` | Archive download with checksum verification, manifest, exclusive-create writes |
| `data.rest_tail` | Recent closed bars and funding from public REST endpoints |
| `data.check` | Manifest SHA check, gap and duplicate detection |
| `engine` | Config, market data loader, backtest loop, result summary |

The data client allows a single host per task, uses https only, ignores environment proxy settings, refuses redirects, bounds response size and time, and stops when free disk space is low.

## Tests

```
python3 -m pytest
```

All tests run offline with synthetic data and fake network responses.

## Known limitations

- Fills are modelled from OHLC bars (decision at a bar close, fill at the next open with slippage and a proportional fee). Real fills depend on order books, latency and size.
- Funding is modelled as position x price x rate at each funding time, using the open price of the bar that contains the event; margin, liquidation and exchange rules are not modelled beyond stopping at zero equity (limited liability: positions are closed at that bar's close and any loss beyond zero is reported as `ruin_shortfall`, so `sum(pnl_by_symbol) + ruin_shortfall` equals the change in equity).
- Survivorship bias: the symbols you choose are the ones that exist today; delisted symbols are not included unless you add them.
- Cost defaults (`base` and `stressed`) are placeholders, not estimates for any venue. Set your own numbers.
- Statistics assume the return series is a fair sample; PSR/DSR/PBO reduce, but do not remove, the risk of overfitting.

## Data terms

This kit ships **code only and never any data**. It downloads from two different sources, which have **different terms**. Nothing here is legal advice; read the current terms yourself before downloading, using or sharing any data.

**1. Archive files from `data.binance.vision`** (`archive` subcommand). These are described in the `binance/binance-public-data` repository (https://github.com/binance/binance-public-data), whose `TERMS_AND_CONDITIONS.md` states a CC BY-NC-SA 4.0 licence (https://creativecommons.org/licenses/by-nc-sa/4.0/). As we read that licence, it means:

- **Attribution**: if you share the data or anything derived from it (cleaned files, aggregated tables, published charts of the data), credit the source, link the licence, and say whether you changed it.
- **NonCommercial**: no commercial use of the data.
- **ShareAlike**: data you derive from it and share must be shared under the same licence.

**2. Recent data from the REST API at `fapi.binance.com`** (`rest` subcommand). This is **not** covered by the archive licence above. It falls under the exchange's own API and website terms of use, which this project has **not** reviewed; we make no statement about what they allow. Check them before using or sharing REST data.

You alone are responsible for complying with the terms of each source and with the exchange's rules.

## License

MIT; see `LICENSE`. The licence covers this code only, not any market data you download with it (see the data terms above).
