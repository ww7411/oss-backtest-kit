"""Readers that turn other tools' backtest output into daily return series."""

from ._series import DailyReturns, LoaderError, from_equity, from_returns
from .csvload import load_csv
from .freqtrade_results import load_freqtrade, load_freqtrade_all
from .matrix import ReturnMatrix, align_returns, load_directory, load_returns

__all__ = [
    "DailyReturns", "LoaderError", "ReturnMatrix", "align_returns", "from_equity",
    "from_returns", "load_csv", "load_directory", "load_freqtrade",
    "load_freqtrade_all", "load_returns",
]
