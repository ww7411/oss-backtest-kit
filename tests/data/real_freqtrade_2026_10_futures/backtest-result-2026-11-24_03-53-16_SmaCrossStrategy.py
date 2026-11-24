from freqtrade.strategy import IStrategy


class SmaCrossStrategy(IStrategy):
    """Small long and short moving-average cross strategy, written for a compatibility check."""
    INTERFACE_VERSION = 3
    timeframe = "4h"
    can_short = True
    minimal_roi = {"0": 100}
    stoploss = -0.10
    startup_candle_count = 30

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs):
        return 2.0

    def populate_indicators(self, dataframe, metadata):
        dataframe["fast"] = dataframe["close"].rolling(6).mean()
        dataframe["slow"] = dataframe["close"].rolling(24).mean()
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        up = dataframe["fast"] > dataframe["slow"]
        dataframe.loc[up & ~up.shift(1, fill_value=False), "enter_long"] = 1
        dataframe.loc[~up & up.shift(1, fill_value=False), "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        up = dataframe["fast"] > dataframe["slow"]
        dataframe.loc[~up & up.shift(1, fill_value=False), "exit_long"] = 1
        dataframe.loc[up & ~up.shift(1, fill_value=False), "exit_short"] = 1
        return dataframe
