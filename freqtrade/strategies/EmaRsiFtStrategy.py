# pragma: no cover
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class EmaRsiFtStrategy(IStrategy):
    timeframe = '5m'
    minimal_roi = {
        "0": 0.02,
        "60": 0.01,
        "120": 0
    }
    stoploss = -0.05
    trailing_stop = False

    use_custom_stoploss = False

    process_only_new_candles = True

    startup_candle_count = 60

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema_fast'] > dataframe['ema_slow']) &
                (dataframe['ema_fast'].shift(1) <= dataframe['ema_slow'].shift(1)) &
                (dataframe['rsi'] >= 52)
            ),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['ema_fast'] < dataframe['ema_slow']) |
                (dataframe['rsi'] <= 48)
            ),
            'sell'
        ] = 1
        return dataframe
