# -*- coding: utf-8 -*-

import backtrader as bt


class EmaRsiStrategy(bt.Strategy):
    params = dict(
        fast_ema=20,
        slow_ema=50,
        rsi_period=14,
        rsi_entry=52,
        rsi_exit=48,
        atr_period=14,
        atr_mult=2.0,
    )

    def __init__(self):
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.fast_ema)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.slow_ema)
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.crossover = bt.ind.CrossOver(self.ema_fast, self.ema_slow)
        self.entry_price = None

    def next(self):
        price = self.data.close[0]
        atr_val = float(self.atr[0]) if len(self.atr) else 0.0

        if not self.position:
            # Entry: EMA fast crosses above slow, RSI above threshold
            if self.crossover[0] > 0 and self.rsi[0] >= self.p.rsi_entry:
                self.buy()
                self.entry_price = price
        else:
            # Exit conditions: cross down or RSI below threshold or ATR stop
            stop_price = (self.entry_price - self.p.atr_mult * atr_val) if self.entry_price and atr_val > 0 else None
            exit_by_cross = self.crossover[0] < 0
            exit_by_rsi = self.rsi[0] <= self.p.rsi_exit
            exit_by_stop = (stop_price is not None and price <= stop_price)
            if exit_by_cross or exit_by_rsi or exit_by_stop:
                self.close()
                self.entry_price = None
