#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import backtrader as bt
import pandas as pd
from loguru import logger

from src.strategies.ema_rsi_backtrader import EmaRsiStrategy


class PandasDataFeed(bt.feeds.PandasData):
    params = (
        ('datetime', 'datetime'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
    )


def load_parquet(symbol_slug: str, timeframe: str) -> pd.DataFrame:
    path = os.path.join('data', 'raw', symbol_slug, f'{timeframe}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}. Run scripts/fetch_ohlcv.py first.")
    df = pd.read_parquet(path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('datetime', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def run_backtest(symbol_slug: str, timeframe: str, cash: float, commission: float, stake_pct: float, plot: bool):
    df = load_parquet(symbol_slug, timeframe)

    cerebro = bt.Cerebro()
    data = PandasDataFeed(dataname=df)
    cerebro.adddata(data, name=f"{symbol_slug}-{timeframe}")
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=max(1.0, min(100.0, stake_pct)))

    cerebro.addstrategy(EmaRsiStrategy)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', tann=365)

    logger.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    results = cerebro.run()
    strat = results[0]

    r_dd = strat.analyzers.dd.get_analysis()
    r_tr = strat.analyzers.trades.get_analysis()
    r_rt = strat.analyzers.returns.get_analysis()

    logger.success(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    logger.info(f"MaxDrawDown: {r_dd.max.drawdown:.2f}%, MaxMoneyDown: {r_dd.max.moneydown:.2f}")
    # 交易统计可能因无交易而缺部分键
    total_trades = r_tr.get('total', {}).get('total', 0)
    won = r_tr.get('won', {}).get('total', 0)
    lost = r_tr.get('lost', {}).get('total', 0)
    winrate = (won / total_trades * 100.0) if total_trades else 0.0
    logger.info(f"Trades: {total_trades}, Won: {won}, Lost: {lost}, WinRate: {winrate:.2f}%")
    logger.info(f"Returns (Annualized): {r_rt.get('rnorm100', 0.0):.2f}%")

    if plot:
        outdir = os.path.join('backtests')
        os.makedirs(outdir, exist_ok=True)
        # Avoid interactive GUI backends in headless envs
        try:
            import matplotlib
            matplotlib.use('Agg')
            fig = cerebro.plot(style='candlestick')[0][0]
            fig.savefig(os.path.join(outdir, f'{symbol_slug}_{timeframe}.png'), dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {outdir}")
        except Exception as e:
            logger.warning(f"Plot failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run Backtrader backtest on Parquet OHLCV')
    parser.add_argument('--symbol-slug', type=str, required=True, help='e.g., btc-usdt-usdt')
    parser.add_argument('--timeframe', type=str, default='5m')
    parser.add_argument('--cash', type=float, default=10000.0)
    parser.add_argument('--commission', type=float, default=0.0005)
    parser.add_argument('--stake-pct', type=float, default=95.0, help='Percent of cash to allocate per trade (1-100)')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    try:
        run_backtest(args.symbol_slug, args.timeframe, args.cash, args.commission, args.stake_pct, args.plot)
    except Exception as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
