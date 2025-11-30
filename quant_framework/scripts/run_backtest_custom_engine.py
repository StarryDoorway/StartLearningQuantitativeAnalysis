#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pandas as pd
from loguru import logger

# 确保项目根目录在系统路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from quant_framework.core.backtest_engine.backtest_engine import BacktestEngine, BacktestConfig, BacktestMode, CommissionModel
from quant_framework.strategies.ema_rsi_strategy import EmaRsiStrategy


def load_parquet(symbol_slug: str, timeframe: str) -> pd.DataFrame:
    """加载历史数据"""
    path = os.path.join('quant_framework', 'data', 'raw', symbol_slug, f'{timeframe}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}. Run scripts/fetch_ohlcv.py first.")
    
    df = pd.read_parquet(path)
    # 将timestamp转换为索引
    df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def run_backtest_with_custom_engine(symbol_slug: str, timeframe: str, cash: float, 
                                   commission: float, stake_pct: float, plot: bool):
    """使用自定义回测引擎运行回测"""
    # 加载数据
    df = load_parquet(symbol_slug, timeframe)
    
    # 确定时间范围
    start_time = df.index[0]
    end_time = df.index[-1]
    
    # 创建回测配置
    config = BacktestConfig(
        start_time=start_time,
        end_time=end_time,
        initial_cash=cash,
        commission_model=CommissionModel.PERCENTAGE,
        commission_rate=commission,
        mode=BacktestMode.EVENT_DRIVEN
    )
    
    # 创建策略实例
    strategy_config = {
        "parameters": {
            "ema_fast": 10,
            "ema_slow": 20,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30
        },
        "symbols": [symbol_slug],
        "frequency": timeframe
    }
    
    strategy = EmaRsiStrategy(strategy_id="custom_engine_ema_rsi", config=strategy_config)
    
    # 创建回测引擎
    engine = BacktestEngine(config)
    
    # 设置策略和数据
    engine.set_strategy(strategy)
    engine.set_historical_data({symbol_slug: df})
    
    # 运行回测
    logger.info(f"Starting backtest for {symbol_slug} {timeframe} with initial cash: {cash:.2f}")
    results = engine.run()
    
    # 获取性能指标
    metrics = results["performance_metrics"]
    
    # 打印结果
    logger.success(f"Final Portfolio Value: {metrics['final_value']:.2f}")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    
    # 如果需要绘图
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            # 提取投资组合历史数据
            portfolio_history = results["portfolio_history"]
            timestamps = [state["timestamp"] for state in portfolio_history]
            values = [state["total_value"] for state in portfolio_history]
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, values, label='Portfolio Value')
            plt.title(f'Backtest Results: {symbol_slug} {timeframe}')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.legend()
            
            # 保存图表
            outdir = os.path.join('backtests')
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, f'{symbol_slug}_{timeframe}_custom_engine.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved plot to {outdir}")
        except Exception as e:
            logger.warning(f"Plot failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run custom backtest engine on Parquet OHLCV')
    parser.add_argument('--symbol-slug', type=str, default='eth-usdt-usdt')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--cash', type=float, default=10000.0)
    parser.add_argument('--commission', type=float, default=0.0005)
    parser.add_argument('--stake-pct', type=float, default=95.0, help='Percent of cash to allocate per trade (1-100)')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    try:
        run_backtest_with_custom_engine(args.symbol_slug, args.timeframe, 
                                       args.cash, args.commission, args.stake_pct, args.plot)
    except Exception as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()