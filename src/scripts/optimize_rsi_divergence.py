#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pathlib
import itertools
import pandas as pd
from datetime import datetime

# 添加项目根目录到Python路径
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import backtrader as bt
from loguru import logger

from src.strategies.rsi_divergence import RsiDivergenceStrategy
from src.scripts.run_backtest import PandasDataFeed, load_parquet


def run_optimization(symbol_slug: str, timeframe: str, cash: float, commission: float,
                    rsi_periods, rsi_overbought_levels, rsi_oversold_levels,
                    divergence_lookbacks, price_change_thresholds, rsi_change_thresholds,
                    risk_per_trades, stop_loss_types, atr_periods, atr_mults,
                    take_profit_mults, confirm_bars):
    """
    运行RSI背离策略参数优化
    """
    # 加载数据
    df = load_parquet(symbol_slug, timeframe)
    
    # 初始化回测引擎
    cerebro = bt.Cerebro()
    data = PandasDataFeed(dataname=df)
    cerebro.adddata(data, name=f"{symbol_slug}-{timeframe}")
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 添加策略和参数
    cerebro.optstrategy(
        RsiDivergenceStrategy,
        rsi_period=rsi_periods,
        rsi_overbought=rsi_overbought_levels,
        rsi_oversold=rsi_oversold_levels,
        divergence_lookback=divergence_lookbacks,
        price_change_threshold=price_change_thresholds,
        rsi_change_threshold=rsi_change_thresholds,
        risk_per_trade=risk_per_trades,
        stop_loss_type=stop_loss_types,
        atr_period=atr_periods,
        atr_mult=atr_mults,
        take_profit_mult=take_profit_mults,
        confirm_bars=confirm_bars
    )
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', tann=365)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1, annualize=True)
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    # 运行优化
    logger.info(f"开始优化，共 {len(list(itertools.product(*[rsi_periods, rsi_overbought_levels, rsi_oversold_levels, divergence_lookbacks, price_change_thresholds, rsi_change_thresholds, risk_per_trades, stop_loss_types, atr_periods, atr_mults, take_profit_mults, confirm_bars])))} 种参数组合")
    results = cerebro.run(maxcpu=1)  # 使用单核以避免内存问题
    
    # 处理结果
    optimization_results = []
    
    for strat in results:
        # 获取参数
        params = strat[0].p._getpairs()
        param_dict = dict(params)
        
        # 获取分析结果
        r_dd = strat[0].analyzers.dd.get_analysis()
        r_tr = strat[0].analyzers.trades.get_analysis()
        r_rt = strat[0].analyzers.returns.get_analysis()
        
        # 计算交易统计
        total_trades = r_tr.get('total', {}).get('total', 0)
        won = r_tr.get('won', {}).get('total', 0)
        lost = r_tr.get('lost', {}).get('total', 0)
        winrate = (won / total_trades * 100.0) if total_trades else 0.0
        
        # 计算盈亏比和盈利因子
        gross_profit = r_tr.get('won', {}).get('pnl', {}).get('total', 0)
        gross_loss = abs(r_tr.get('lost', {}).get('pnl', {}).get('total', 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 夏普比率
        try:
            r_sharpe = strat[0].analyzers.sharpe.get_analysis()
            sharpe_ratio = r_sharpe.get('sharperatio', 0.0) or 0.0
        except:
            sharpe_ratio = 0.0
        
        # SQN (System Quality Number)
        try:
            r_sqn = strat[0].analyzers.sqn.get_analysis()
            sqn = r_sqn.get('sqn', 0.0) or 0.0
        except:
            sqn = 0.0
        
        # 最终资产
        final_value = strat[0].broker.getvalue()
        
        # 计算收益率
        return_pct = (final_value / cash - 1) * 100
        
        # 最大回撤
        max_dd = r_dd.max.drawdown
        
        # 添加到结果列表
        optimization_results.append({
            **param_dict,
            'total_trades': total_trades,
            'winrate': winrate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sqn': sqn,
            'return_pct': return_pct,
            'max_drawdown': max_dd,
            'final_value': final_value
        })
    
    # 转换为DataFrame
    df_results = pd.DataFrame(optimization_results)
    
    # 过滤掉没有交易的组合
    df_results = df_results[df_results['total_trades'] > 0]
    
    # 按夏普比率和收益率排序
    df_results = df_results.sort_values(['sharpe_ratio', 'return_pct'], ascending=False)
    
    # 保存结果
    outdir = os.path.join('backtests', 'optimization')
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'rsi_divergence_optimization_{timestamp}.csv'
    filepath = os.path.join(outdir, filename)
    df_results.to_csv(filepath, index=False)
    
    # 打印前10个结果
    logger.success(f"优化完成，结果已保存至 {filepath}")
    logger.info("前10个最佳参数组合:")
    print(df_results.head(10).to_string(index=False))
    
    return df_results


def main():
    parser = argparse.ArgumentParser(description='RSI背离策略参数优化')
    
    # 基本参数
    parser.add_argument('--symbol-slug', type=str, required=True, help='交易对，例如: btc-usdt-usdt, eth-usdt-usdt')
    parser.add_argument('--timeframe', type=str, default='1h', help='时间周期')
    parser.add_argument('--cash', type=float, default=10000.0, help='初始资金')
    parser.add_argument('--commission', type=float, default=0.0005, help='交易佣金')
    
    args = parser.parse_args()
    
    # 定义参数范围
    rsi_periods = [10, 14, 21]
    rsi_overbought_levels = [65, 70, 75]
    rsi_oversold_levels = [25, 30, 35]
    divergence_lookbacks = [3, 5, 8]
    price_change_thresholds = [0.003, 0.005, 0.01]
    rsi_change_thresholds = [1.0, 2.0, 3.0]
    risk_per_trades = [0.01, 0.02, 0.03]
    stop_loss_types = ['atr', 'fixed_percent']
    atr_periods = [14, 21]
    atr_mults = [1.5, 2.0, 2.5]
    take_profit_mults = [2.0, 3.0, 4.0]
    confirm_bars = [0, 1, 2]
    
    try:
        # 运行优化
        results = run_optimization(
            args.symbol_slug, args.timeframe, args.cash, args.commission,
            rsi_periods, rsi_overbought_levels, rsi_oversold_levels,
            divergence_lookbacks, price_change_thresholds, rsi_change_thresholds,
            risk_per_trades, stop_loss_types, atr_periods, atr_mults,
            take_profit_mults, confirm_bars
        )
    except Exception as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()