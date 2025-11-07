#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from loguru import logger

import backtrader as bt

from src.strategies.trend_following import TrendFollowingStrategy

# 复用现有的数据加载功能
from src.scripts.run_backtest import PandasDataFeed, load_parquet


def optimize_strategy(params_list, symbol_slug, timeframe, cash, commission, results_dir):
    """
    运行策略优化，测试不同参数组合
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备结果表格
    results_columns = [
        'ema_fast', 'ema_slow', 'signal_combination', 'confirm_threshold',
        'risk_per_trade', 'stop_loss_type', 'atr_mult', 'total_trades',
        'win_rate', 'profit_factor', 'max_drawdown', 'annual_return', 'final_value'
    ]
    results_df = pd.DataFrame(columns=results_columns)
    
    # 记录开始时间
    start_time = time.time()
    total_params = len(params_list)
    
    logger.info(f"开始优化，总共有 {total_params} 组参数")
    
    # 测试每组参数
    for i, params in enumerate(params_list):
        try:
            # 设置日志显示优化进度
            elapsed_time = time.time() - start_time
            avg_time_per_param = elapsed_time / (i + 1) if i > 0 else 0
            remaining_time = avg_time_per_param * (total_params - i - 1)
            logger.info(f"优化进度: {i+1}/{total_params} - 已用时: {elapsed_time:.1f}s - 预计剩余: {remaining_time:.1f}s")
            
            # 运行回测并获取结果
            result = run_backtest_with_params(
                symbol_slug, timeframe, cash, commission,
                params['ema_fast'], params['ema_slow'],
                params['signal_combination'], params['confirm_threshold'],
                params['risk_per_trade'], params['stop_loss_type'],
                params['atr_mult']
            )
            
            # 添加结果到表格
            results_df.loc[len(results_df)] = [
                params['ema_fast'], params['ema_slow'],
                params['signal_combination'], params['confirm_threshold'],
                params['risk_per_trade'], params['stop_loss_type'],
                params['atr_mult'], result['total_trades'],
                result['win_rate'], result['profit_factor'],
                result['max_drawdown'], result['annual_return'],
                result['final_value']
            ]
            
            # 每10组参数保存一次结果
            if (i + 1) % 10 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_df.to_csv(os.path.join(results_dir, f"optimization_results_{timestamp}.csv"), index=False)
                logger.info(f"已保存中间结果，完成 {i+1}/{total_params} 组参数测试")
                
        except Exception as e:
            logger.error(f"测试参数组合 {params} 时出错: {str(e)}")
    
    # 排序结果：优先考虑年化收益率，其次是最大回撤
    results_df = results_df.sort_values(
        by=['annual_return', 'max_drawdown'],
        ascending=[False, True]
    )
    
    # 保存最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"optimization_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    
    logger.success(f"优化完成，结果已保存至 {results_file}")
    logger.info(f"最优参数组合:")
    if not results_df.empty:
        best_params = results_df.iloc[0]
        for col in results_columns:
            logger.info(f"{col}: {best_params[col]}")
    
    return results_df


def run_backtest_with_params(
    symbol_slug, timeframe, cash, commission,
    ema_fast, ema_slow, signal_combination, confirm_threshold,
    risk_per_trade, stop_loss_type, atr_mult
):
    """
    使用指定参数运行回测
    """
    # 加载数据
    df = load_parquet(symbol_slug, timeframe)
    
    # 初始化回测引擎
    cerebro = bt.Cerebro()
    
    # 添加数据
    data = PandasDataFeed(dataname=df)
    cerebro.adddata(data, name=f"{symbol_slug}-{timeframe}")
    
    # 设置初始资金和佣金
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_fixed(0.001)  # 滑点设置
    
    # 添加策略和参数
    cerebro.addstrategy(TrendFollowingStrategy,
                       ema_fast=ema_fast,
                       ema_slow=ema_slow,
                       signal_combination=signal_combination,
                       confirm_threshold=confirm_threshold,
                       risk_per_trade=risk_per_trade,
                       stop_loss_type=stop_loss_type,
                       atr_mult=atr_mult)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', tann=365)
    
    # 运行回测
    results = cerebro.run()
    strat = results[0]
    
    # 解析回测结果
    r_dd = strat.analyzers.dd.get_analysis()
    r_tr = strat.analyzers.trades.get_analysis()
    r_rt = strat.analyzers.returns.get_analysis()
    
    # 计算交易统计指标
    total_trades = r_tr.get('total', {}).get('total', 0)
    won_trades = r_tr.get('won', {}).get('total', 0)
    lost_trades = r_tr.get('lost', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    
    # 计算盈亏比和盈利因子
    gross_profit = r_tr.get('won', {}).get('pnl', {}).get('total', 0)
    gross_loss = abs(r_tr.get('lost', {}).get('pnl', {}).get('total', 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # 最大回撤
    max_drawdown = r_dd.max.drawdown
    
    # 年化收益率
    annual_return = r_rt.get('rnorm100', 0.0)
    
    # 最终组合价值
    final_value = cerebro.broker.getvalue()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'annual_return': annual_return,
        'final_value': final_value
    }


def generate_param_combinations(param_ranges):
    """
    生成参数组合
    """
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # 生成所有组合
    combinations = product(*param_values)
    
    # 转换为字典列表
    params_list = []
    for combo in combinations:
        param_dict = {}
        for name, value in zip(param_names, combo):
            param_dict[name] = value
        params_list.append(param_dict)
    
    return params_list


def main():
    parser = argparse.ArgumentParser(description='优化趋势跟踪策略参数')
    parser.add_argument('--symbol-slug', type=str, required=True, help='e.g., btc-usdt-usdt, eth-usdt-usdt')
    parser.add_argument('--timeframe', type=str, default='1h', help='时间周期')
    parser.add_argument('--cash', type=float, default=10000.0, help='初始资金')
    parser.add_argument('--commission', type=float, default=0.0005, help='交易佣金')
    parser.add_argument('--num-sets', type=int, default=10, help='要测试的参数组合数量')
    parser.add_argument('--results-dir', type=str, default='backtests/optimization', help='结果保存目录')
    args = parser.parse_args()
    
    # 参数范围定义
    # 为了避免参数组合过多，我们使用合理的参数范围
    param_ranges = {
        'ema_fast': [10, 20, 30],
        'ema_slow': [40, 50, 60],
        'signal_combination': ['any', 'ema_macd', 'ema_bollinger'],
        'confirm_threshold': [1, 2],
        'risk_per_trade': [0.01, 0.02],
        'stop_loss_type': ['atr'],
        'atr_mult': [1.5, 2.0, 2.5]
    }
    
    # 生成参数组合
    all_params = generate_param_combinations(param_ranges)
    
    # 限制参数组合数量
    if len(all_params) > args.num_sets:
        # 随机采样参数组合
        np.random.seed(42)  # 设置随机种子以确保结果可重复
        selected_indices = np.random.choice(len(all_params), args.num_sets, replace=False)
        params_list = [all_params[i] for i in selected_indices]
    else:
        params_list = all_params
    
    logger.info(f"生成了 {len(params_list)} 组参数组合进行测试")
    
    # 执行优化
    optimize_strategy(params_list, args.symbol_slug, args.timeframe, args.cash, args.commission, args.results_dir)


if __name__ == '__main__':
    main()