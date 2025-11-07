#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import backtrader as bt
import pandas as pd
from loguru import logger

from src.strategies.trend_following import TrendFollowingStrategy

# 复用现有的数据加载功能
from src.scripts.run_backtest import PandasDataFeed, load_parquet

def run_backtest(symbol_slug: str, timeframe: str, cash: float, commission: float, 
                ema_fast: int, ema_slow: int, signal_combination: str,
                confirm_threshold: int, risk_per_trade: float, 
                stop_loss_type: str, atr_mult: float, plot: bool):
    """
    运行趋势跟踪策略回测
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
    cerebro.addstrategy(
        TrendFollowingStrategy,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        signal_combination=signal_combination,
        confirm_threshold=confirm_threshold,
        risk_per_trade=risk_per_trade,
        stop_loss_type=stop_loss_type,
        atr_mult=atr_mult
    )
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', tann=365)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1, annualize=True)
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    # 记录初始资产
    logger.info(f"初始资产价值: {cerebro.broker.getvalue():.2f}")
    
    # 运行回测
    results = cerebro.run()
    strat = results[0]

    # 获取分析结果
    r_dd = strat.analyzers.dd.get_analysis()
    r_tr = strat.analyzers.trades.get_analysis()
    r_rt = strat.analyzers.returns.get_analysis()
    
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
        r_sharpe = strat.analyzers.sharpe.get_analysis()
        sharpe_ratio = r_sharpe.get('sharperatio', 0.0)
    except:
        sharpe_ratio = 0.0
    
    # SQN (System Quality Number)
    try:
        r_sqn = strat.analyzers.sqn.get_analysis()
        sqn = r_sqn.get('sqn', 0.0)
    except:
        sqn = 0.0
    
    # 打印结果
    logger.success(f"最终资产价值: {cerebro.broker.getvalue():.2f}")
    logger.info(f"最大回撤: {r_dd.max.drawdown:.2f}%, 最大资金回撤: {r_dd.max.moneydown:.2f}")
    logger.info(f"交易次数: {total_trades}, 盈利: {won}, 亏损: {lost}, 胜率: {winrate:.2f}%")
    logger.info(f"年化收益率: {r_rt.get('rnorm100', 0.0):.2f}%")
    logger.info(f"盈利因子: {profit_factor:.2f}")
    logger.info(f"夏普比率: {sharpe_ratio:.2f}")
    logger.info(f"SQN (系统质量数): {sqn:.2f}")
    
    # 绘制图表
    if plot:
        outdir = os.path.join('backtests')
        os.makedirs(outdir, exist_ok=True)
        # 避免在无头环境中使用交互式GUI后端
        try:
            import matplotlib
            matplotlib.use('Agg')
            fig = cerebro.plot(style='candlestick')[0][0]
            # 生成唯一的文件名，包含所有关键参数
            filename = f'{symbol_slug}_{timeframe}_ema{ema_fast}-{ema_slow}_{signal_combination}.png'
            fig.savefig(os.path.join(outdir, filename), dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存至 {os.path.join(outdir, filename)}")
        except Exception as e:
            logger.warning(f"图表生成失败: {e}")


def run_optimization(symbol_slug: str, timeframe: str, cash: float, commission: float, num_sets: int):
    """
    运行参数优化
    """
    # 导入优化模块
    from src.scripts.optimize_trend_following import main as run_optimize
    
    # 构建命令行参数
    sys.argv = [
        sys.argv[0],  # 脚本名称
        '--symbol-slug', symbol_slug,
        '--timeframe', timeframe,
        '--cash', str(cash),
        '--commission', str(commission),
        '--num-sets', str(num_sets)
    ]
    
    # 运行优化
    run_optimize()

def main():
    parser = argparse.ArgumentParser(description='运行趋势跟踪策略回测')
    
    # 基本参数
    parser.add_argument('--symbol-slug', type=str, required=True, help='交易对，例如: btc-usdt-usdt, eth-usdt-usdt')
    parser.add_argument('--timeframe', type=str, default='1h', help='时间周期')
    parser.add_argument('--cash', type=float, default=10000.0, help='初始资金')
    parser.add_argument('--commission', type=float, default=0.0005, help='交易佣金')
    
    # 策略参数
    parser.add_argument('--ema-fast', type=int, default=20, help='快速EMA周期')
    parser.add_argument('--ema-slow', type=int, default=50, help='慢速EMA周期')
    parser.add_argument('--signal-combination', type=str, default='any', 
                        choices=['any', 'all', 'ema_bollinger', 'ema_macd', 'bollinger_macd'], 
                        help='信号组合方式')
    parser.add_argument('--confirm-threshold', type=int, default=1, help='信号确认周期数')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='每笔交易风险比例')
    parser.add_argument('--stop-loss-type', type=str, default='atr', 
                        choices=['atr', 'fixed_percent', 'trailing'], 
                        help='止损类型')
    parser.add_argument('--atr-mult', type=float, default=2.0, help='ATR止损倍数')
    
    # 操作选项
    parser.add_argument('--plot', action='store_true', help='生成图表')
    parser.add_argument('--optimize', action='store_true', help='运行参数优化')
    parser.add_argument('--num-optimize-sets', type=int, default=10, help='优化测试的参数组合数量')
    
    args = parser.parse_args()
    
    try:
        if args.optimize:
            # 运行优化
            run_optimization(args.symbol_slug, args.timeframe, args.cash, args.commission, args.num_optimize_sets)
        else:
            # 运行单个回测
            run_backtest(
                args.symbol_slug, args.timeframe, args.cash, args.commission,
                args.ema_fast, args.ema_slow, args.signal_combination,
                args.confirm_threshold, args.risk_per_trade,
                args.stop_loss_type, args.atr_mult, args.plot
            )
    except Exception as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()