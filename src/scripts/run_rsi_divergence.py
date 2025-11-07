#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pathlib

# 添加项目根目录到Python路径
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import backtrader as bt
import pandas as pd
from loguru import logger

from src.strategies.rsi_divergence import RsiDivergenceStrategy

# 复用现有的数据加载功能
from src.scripts.run_backtest import PandasDataFeed, load_parquet


def run_backtest(symbol_slug: str, timeframe: str, cash: float, commission: float, 
                rsi_period: int, rsi_overbought: int, rsi_oversold: int,
                divergence_lookback: int, price_change_threshold: float, 
                rsi_change_threshold: float, risk_per_trade: float,
                stop_loss_type: str, atr_period: int, atr_mult: float,
                take_profit_mult: float, confirm_bars: int, plot: bool):
    """
    运行RSI背离策略回测
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
        RsiDivergenceStrategy,
        rsi_period=rsi_period,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        divergence_lookback=divergence_lookback,
        price_change_threshold=price_change_threshold,
        rsi_change_threshold=rsi_change_threshold,
        risk_per_trade=risk_per_trade,
        stop_loss_type=stop_loss_type,
        atr_period=atr_period,
        atr_mult=atr_mult,
        take_profit_mult=take_profit_mult,
        confirm_bars=confirm_bars
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
        sharpe_ratio = r_sharpe.get('sharperatio', 0.0) or 0.0
    except:
        sharpe_ratio = 0.0
    
    # SQN (System Quality Number)
    try:
        r_sqn = strat.analyzers.sqn.get_analysis()
        sqn = r_sqn.get('sqn', 0.0) or 0.0
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
            filename = f'{symbol_slug}_{timeframe}_rsi{rsi_period}_divergence.png'
            fig.savefig(os.path.join(outdir, filename), dpi=150, bbox_inches='tight')
            logger.info(f"图表已保存至 {os.path.join(outdir, filename)}")
        except Exception as e:
            logger.warning(f"图表生成失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='运行RSI背离策略回测')
    
    # 基本参数
    parser.add_argument('--symbol-slug', type=str, required=True, help='交易对，例如: btc-usdt-usdt, eth-usdt-usdt')
    parser.add_argument('--timeframe', type=str, default='1h', help='时间周期')
    parser.add_argument('--cash', type=float, default=10000.0, help='初始资金')
    parser.add_argument('--commission', type=float, default=0.0005, help='交易佣金')
    
    # RSI参数
    parser.add_argument('--rsi-period', type=int, default=14, help='RSI周期')
    parser.add_argument('--rsi-overbought', type=int, default=70, help='RSI超买阈值')
    parser.add_argument('--rsi-oversold', type=int, default=30, help='RSI超卖阈值')
    
    # 背离检测参数
    parser.add_argument('--divergence-lookback', type=int, default=5, help='背离检测回看K线数')
    parser.add_argument('--price-change-threshold', type=float, default=0.005, help='价格变化阈值')
    parser.add_argument('--rsi-change-threshold', type=float, default=2.0, help='RSI变化阈值')
    
    # 风险管理参数
    parser.add_argument('--risk-per-trade', type=float, default=0.02, help='每笔交易风险比例')
    parser.add_argument('--stop-loss-type', type=str, default='atr', 
                        choices=['atr', 'fixed_percent'], 
                        help='止损类型')
    parser.add_argument('--atr-period', type=int, default=14, help='ATR周期')
    parser.add_argument('--atr-mult', type=float, default=2.0, help='ATR止损倍数')
    parser.add_argument('--take-profit-mult', type=float, default=3.0, help='盈利倍数')
    
    # 信号确认参数
    parser.add_argument('--confirm-bars', type=int, default=1, help='信号确认K线数')
    
    # 操作选项
    parser.add_argument('--plot', action='store_true', help='生成图表')
    
    args = parser.parse_args()
    
    try:
        # 运行回测
        run_backtest(
            args.symbol_slug, args.timeframe, args.cash, args.commission,
            args.rsi_period, args.rsi_overbought, args.rsi_oversold,
            args.divergence_lookback, args.price_change_threshold,
            args.rsi_change_threshold, args.risk_per_trade,
            args.stop_loss_type, args.atr_period, args.atr_mult,
            args.take_profit_mult, args.confirm_bars, args.plot
        )
    except Exception as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()