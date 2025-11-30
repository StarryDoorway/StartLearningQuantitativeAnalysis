#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from datetime import datetime
import pandas as pd
from loguru import logger

# 确保项目根目录在系统路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from quant_framework.strategies.ema_rsi_strategy import EmaRsiStrategy
from quant_framework.strategies.strategy_base import SignalType


def load_parquet(symbol_slug: str, timeframe: str) -> pd.DataFrame:
    """加载历史数据"""
    path = os.path.join('quant_framework', 'data', 'raw', symbol_slug, f'{timeframe}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parquet not found: {path}. Run scripts/fetch_ohlcv.py first.")
    
    df = pd.read_parquet(path)
    # 将timestamp转换为索引
    df.index = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


class SimpleBacktestEngine:
    """简化的回测引擎，不依赖配置文件和复杂的数据引擎"""
    
    def __init__(self, initial_cash: float, commission_rate: float):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.cash = initial_cash
        self.position = 0  # 持仓数量
        self.trades = []  # 交易记录
        self.portfolio_history = []  # 投资组合历史
        self.current_price = 0
    
    def execute_buy(self, quantity: float, price: float, timestamp: datetime):
        """执行买入操作"""
        if quantity <= 0:
            return
        
        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.position += quantity
            
            trade = {
                'timestamp': timestamp,
                'type': 'buy',
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'value': total_cost
            }
            self.trades.append(trade)
            logger.info(f"BUY {quantity:.4f} at {price:.4f}, commission: {commission:.4f}")
    
    def execute_sell(self, quantity: float, price: float, timestamp: datetime):
        """执行卖出操作"""
        if quantity <= 0 or self.position <= 0:
            return
        
        sell_quantity = min(quantity, self.position)
        proceeds = sell_quantity * price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission
        
        self.cash += net_proceeds
        self.position -= sell_quantity
        
        trade = {
            'timestamp': timestamp,
            'type': 'sell',
            'quantity': sell_quantity,
            'price': price,
            'commission': commission,
            'value': net_proceeds
        }
        self.trades.append(trade)
        logger.info(f"SELL {sell_quantity:.4f} at {price:.4f}, commission: {commission:.4f}")
    
    def update_portfolio_value(self, price: float, timestamp: datetime):
        """更新投资组合价值"""
        self.current_price = price
        portfolio_value = self.cash + (self.position * price)
        
        self.portfolio_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position': self.position,
            'price': price,
            'portfolio_value': portfolio_value
        })
    
    def get_performance_metrics(self):
        """计算性能指标"""
        if not self.portfolio_history:
            return {}
        
        final_value = self.portfolio_history[-1]['portfolio_value']
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # 计算最大回撤
        peak = self.initial_cash
        max_drawdown = 0
        for record in self.portfolio_history:
            if record['portfolio_value'] > peak:
                peak = record['portfolio_value']
            drawdown = (peak - record['portfolio_value']) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 计算交易统计
        total_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t['type'] == 'buy']
        sell_trades = [t for t in self.trades if t['type'] == 'sell']
        
        # 简单的盈亏计算
        profit_loss = 0
        winning_trades = 0
        losing_trades = 0
        
        # 配对买卖交易计算盈亏
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy = buy_trades[i]
            sell = sell_trades[i]
            pl = (sell['quantity'] * sell['price']) - (buy['quantity'] * buy['price']) - sell['commission'] - buy['commission']
            profit_loss += pl
            if pl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'initial_cash': self.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_loss': profit_loss
        }


def run_backtest_with_simple_engine(symbol_slug: str, timeframe: str, cash: float, 
                                   commission: float, stake_pct: float, plot: bool):
    """使用简化回测引擎运行回测"""
    # 加载数据
    df = load_parquet(symbol_slug, timeframe)
    
    # 创建策略实例
    strategy_config = {
        "parameters": {
            "ema_fast": 10,
            "ema_slow": 20,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30
        }
    }
    
    strategy = EmaRsiStrategy(strategy_id="simple_engine_ema_rsi", config=strategy_config)
    
    # 创建简化回测引擎
    engine = SimpleBacktestEngine(cash, commission)
    
    # 运行回测
    logger.info(f"Starting backtest for {symbol_slug} {timeframe} with initial cash: {cash:.2f}")
    
    # 需要足够的历史数据来计算指标
    min_history = 50  # 至少需要50根K线
    
    for i in range(min_history, len(df)):
        current_time = df.index[i]
        current_price = df.iloc[i]['close']
        
        # 获取历史数据用于策略计算
        history_df = df.iloc[i-min_history:i+1].copy()
        
        # 计算策略信号
        signals = strategy.calculate_signals(symbol_slug, history_df)
        
        # 更新投资组合价值
        engine.update_portfolio_value(current_price, current_time)
        
        # 执行信号
        for signal in signals:
            if signal.signal_type == SignalType.BUY:
                # 计算买入数量（使用固定百分比的资金）
                if engine.cash > 0:
                    buy_value = engine.cash * (stake_pct / 100)
                    quantity = buy_value / current_price
                    engine.execute_buy(quantity, current_price, current_time)
            
            elif signal.signal_type == SignalType.SELL:
                # 卖出所有持仓
                if engine.position > 0:
                    engine.execute_sell(engine.position, current_price, current_time)
    
    # 获取性能指标
    metrics = engine.get_performance_metrics()
    
    # 打印结果
    logger.success(f"Final Portfolio Value: {metrics['final_value']:.2f}")
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Win Rate: {metrics['win_rate']:.2%}")
    
    # 如果需要绘图
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            # 提取投资组合历史数据
            timestamps = [record['timestamp'] for record in engine.portfolio_history]
            values = [record['portfolio_value'] for record in engine.portfolio_history]
            
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
            plt.savefig(os.path.join(outdir, f'{symbol_slug}_{timeframe}_simple_engine.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved plot to {outdir}")
        except Exception as e:
            logger.warning(f"Plot failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run simple backtest engine on Parquet OHLCV')
    parser.add_argument('--symbol-slug', type=str, default='eth-usdt-usdt')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--cash', type=float, default=10000.0)
    parser.add_argument('--commission', type=float, default=0.0005)
    parser.add_argument('--stake-pct', type=float, default=95.0, help='Percent of cash to allocate per trade (1-100)')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    try:
        run_backtest_with_simple_engine(args.symbol_slug, args.timeframe, 
                                       args.cash, args.commission, args.stake_pct, args.plot)
    except Exception as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()