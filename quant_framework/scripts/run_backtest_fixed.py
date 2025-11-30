"""
修复策略模块的循环导入问题

这个脚本演示了如何解决策略模块中的循环导入问题，并提供了一个简化的回测引擎实现。
"""

import os
import sys
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 直接导入必要的组件，避免通过__init__.py导入
from quant_framework.strategies.ema_rsi_strategy import EmaRsiStrategy
from quant_framework.strategies.base.signal_types import SignalType

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# 删除SimpleEmaRsiStrategy类，因为我们现在直接使用strategies目录下的EmaRsiStrategy


def load_parquet(symbol_slug, timeframe):
    """加载Parquet格式的市场数据"""
    data_path = f"quant_framework/data/raw/{symbol_slug}/{timeframe}.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Parquet not found: {data_path}. Run scripts/fetch_ohlcv.py first.")
    
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df


def run_fixed_backtest(symbol_slug: str, timeframe: str, cash: float, commission: float, 
                       plot: bool = True, strategy_config: Optional[Dict[str, Any]] = None):
    """
    运行修复循环导入问题的回测
    
    Args:
        symbol_slug: 交易对符号
        timeframe: 时间框架
        cash: 初始资金
        commission: 手续费率
        plot: 是否绘制图表
        strategy_config: 策略配置（可选，如果不提供则使用策略默认配置）
    """
    # 加载数据
    data = load_parquet(symbol_slug, timeframe)
    
    # 创建策略实例，如果没有提供配置则使用默认配置
    strategy = EmaRsiStrategy("ema_rsi_fixed", strategy_config)
    
    # 初始化回测变量
    position = 0.0  # 持仓数量
    cash_balance = cash  # 现金余额
    portfolio_values = []  # 投资组合价值历史
    dates = []  # 日期历史
    trades = []  # 交易记录
    
    # 运行回测
    print(f"Starting backtest for {symbol_slug} {timeframe}, initial cash: {cash}")
    print(f"Total data points: {len(data)}")
    
    # 确保有足够的数据开始回测
    min_bars = max(
        strategy.parameters.get("ema_slow", 20),
        strategy.parameters.get("rsi_period", 14)
    )
    print(f"Starting from bar {min_bars}")
    
    # 限制处理的数据量，避免过多交易
    max_bars_to_process = min(500, len(data))  # 只处理前500个K线
    print(f"Processing {max_bars_to_process} bars out of {len(data)} total")
    
    for i in range(min_bars, max_bars_to_process):
        current_data = data.iloc[:i+1]  # 使用到当前K线的所有数据
        
        # 生成交易信号
        signals = strategy.calculate_signals(symbol_slug, current_data)
        
        # 获取当前价格
        current_price = current_data['close'].iloc[-1]
        current_date = current_data.index[-1]
        
        # 处理交易信号
        for signal in signals:
            if signal.signal_type == SignalType.BUY and position == 0:
                # 买入信号，使用90%的现金
                buy_amount = cash_balance * 0.9
                buy_quantity = buy_amount / current_price
                commission_cost = buy_amount * commission
                
                position = buy_quantity
                cash_balance -= (buy_amount + commission_cost)
                
                trades.append({
                    'date': current_date,
                    'type': 'BUY',
                    'price': current_price,
                    'quantity': buy_quantity,
                    'value': buy_amount,
                    'commission': commission_cost
                })
                
                print(f"{current_date}: BUY {buy_quantity:.4f} @ {current_price:.2f}, commission: {commission_cost:.2f}")
                
            elif signal.signal_type == SignalType.SELL and position > 0:
                # 卖出信号，卖出全部持仓
                sell_value = position * current_price
                commission_cost = sell_value * commission
                
                cash_balance += (sell_value - commission_cost)
                
                trades.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': current_price,
                    'quantity': position,
                    'value': sell_value,
                    'commission': commission_cost
                })
                
                print(f"{current_date}: SELL {position:.4f} @ {current_price:.2f}, commission: {commission_cost:.2f}")
                
                position = 0.0
        
        # 计算当前投资组合价值
        portfolio_value = cash_balance + (position * current_price)
        portfolio_values.append(portfolio_value)
        dates.append(current_date)
    
    # 计算回测结果
    final_value = portfolio_values[-1] if portfolio_values else cash
    total_return = (final_value / cash - 1) * 100
    
    # 计算最大回撤
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak * 100
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # 计算胜率
    winning_trades = [t for t in trades if t['type'] == 'SELL' and t['value'] > t['commission']]
    win_rate = len(winning_trades) / len([t for t in trades if t['type'] == 'SELL']) * 100 if trades else 0
    
    # 输出结果
    print("\n===== Backtest Results =====")
    print(f"Initial Cash: {cash:.2f}")
    print(f"Final Value: {final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {len(trades)}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    # 绘制图表
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(dates, portfolio_values, label='Portfolio Value')
        plt.title(f'{symbol_slug} {timeframe} Backtest Results')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        output_dir = Path("backtests")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{symbol_slug}_{timeframe}_fixed.png"
        plt.savefig(output_file)
        print(f"Chart saved to: {output_file}")
        # 不显示图表，直接关闭
        plt.close()
    
    return {
        'initial_cash': cash,
        'final_value': final_value,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'portfolio_values': portfolio_values,
        'dates': dates
    }


def main():
    parser = argparse.ArgumentParser(description='Run fixed backtest without circular import issues')
    parser.add_argument('--symbol-slug', type=str, default='btc-usdt-usdt', help='Symbol slug for data file')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe for data file')
    parser.add_argument('--cash', type=float, default=10000, help='Initial cash amount')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--use-default-config', action='store_true', help='Use strategy default configuration')
    
    args = parser.parse_args()
    
    # 根据参数决定是否使用默认配置
    strategy_config = None if args.use_default_config else {
        "parameters": {
            "ema_fast": 12,  # 修改默认值以区分
            "ema_slow": 26,  # 修改默认值以区分
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_neutral_high": 60,
            "rsi_neutral_low": 40
        }
    }
    
    run_fixed_backtest(
        symbol_slug=args.symbol_slug,
        timeframe=args.timeframe,
        cash=args.cash,
        commission=args.commission,
        plot=args.plot,
        strategy_config=strategy_config
    )


if __name__ == "__main__":
    main()