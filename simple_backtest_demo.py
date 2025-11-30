#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单回测示例
演示如何使用不同参数进行回测
"""

import subprocess
import os

def run_backtest_and_print(symbol_slug, timeframe, cash, commission, stake_pct, test_name):
    """运行回测并打印结果"""
    print(f"\n{'='*50}")
    print(f"运行回测: {test_name}")
    print(f"参数: {symbol_slug} {timeframe} 资金:{cash} 手续费:{commission} 仓位:{stake_pct}%")
    print(f"{'='*50}")
    
    cmd = [
        "python", "quant_framework/scripts/run_backtest.py",
        "--symbol-slug", symbol_slug,
        "--timeframe", timeframe,
        "--cash", str(cash),
        "--commission", str(commission),
        "--stake-pct", str(stake_pct),
        "--plot"
    ]
    
    # 运行命令并显示输出
    subprocess.run(cmd)

if __name__ == "__main__":
    # 确保回测目录存在
    os.makedirs("backtests", exist_ok=True)
    
    # 运行两个示例回测
    run_backtest_and_print(
        symbol_slug="btc-usdt-usdt",
        timeframe="5m",
        cash=10000,
        commission=0.0005,
        stake_pct=95.0,
        test_name="BTC 5分钟 高频交易"
    )
    
    run_backtest_and_print(
        symbol_slug="eth-usdt-usdt",
        timeframe="15m",
        cash=20000,
        commission=0.001,
        stake_pct=80.0,
        test_name="ETH 15分钟 中频交易"
    )
    
    print(f"\n{'='*50}")
    print("回测完成! 生成的图表保存在 backtests/ 目录下")
    print(f"{'='*50}")