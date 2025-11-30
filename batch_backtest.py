#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量回测脚本示例
用于测试不同参数组合的策略表现
"""

import subprocess
import json
import os
from datetime import datetime

def run_single_backtest(symbol_slug, timeframe, cash, commission, stake_pct, plot=False):
    """运行单次回测并返回结果"""
    cmd = [
        "python", "quant_framework/scripts/run_backtest.py",
        "--symbol-slug", symbol_slug,
        "--timeframe", timeframe,
        "--cash", str(cash),
        "--commission", str(commission),
        "--stake-pct", str(stake_pct)
    ]
    
    if plot:
        cmd.append("--plot")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def parse_backtest_output(output):
    """解析回测输出，提取关键指标"""
    lines = output.split('\n')
    results = {}
    
    for line in lines:
        if "Starting Portfolio Value:" in line:
            results["start_value"] = float(line.split(":")[1].strip())
        elif "Final Portfolio Value:" in line:
            results["final_value"] = float(line.split(":")[1].strip())
        elif "MaxDrawDown:" in line:
            results["max_drawdown"] = float(line.split(":")[1].split("%")[0].strip())
        elif "Trades:" in line:
            parts = line.split(", ")
            results["total_trades"] = int(parts[0].split(":")[1].strip())
            results["won_trades"] = int(parts[1].split(":")[1].strip())
            results["lost_trades"] = int(parts[2].split(":")[1].strip())
            results["win_rate"] = float(parts[3].split(":")[1].replace("%", "").strip())
        elif "Returns (Annualized):" in line:
            results["annual_returns"] = float(line.split(":")[1].replace("%", "").strip())
    
    # 计算额外指标
    if "start_value" in results and "final_value" in results:
        results["total_return"] = (results["final_value"] - results["start_value"]) / results["start_value"] * 100
    
    return results

def run_batch_backtest():
    """运行批量回测"""
    # 定义测试参数组合
    test_cases = [
        {
            "name": "BTC 5分钟 高频交易",
            "symbol_slug": "btc-usdt-usdt",
            "timeframe": "5m",
            "cash": 10000,
            "commission": 0.0005,
            "stake_pct": 95.0
        },
        {
            "name": "BTC 1小时 中长期交易",
            "symbol_slug": "btc-usdt-usdt",
            "timeframe": "1h",
            "cash": 10000,
            "commission": 0.0005,
            "stake_pct": 80.0
        },
        {
            "name": "ETH 15分钟 中频交易",
            "symbol_slug": "eth-usdt-usdt",
            "timeframe": "15m",
            "cash": 20000,
            "commission": 0.001,
            "stake_pct": 85.0
        },
        {
            "name": "ETH 1天 长期交易",
            "symbol_slug": "eth-usdt-usdt",
            "timeframe": "1d",
            "cash": 15000,
            "commission": 0.001,
            "stake_pct": 70.0
        }
    ]
    
    # 运行所有测试用例
    all_results = []
    
    for test_case in test_cases:
        print(f"\n正在运行测试: {test_case['name']}")
        print(f"参数: {test_case['symbol_slug']} {test_case['timeframe']} 资金:{test_case['cash']} 手续费:{test_case['commission']} 仓位:{test_case['stake_pct']}%")
        
        output = run_single_backtest(
            test_case["symbol_slug"],
            test_case["timeframe"],
            test_case["cash"],
            test_case["commission"],
            test_case["stake_pct"],
            plot=True  # 生成图表
        )
        
        results = parse_backtest_output(output)
        results["test_name"] = test_case["name"]
        results["parameters"] = test_case
        
        all_results.append(results)
        
        # 打印关键结果
        print(f"总收益率: {results.get('total_return', 0):.2f}%")
        print(f"年化收益率: {results.get('annual_returns', 0):.2f}%")
        print(f"最大回撤: {results.get('max_drawdown', 0):.2f}%")
        print(f"交易次数: {results.get('total_trades', 0)}")
        print(f"胜率: {results.get('win_rate', 0):.2f}%")
    
    # 保存结果到JSON文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"backtests/batch_results_{timestamp}.json"
    
    os.makedirs("backtests", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n批量回测完成! 结果已保存到: {results_file}")
    
    # 生成汇总报告
    generate_summary_report(all_results, f"backtests/summary_report_{timestamp}.txt")
    
    return all_results

def generate_summary_report(results, filename):
    """生成汇总报告"""
    with open(filename, "w") as f:
        f.write("策略回测汇总报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 按总收益率排序
        sorted_results = sorted(results, key=lambda x: x.get('total_return', 0), reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            f.write(f"{i}. {result['test_name']}\n")
            f.write(f"   总收益率: {result.get('total_return', 0):.2f}%\n")
            f.write(f"   年化收益率: {result.get('annual_returns', 0):.2f}%\n")
            f.write(f"   最大回撤: {result.get('max_drawdown', 0):.2f}%\n")
            f.write(f"   交易次数: {result.get('total_trades', 0)}\n")
            f.write(f"   胜率: {result.get('win_rate', 0):.2f}%\n")
            f.write(f"   参数: {result['parameters']['symbol_slug']} {result['parameters']['timeframe']} 资金:{result['parameters']['cash']} 仓位:{result['parameters']['stake_pct']}%\n\n")
        
        # 找出最佳策略
        best_return = max(results, key=lambda x: x.get('total_return', 0))
        best_sharpe = max(results, key=lambda x: x.get('annual_returns', 0) / (abs(x.get('max_drawdown', 0.01)) + 0.01))
        
        f.write("最佳策略 (按总收益率):\n")
        f.write(f"   {best_return['test_name']}: {best_return.get('total_return', 0):.2f}%\n\n")
        
        f.write("最佳策略 (按风险调整收益):\n")
        f.write(f"   {best_sharpe['test_name']}: 年化{best_sharpe.get('annual_returns', 0):.2f}% / 回撤{best_sharpe.get('max_drawdown', 0):.2f}%\n")
    
    print(f"汇总报告已保存到: {filename}")

if __name__ == "__main__":
    run_batch_backtest()