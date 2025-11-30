#!/usr/bin/env python3
"""
Example script demonstrating the use of migrated strategies with the new framework.

This script shows how to:
1. Load market data
2. Create and configure strategies
3. Run backtests
4. Analyze results
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant_framework.data.data_engine import DataEngine
from quant_framework.strategies import RsiDivergenceStrategy, TrendFollowingStrategy, EmaRsiStrategy
from quant_framework.backtesting.backtest_engine import BacktestEngine
from quant_framework.analysis.performance_analyzer import PerformanceAnalyzer
from quant_framework.analysis.visualizer import Visualizer


def generate_sample_data(symbol="AAPL", days=252):
    """Generate sample market data for demonstration."""
    np.random.seed(42)
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with some trend and volatility
    returns = np.random.normal(0.0005, 0.02, len(date_range))
    prices = [100]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(date_range))
    }, index=date_range)
    
    return data


def run_strategy_example(strategy_class, strategy_name, strategy_params, data):
    """Run a strategy example and return results."""
    print(f"\n=== Running {strategy_name} Strategy ===")
    
    # Create strategy instance
    strategy = strategy_class(
        name=strategy_name,
        symbols=["AAPL"],
        parameters=strategy_params
    )
    
    # Create backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001
    )
    
    # Run backtest
    results = backtest_engine.run_backtest(
        strategy=strategy,
        data={"AAPL": data}
    )
    
    # Analyze performance
    analyzer = PerformanceAnalyzer()
    performance_metrics = analyzer.analyze_performance(results)
    
    # Print key metrics
    print(f"Total Return: {performance_metrics.get('total_return', 0):.2%}")
    print(f"Annual Return: {performance_metrics.get('annual_return', 0):.2%}")
    print(f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {performance_metrics.get('win_rate', 0):.2%}")
    print(f"Total Trades: {performance_metrics.get('total_trades', 0)}")
    
    return results, performance_metrics


def main():
    """Main function to run strategy examples."""
    print("=== Strategy Migration Example ===")
    
    # Generate sample data
    print("Generating sample market data...")
    data = generate_sample_data()
    
    # Define strategy parameters
    rsi_divergence_params = {
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "pivot_period": 5,
        "signal_strength": 2
    }
    
    trend_following_params = {
        "ema_fast": 20,
        "ema_slow": 50,
        "bollinger_period": 20,
        "bollinger_dev": 2.0,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "channel_length": 20,
        "signal_combination": "any",
        "confirm_threshold": 1,
        "signal_strength": 2
    }
    
    ema_rsi_params = {
        "ema_fast": 10,
        "ema_slow": 20,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "rsi_neutral_high": 60,
        "rsi_neutral_low": 40,
        "signal_strength": 2
    }
    
    # Run strategy examples
    results_rsi, metrics_rsi = run_strategy_example(
        RsiDivergenceStrategy, "RSI Divergence", rsi_divergence_params, data
    )
    
    results_trend, metrics_trend = run_strategy_example(
        TrendFollowingStrategy, "Trend Following", trend_following_params, data
    )
    
    results_ema_rsi, metrics_ema_rsi = run_strategy_example(
        EmaRsiStrategy, "EMA RSI", ema_rsi_params, data
    )
    
    # Compare strategies
    print("\n=== Strategy Comparison ===")
    comparison = pd.DataFrame({
        "RSI Divergence": [
            metrics_rsi.get('total_return', 0),
            metrics_rsi.get('sharpe_ratio', 0),
            metrics_rsi.get('max_drawdown', 0),
            metrics_rsi.get('win_rate', 0)
        ],
        "Trend Following": [
            metrics_trend.get('total_return', 0),
            metrics_trend.get('sharpe_ratio', 0),
            metrics_trend.get('max_drawdown', 0),
            metrics_trend.get('win_rate', 0)
        ],
        "EMA RSI": [
            metrics_ema_rsi.get('total_return', 0),
            metrics_ema_rsi.get('sharpe_ratio', 0),
            metrics_ema_rsi.get('max_drawdown', 0),
            metrics_ema_rsi.get('win_rate', 0)
        ]
    }, index=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'])
    
    print(comparison)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer = Visualizer()
    
    # Plot equity curves
    visualizer.plot_equity_curve({
        "RSI Divergence": results_rsi['equity_curve'],
        "Trend Following": results_trend['equity_curve'],
        "EMA RSI": results_ema_rsi['equity_curve']
    }, save_path="equity_curves_comparison.png")
    
    # Generate performance summary
    visualizer.plot_performance_summary(
        results_rsi, 
        save_path="rsi_divergence_summary.png"
    )
    
    print("Visualizations saved to current directory.")
    print("\nStrategy migration example completed successfully!")


if __name__ == "__main__":
    main()