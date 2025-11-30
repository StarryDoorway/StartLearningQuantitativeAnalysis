"""
Example usage of the quantitative trading framework.

This script demonstrates how to use the various components of the framework
to run a backtest with multiple strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import framework components
from quant_framework.strategies import RsiDivergenceStrategy, TrendFollowingStrategy, EmaRsiStrategy
from quant_framework.core.backtest_engine.backtest_engine import BacktestEngine
from quant_framework.data import DataManager
from quant_framework.execution import PortfolioManager, OrderManager
from quant_framework.core.risk_engine.risk_engine import RiskEngine
from quant_framework.utils import sma, ema, rsi


def generate_sample_data(days=252, symbols=['AAPL', 'MSFT', 'GOOG']):
    """
    Generate sample market data for demonstration.
    
    Args:
        days: Number of days of data to generate
        symbols: List of symbols to generate data for
        
    Returns:
        Dictionary of DataFrames with OHLCV data
    """
    data = {}
    
    for symbol in symbols:
        # Generate date range
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate random price data with trend
        np.random.seed(hash(symbol) % 2**32)  # Reproducible data per symbol
        
        # Starting price based on symbol
        if symbol == 'AAPL':
            start_price = 150.0
            trend = 0.0005  # Slight upward trend
        elif symbol == 'MSFT':
            start_price = 300.0
            trend = 0.0003  # Slight upward trend
        else:  # GOOG
            start_price = 2500.0
            trend = 0.0002  # Slight upward trend
        
        # Generate random walk with trend
        returns = np.random.normal(trend, 0.02, days)
        prices = [start_price]
        
        for i in range(1, days):
            prices.append(prices[-1] * (1 + returns[i]))
        
        prices = np.array(prices)
        
        # Generate OHLC data
        high_noise = np.random.uniform(0, 0.02, days)
        low_noise = np.random.uniform(0, 0.02, days)
        
        open_prices = prices * (1 + np.random.uniform(-0.01, 0.01, days))
        close_prices = prices
        high_prices = np.maximum(open_prices, close_prices) * (1 + high_noise)
        low_prices = np.minimum(open_prices, close_prices) * (1 - low_noise)
        
        # Generate volume data
        base_volume = 1000000 if symbol == 'AAPL' else 500000
        volume = np.random.normal(base_volume, base_volume * 0.3, days).astype(int)
        volume = np.maximum(volume, 100000)  # Minimum volume
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume
        }, index=dates)
        
        data[symbol] = df
    
    return data


def run_backtest_example():
    """Run a backtest example with multiple strategies."""
    print("Generating sample market data...")
    data = generate_sample_data(days=252, symbols=['AAPL', 'MSFT', 'GOOG'])
    
    # Initialize framework components
    print("Initializing framework components...")
    data_manager = DataManager()
    portfolio_manager = PortfolioManager()
    order_manager = OrderManager()
    risk_engine = RiskEngine(risk_config={})
    
    # Initialize strategies
    print("Initializing strategies...")
    strategies = [
        RsiDivergenceStrategy(
            strategy_id="RSI_Divergence",
            config={
                "symbols": ['AAPL', 'MSFT'],
                "parameters": {
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "price_change_threshold": 0.01
                }
            }
        ),
        TrendFollowingStrategy(
            strategy_id="Trend_Following",
            config={
                "symbols": ['MSFT', 'GOOG'],
                "parameters": {
                    "ema_fast": 10,
                    "ema_slow": 20,
                    "bb_period": 20,
                    "bb_std": 2
                }
            }
        ),
        EmaRsiStrategy(
            strategy_id="EMA_RSI",
            config={
                "symbols": ['AAPL', 'GOOG'],
                "parameters": {
                    "ema_fast": 12,
                    "ema_slow": 26,
                    "rsi_period": 14
                }
            }
        )
    ]
    
    # Initialize backtest engine
    print("Setting up backtest engine...")
    from datetime import datetime
    from quant_framework.core.backtest_engine.backtest_engine import BacktestConfig, BacktestMode, CommissionModel
    
    # Create backtest configuration
    backtest_config = BacktestConfig(
        start_time=data['AAPL'].index[0],
        end_time=data['AAPL'].index[-1],
        initial_cash=100000.0,
        commission_model=CommissionModel.PERCENTAGE,
        commission_rate=0.001,
        mode=BacktestMode.EVENT_DRIVEN
    )
    
    backtest_engine = BacktestEngine(config=backtest_config)
    
    # Load data
    print("Loading data...")
    # Data is loaded automatically by the BacktestEngine through the DataEngine
    
    # Run backtest for each strategy
    print("Running backtest...")
    results = {}
    
    for strategy in strategies:
        print(f"Running backtest for {strategy.strategy_id}...")
        
        # Create a new backtest engine for each strategy
        backtest_engine = BacktestEngine(config=backtest_config)
        
        # Set the strategy
        backtest_engine.set_strategy(strategy)
        
        # Set historical data for the strategy symbols
        strategy_data = {symbol: data[symbol] for symbol in strategy.config.get("symbols", [])}
        backtest_engine.set_historical_data(strategy_data)
        
        # Run the backtest
        strategy_results = backtest_engine.run()
        
        # Store results
        if "portfolio_history" in strategy_results and strategy_results["portfolio_history"]:
            print(f"Portfolio history keys: {strategy_results['portfolio_history'][0].keys()}")
            
            results[strategy.strategy_id] = {
                "total_return": strategy_results["performance_metrics"].get("total_return", 0),
                "sharpe_ratio": strategy_results["performance_metrics"].get("sharpe_ratio", 0),
                "max_drawdown": strategy_results["performance_metrics"].get("max_drawdown", 0),
                "win_rate": strategy_results["performance_metrics"].get("win_rate", 0),
                "total_trades": strategy_results["performance_metrics"].get("total_trades", 0),
                "equity_curve": pd.Series([state["total_value"] for state in strategy_results["portfolio_history"]]),
                "drawdown": pd.Series([state.get("drawdown", 0) for state in strategy_results["portfolio_history"]])
            }
        else:
            print(f"No portfolio history available for {strategy.strategy_id}")
            results[strategy.strategy_id] = {
                "total_return": strategy_results["performance_metrics"].get("total_return", 0),
                "sharpe_ratio": strategy_results["performance_metrics"].get("sharpe_ratio", 0),
                "max_drawdown": strategy_results["performance_metrics"].get("max_drawdown", 0),
                "win_rate": strategy_results["performance_metrics"].get("win_rate", 0),
                "total_trades": strategy_results["performance_metrics"].get("total_trades", 0),
                "equity_curve": pd.Series(),
                "drawdown": pd.Series()
            }
    
    # Print results
    print("\nBacktest Results:")
    print(f"Initial Capital: ${100000.0:,.2f}")
    
    # Get the first strategy's final portfolio value
    if results:
        first_strategy = list(results.keys())[0]
        final_value = results[first_strategy]["equity_curve"].iloc[-1] if len(results[first_strategy]["equity_curve"]) > 0 else 100000.0
        total_return = results[first_strategy]["total_return"]
        total_trades = results[first_strategy]["total_trades"]
        
        print(f"Final Capital: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Total Trades: {total_trades}")
    else:
        print("No results available")
    
    # Print strategy performance
    print("\nStrategy Performance:")
    for strategy_name, strategy_results in results.items():
        print(f"{strategy_name}:")
        print(f"  Total Return: {strategy_results['total_return']:.2%}")
        print(f"  Sharpe Ratio: {strategy_results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {strategy_results['max_drawdown']:.2%}")
        print(f"  Win Rate: {strategy_results['win_rate']:.2%}")
        print(f"  Total Trades: {strategy_results['total_trades']}")
    
    # Plot results
    print("\nPlotting results...")
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves
    plt.subplot(2, 1, 1)
    for strategy_name, strategy_results in results.items():
        equity_curve = strategy_results['equity_curve']
        plt.plot(equity_curve.index, equity_curve.values, label=strategy_name)
    
    plt.title('Strategy Equity Curves')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot drawdowns
    plt.subplot(2, 1, 2)
    for strategy_name, strategy_results in results.items():
        drawdown = strategy_results['drawdown']
        plt.plot(drawdown.index, drawdown.values, label=strategy_name)
    
    plt.title('Strategy Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    print("Results saved to backtest_results.png")
    
    return results


def demonstrate_indicators():
    """Demonstrate the use of technical indicators."""
    print("\nDemonstrating technical indicators...")
    
    # Generate sample data
    data = generate_sample_data(days=100, symbols=['AAPL'])
    df = data['AAPL']
    
    # Calculate indicators
    df['sma_20'] = sma(df['close'], 20)
    df['ema_12'] = ema(df['close'], 12)
    df['ema_26'] = ema(df['close'], 26)
    df['rsi'] = rsi(df['close'], 14)
    
    # Plot indicators
    plt.figure(figsize=(12, 10))
    
    # Plot price and moving averages
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['close'], label='Close Price')
    plt.plot(df.index, df['sma_20'], label='SMA(20)')
    plt.plot(df.index, df['ema_12'], label='EMA(12)')
    plt.plot(df.index, df['ema_26'], label='EMA(26)')
    plt.title('Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot RSI
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['rsi'], label='RSI(14)')
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold')
    plt.title('RSI Indicator')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    # Plot volume
    plt.subplot(3, 1, 3)
    plt.bar(df.index, df['volume'], label='Volume')
    plt.title('Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('indicators_demo.png')
    print("Indicators demo saved to indicators_demo.png")


if __name__ == "__main__":
    print("Quantitative Trading Framework Example")
    print("=" * 40)
    
    # Run backtest example
    results = run_backtest_example()
    
    # Demonstrate indicators
    demonstrate_indicators()
    
    print("\nExample completed successfully!")
    print("Check backtest_results.png and indicators_demo.png for visualizations.")