"""
Backtest engine module for the quantitative trading framework.

This module provides a comprehensive backtesting system that simulates trading
strategies on historical data with realistic market conditions and cost modeling.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

from ..event_bus import get_event_bus, EventType, Event
from ..data_engine.data_engine import get_data_engine, DataFrequency, MarketData
from quant_framework.strategies.strategy_base import StrategyBase

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Enumeration of backtest modes."""
    VECTORIZED = "vectorized"  # Process all data at once (fastest)
    EVENT_DRIVEN = "event_driven"  # Process bar by bar (most realistic)
    HYBRID = "hybrid"  # Use vectorized for indicators, event-driven for execution


class CommissionModel(Enum):
    """Enumeration of commission models."""
    FIXED = "fixed"  # Fixed amount per trade
    PERCENTAGE = "percentage"  # Percentage of trade value
    TIERED = "tiered"  # Tiered commission based on volume


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.
    
    Attributes:
        start_time: Backtest start time
        end_time: Backtest end time
        initial_cash: Initial cash balance
        commission_model: Commission model
        commission_rate: Commission rate (percentage or fixed amount)
        slippage_model: Slippage model
        slippage_rate: Slippage rate
        mode: Backtest mode
        benchmark: Benchmark symbol for comparison
        rebalance_frequency: Portfolio rebalancing frequency
        allow_short_selling: Whether to allow short selling
        allow_leverage: Whether to allow leverage
        max_leverage: Maximum leverage ratio
    """
    start_time: datetime
    end_time: datetime
    initial_cash: float = 100000.0
    commission_model: CommissionModel = CommissionModel.PERCENTAGE
    commission_rate: float = 0.001  # 0.1%
    slippage_model: str = "fixed_percentage"
    slippage_rate: float = 0.0005  # 0.05%
    mode: BacktestMode = BacktestMode.EVENT_DRIVEN
    benchmark: Optional[str] = None
    rebalance_frequency: Optional[str] = None
    allow_short_selling: bool = False
    allow_leverage: bool = False
    max_leverage: float = 1.0


@dataclass
class Trade:
    """
    Trade record.
    
    Attributes:
        trade_id: Unique trade identifier
        symbol: Trading symbol
        side: Trade side (buy/sell)
        quantity: Trade quantity
        price: Trade price
        timestamp: Trade timestamp
        commission: Commission amount
        slippage: Slippage amount
        pnl: Trade PnL
    """
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    slippage: float
    pnl: float = 0.0


@dataclass
class Position:
    """
    Position record.
    
    Attributes:
        symbol: Trading symbol
        quantity: Position quantity
        avg_price: Average entry price
        market_value: Current market value
        unrealized_pnl: Unrealized PnL
        realized_pnl: Realized PnL
        last_update: Last update timestamp
    """
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: datetime
    
    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized PnL based on current price."""
        self.market_value = self.quantity * current_price
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
        else:
            self.unrealized_pnl = 0.0
        self.last_update = datetime.now()


@dataclass
class Portfolio:
    """
    Portfolio record.
    
    Attributes:
        cash: Cash balance
        positions: Dictionary of positions
        total_value: Total portfolio value
        total_pnl: Total PnL
        last_update: Last update timestamp
    """
    cash: float
    positions: Dict[str, Position]
    total_value: float
    total_pnl: float
    last_update: datetime
    
    def update_total_value(self) -> None:
        """Update total portfolio value."""
        self.total_value = self.cash + sum(pos.market_value for pos in self.positions.values())
        self.total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in self.positions.values())
        self.last_update = datetime.now()


class Broker:
    """
    Broker simulation for backtesting.
    
    This class simulates order execution with realistic market conditions,
    including commissions, slippage, and order fills.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the broker.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.commission_model = config.commission_model
        self.commission_rate = config.commission_rate
        self.slippage_model = config.slippage_model
        self.slippage_rate = config.slippage_rate
    
    def execute_order(self, symbol: str, side: str, quantity: float, 
                     price: float, timestamp: datetime) -> Trade:
        """
        Execute an order with realistic market conditions.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Order price
            timestamp: Order timestamp
            
        Returns:
            Trade object with execution details
        """
        # Calculate slippage
        slippage = self._calculate_slippage(price, side)
        
        # Apply slippage to price
        if side == 'buy':
            execution_price = price * (1 + slippage)
        else:  # sell
            execution_price = price * (1 - slippage)
        
        # Calculate commission
        commission = self._calculate_commission(execution_price, quantity)
        
        # Create trade record
        trade_id = str(uuid.uuid4())
        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=execution_price,
            timestamp=timestamp,
            commission=commission,
            slippage=price * slippage
        )
        
        return trade
    
    def _calculate_slippage(self, price: float, side: str) -> float:
        """Calculate slippage based on model."""
        if self.slippage_model == "fixed_percentage":
            return self.slippage_rate
        elif self.slippage_model == "volume_based":
            # More complex model based on volume
            return self.slippage_rate
        else:
            return 0.0
    
    def _calculate_commission(self, price: float, quantity: float) -> float:
        """Calculate commission based on model."""
        trade_value = price * quantity
        
        if self.commission_model == CommissionModel.FIXED:
            return self.commission_rate
        elif self.commission_model == CommissionModel.PERCENTAGE:
            return trade_value * self.commission_rate
        elif self.commission_model == CommissionModel.TIERED:
            # Tiered commission based on trade value
            if trade_value < 1000:
                return trade_value * 0.001
            elif trade_value < 10000:
                return trade_value * 0.0005
            else:
                return trade_value * 0.0002
        else:
            return 0.0


class BacktestEngine:
    """
    Main backtest engine class.
    
    This class orchestrates the backtesting process, managing data flow,
    strategy execution, and portfolio tracking.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtest engine.
        
        Args:
            config: Backtest configuration
        """
        self.config = config
        self.data_engine = get_data_engine()
        self.event_bus = get_event_bus()
        self.broker = Broker(config)
        
        # Portfolio and tracking
        self.portfolio = Portfolio(
            cash=config.initial_cash,
            positions={},
            total_value=config.initial_cash,
            total_pnl=0.0,
            last_update=config.start_time
        )
        
        # Trade history
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Strategy
        self.strategy: Optional[StrategyBase] = None
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        
        # Event handlers
        self._setup_event_handlers()
        
        # Start event bus for processing events
        self.event_bus.start()
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for the backtest."""
        self.event_bus.subscribe(EventType.STRATEGY_SIGNAL, self._handle_strategy_signal)
    
    def set_strategy(self, strategy: StrategyBase) -> None:
        """
        Set the strategy for backtesting.
        
        Args:
            strategy: Strategy instance
        """
        self.strategy = strategy
        self.strategy.set_backtest_mode(True)
    
    def set_historical_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Set historical data for backtesting.
        
        Args:
            data: Dictionary of DataFrames by symbol
        """
        self._historical_data = data
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dictionary with backtest results
        """
        if not self.strategy:
            raise ValueError("No strategy set for backtesting")
        
        logger.info(f"Starting backtest from {self.config.start_time} to {self.config.end_time}")
        
        # Initialize strategy
        self.strategy.on_start()
        
        # Run based on backtest mode
        if self.config.mode == BacktestMode.EVENT_DRIVEN:
            self._run_event_driven()
        elif self.config.mode == BacktestMode.VECTORIZED:
            self._run_vectorized()
        elif self.config.mode == BacktestMode.HYBRID:
            self._run_hybrid()
        
        # Finalize strategy
        self.strategy.on_stop()
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        # Stop event bus
        self.event_bus.stop()
        
        logger.info("Backtest completed")
        
        return {
            "config": self.config,
            "portfolio": self.portfolio,
            "trades": self.trades,
            "portfolio_history": self.portfolio_history,
            "performance_metrics": self.performance_metrics
        }
    
    def _run_event_driven(self) -> None:
        """Run the backtest in event-driven mode."""
        # Get all symbols from strategy
        symbols = self.strategy.get_symbols()
        frequency = self.strategy.get_frequency()
        
        # Use historical data if available, otherwise try to get from data engine
        if hasattr(self, '_historical_data'):
            data = self._historical_data
        else:
            # Get historical data for all symbols
            data = {}
            for symbol in symbols:
                symbol_data = self.data_engine.get_historical_data(
                    symbol, frequency, self.config.start_time, self.config.end_time
                )
                data[symbol] = symbol_data
        
        # Process each timestamp
        all_timestamps = sorted(set().union(*[df.index for df in data.values()]))
        
        for timestamp in all_timestamps:
            # Get current bar data for all symbols
            current_data = {}
            for symbol, df in data.items():
                if timestamp in df.index:
                    current_data[symbol] = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=df.loc[timestamp, 'open'],
                        high=df.loc[timestamp, 'high'],
                        low=df.loc[timestamp, 'low'],
                        close=df.loc[timestamp, 'close'],
                        volume=df.loc[timestamp, 'volume'],
                        frequency=frequency,
                        source="backtest"
                    )
            
            # Update portfolio values
            self._update_portfolio_values(current_data)
            
            # Feed data to strategy
            self.strategy.on_bar(current_data)
            
            # Record portfolio state
            self._record_portfolio_state(timestamp)
    
    def _run_vectorized(self) -> None:
        """Run the backtest in vectorized mode."""
        # Get all symbols from strategy
        symbols = self.strategy.get_symbols()
        frequency = self.strategy.get_frequency()
        
        # Use historical data if available, otherwise try to get from data engine
        if hasattr(self, '_historical_data'):
            data = self._historical_data
        else:
            # Get historical data for all symbols
            data = {}
            for symbol in symbols:
                symbol_data = self.data_engine.get_historical_data(
                    symbol, frequency, self.config.start_time, self.config.end_time
                )
                data[symbol] = symbol_data
        
        # Get all timestamps
        all_timestamps = sorted(set().union(*[df.index for df in data.values()]))
        
        # Process all data at once
        for timestamp in all_timestamps:
            # Get current bar data for all symbols
            current_data = {}
            for symbol, df in data.items():
                if timestamp in df.index:
                    current_data[symbol] = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=df.loc[timestamp, 'open'],
                        high=df.loc[timestamp, 'high'],
                        low=df.loc[timestamp, 'low'],
                        close=df.loc[timestamp, 'close'],
                        volume=df.loc[timestamp, 'volume'],
                        frequency=frequency,
                        source="backtest"
                    )
            
            # Update portfolio values
            self._update_portfolio_values(current_data)
            
            # Feed data to strategy
            self.strategy.on_bar(current_data)
            
            # Record portfolio state
            self._record_portfolio_state(timestamp)
    
    def _run_hybrid(self) -> None:
        """Run the backtest in hybrid mode."""
        # Similar to event-driven but with optimizations
        self._run_event_driven()
    
    def _update_portfolio_values(self, current_data: Dict[str, MarketData]) -> None:
        """Update portfolio values based on current market data."""
        for symbol, market_data in current_data.items():
            if symbol in self.portfolio.positions:
                self.portfolio.positions[symbol].update_market_value(market_data.close)
        
        self.portfolio.update_total_value()
    
    def _record_portfolio_state(self, timestamp: datetime) -> None:
        """Record the current portfolio state."""
        state = {
            "timestamp": timestamp,
            "cash": self.portfolio.cash,
            "total_value": self.portfolio.total_value,
            "total_pnl": self.portfolio.total_pnl,
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl
                }
                for symbol, pos in self.portfolio.positions.items()
            }
        }
        
        self.portfolio_history.append(state)
    
    def _handle_strategy_signal(self, event: Event) -> None:
        """
        Handle strategy signal events.
        
        Args:
            event: Strategy signal event
        """
        signal_data = event.data.get("signal", {})
        symbol = signal_data.get("symbol")
        signal_type = signal_data.get("signal_type")  # 'buy', 'sell', 'hold'
        quantity = signal_data.get("quantity", 0)
        timestamp = signal_data.get("timestamp", datetime.now())
        price = signal_data.get("price")  # Get price from signal if available
        
        if signal_type == "hold" or quantity == 0:
            return
        
        # Use price from signal if available, otherwise try to get from historical data
        if price is None:
            # Try to get from historical data if available
            if hasattr(self, '_historical_data') and symbol in self._historical_data:
                # Get the most recent price at or before the signal timestamp
                df = self._historical_data[symbol]
                valid_times = df[df.index <= timestamp]
                if not valid_times.empty:
                    price = valid_times.iloc[-1]["close"]
            
            # If still no price, try data engine
            if price is None:
                try:
                    current_data = self.data_engine.get_latest_data(symbol, DataFrequency.MINUTE, 1)
                    if not current_data.empty:
                        price = current_data.iloc[-1]["close"]
                except Exception as e:
                    logger.warning(f"Could not get price for {symbol}: {str(e)}")
        
        # If still no price, skip this signal
        if price is None:
            logger.warning(f"No price available for {symbol}, skipping signal")
            return
        
        # Execute order
        trade = self.broker.execute_order(symbol, signal_type, quantity, price, timestamp)
        self.trades.append(trade)
        
        # Update portfolio
        self._update_portfolio_after_trade(trade)
        
        # Log trade
        logger.info(f"Executed {signal_type} {quantity} {symbol} at {trade.price:.4f}")
    
    def _update_portfolio_after_trade(self, trade: Trade) -> None:
        """Update portfolio after a trade."""
        symbol = trade.symbol
        
        # Update cash
        if trade.side == "buy":
            self.portfolio.cash -= (trade.quantity * trade.price) + trade.commission
        else:  # sell
            self.portfolio.cash += (trade.quantity * trade.price) - trade.commission
        
        # Update or create position
        if symbol in self.portfolio.positions:
            position = self.portfolio.positions[symbol]
            
            if trade.side == "buy":
                # Update average price
                total_cost = (position.quantity * position.avg_price) + (trade.quantity * trade.price)
                position.quantity += trade.quantity
                position.avg_price = total_cost / position.quantity if position.quantity != 0 else 0
            else:  # sell
                # Calculate realized PnL
                realized_pnl = trade.quantity * (trade.price - position.avg_price)
                position.realized_pnl += realized_pnl
                position.quantity -= trade.quantity
                
                # If position is closed, reset avg price
                if position.quantity == 0:
                    position.avg_price = 0
        else:
            # Create new position
            position = Position(
                symbol=symbol,
                quantity=trade.quantity if trade.side == "buy" else -trade.quantity,
                avg_price=trade.price,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                last_update=trade.timestamp
            )
            
            if trade.side == "sell":
                position.avg_price = trade.price
                position.quantity = -trade.quantity
            
            self.portfolio.positions[symbol] = position
        
        # Update portfolio values
        self.portfolio.update_total_value()
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics for the backtest."""
        if not self.portfolio_history:
            return
        
        # Extract portfolio values
        values = [state["total_value"] for state in self.portfolio_history]
        timestamps = [state["timestamp"] for state in self.portfolio_history]
        
        # Calculate returns
        returns = pd.Series(values).pct_change().dropna()
        
        # Basic metrics
        total_return = (values[-1] - self.config.initial_cash) / self.config.initial_cash
        annualized_return = (1 + total_return) ** (252 / len(values)) - 1  # Assuming 252 trading days per year
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        max_drawdown = self._calculate_max_drawdown(values)
        
        # Trade statistics
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_average_win()
        avg_loss = self._calculate_average_loss()
        profit_factor = self._calculate_profit_factor()
        
        # Store metrics
        self.performance_metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "start_date": self.config.start_time,
            "end_date": self.config.end_time,
            "initial_cash": self.config.initial_cash,
            "final_value": values[-1] if values else self.config.initial_cash
        }
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown."""
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        if not self.trades:
            return 0.0
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        return len(winning_trades) / len(self.trades)
    
    def _calculate_average_win(self) -> float:
        """Calculate average winning trade."""
        winning_trades = [t for t in self.trades if t.pnl > 0]
        return sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
    
    def _calculate_average_loss(self) -> float:
        """Calculate average losing trade."""
        losing_trades = [t for t in self.trades if t.pnl < 0]
        return sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get backtest results.
        
        Returns:
            Dictionary with backtest results
        """
        return {
            "config": self.config,
            "portfolio": self.portfolio,
            "trades": self.trades,
            "portfolio_history": self.portfolio_history,
            "performance_metrics": self.performance_metrics
        }