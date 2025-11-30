"""
Live trading engine module for the quantitative trading framework.

This module provides a comprehensive live trading system that executes strategies
in real market conditions with proper risk management and monitoring.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import threading

from ...utils.config_loader import get_config
from ..event_bus import get_event_bus, EventType, Event
from ..data_engine.data_engine import get_data_engine, DataFrequency, MarketData
from ...strategies.strategy_base import StrategyBase
from ...execution.order_manager import OrderManager
from ..common import Order, Position, OrderStatus, OrderSide, OrderType
from ..risk_engine.risk_engine import RiskEngine

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Enumeration of trading modes."""
    SIMULATION = "simulation"  # Paper trading without real money
    LIVE = "live"  # Live trading with real money
    MONITOR = "monitor"  # Monitor only, no trading


@dataclass
class LiveTradingConfig:
    """
    Configuration for live trading.
    
    Attributes:
        trading_mode: Trading mode (simulation, live, monitor)
        symbols: List of symbols to trade
        strategies: List of strategies to run
        risk_limits: Risk limits configuration
        order_settings: Order execution settings
        data_settings: Data settings
        notification_settings: Notification settings
    """
    trading_mode: TradingMode
    symbols: List[str]
    strategies: List[str]
    risk_limits: Dict[str, Any]
    order_settings: Dict[str, Any]
    data_settings: Dict[str, Any]
    notification_settings: Dict[str, Any]


class LiveTradingEngine:
    """
    Main live trading engine class.
    
    This class orchestrates the live trading process, managing strategy execution,
    order management, risk control, and monitoring.
    """
    
    def __init__(self, config: LiveTradingConfig):
        """
        Initialize the live trading engine.
        
        Args:
            config: Live trading configuration
        """
        self.config = config
        self.data_engine = get_data_engine()
        self.event_bus = get_event_bus()
        self.order_manager = OrderManager(config.order_settings)
        self.risk_engine = RiskEngine(config.risk_limits)
        
        # Trading state
        self.is_running = False
        self.start_time = None
        self.stop_time = None
        
        # Portfolio and positions
        self.cash = 0.0
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        
        # Strategies
        self.strategies: Dict[str, StrategyBase] = {}
        
        # Data subscriptions
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        
        # Event handlers
        self._setup_event_handlers()
        
        # Worker thread
        self._worker_thread = None
        self._stop_event = threading.Event()
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for live trading."""
        self.event_bus.subscribe(EventType.STRATEGY_SIGNAL, self._handle_strategy_signal)
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self.event_bus.subscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        self.event_bus.subscribe(EventType.ORDER_REJECTED, self._handle_order_rejected)
        self.event_bus.subscribe(EventType.MARKET_DATA, self._handle_market_data)
    
    def add_strategy(self, strategy: StrategyBase) -> None:
        """
        Add a strategy to the engine.
        
        Args:
            strategy: Strategy instance
        """
        strategy_id = strategy.get_strategy_id()
        self.strategies[strategy_id] = strategy
        strategy.set_live_mode(True)
        logger.info(f"Added strategy {strategy_id} to live trading engine")
    
    def start(self) -> None:
        """Start the live trading engine."""
        if self.is_running:
            logger.warning("Live trading engine is already running")
            return
        
        logger.info(f"Starting live trading engine in {self.config.trading_mode.value} mode")
        
        # Initialize strategies
        for strategy_id, strategy in self.strategies.items():
            strategy.on_start()
        
        # Initialize positions and cash
        self._initialize_portfolio()
        
        # Subscribe to market data
        self._subscribe_to_market_data()
        
        # Start worker thread
        self.is_running = True
        self.start_time = datetime.now()
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("Live trading engine started")
    
    def stop(self) -> None:
        """Stop the live trading engine."""
        if not self.is_running:
            logger.warning("Live trading engine is not running")
            return
        
        logger.info("Stopping live trading engine")
        
        # Set stop event
        self._stop_event.set()
        
        # Cancel all pending orders
        self._cancel_all_orders()
        
        # Stop strategies
        for strategy_id, strategy in self.strategies.items():
            strategy.on_stop()
        
        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        
        self.is_running = False
        self.stop_time = datetime.now()
        
        logger.info("Live trading engine stopped")
    
    def _run_loop(self) -> None:
        """Main run loop for the live trading engine."""
        while self.is_running and not self._stop_event.is_set():
            try:
                # Process any pending tasks
                self._process_pending_tasks()
                
                # Check risk limits
                self._check_risk_limits()
                
                # Small delay to prevent busy waiting
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in live trading loop: {str(e)}")
                # Continue running even if there's an error
    
    def _process_pending_tasks(self) -> None:
        """Process any pending tasks."""
        # This would handle any pending tasks like order updates, etc.
        pass
    
    def _check_risk_limits(self) -> None:
        """Check if any risk limits are breached."""
        # Get current portfolio state
        portfolio_state = {
            "cash": self.cash,
            "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            "orders": {order_id: order.to_dict() for order_id, order in self.orders.items()}
        }
        
        # Check risk limits
        risk_check = self.risk_engine.check_risk_limits(portfolio_state)
        
        if not risk_check["passed"]:
            # Risk limit breached
            logger.warning(f"Risk limit breached: {risk_check['message']}")
            
            # Publish risk event
            event = Event(
                event_type=EventType.RISK_LIMIT_BREACH,
                timestamp=time.time(),
                data=risk_check,
                source="live_engine"
            )
            self.event_bus.publish(event)
            
            # Take action based on risk limit type
            self._handle_risk_limit_breach(risk_check)
    
    def _handle_risk_limit_breach(self, risk_check: Dict[str, Any]) -> None:
        """Handle risk limit breach."""
        limit_type = risk_check.get("limit_type")
        
        if limit_type == "max_position_size":
            # Reduce position size
            self._reduce_positions()
        elif limit_type == "max_drawdown":
            # Stop all strategies
            self._stop_all_strategies()
        elif limit_type == "max_loss":
            # Liquidate all positions
            self._liquidate_all_positions()
    
    def _reduce_positions(self) -> None:
        """Reduce positions to meet risk limits."""
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                # Reduce position by half
                reduce_quantity = abs(position.quantity) * 0.5
                side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                
                # Create order to reduce position
                order = Order(
                    order_id=str(uuid.uuid4()),
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=reduce_quantity,
                    strategy_id="risk_management"
                )
                
                # Submit order
                self._submit_order(order)
    
    def _stop_all_strategies(self) -> None:
        """Stop all strategies."""
        for strategy_id, strategy in self.strategies.items():
            strategy.on_stop()
        logger.info("All strategies stopped due to risk limit breach")
    
    def _liquidate_all_positions(self) -> None:
        """Liquidate all positions."""
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                # Create order to liquidate position
                side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                
                order = Order(
                    order_id=str(uuid.uuid4()),
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=abs(position.quantity),
                    strategy_id="risk_management"
                )
                
                # Submit order
                self._submit_order(order)
        logger.info("All positions liquidated due to risk limit breach")
    
    def _initialize_portfolio(self) -> None:
        """Initialize portfolio with current positions and cash."""
        # This would fetch current portfolio state from the broker
        # For now, we'll use default values
        self.cash = 100000.0  # Default cash
        logger.info(f"Initialized portfolio with cash: {self.cash}")
    
    def _subscribe_to_market_data(self) -> None:
        """Subscribe to market data for all symbols."""
        for symbol in self.config.symbols:
            # Subscribe to real-time data
            self.data_engine.subscribe_realtime(
                symbol=symbol,
                frequency=DataFrequency.MINUTE,  # Default to minute data
                callback=self._handle_market_data
            )
            
            self.subscriptions[symbol] = {
                "frequency": DataFrequency.MINUTE,
                "subscribed": True
            }
        
        logger.info(f"Subscribed to market data for {len(self.config.symbols)} symbols")
    
    def _handle_market_data(self, event: Event) -> None:
        """
        Handle market data events.
        
        Args:
            event: Market data event
        """
        data = event.data
        
        # Convert to MarketData object
        market_data = MarketData.from_dict(data)
        
        # Update position values
        if market_data.symbol in self.positions:
            self.positions[market_data.symbol].update_market_value(market_data.close)
        
        # Feed data to all strategies
        for strategy_id, strategy in self.strategies.items():
            strategy.on_bar({market_data.symbol: market_data})
    
    def _handle_strategy_signal(self, event: Event) -> None:
        """
        Handle strategy signal events.
        
        Args:
            event: Strategy signal event
        """
        signal_data = event.data
        symbol = signal_data["symbol"]
        action = signal_data["action"]  # 'buy', 'sell', 'hold'
        quantity = signal_data.get("quantity", 0)
        strategy_id = signal_data.get("strategy_id", "unknown")
        
        if action == "hold" or quantity == 0:
            return
        
        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.BUY if action == "buy" else OrderSide.SELL,
            order_type=OrderType.MARKET,  # Default to market orders
            quantity=quantity,
            strategy_id=strategy_id
        )
        
        # Check risk limits before submitting
        if self._check_order_risk(order):
            self._submit_order(order)
        else:
            logger.warning(f"Order rejected by risk engine: {order.order_id}")
    
    def _check_order_risk(self, order: Order) -> bool:
        """
        Check if order passes risk checks.
        
        Args:
            order: Order to check
            
        Returns:
            True if order passes risk checks, False otherwise
        """
        # Get current portfolio state
        portfolio_state = {
            "cash": self.cash,
            "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            "orders": {order_id: order.to_dict() for order_id, order in self.orders.items()}
        }
        
        # Check order risk
        risk_check = self.risk_engine.check_order_risk(order, portfolio_state)
        
        return risk_check["passed"]
    
    def _submit_order(self, order: Order) -> None:
        """
        Submit an order to the broker.
        
        Args:
            order: Order to submit
        """
        # Update order status
        order.status = OrderStatus.SUBMITTED
        order.updated_timestamp = datetime.now()
        
        # Store order
        self.orders[order.order_id] = order
        
        # Submit to order manager
        if self.config.trading_mode == TradingMode.LIVE:
            self.order_manager.submit_order(order)
        elif self.config.trading_mode == TradingMode.SIMULATION:
            # Simulate order execution
            self._simulate_order_execution(order)
        
        # Publish order event
        event = Event(
            event_type=EventType.ORDER_CREATED,
            timestamp=time.time(),
            data=order.to_dict(),
            source="live_engine"
        )
        self.event_bus.publish(event)
        
        logger.info(f"Submitted order {order.order_id}: {order.side.value} {order.quantity} {order.symbol}")
    
    def _simulate_order_execution(self, order: Order) -> None:
        """
        Simulate order execution for paper trading.
        
        Args:
            order: Order to simulate
        """
        # Get current price
        current_data = self.data_engine.get_latest_data(order.symbol, DataFrequency.MINUTE, 1)
        if current_data.empty:
            logger.warning(f"No current data available for {order.symbol}")
            return
        
        current_price = current_data.iloc[-1]["close"]
        
        # Simulate fill
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = current_price
        order.updated_timestamp = datetime.now()
        
        # Calculate commission (simplified)
        order.commission = order.filled_quantity * order.avg_fill_price * 0.001  # 0.1% commission
        
        # Update portfolio
        self._update_portfolio_after_fill(order)
        
        # Publish fill event
        event = Event(
            event_type=EventType.ORDER_FILLED,
            timestamp=time.time(),
            data=order.to_dict(),
            source="live_engine"
        )
        self.event_bus.publish(event)
    
    def _update_portfolio_after_fill(self, order: Order) -> None:
        """
        Update portfolio after an order fill.
        
        Args:
            order: Filled order
        """
        symbol = order.symbol
        
        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= (order.filled_quantity * order.avg_fill_price) + order.commission
        else:  # SELL
            self.cash += (order.filled_quantity * order.avg_fill_price) - order.commission
        
        # Update or create position
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if order.side == OrderSide.BUY:
                # Update average price
                total_cost = (position.quantity * position.avg_price) + (order.filled_quantity * order.avg_fill_price)
                position.quantity += order.filled_quantity
                position.avg_price = total_cost / position.quantity if position.quantity != 0 else 0
            else:  # SELL
                # Calculate realized PnL
                realized_pnl = order.filled_quantity * (order.avg_fill_price - position.avg_price)
                position.realized_pnl += realized_pnl
                position.quantity -= order.filled_quantity
                
                # If position is closed, reset avg price
                if position.quantity == 0:
                    position.avg_price = 0
        else:
            # Create new position
            position = Position(
                symbol=symbol,
                quantity=order.filled_quantity if order.side == OrderSide.BUY else -order.filled_quantity,
                avg_price=order.avg_fill_price,
                market_value=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                last_update=order.updated_timestamp
            )
            
            if order.side == OrderSide.SELL:
                position.avg_price = order.avg_fill_price
                position.quantity = -order.filled_quantity
            
            self.positions[symbol] = position
        
        # Update market value
        current_data = self.data_engine.get_latest_data(symbol, DataFrequency.MINUTE, 1)
        if not current_data.empty:
            current_price = current_data.iloc[-1]["close"]
            position.update_market_value(current_price)
    
    def _handle_order_filled(self, event: Event) -> None:
        """
        Handle order filled events.
        
        Args:
            event: Order filled event
        """
        data = event.data
        order_id = data["order_id"]
        
        if order_id in self.orders:
            # Update order
            order = self.orders[order_id]
            order.status = OrderStatus.FILLED
            order.filled_quantity = data["filled_quantity"]
            order.avg_fill_price = data["avg_fill_price"]
            order.commission = data["commission"]
            order.updated_timestamp = datetime.now()
            
            # Update portfolio
            self._update_portfolio_after_fill(order)
            
            logger.info(f"Order filled: {order_id}")
    
    def _handle_order_cancelled(self, event: Event) -> None:
        """
        Handle order cancelled events.
        
        Args:
            event: Order cancelled event
        """
        data = event.data
        order_id = data["order_id"]
        
        if order_id in self.orders:
            # Update order
            order = self.orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_timestamp = datetime.now()
            
            logger.info(f"Order cancelled: {order_id}")
    
    def _handle_order_rejected(self, event: Event) -> None:
        """
        Handle order rejected events.
        
        Args:
            event: Order rejected event
        """
        data = event.data
        order_id = data["order_id"]
        
        if order_id in self.orders:
            # Update order
            order = self.orders[order_id]
            order.status = OrderStatus.REJECTED
            order.updated_timestamp = datetime.now()
            
            logger.warning(f"Order rejected: {order_id}, reason: {data.get('reason', 'Unknown')}")
    
    def _cancel_all_orders(self) -> None:
        """Cancel all pending orders."""
        for order_id, order in self.orders.items():
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]:
                if self.config.trading_mode == TradingMode.LIVE:
                    self.order_manager.cancel_order(order_id)
                
                # Update order status
                order.status = OrderStatus.CANCELLED
                order.updated_timestamp = datetime.now()
                
                # Publish cancel event
                event = Event(
                    event_type=EventType.ORDER_CANCELLED,
                    timestamp=time.time(),
                    data=order.to_dict(),
                    source="live_engine"
                )
                self.event_bus.publish(event)
        
        logger.info("All pending orders cancelled")
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Dictionary with portfolio state
        """
        total_value = self.cash + sum(pos.market_value for pos in self.positions.values())
        total_pnl = sum(pos.realized_pnl + pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            "cash": self.cash,
            "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            "orders": {order_id: order.to_dict() for order_id, order in self.orders.items()},
            "total_value": total_value,
            "total_pnl": total_pnl,
            "is_running": self.is_running,
            "start_time": self.start_time,
            "stop_time": self.stop_time
        }
    
    def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Dictionary with strategy performance
        """
        if strategy_id not in self.strategies:
            return {}
        
        strategy = self.strategies[strategy_id]
        return strategy.get_performance_metrics()