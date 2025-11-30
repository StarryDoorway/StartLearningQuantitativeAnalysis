"""
Refactored strategy base module for the quantitative trading framework.

This module provides the base strategy class that all trading strategies should inherit from.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import pandas as pd

from quant_framework.utils.config_loader import get_config
from quant_framework.core.event_bus import get_event_bus, EventType, Event
from quant_framework.strategies.base.signal_types import Signal, SignalType, SignalStrength
from quant_framework.strategies.base.strategy_state import StrategyState
from quant_framework.strategies.base.strategy_data_manager import StrategyDataManager
from quant_framework.strategies.base.strategy_performance_manager import StrategyPerformanceManager
from quant_framework.strategies.base.strategy_executor import StrategyExecutor
from quant_framework.strategies.risk_managers.strategy_risk_manager import StrategyRiskManager

logger = logging.getLogger(__name__)


class StrategyBase(ABC):
    """
    Base class for all trading strategies.
    
    This class provides the basic structure and functionality that all
    trading strategies should inherit from.
    """
    
    def __init__(self, strategy_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            config: Strategy configuration (optional, will use default if None)
        """
        self.strategy_id = strategy_id
        
        # Use default config if none provided
        if config is None:
            config = self.get_default_config()
        
        self.config = config
        
        # Event bus
        self.event_bus = get_event_bus()
        
        # Strategy state
        self.is_running = False
        self.is_live = False
        self.is_backtest = False
        self.current_positions = {}
        
        # Strategy parameters
        self.parameters = config.get("parameters", {})
        
        # Initialize managers
        self.data_manager = StrategyDataManager(
            strategy_id=strategy_id,
            max_data_rows=config.get("max_data_rows", 10000)
        )
        
        self.performance_manager = StrategyPerformanceManager(
            strategy_id=strategy_id
        )
        
        self.risk_manager = StrategyRiskManager(strategy_id=strategy_id)
        
        # Initialize executor if order and position managers are available
        self.executor = None
        self.logger = logging.getLogger(f"{__name__}.{strategy_id}")
        
        try:
            from quant_framework.execution.order_manager import get_order_manager
            from quant_framework.execution.portfolio_manager import get_position_manager
            self.executor = StrategyExecutor(
                strategy_id=strategy_id,
                order_manager=get_order_manager(),
                position_manager=get_position_manager()
            )
        except ImportError:
            # Create mock managers for testing
            class MockOrderManager:
                def submit_order(self, order):
                    return "mock_order_id"
                
                def cancel_order(self, order_id):
                    return True
            
            class MockPositionManager:
                def get_position(self, symbol):
                    return 0.0
            
            self.executor = StrategyExecutor(
                strategy_id=strategy_id,
                order_manager=MockOrderManager(),
                position_manager=MockPositionManager()
            )
            self.logger.warning("Using mock order and position managers for testing")
        
        # Logger
        self.last_update = datetime.now()
        
        # Initialize strategy
        self._initialize()
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for this strategy.
        
        This method should be overridden by subclasses to provide
        strategy-specific default parameters.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "parameters": {},
            "max_data_rows": 10000,
            "symbols": []
        }
    
    def _initialize(self) -> None:
        """Initialize strategy-specific components."""
        # Initialize symbols from config if provided
        symbols = self.config.get("symbols", [])
        for symbol in symbols:
            if symbol not in self.data_manager.data:
                # Initialize empty DataFrame for each symbol
                self.data_manager.data[symbol] = pd.DataFrame()
        
        # This can be overridden by subclasses for additional initialization
        pass
    
    @abstractmethod
    def calculate_signals(self, symbol: str, data) -> List[Signal]:
        """
        Calculate trading signals for a symbol.
        
        This is the main method that subclasses must implement.
        
        Args:
            symbol: Trading symbol
            data: Historical market data
            
        Returns:
            List of trading signals
        """
        pass
    
    def on_bar(self, market_data: Dict[str, Any]) -> None:
        """
        Handle new bar data.
        
        Args:
            market_data: Dictionary of market data by symbol
        """
        # Log the incoming data
        self.logger.debug(f"Received bar data for {len(market_data)} symbols: {list(market_data.keys())}")
        
        # Update data for each symbol
        for symbol, data in market_data.items():
            self.data_manager.update_data(symbol, data)
        
        # Calculate signals for each symbol
        for symbol in market_data.keys():
            if symbol in self.data_manager.data and len(self.data_manager.data[symbol]) > 0:
                self.logger.debug(f"Calculating signals for {symbol}, data length: {len(self.data_manager.data[symbol])}")
                signals = self.calculate_signals(symbol, self.data_manager.data[symbol])
                self.logger.debug(f"Generated {len(signals)} signals for {symbol}")
                
                # Process signals
                for signal in signals:
                    self._process_signal(signal)
    
    def _calculate_quantity(self, signal: Signal) -> float:
        """
        Calculate the quantity for a trading signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Quantity to trade
        """
        # Default quantity calculation - can be overridden by subclasses
        if signal.price is None or signal.price <= 0:
            return 0.0
        
        # Get default quantity from parameters or use a fixed amount
        default_quantity = self.parameters.get("default_quantity", 100)
        
        # Calculate based on signal strength
        strength_multiplier = signal.strength.value / SignalStrength.MODERATE.value
        
        # Apply strength multiplier
        quantity = default_quantity * strength_multiplier
        
        # Ensure quantity is positive and reasonable
        quantity = max(0.0, min(quantity, 10000))
        
        return quantity
    
    def _register_event_handlers(self):
        """Register event handlers with the event bus."""
        try:
            event_bus = get_event_bus()
            
            # Register for order updates if executor is available
            if self.executor:
                event_bus.subscribe(EventType.ORDER_UPDATE, self._handle_order_update)
                
            # Register for trade updates
            event_bus.subscribe(EventType.TRADE, self._handle_trade_update)
            
        except Exception as e:
            self.logger.error(f"Failed to register event handlers: {str(e)}")
    
    def _handle_order_update(self, event: Event):
        """
        Handle order update events.
        
        Args:
            event: Order update event
        """
        if self.executor:
            self.executor.handle_order_update(event.data)
    
    def _handle_trade_update(self, event: Event):
        """
        Handle trade update events.
        
        Args:
            event: Trade update event
        """
        trade_data = event.data
        
        # Check if this trade is for our strategy
        if trade_data.get("strategy_id") == self.strategy_id:
            # Add trade to history
            self.data_manager.add_trade(trade_data)
            
            # Update performance metrics
            self.performance_manager.update_metrics(self.data_manager.get_trade_history())
            
            # Log trade
            self.logger.info(f"Trade executed: {trade_data}")
    
    def _process_signal(self, signal: Signal, current_price: float = None):
        """
        Process a trading signal.
        
        Args:
            signal: Trading signal to process
            current_price: Current market price (optional)
        """
        # Add signal to history
        self.data_manager.add_signal(signal)
        
        # If executor is available, execute the signal
        if self.executor and signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            # Get current price if not provided
            if current_price is None:
                data = self.data_manager.get_data(signal.symbol, 1)
                if not data.empty:
                    current_price = data['close'].iloc[-1]
                else:
                    self.logger.warning(f"No data available for {signal.symbol}, cannot execute signal")
                    return
            
            # Perform risk check
            current_positions = {}  # Would get from position manager
            portfolio_value = 10000.0  # Would get from portfolio manager
            
            is_allowed, reason = self.risk_manager.check_signal_risk(
                signal, current_price, current_positions, portfolio_value
            )
            
            if not is_allowed:
                self.logger.warning(f"Signal rejected by risk manager: {reason}")
                return
            
            # Calculate position size based on risk parameters
            quantity = self.risk_manager.calculate_position_size(signal, current_price, portfolio_value)
            
            # Execute the signal
            success = self.executor.execute_signal(signal, current_price, quantity)
            
            if success:
                self.logger.info(f"Executed signal: {signal}")
            else:
                self.logger.error(f"Failed to execute signal: {signal}")
        else:
            # Just log the signal if no executor
            self.logger.info(f"Signal generated: {signal}")
    
    def cancel_pending_orders(self):
        """Cancel all pending orders."""
        if self.executor:
            self.executor.cancel_pending_orders()
            self.logger.info("All pending orders canceled")
        else:
            self.logger.warning("No executor available, cannot cancel orders")
    
    def on_start(self) -> None:
        """Called when the strategy starts."""
        self.is_running = True
        self.last_update = datetime.now()
        self.logger.info(f"Strategy {self.strategy_id} started")
    
    def on_stop(self) -> None:
        """Called when the strategy stops."""
        self.is_running = False
        self.last_update = datetime.now()
        self.logger.info(f"Strategy {self.strategy_id} stopped")
    
    def on_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Called when a trade is executed.
        
        Args:
            trade_data: Trade execution data
        """
        # Add to trade history
        self.data_manager.add_trade(trade_data)
        
        # Update performance metrics
        self.performance_manager.update_metrics(self.data_manager.trade_history)
        
        self.logger.info(f"Trade executed: {trade_data}")
    
    def set_live_mode(self, is_live: bool) -> None:
        """
        Set live trading mode.
        
        Args:
            is_live: Whether strategy is in live trading mode
        """
        self.is_live = is_live
        self.logger.info(f"Strategy {self.strategy_id} live mode set to {is_live}")
    
    def set_backtest_mode(self, is_backtest: bool) -> None:
        """
        Set backtest mode.
        
        Args:
            is_backtest: Whether strategy is in backtest mode
        """
        self.is_backtest = is_backtest
        self.logger.info(f"Strategy {self.strategy_id} backtest mode set to {is_backtest}")
    
    def get_strategy_id(self) -> str:
        """
        Get strategy ID.
        
        Returns:
            Strategy ID
        """
        return self.strategy_id
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters.
        
        Returns:
            Strategy parameters
        """
        return self.parameters.copy()
    
    def get_symbols(self) -> List[str]:
        """
        Get list of symbols the strategy is tracking.
        
        Returns:
            List of symbols
        """
        return self.data_manager.get_symbols()
    
    def get_frequency(self) -> str:
        """
        Get the data frequency for the strategy.
        
        Returns:
            Data frequency (e.g., '1d', '1h', '5m')
        """
        return self.parameters.get("frequency", "1d")
    
    def get_strategy_state(self) -> StrategyState:
        """
        Get current strategy state.
        
        Returns:
            Strategy state
        """
        return StrategyState(
            strategy_id=self.strategy_id,
            is_running=self.is_running,
            is_live_mode=self.is_live,
            last_update=self.last_update,
            current_positions=self.current_positions.copy(),
            performance_metrics=self.performance_manager.get_metrics(),
            parameters=self.parameters.copy(),
            metadata={}
        )
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        return self.performance_manager.get_metrics()
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dict: Current risk metrics
        """
        return self.risk_manager.get_risk_metrics()
    
    def get_risk_params(self) -> Dict[str, Any]:
        """
        Get current risk parameters.
        
        Returns:
            Dict: Current risk parameters
        """
        return self.risk_manager.get_risk_params()
    
    def update_risk_param(self, param_name: str, param_value: Any):
        """
        Update a risk parameter.
        
        Args:
            param_name: Name of the parameter to update
            param_value: New value for the parameter
        """
        self.risk_manager.update_risk_param(param_name, param_value)
    
    def reset_daily_risk(self, portfolio_value: float):
        """
        Reset daily risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
        """
        self.risk_manager.reset_daily_risk(portfolio_value)
    
    def get_signal_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get signal history.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            Signal history
        """
        return self.data_manager.get_signal_history(limit)
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            Trade history
        """
        return self.data_manager.get_trade_history(limit)
    
    def get_data(self, symbol: str, limit: Optional[int] = None):
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Maximum number of rows to return
            
        Returns:
            Historical data
        """
        return self.data_manager.get_data(symbol, limit)
    
    def update_parameter(self, name: str, value: Any) -> None:
        """
        Update a strategy parameter.
        
        Args:
            name: Parameter name
            value: New parameter value
        """
        self.parameters[name] = value
        self.logger.info(f"Updated parameter {name} to {value}")
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.is_running = False
        self.is_live = False
        self.is_backtest = False
        self.current_positions = {}
        
        # Reset managers
        self.data_manager.reset()
        self.performance_manager.reset()
        
        self.last_update = datetime.now()
        self.logger.info(f"Strategy {self.strategy_id} reset")
    
    def save_state(self, filepath: str) -> None:
        """
        Save strategy state to file.
        
        Args:
            filepath: Path to save state
        """
        state = {
            "strategy_id": self.strategy_id,
            "parameters": self.parameters,
            "current_positions": self.current_positions,
            "performance_metrics": self.performance_manager.get_metrics(),
            "trade_history": self.data_manager.get_trade_history(),
            "signal_history": self.data_manager.get_signal_history(),
            "last_update": self.last_update
        }
        
        # In a real implementation, this would save to a file
        # For now, just log
        self.logger.info(f"Saving strategy state to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """
        Load strategy state from file.
        
        Args:
            filepath: Path to load state from
        """
        # In a real implementation, this would load from a file
        # For now, just log
        self.logger.info(f"Loading strategy state from {filepath}")
    
    def validate_parameters(self) -> List[str]:
        """
        Validate strategy parameters.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for required parameters
        required_params = self.config.get("required_parameters", [])
        for param in required_params:
            if param not in self.parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Check parameter ranges
        param_ranges = self.config.get("parameter_ranges", {})
        for param, (min_val, max_val) in param_ranges.items():
            if param in self.parameters:
                value = self.parameters[param]
                if not (min_val <= value <= max_val):
                    errors.append(f"Parameter {param} value {value} outside range [{min_val}, {max_val}]")
        
        return errors