"""
Portfolio management for the quantitative trading framework.

This module provides portfolio management capabilities, including position tracking,
performance calculation, and risk metrics.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd

from ..utils.config_loader import get_config
from ..core.event_bus import EventBus, EventType, Event, get_event_bus
from .order_manager import Order, OrderSide, OrderStatus, get_order_manager

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Enumeration of position sides."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Position:
    """Position class representing a holding in a symbol."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float = 0.0,
        avg_price: float = 0.0,
        market_price: float = 0.0,
        currency: str = "USD",
        strategy_id: Optional[str] = None
    ):
        """
        Initialize a position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity (positive for long, negative for short)
            avg_price: Average price
            market_price: Current market price
            currency: Position currency
            strategy_id: Strategy ID
        """
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.market_price = market_price
        self.currency = currency
        self.strategy_id = strategy_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Calculate cost basis
        self.cost_basis = abs(quantity) * avg_price if quantity != 0 else 0.0
        
        # Track realized PnL
        self.realized_pnl = 0.0
        
        # Track trades
        self.trades = []
    
    def update_market_price(self, market_price: float) -> None:
        """
        Update market price.
        
        Args:
            market_price: New market price
        """
        self.market_price = market_price
        self.updated_at = datetime.now()
    
    def add_trade(
        self,
        side: OrderSide,
        quantity: float,
        price: float,
        commission: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> float:
        """
        Add a trade to the position.
        
        Args:
            side: Trade side
            quantity: Trade quantity
            price: Trade price
            commission: Trade commission
            timestamp: Trade timestamp
            
        Returns:
            Realized PnL from this trade
        """
        trade_timestamp = timestamp or datetime.now()
        
        # Create trade record
        trade = {
            "side": side,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "timestamp": trade_timestamp
        }
        
        # Calculate realized PnL
        realized_pnl = 0.0
        
        if self.quantity == 0:
            # Opening a new position
            if side == OrderSide.BUY:
                self.quantity += quantity
                self.avg_price = (self.avg_price * abs(self.quantity - quantity) + price * quantity) / self.quantity
            else:
                self.quantity -= quantity
                self.avg_price = (self.avg_price * abs(self.quantity + quantity) + price * quantity) / abs(self.quantity)
        else:
            # Adjusting or closing position
            if (self.quantity > 0 and side == OrderSide.BUY) or (self.quantity < 0 and side == OrderSide.SELL):
                # Adding to position
                if side == OrderSide.BUY:
                    self.quantity += quantity
                    self.avg_price = (self.avg_price * (self.quantity - quantity) + price * quantity) / self.quantity
                else:
                    self.quantity -= quantity
                    self.avg_price = (self.avg_price * (abs(self.quantity) - quantity) + price * quantity) / abs(self.quantity)
            else:
                # Reducing or closing position
                trade_quantity = min(quantity, abs(self.quantity))
                if self.quantity > 0:
                    realized_pnl = trade_quantity * (price - self.avg_price)
                    self.quantity -= trade_quantity
                else:
                    realized_pnl = trade_quantity * (self.avg_price - price)
                    self.quantity += trade_quantity
                
                # Adjust average price if position not fully closed
                if self.quantity != 0:
                    remaining_quantity = quantity - trade_quantity
                    if side == OrderSide.BUY:
                        self.quantity += remaining_quantity
                        self.avg_price = (self.avg_price * abs(self.quantity - remaining_quantity) + price * remaining_quantity) / abs(self.quantity)
                    else:
                        self.quantity -= remaining_quantity
                        self.avg_price = (self.avg_price * (abs(self.quantity) - remaining_quantity) + price * remaining_quantity) / abs(self.quantity)
        
        # Update cost basis
        self.cost_basis = abs(self.quantity) * self.avg_price if self.quantity != 0 else 0.0
        
        # Update realized PnL
        self.realized_pnl += realized_pnl - commission
        
        # Add trade to history
        self.trades.append(trade)
        self.updated_at = trade_timestamp
        
        return realized_pnl
    
    def get_side(self) -> PositionSide:
        """
        Get position side.
        
        Returns:
            Position side
        """
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT
    
    def get_market_value(self) -> float:
        """
        Get market value of position.
        
        Returns:
            Market value
        """
        return self.quantity * self.market_price
    
    def get_unrealized_pnl(self) -> float:
        """
        Get unrealized PnL.
        
        Returns:
            Unrealized PnL
        """
        if self.quantity == 0:
            return 0.0
        
        if self.quantity > 0:
            return self.quantity * (self.market_price - self.avg_price)
        else:
            return abs(self.quantity) * (self.avg_price - self.market_price)
    
    def get_total_pnl(self) -> float:
        """
        Get total PnL (realized + unrealized).
        
        Returns:
            Total PnL
        """
        return self.realized_pnl + self.get_unrealized_pnl()
    
    def get_pnl_percentage(self) -> float:
        """
        Get PnL as percentage of cost basis.
        
        Returns:
            PnL percentage
        """
        if self.cost_basis == 0:
            return 0.0
        
        return (self.get_total_pnl() / self.cost_basis) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary.
        
        Returns:
            Dictionary representation of position
        """
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "market_price": self.market_price,
            "currency": self.currency,
            "strategy_id": self.strategy_id,
            "side": self.get_side().value,
            "cost_basis": self.cost_basis,
            "market_value": self.get_market_value(),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.get_unrealized_pnl(),
            "total_pnl": self.get_total_pnl(),
            "pnl_percentage": self.get_pnl_percentage(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "trades_count": len(self.trades)
        }


class Portfolio:
    """Portfolio class representing a collection of positions."""
    
    def __init__(
        self,
        portfolio_id: str,
        initial_cash: float = 100000.0,
        currency: str = "USD",
        strategy_id: Optional[str] = None
    ):
        """
        Initialize a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            initial_cash: Initial cash amount
            currency: Portfolio currency
            strategy_id: Strategy ID
        """
        self.portfolio_id = portfolio_id
        self.initial_cash = initial_cash
        self.currency = currency
        self.strategy_id = strategy_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Positions
        self.positions = {}  # symbol -> Position
        
        # Cash
        self.cash = initial_cash
        
        # Performance tracking
        self.high_watermark = initial_cash
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Transaction history
        self.transactions = []
    
    def update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        commission: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update a position in the portfolio.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            quantity: Trade quantity
            price: Trade price
            commission: Trade commission
            timestamp: Trade timestamp
            
        Returns:
            Dictionary with update results
        """
        trade_timestamp = timestamp or datetime.now()
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                currency=self.currency,
                strategy_id=self.strategy_id
            )
        
        position = self.positions[symbol]
        
        # Calculate trade value
        trade_value = quantity * price
        if side == OrderSide.BUY:
            trade_value = -trade_value  # Cash outflow
        else:
            trade_value = trade_value  # Cash inflow
        
        # Update position
        realized_pnl = position.add_trade(side, quantity, price, commission, trade_timestamp)
        
        # Update cash
        self.cash += trade_value - commission
        
        # Create transaction record
        transaction = {
            "timestamp": trade_timestamp,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": price,
            "value": trade_value,
            "commission": commission,
            "realized_pnl": realized_pnl
        }
        self.transactions.append(transaction)
        
        # Update portfolio value and drawdown
        self._update_performance_metrics()
        
        # Update timestamp
        self.updated_at = trade_timestamp
        
        # Return update results
        return {
            "symbol": symbol,
            "position": position.to_dict(),
            "cash": self.cash,
            "portfolio_value": self.get_total_value(),
            "realized_pnl": realized_pnl
        }
    
    def update_market_prices(self, market_prices: Dict[str, float]) -> None:
        """
        Update market prices for positions.
        
        Args:
            market_prices: Dictionary of symbol -> price
        """
        for symbol, price in market_prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_market_price(price)
        
        # Update portfolio value and drawdown
        self._update_performance_metrics()
        self.updated_at = datetime.now()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position or None if not found
        """
        return self.positions.get(symbol)
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all positions.
        
        Returns:
            Dictionary of symbol -> Position
        """
        return self.positions.copy()
    
    def get_long_positions(self) -> Dict[str, Position]:
        """
        Get long positions.
        
        Returns:
            Dictionary of symbol -> Position
        """
        return {symbol: pos for symbol, pos in self.positions.items() if pos.get_side() == PositionSide.LONG}
    
    def get_short_positions(self) -> Dict[str, Position]:
        """
        Get short positions.
        
        Returns:
            Dictionary of symbol -> Position
        """
        return {symbol: pos for symbol, pos in self.positions.items() if pos.get_side() == PositionSide.SHORT}
    
    def get_total_value(self) -> float:
        """
        Get total portfolio value.
        
        Returns:
            Total portfolio value
        """
        total_value = self.cash
        
        for position in self.positions.values():
            total_value += position.get_market_value()
        
        return total_value
    
    def get_total_pnl(self) -> float:
        """
        Get total PnL for the portfolio.
        
        Returns:
            Total PnL
        """
        total_pnl = 0.0
        
        for position in self.positions.values():
            total_pnl += position.get_total_pnl()
        
        return total_pnl
    
    def get_pnl_percentage(self) -> float:
        """
        Get PnL as percentage of initial capital.
        
        Returns:
            PnL percentage
        """
        if self.initial_cash == 0:
            return 0.0
        
        return (self.get_total_pnl() / self.initial_cash) * 100
    
    def get_return(self) -> float:
        """
        Get portfolio return.
        
        Returns:
            Portfolio return
        """
        return (self.get_total_value() / self.initial_cash) - 1
    
    def get_return_percentage(self) -> float:
        """
        Get portfolio return as percentage.
        
        Returns:
            Portfolio return percentage
        """
        return self.get_return() * 100
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        current_value = self.get_total_value()
        
        # Update high watermark
        if current_value > self.high_watermark:
            self.high_watermark = current_value
        
        # Calculate drawdown
        self.current_drawdown = (self.high_watermark - current_value) / self.high_watermark
        
        # Update max drawdown
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            "portfolio_id": self.portfolio_id,
            "initial_cash": self.initial_cash,
            "current_cash": self.cash,
            "total_value": self.get_total_value(),
            "total_pnl": self.get_total_pnl(),
            "pnl_percentage": self.get_pnl_percentage(),
            "return": self.get_return(),
            "return_percentage": self.get_return_percentage(),
            "high_watermark": self.high_watermark,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "positions_count": len(self.positions),
            "long_positions_count": len(self.get_long_positions()),
            "short_positions_count": len(self.get_short_positions()),
            "transactions_count": len(self.transactions),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def get_position_summary(self) -> List[Dict[str, Any]]:
        """
        Get position summary.
        
        Returns:
            List of position dictionaries
        """
        return [position.to_dict() for position in self.positions.values()]
    
    def get_transaction_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get transaction history.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction dictionaries
        """
        transactions = sorted(self.transactions, key=lambda x: x["timestamp"], reverse=True)
        
        if limit:
            transactions = transactions[:limit]
        
        return transactions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert portfolio to dictionary.
        
        Returns:
            Dictionary representation of portfolio
        """
        return {
            "portfolio_id": self.portfolio_id,
            "initial_cash": self.initial_cash,
            "currency": self.currency,
            "strategy_id": self.strategy_id,
            "cash": self.cash,
            "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()},
            "performance_metrics": self.get_performance_metrics(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class PortfolioManager:
    """Portfolio manager for handling multiple portfolios."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the portfolio manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().get("portfolio", {})
        self.portfolios = {}  # portfolio_id -> Portfolio
        self.strategy_portfolios = {}  # strategy_id -> portfolio_id
        self.order_manager = get_order_manager()
        self.event_bus = get_event_bus()
        self.logger = logging.getLogger(__name__)
        
        # Subscribe to order events
        self.event_bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
    
    def create_portfolio(
        self,
        portfolio_id: str,
        initial_cash: float = 100000.0,
        currency: str = "USD",
        strategy_id: Optional[str] = None
    ) -> Portfolio:
        """
        Create a new portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            initial_cash: Initial cash amount
            currency: Portfolio currency
            strategy_id: Strategy ID
            
        Returns:
            Created portfolio
        """
        if portfolio_id in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} already exists")
        
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            initial_cash=initial_cash,
            currency=currency,
            strategy_id=strategy_id
        )
        
        self.portfolios[portfolio_id] = portfolio
        
        if strategy_id:
            self.strategy_portfolios[strategy_id] = portfolio_id
        
        self.logger.info(f"Created portfolio: {portfolio_id}")
        
        return portfolio
    
    def get_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """
        Get portfolio by ID.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio or None if not found
        """
        return self.portfolios.get(portfolio_id)
    
    def get_portfolio_by_strategy(self, strategy_id: str) -> Optional[Portfolio]:
        """
        Get portfolio by strategy ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Portfolio or None if not found
        """
        portfolio_id = self.strategy_portfolios.get(strategy_id)
        if portfolio_id:
            return self.portfolios.get(portfolio_id)
        return None
    
    def get_all_portfolios(self) -> Dict[str, Portfolio]:
        """
        Get all portfolios.
        
        Returns:
            Dictionary of portfolio_id -> Portfolio
        """
        return self.portfolios.copy()
    
    def update_market_prices(self, market_prices: Dict[str, float]) -> None:
        """
        Update market prices for all portfolios.
        
        Args:
            market_prices: Dictionary of symbol -> price
        """
        for portfolio in self.portfolios.values():
            portfolio.update_market_prices(market_prices)
    
    def _on_order_filled(self, event: Event) -> None:
        """
        Handle order filled event.
        
        Args:
            event: Order filled event
        """
        data = event.data
        order_dict = data.get("order", {})
        fill_quantity = data.get("fill_quantity", 0.0)
        fill_price = data.get("fill_price", 0.0)
        commission = data.get("commission", 0.0)
        
        # Get order
        order = self.order_manager.get_order(order_dict.get("client_order_id"))
        if not order:
            return
        
        # Get portfolio
        portfolio = self.get_portfolio_by_strategy(order.strategy_id)
        if not portfolio:
            return
        
        # Update position
        portfolio.update_position(
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission
        )
        
        self.logger.info(f"Updated portfolio {portfolio.portfolio_id} for filled order {order.client_order_id}")
    
    def get_portfolio_performance(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """
        Get portfolio performance metrics.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            Performance metrics or None if portfolio not found
        """
        portfolio = self.get_portfolio(portfolio_id)
        if portfolio:
            return portfolio.get_performance_metrics()
        return None
    
    def get_all_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all portfolios.
        
        Returns:
            Dictionary of portfolio_id -> performance metrics
        """
        return {
            portfolio_id: portfolio.get_performance_metrics()
            for portfolio_id, portfolio in self.portfolios.items()
        }
    
    def delete_portfolio(self, portfolio_id: str) -> bool:
        """
        Delete a portfolio.
        
        Args:
            portfolio_id: Portfolio ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        if portfolio_id not in self.portfolios:
            return False
        
        portfolio = self.portfolios[portfolio_id]
        
        # Remove from strategy mapping
        if portfolio.strategy_id and portfolio.strategy_id in self.strategy_portfolios:
            del self.strategy_portfolios[portfolio.strategy_id]
        
        # Delete portfolio
        del self.portfolios[portfolio_id]
        
        self.logger.info(f"Deleted portfolio: {portfolio_id}")
        
        return True


# Global portfolio manager instance
_portfolio_manager: Optional[PortfolioManager] = None


def get_portfolio_manager() -> PortfolioManager:
    """
    Get the global portfolio manager instance.
    
    Returns:
        Global portfolio manager instance
    """
    global _portfolio_manager
    if _portfolio_manager is None:
        _portfolio_manager = PortfolioManager()
    return _portfolio_manager


def initialize_portfolio_manager(config: Optional[Dict[str, Any]] = None) -> PortfolioManager:
    """
    Initialize the global portfolio manager.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized portfolio manager
    """
    global _portfolio_manager
    _portfolio_manager = PortfolioManager(config)
    return _portfolio_manager