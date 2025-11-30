"""
Order management for the quantitative trading framework.

This module provides order management capabilities, including order creation,
tracking, and status updates.
"""

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd

from ...utils.config_loader import get_config
from ...core.event_bus import EventBus, EventType, Event, get_event_bus

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Enumeration of order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Enumeration of order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Enumeration of order statuses."""
    NEW = "new"
    PENDING_NEW = "pending_new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    PENDING_CANCEL = "pending_cancel"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Enumeration of time in force types."""
    GTC = "GTC"  # Good Till Canceled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day Order


class Order:
    """Order class representing a trading order."""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        exchange_order_id: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            client_order_id: Client order ID
            exchange_order_id: Exchange order ID
            exchange: Exchange name
            strategy_id: Strategy ID
            metadata: Additional metadata
        """
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.client_order_id = client_order_id or str(uuid.uuid4())
        self.exchange_order_id = exchange_order_id
        self.exchange = exchange
        self.strategy_id = strategy_id
        self.metadata = metadata or {}
        
        # Order state
        self.status = OrderStatus.NEW
        self.filled_quantity = 0.0
        self.average_fill_price = 0.0
        self.commission = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.fills = []
        
        # Validation
        self._validate()
    
    def _validate(self) -> None:
        """Validate order parameters."""
        # Validate order type and required prices
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT] and self.price is None:
            raise ValueError(f"Price is required for {self.order_type.value} orders")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"Stop price is required for {self.order_type.value} orders")
        
        # Validate quantity
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Validate price
        if self.price is not None and self.price <= 0:
            raise ValueError("Price must be positive")
        
        # Validate stop price
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("Stop price must be positive")
    
    def update_status(self, status: OrderStatus) -> None:
        """
        Update order status.
        
        Args:
            status: New order status
        """
        self.status = status
        self.updated_at = datetime.now()
    
    def add_fill(self, quantity: float, price: float, commission: float = 0.0, 
                 timestamp: Optional[datetime] = None) -> None:
        """
        Add a fill to the order.
        
        Args:
            quantity: Fill quantity
            price: Fill price
            commission: Commission for this fill
            timestamp: Fill timestamp
        """
        fill_timestamp = timestamp or datetime.now()
        
        # Create fill record
        fill = {
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "timestamp": fill_timestamp
        }
        
        self.fills.append(fill)
        self.filled_quantity += quantity
        self.commission += commission
        
        # Calculate average fill price
        total_value = sum(f["quantity"] * f["price"] for f in self.fills)
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.updated_at = datetime.now()
    
    def get_remaining_quantity(self) -> float:
        """
        Get remaining quantity to be filled.
        
        Returns:
            Remaining quantity
        """
        return self.quantity - self.filled_quantity
    
    def is_filled(self) -> bool:
        """
        Check if order is fully filled.
        
        Returns:
            True if order is fully filled
        """
        return self.status == OrderStatus.FILLED
    
    def is_partially_filled(self) -> bool:
        """
        Check if order is partially filled.
        
        Returns:
            True if order is partially filled
        """
        return self.status == OrderStatus.PARTIALLY_FILLED
    
    def is_active(self) -> bool:
        """
        Check if order is still active.
        
        Returns:
            True if order is active
        """
        return self.status in [OrderStatus.NEW, OrderStatus.PENDING_NEW, OrderStatus.PARTIALLY_FILLED]
    
    def is_done(self) -> bool:
        """
        Check if order is done (filled, canceled, rejected, or expired).
        
        Returns:
            True if order is done
        """
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order to dictionary.
        
        Returns:
            Dictionary representation of order
        """
        return {
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "exchange": self.exchange,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
            "fills": self.fills
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """
        Create order from dictionary.
        
        Args:
            data: Dictionary representation of order
            
        Returns:
            Order instance
        """
        order = cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            quantity=data["quantity"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            time_in_force=TimeInForce(data.get("time_in_force", "GTC")),
            client_order_id=data.get("client_order_id"),
            exchange_order_id=data.get("exchange_order_id"),
            exchange=data.get("exchange"),
            strategy_id=data.get("strategy_id"),
            metadata=data.get("metadata", {})
        )
        
        order.status = OrderStatus(data.get("status", "new"))
        order.filled_quantity = data.get("filled_quantity", 0.0)
        order.average_fill_price = data.get("average_fill_price", 0.0)
        order.commission = data.get("commission", 0.0)
        order.created_at = datetime.fromisoformat(data["created_at"])
        order.updated_at = datetime.fromisoformat(data["updated_at"])
        order.fills = data.get("fills", [])
        
        return order


class OrderManager:
    """Order manager for handling order lifecycle."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the order manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or get_config().get("execution", {})
        self.orders = {}  # client_order_id -> Order
        self.exchange_order_ids = {}  # exchange_order_id -> client_order_id
        self.strategy_orders = {}  # strategy_id -> list of client_order_ids
        self.symbol_orders = {}  # symbol -> list of client_order_ids
        self.event_bus = get_event_bus()
        self.logger = logging.getLogger(__name__)
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        exchange: Optional[str] = None,
        strategy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            client_order_id: Client order ID
            exchange: Exchange name
            strategy_id: Strategy ID
            metadata: Additional metadata
            
        Returns:
            Created order
        """
        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            exchange=exchange,
            strategy_id=strategy_id,
            metadata=metadata
        )
        
        # Store order
        self.orders[order.client_order_id] = order
        
        # Update mappings
        if order.strategy_id:
            if order.strategy_id not in self.strategy_orders:
                self.strategy_orders[order.strategy_id] = []
            self.strategy_orders[order.strategy_id].append(order.client_order_id)
        
        if order.symbol not in self.symbol_orders:
            self.symbol_orders[order.symbol] = []
        self.symbol_orders[order.symbol].append(order.client_order_id)
        
        # Emit event
        self.event_bus.publish(Event(
            event_type=EventType.ORDER_CREATED,
            data={"order": order.to_dict()}
        ))
        
        self.logger.info(f"Created order: {order.client_order_id}")
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by client order ID.
        
        Args:
            order_id: Client order ID
            
        Returns:
            Order or None if not found
        """
        return self.orders.get(order_id)
    
    def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[Order]:
        """
        Get order by exchange order ID.
        
        Args:
            exchange_order_id: Exchange order ID
            
        Returns:
            Order or None if not found
        """
        client_order_id = self.exchange_order_ids.get(exchange_order_id)
        if client_order_id:
            return self.orders.get(client_order_id)
        return None
    
    def update_order_status(self, order_id: str, status: OrderStatus, 
                           exchange_order_id: Optional[str] = None) -> bool:
        """
        Update order status.
        
        Args:
            order_id: Client order ID
            status: New order status
            exchange_order_id: Exchange order ID
            
        Returns:
            True if update successful, False otherwise
        """
        order = self.get_order(order_id)
        if not order:
            self.logger.error(f"Order not found: {order_id}")
            return False
        
        # Update status
        order.update_status(status)
        
        # Update exchange order ID if provided
        if exchange_order_id:
            order.exchange_order_id = exchange_order_id
            self.exchange_order_ids[exchange_order_id] = order_id
        
        # Emit event
        self.event_bus.publish(Event(
            event_type=EventType.ORDER_UPDATED,
            data={"order": order.to_dict()}
        ))
        
        self.logger.info(f"Updated order status: {order_id} -> {status.value}")
        
        return True
    
    def add_fill(self, order_id: str, quantity: float, price: float, 
                commission: float = 0.0, timestamp: Optional[datetime] = None) -> bool:
        """
        Add a fill to an order.
        
        Args:
            order_id: Client order ID
            quantity: Fill quantity
            price: Fill price
            commission: Commission for this fill
            timestamp: Fill timestamp
            
        Returns:
            True if fill added successfully, False otherwise
        """
        order = self.get_order(order_id)
        if not order:
            self.logger.error(f"Order not found: {order_id}")
            return False
        
        # Add fill
        order.add_fill(quantity, price, commission, timestamp)
        
        # Emit event
        self.event_bus.publish(Event(
            event_type=EventType.ORDER_FILLED,
            data={
                "order": order.to_dict(),
                "fill_quantity": quantity,
                "fill_price": price,
                "commission": commission
            }
        ))
        
        self.logger.info(f"Added fill to order: {order_id}, qty: {quantity}, price: {price}")
        
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Client order ID
            
        Returns:
            True if cancellation successful, False otherwise
        """
        order = self.get_order(order_id)
        if not order:
            self.logger.error(f"Order not found: {order_id}")
            return False
        
        if not order.is_active():
            self.logger.warning(f"Cannot cancel non-active order: {order_id}")
            return False
        
        # Update status
        order.update_status(OrderStatus.CANCELED)
        
        # Emit event
        self.event_bus.publish(Event(
            event_type=EventType.ORDER_CANCELED,
            data={"order": order.to_dict()}
        ))
        
        self.logger.info(f"Canceled order: {order_id}")
        
        return True
    
    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """
        Get orders by strategy ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            List of orders
        """
        order_ids = self.strategy_orders.get(strategy_id, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """
        Get orders by symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of orders
        """
        order_ids = self.symbol_orders.get(symbol, [])
        return [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get active orders.
        
        Args:
            symbol: Trading symbol (if None, returns all active orders)
            
        Returns:
            List of active orders
        """
        if symbol:
            order_ids = self.symbol_orders.get(symbol, [])
            orders = [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
        else:
            orders = list(self.orders.values())
        
        return [order for order in orders if order.is_active()]
    
    def get_filled_orders(self, symbol: Optional[str] = None, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Order]:
        """
        Get filled orders.
        
        Args:
            symbol: Trading symbol (if None, returns all filled orders)
            start_time: Start time for filtering
            end_time: End time for filtering
            
        Returns:
            List of filled orders
        """
        if symbol:
            order_ids = self.symbol_orders.get(symbol, [])
            orders = [self.orders[order_id] for order_id in order_ids if order_id in self.orders]
        else:
            orders = list(self.orders.values())
        
        filled_orders = [order for order in orders if order.is_filled()]
        
        # Filter by time
        if start_time:
            filled_orders = [order for order in filled_orders if order.updated_at >= start_time]
        
        if end_time:
            filled_orders = [order for order in filled_orders if order.updated_at <= end_time]
        
        return filled_orders
    
    def get_order_statistics(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get order statistics.
        
        Args:
            strategy_id: Strategy ID (if None, returns statistics for all orders)
            
        Returns:
            Dictionary with order statistics
        """
        if strategy_id:
            orders = self.get_orders_by_strategy(strategy_id)
        else:
            orders = list(self.orders.values())
        
        # Count orders by status
        status_counts = {}
        for status in OrderStatus:
            status_counts[status.value] = sum(1 for order in orders if order.status == status)
        
        # Calculate total filled quantity and commission
        total_filled_quantity = sum(order.filled_quantity for order in orders if order.is_filled())
        total_commission = sum(order.commission for order in orders if order.is_filled())
        
        return {
            "total_orders": len(orders),
            "status_counts": status_counts,
            "total_filled_quantity": total_filled_quantity,
            "total_commission": total_commission
        }


# Global order manager instance
_order_manager: Optional[OrderManager] = None


def get_order_manager() -> OrderManager:
    """
    Get the global order manager instance.
    
    Returns:
        Order manager instance
    """
    global _order_manager
    if _order_manager is None:
        _order_manager = OrderManager()
    return _order_manager


def initialize_order_manager(config: Optional[Dict[str, Any]] = None) -> OrderManager:
    """
    Initialize the global order manager instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Order manager instance
    """
    global _order_manager
    _order_manager = OrderManager(config)
    return _order_manager


__all__ = [
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    "Order",
    "OrderManager",
    "get_order_manager",
    "initialize_order_manager"
]