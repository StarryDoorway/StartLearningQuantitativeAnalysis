"""
Common data structures for the quantitative trading framework.

This module provides shared data structures used across multiple components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional


class OrderStatus(Enum):
    """Enumeration of order statuses."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Enumeration of order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Enumeration of order sides."""
    BUY = "buy"
    SELL = "sell"


class PositionSide(Enum):
    """Enumeration of position sides."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """
    Order object for trading.
    
    Attributes:
        order_id: Unique order identifier
        symbol: Trading symbol
        side: Order side
        order_type: Order type
        quantity: Order quantity
        price: Order price (for limit orders)
        stop_price: Stop price (for stop orders)
        time_in_force: Time in force
        status: Order status
        filled_quantity: Filled quantity
        avg_fill_price: Average fill price
        commission: Total commission
        timestamp: Order creation timestamp
        updated_timestamp: Last update timestamp
        strategy_id: Strategy ID that created the order
        metadata: Additional order metadata
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    updated_timestamp: datetime = field(default_factory=datetime.now)
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "commission": self.commission,
            "timestamp": self.timestamp,
            "updated_timestamp": self.updated_timestamp,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create from dictionary."""
        return cls(
            order_id=data["order_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            quantity=data["quantity"],
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            time_in_force=data.get("time_in_force", "GTC"),
            status=OrderStatus(data.get("status", "pending")),
            filled_quantity=data.get("filled_quantity", 0.0),
            avg_fill_price=data.get("avg_fill_price", 0.0),
            commission=data.get("commission", 0.0),
            timestamp=data.get("timestamp", datetime.now()),
            updated_timestamp=data.get("updated_timestamp", datetime.now()),
            strategy_id=data.get("strategy_id"),
            metadata=data.get("metadata", {})
        )


@dataclass
class Position:
    """
    Position object for trading.
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "last_update": self.last_update
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            quantity=data["quantity"],
            avg_price=data["avg_price"],
            market_value=data["market_value"],
            unrealized_pnl=data["unrealized_pnl"],
            realized_pnl=data["realized_pnl"],
            last_update=data["last_update"]
        )
    
    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized PnL based on current price."""
        self.market_value = self.quantity * current_price
        if self.quantity != 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
        else:
            self.unrealized_pnl = 0.0
        self.last_update = datetime.now()