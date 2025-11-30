"""
Execution module for the quantitative trading framework.

This module provides trade execution capabilities, including order management,
execution algorithms, broker integration, and portfolio management.
"""

from .order_manager import (
    OrderType, OrderSide, OrderStatus, TimeInForce, Order, OrderManager,
    get_order_manager, initialize_order_manager
)

from .execution_engine import (
    ExecutionMode,
    ExecutionAlgorithm,
    BrokerType,
    ExecutionResult,
    BrokerInterface,
    SimulatedBroker,
    ExecutionEngine,
    get_execution_engine,
    initialize_execution_engine
)

from .portfolio_manager import (
    PositionSide,
    Position,
    Portfolio,
    PortfolioManager,
    get_portfolio_manager,
    initialize_portfolio_manager
)

__all__ = [
    # Order management
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    "Order",
    "OrderManager",
    "get_order_manager",
    "initialize_order_manager",
    
    # Execution engine
    "ExecutionMode",
    "ExecutionAlgorithm",
    "BrokerType",
    "ExecutionResult",
    "BrokerInterface",
    "SimulatedBroker",
    "ExecutionEngine",
    "get_execution_engine",
    "initialize_execution_engine",
    
    # Portfolio management
    "PositionSide",
    "Position",
    "Portfolio",
    "PortfolioManager",
    "get_portfolio_manager",
    "initialize_portfolio_manager"
]