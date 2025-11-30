"""
Risk management engine module for the quantitative trading framework.

This module provides comprehensive risk management functionality including
position limits, drawdown controls, and real-time risk monitoring.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ...utils.config_loader import get_config
from ..event_bus import get_event_bus, EventType, Event
from ..common import Order, Position, OrderStatus, OrderSide

logger = logging.getLogger(__name__)


class RiskLimitType(Enum):
    """Enumeration of risk limit types."""
    MAX_POSITION_SIZE = "max_position_size"
    MAX_POSITION_VALUE = "max_position_value"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_LEVERAGE = "max_leverage"
    MAX_CONCENTRATION = "max_concentration"
    MAX_CORRELATION = "max_correlation"
    MAX_ORDERS_PER_DAY = "max_orders_per_day"
    MAX_ORDER_SIZE = "max_order_size"
    MIN_ACCOUNT_BALANCE = "min_account_balance"


@dataclass
class RiskLimit:
    """
    Risk limit configuration.
    
    Attributes:
        limit_type: Type of risk limit
        limit_value: Limit value
        time_window: Time window for the limit (if applicable)
        action: Action to take when limit is breached
        enabled: Whether the limit is enabled
        description: Description of the limit
    """
    limit_type: RiskLimitType
    limit_value: float
    time_window: Optional[timedelta] = None
    action: str = "warn"  # 'warn', 'reduce', 'stop', 'liquidate'
    enabled: bool = True
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "limit_type": self.limit_type.value,
            "limit_value": self.limit_value,
            "time_window": self.time_window.total_seconds() if self.time_window else None,
            "action": self.action,
            "enabled": self.enabled,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskLimit':
        """Create from dictionary."""
        time_window = None
        if data.get("time_window"):
            time_window = timedelta(seconds=data["time_window"])
        
        return cls(
            limit_type=RiskLimitType(data["limit_type"]),
            limit_value=data["limit_value"],
            time_window=time_window,
            action=data.get("action", "warn"),
            enabled=data.get("enabled", True),
            description=data.get("description", "")
        )


@dataclass
class RiskCheckResult:
    """
    Result of a risk check.
    
    Attributes:
        passed: Whether the check passed
        limit_type: Type of limit that was checked
        current_value: Current value
        limit_value: Limit value
        message: Descriptive message
        action: Recommended action
        data: Additional data
    """
    passed: bool
    limit_type: RiskLimitType
    current_value: float
    limit_value: float
    message: str
    action: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "limit_type": self.limit_type.value,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "message": self.message,
            "action": self.action,
            "data": self.data
        }


@dataclass
class RiskMetrics:
    """
    Risk metrics for the portfolio.
    
    Attributes:
        portfolio_value: Total portfolio value
        cash: Cash balance
        total_position_value: Total value of all positions
        total_pnl: Total PnL
        daily_pnl: Daily PnL
        max_drawdown: Maximum drawdown
        current_drawdown: Current drawdown
        leverage: Current leverage
        position_count: Number of positions
        largest_position: Largest position by value
        concentration_ratio: Concentration ratio
        correlation_risk: Correlation risk score
        order_count_today: Number of orders placed today
        last_updated: Last update timestamp
    """
    portfolio_value: float
    cash: float
    total_position_value: float
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    current_drawdown: float
    leverage: float
    position_count: int
    largest_position: float
    concentration_ratio: float
    correlation_risk: float
    order_count_today: int
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "total_position_value": self.total_position_value,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "leverage": self.leverage,
            "position_count": self.position_count,
            "largest_position": self.largest_position,
            "concentration_ratio": self.concentration_ratio,
            "correlation_risk": self.correlation_risk,
            "order_count_today": self.order_count_today,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskMetrics':
        """Create from dictionary."""
        return cls(
            portfolio_value=data["portfolio_value"],
            cash=data["cash"],
            total_position_value=data["total_position_value"],
            total_pnl=data["total_pnl"],
            daily_pnl=data["daily_pnl"],
            max_drawdown=data["max_drawdown"],
            current_drawdown=data["current_drawdown"],
            leverage=data["leverage"],
            position_count=data["position_count"],
            largest_position=data["largest_position"],
            concentration_ratio=data["concentration_ratio"],
            correlation_risk=data["correlation_risk"],
            order_count_today=data["order_count_today"],
            last_updated=data["last_updated"]
        )


class RiskEngine:
    """
    Risk management engine.
    
    This class provides comprehensive risk management functionality including
    position limits, drawdown controls, and real-time risk monitoring.
    """
    
    def __init__(self, risk_config: Dict[str, Any]):
        """
        Initialize the risk engine.
        
        Args:
            risk_config: Risk management configuration
        """
        self.config = risk_config
        self.event_bus = get_event_bus()
        
        # Risk limits
        self.risk_limits: Dict[RiskLimitType, RiskLimit] = {}
        self._initialize_risk_limits()
        
        # Risk metrics history
        self.risk_metrics_history: List[RiskMetrics] = []
        
        # Order history for tracking daily limits
        self.order_history: List[Dict[str, Any]] = []
        
        # Portfolio history for drawdown calculation
        self.portfolio_history: List[Tuple[datetime, float]] = []
        
        # Correlation matrix cache
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_correlation_update: Optional[datetime] = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _initialize_risk_limits(self) -> None:
        """Initialize risk limits from configuration."""
        limits_config = self.config.get("limits", {})
        
        for limit_name, limit_config in limits_config.items():
            try:
                limit_type = RiskLimitType(limit_name)
                risk_limit = RiskLimit.from_dict(limit_config)
                self.risk_limits[limit_type] = risk_limit
                self.logger.info(f"Initialized risk limit: {limit_name} = {risk_limit.limit_value}")
            except ValueError:
                self.logger.warning(f"Unknown risk limit type: {limit_name}")
    
    def check_risk_limits(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check all risk limits against the current portfolio state.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with risk check results
        """
        # Calculate current risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_state)
        
        # Check each enabled risk limit
        results = []
        for limit_type, risk_limit in self.risk_limits.items():
            if not risk_limit.enabled:
                continue
            
            result = self._check_risk_limit(limit_type, risk_limit, risk_metrics)
            results.append(result)
            
            # Publish risk event if limit is breached
            if not result.passed:
                event = Event(
                    event_type=EventType.RISK_LIMIT_BREACH,
                    timestamp=datetime.now().timestamp(),
                    data=result.to_dict(),
                    source="risk_engine"
                )
                self.event_bus.publish(event)
        
        # Store risk metrics in history
        self.risk_metrics_history.append(risk_metrics)
        
        # Keep only last 1000 entries
        if len(self.risk_metrics_history) > 1000:
            self.risk_metrics_history = self.risk_metrics_history[-1000:]
        
        # Update portfolio history for drawdown calculation
        self.portfolio_history.append((risk_metrics.last_updated, risk_metrics.portfolio_value))
        
        # Keep only last 1000 entries
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
        
        # Return overall result
        passed = all(result.passed for result in results)
        
        return {
            "passed": passed,
            "metrics": risk_metrics.to_dict(),
            "checks": [result.to_dict() for result in results],
            "timestamp": datetime.now()
        }
    
    def check_order_risk(self, order: Order, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an order passes risk checks.
        
        Args:
            order: Order to check
            portfolio_state: Current portfolio state
            
        Returns:
            Dictionary with order risk check result
        """
        # Calculate current risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_state)
        
        # Simulate portfolio state after order execution
        simulated_state = self._simulate_order_execution(order, portfolio_state)
        simulated_metrics = self._calculate_risk_metrics(simulated_state)
        
        # Check relevant risk limits
        results = []
        
        # Check order size limit
        if RiskLimitType.MAX_ORDER_SIZE in self.risk_limits:
            limit = self.risk_limits[RiskLimitType.MAX_ORDER_SIZE]
            order_value = order.quantity * (order.price or 0)
            
            result = RiskCheckResult(
                passed=order_value <= limit.limit_value,
                limit_type=RiskLimitType.MAX_ORDER_SIZE,
                current_value=order_value,
                limit_value=limit.limit_value,
                message=f"Order value {order_value:.2f} exceeds limit {limit.limit_value:.2f}" if order_value > limit.limit_value else f"Order value {order_value:.2f} within limit {limit.limit_value:.2f}",
                action=limit.action,
                data={"order_id": order.order_id, "symbol": order.symbol}
            )
            results.append(result)
        
        # Check position size limit
        if RiskLimitType.MAX_POSITION_SIZE in self.risk_limits:
            limit = self.risk_limits[RiskLimitType.MAX_POSITION_SIZE]
            
            # Get current position
            current_position = 0.0
            if order.symbol in portfolio_state.get("positions", {}):
                position = portfolio_state["positions"][order.symbol]
                current_position = position["quantity"]
            
            # Calculate new position size
            new_position = current_position
            if order.side == OrderSide.BUY:
                new_position += order.quantity
            else:  # SELL
                new_position -= order.quantity
            
            result = RiskCheckResult(
                passed=abs(new_position) <= limit.limit_value,
                limit_type=RiskLimitType.MAX_POSITION_SIZE,
                current_value=abs(new_position),
                limit_value=limit.limit_value,
                message=f"Position size {abs(new_position):.2f} exceeds limit {limit.limit_value:.2f}" if abs(new_position) > limit.limit_value else f"Position size {abs(new_position):.2f} within limit {limit.limit_value:.2f}",
                action=limit.action,
                data={"order_id": order.order_id, "symbol": order.symbol, "current_position": current_position, "new_position": new_position}
            )
            results.append(result)
        
        # Check position value limit
        if RiskLimitType.MAX_POSITION_VALUE in self.risk_limits:
            limit = self.risk_limits[RiskLimitType.MAX_POSITION_VALUE]
            
            # Get current position value
            current_value = 0.0
            if order.symbol in portfolio_state.get("positions", {}):
                position = portfolio_state["positions"][order.symbol]
                current_value = position["market_value"]
            
            # Calculate new position value
            order_value = order.quantity * (order.price or 0)
            new_value = current_value
            
            if order.side == OrderSide.BUY:
                new_value += order_value
            else:  # SELL
                new_value -= order_value
            
            result = RiskCheckResult(
                passed=abs(new_value) <= limit.limit_value,
                limit_type=RiskLimitType.MAX_POSITION_VALUE,
                current_value=abs(new_value),
                limit_value=limit.limit_value,
                message=f"Position value {abs(new_value):.2f} exceeds limit {limit.limit_value:.2f}" if abs(new_value) > limit.limit_value else f"Position value {abs(new_value):.2f} within limit {limit.limit_value:.2f}",
                action=limit.action,
                data={"order_id": order.order_id, "symbol": order.symbol, "current_value": current_value, "new_value": new_value}
            )
            results.append(result)
        
        # Check leverage limit
        if RiskLimitType.MAX_LEVERAGE in self.risk_limits:
            limit = self.risk_limits[RiskLimitType.MAX_LEVERAGE]
            
            result = RiskCheckResult(
                passed=simulated_metrics.leverage <= limit.limit_value,
                limit_type=RiskLimitType.MAX_LEVERAGE,
                current_value=simulated_metrics.leverage,
                limit_value=limit.limit_value,
                message=f"Leverage {simulated_metrics.leverage:.2f} exceeds limit {limit.limit_value:.2f}" if simulated_metrics.leverage > limit.limit_value else f"Leverage {simulated_metrics.leverage:.2f} within limit {limit.limit_value:.2f}",
                action=limit.action,
                data={"order_id": order.order_id, "current_leverage": risk_metrics.leverage, "new_leverage": simulated_metrics.leverage}
            )
            results.append(result)
        
        # Check daily order count limit
        if RiskLimitType.MAX_ORDERS_PER_DAY in self.risk_limits:
            limit = self.risk_limits[RiskLimitType.MAX_ORDERS_PER_DAY]
            
            # Count orders placed today
            today = datetime.now().date()
            today_orders = sum(1 for order in self.order_history if order["timestamp"].date() == today)
            
            result = RiskCheckResult(
                passed=today_orders < limit.limit_value,
                limit_type=RiskLimitType.MAX_ORDERS_PER_DAY,
                current_value=today_orders,
                limit_value=limit.limit_value,
                message=f"Daily order count {today_orders} exceeds limit {limit.limit_value}" if today_orders >= limit.limit_value else f"Daily order count {today_orders} within limit {limit.limit_value}",
                action=limit.action,
                data={"order_id": order.order_id, "today_orders": today_orders}
            )
            results.append(result)
        
        # Add order to history
        self.order_history.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": order.price,
            "timestamp": datetime.now()
        })
        
        # Keep only last 1000 entries
        if len(self.order_history) > 1000:
            self.order_history = self.order_history[-1000:]
        
        # Return overall result
        passed = all(result.passed for result in results)
        
        return {
            "passed": passed,
            "order_id": order.order_id,
            "checks": [result.to_dict() for result in results],
            "timestamp": datetime.now()
        }
    
    def _check_risk_limit(self, limit_type: RiskLimitType, risk_limit: RiskLimit, risk_metrics: RiskMetrics) -> RiskCheckResult:
        """
        Check a specific risk limit.
        
        Args:
            limit_type: Type of risk limit
            risk_limit: Risk limit configuration
            risk_metrics: Current risk metrics
            
        Returns:
            Risk check result
        """
        current_value = 0.0
        passed = True
        message = ""
        
        if limit_type == RiskLimitType.MAX_POSITION_SIZE:
            # This is checked at order level, not portfolio level
            passed = True
            message = "Position size limit checked at order level"
        elif limit_type == RiskLimitType.MAX_POSITION_VALUE:
            # This is checked at order level, not portfolio level
            passed = True
            message = "Position value limit checked at order level"
        elif limit_type == RiskLimitType.MAX_DAILY_LOSS:
            current_value = risk_metrics.daily_pnl
            passed = current_value >= -risk_limit.limit_value
            message = f"Daily loss {abs(current_value):.2f} exceeds limit {risk_limit.limit_value:.2f}" if not passed else f"Daily loss {abs(current_value):.2f} within limit {risk_limit.limit_value:.2f}"
        elif limit_type == RiskLimitType.MAX_DRAWDOWN:
            current_value = risk_metrics.current_drawdown
            passed = current_value <= risk_limit.limit_value
            message = f"Current drawdown {current_value:.2f}% exceeds limit {risk_limit.limit_value:.2f}%" if not passed else f"Current drawdown {current_value:.2f}% within limit {risk_limit.limit_value:.2f}%"
        elif limit_type == RiskLimitType.MAX_LEVERAGE:
            current_value = risk_metrics.leverage
            passed = current_value <= risk_limit.limit_value
            message = f"Leverage {current_value:.2f} exceeds limit {risk_limit.limit_value:.2f}" if not passed else f"Leverage {current_value:.2f} within limit {risk_limit.limit_value:.2f}"
        elif limit_type == RiskLimitType.MAX_CONCENTRATION:
            current_value = risk_metrics.concentration_ratio
            passed = current_value <= risk_limit.limit_value
            message = f"Concentration ratio {current_value:.2f} exceeds limit {risk_limit.limit_value:.2f}" if not passed else f"Concentration ratio {current_value:.2f} within limit {risk_limit.limit_value:.2f}"
        elif limit_type == RiskLimitType.MAX_CORRELATION:
            current_value = risk_metrics.correlation_risk
            passed = current_value <= risk_limit.limit_value
            message = f"Correlation risk {current_value:.2f} exceeds limit {risk_limit.limit_value:.2f}" if not passed else f"Correlation risk {current_value:.2f} within limit {risk_limit.limit_value:.2f}"
        elif limit_type == RiskLimitType.MAX_ORDERS_PER_DAY:
            # This is checked at order level, not portfolio level
            passed = True
            message = "Daily order count limit checked at order level"
        elif limit_type == RiskLimitType.MAX_ORDER_SIZE:
            # This is checked at order level, not portfolio level
            passed = True
            message = "Order size limit checked at order level"
        elif limit_type == RiskLimitType.MIN_ACCOUNT_BALANCE:
            current_value = risk_metrics.cash
            passed = current_value >= risk_limit.limit_value
            message = f"Account balance {current_value:.2f} below minimum {risk_limit.limit_value:.2f}" if not passed else f"Account balance {current_value:.2f} above minimum {risk_limit.limit_value:.2f}"
        
        return RiskCheckResult(
            passed=passed,
            limit_type=limit_type,
            current_value=current_value,
            limit_value=risk_limit.limit_value,
            message=message,
            action=risk_limit.action
        )
    
    def _calculate_risk_metrics(self, portfolio_state: Dict[str, Any]) -> RiskMetrics:
        """
        Calculate risk metrics for the portfolio.
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Risk metrics
        """
        # Extract portfolio data
        cash = portfolio_state.get("cash", 0.0)
        positions = portfolio_state.get("positions", {})
        orders = portfolio_state.get("orders", {})
        
        # Calculate position values
        total_position_value = 0.0
        total_pnl = 0.0
        position_values = {}
        largest_position = 0.0
        
        for symbol, position_data in positions.items():
            position_value = position_data.get("market_value", 0.0)
            position_values[symbol] = abs(position_value)
            total_position_value += position_value
            total_pnl += position_data.get("realized_pnl", 0.0) + position_data.get("unrealized_pnl", 0.0)
            largest_position = max(largest_position, abs(position_value))
        
        # Calculate portfolio value
        portfolio_value = cash + total_position_value
        
        # Calculate daily PnL
        daily_pnl = 0.0
        if self.risk_metrics_history:
            yesterday_metrics = self.risk_metrics_history[0]
            daily_pnl = total_pnl - yesterday_metrics.total_pnl
        
        # Calculate drawdown
        max_drawdown = 0.0
        current_drawdown = 0.0
        
        if len(self.portfolio_history) > 1:
            # Find peak portfolio value
            peak_value = max(value for _, value in self.portfolio_history)
            
            # Calculate current drawdown
            if peak_value > 0:
                current_drawdown = (peak_value - portfolio_value) / peak_value * 100
            
            # Calculate maximum drawdown
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1][1]
                curr_value = self.portfolio_history[i][1]
                
                if curr_value > prev_value:
                    # New peak, reset drawdown
                    peak = curr_value
                else:
                    # Calculate drawdown from peak
                    drawdown = (peak - curr_value) / peak * 100
                    max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate leverage
        leverage = 0.0
        if portfolio_value > 0:
            leverage = total_position_value / portfolio_value
        
        # Calculate concentration ratio (largest position as percentage of total)
        concentration_ratio = 0.0
        if total_position_value > 0:
            concentration_ratio = largest_position / total_position_value
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(positions)
        
        # Count orders placed today
        today = datetime.now().date()
        order_count_today = sum(1 for order in orders.values() if order.get("timestamp", datetime.now()).date() == today)
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            cash=cash,
            total_position_value=total_position_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            leverage=leverage,
            position_count=len(positions),
            largest_position=largest_position,
            concentration_ratio=concentration_ratio,
            correlation_risk=correlation_risk,
            order_count_today=order_count_today,
            last_updated=datetime.now()
        )
    
    def _simulate_order_execution(self, order: Order, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate the portfolio state after order execution.
        
        Args:
            order: Order to simulate
            portfolio_state: Current portfolio state
            
        Returns:
            Simulated portfolio state
        """
        # Deep copy the portfolio state
        simulated_state = {
            "cash": portfolio_state.get("cash", 0.0),
            "positions": {symbol: pos.copy() for symbol, pos in portfolio_state.get("positions", {}).items()},
            "orders": portfolio_state.get("orders", {})
        }
        
        # Calculate order value
        order_value = order.quantity * (order.price or 0)
        
        # Update cash
        if order.side == OrderSide.BUY:
            simulated_state["cash"] -= order_value
        else:  # SELL
            simulated_state["cash"] += order_value
        
        # Update position
        if order.symbol in simulated_state["positions"]:
            position = simulated_state["positions"][order.symbol]
            
            if order.side == OrderSide.BUY:
                position["quantity"] += order.quantity
                # Update average price
                total_cost = (position["quantity"] - order.quantity) * position["avg_price"] + order_value
                position["avg_price"] = total_cost / position["quantity"] if position["quantity"] != 0 else 0
            else:  # SELL
                position["quantity"] -= order.quantity
                # Calculate realized PnL
                realized_pnl = order.quantity * (order.price - position["avg_price"])
                position["realized_pnl"] = position.get("realized_pnl", 0.0) + realized_pnl
                
                # If position is closed, reset avg price
                if position["quantity"] == 0:
                    position["avg_price"] = 0
            
            # Update market value
            position["market_value"] = position["quantity"] * (order.price or 0)
        else:
            # Create new position
            position = {
                "symbol": order.symbol,
                "quantity": order.quantity if order.side == OrderSide.BUY else -order.quantity,
                "avg_price": order.price or 0,
                "market_value": order_value,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "last_update": datetime.now()
            }
            
            if order.side == OrderSide.SELL:
                position["avg_price"] = order.price or 0
                position["quantity"] = -order.quantity
            
            simulated_state["positions"][order.symbol] = position
        
        return simulated_state
    
    def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """
        Calculate correlation risk score.
        
        Args:
            positions: Current positions
            
        Returns:
            Correlation risk score (0-1)
        """
        # For simplicity, return a default value
        # In a real implementation, this would calculate the correlation matrix
        # of position returns and derive a risk score
        return 0.5
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary with current risk metrics
        """
        if not self.risk_metrics_history:
            return {}
        
        return self.risk_metrics_history[-1].to_dict()
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """
        Get current risk limits.
        
        Returns:
            Dictionary with current risk limits
        """
        return {limit_type.value: limit.to_dict() for limit_type, limit in self.risk_limits.items()}
    
    def update_risk_limit(self, limit_type: RiskLimitType, limit_value: float) -> None:
        """
        Update a risk limit value.
        
        Args:
            limit_type: Type of risk limit
            limit_value: New limit value
        """
        if limit_type in self.risk_limits:
            self.risk_limits[limit_type].limit_value = limit_value
            self.logger.info(f"Updated risk limit {limit_type.value} to {limit_value}")
        else:
            self.logger.warning(f"Unknown risk limit type: {limit_type.value}")
    
    def enable_risk_limit(self, limit_type: RiskLimitType) -> None:
        """
        Enable a risk limit.
        
        Args:
            limit_type: Type of risk limit
        """
        if limit_type in self.risk_limits:
            self.risk_limits[limit_type].enabled = True
            self.logger.info(f"Enabled risk limit {limit_type.value}")
        else:
            self.logger.warning(f"Unknown risk limit type: {limit_type.value}")
    
    def disable_risk_limit(self, limit_type: RiskLimitType) -> None:
        """
        Disable a risk limit.
        
        Args:
            limit_type: Type of risk limit
        """
        if limit_type in self.risk_limits:
            self.risk_limits[limit_type].enabled = False
            self.logger.info(f"Disabled risk limit {limit_type.value}")
        else:
            self.logger.warning(f"Unknown risk limit type: {limit_type.value}")


# Global risk engine instance
_risk_engine = None


def get_risk_engine() -> RiskEngine:
    """
    Get the global risk engine instance.
    
    Returns:
        Risk engine instance
    """
    global _risk_engine
    if _risk_engine is None:
        config = get_config()
        risk_config = config.get("risk_management", {})
        _risk_engine = RiskEngine(risk_config)
    return _risk_engine


def initialize_risk_engine(risk_config: Dict[str, Any]) -> None:
    """
    Initialize the global risk engine.
    
    Args:
        risk_config: Risk management configuration
    """
    global _risk_engine
    _risk_engine = RiskEngine(risk_config)