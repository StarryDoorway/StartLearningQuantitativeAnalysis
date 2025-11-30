"""
Strategy risk manager module for the quantitative trading framework.

This module provides risk management functionality for trading strategies.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from ..base.signal_types import Signal, SignalType, SignalStrength


class StrategyRiskManager:
    """
    Risk manager for trading strategies.
    
    This class is responsible for:
    - Pre-trade risk checks
    - Position sizing based on risk parameters
    - Portfolio risk monitoring
    - Risk limit enforcement
    """
    
    def __init__(self, strategy_id: str, risk_params: Dict[str, Any] = None):
        """
        Initialize the strategy risk manager.
        
        Args:
            strategy_id: Unique identifier for the strategy
            risk_params: Risk management parameters
        """
        self.logger = logging.getLogger(f"{__name__}.{strategy_id}")
        self.strategy_id = strategy_id
        
        # Default risk parameters
        self.risk_params = {
            "max_position_size": 10000.0,  # Maximum position size in currency units
            "max_portfolio_risk": 0.02,     # Maximum portfolio risk (2%)
            "max_daily_loss": 0.05,         # Maximum daily loss (5%)
            "max_drawdown": 0.20,           # Maximum drawdown (20%)
            "position_size_method": "fixed", # Position sizing method: "fixed", "volatility", "kelly"
            "volatility_lookback": 20,      # Lookback period for volatility calculation
            "risk_free_rate": 0.02,        # Risk-free rate for Sharpe ratio
            "confidence_level": 0.95,       # Confidence level for VaR
        }
        
        # Update with provided parameters
        if risk_params:
            self.risk_params.update(risk_params)
        
        # Risk state
        self.daily_pnl = 0.0
        self.current_drawdown = 0.0
        self.peak_portfolio_value = 0.0
        self.daily_start_value = 0.0
        self.last_risk_check = datetime.now()
        
        # Risk violations
        self.risk_violations = []
    
    def check_signal_risk(self, signal: Signal, current_price: float, 
                         current_positions: Dict[str, float] = None,
                         portfolio_value: float = None) -> Tuple[bool, str]:
        """
        Check if a signal passes risk checks.
        
        Args:
            signal: Trading signal to check
            current_price: Current market price
            current_positions: Current positions by symbol
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple[bool, str]: (is_allowed, reason_if_denied)
        """
        # Initialize default values
        if current_positions is None:
            current_positions = {}
        if portfolio_value is None:
            portfolio_value = self.risk_params.get("max_position_size", 10000.0)
        
        # Check if signal is valid
        if signal.signal_type not in [SignalType.BUY, SignalType.SELL]:
            return True, "HOLD signal always passes"
        
        # Calculate position size
        quantity = self.calculate_position_size(signal, current_price, portfolio_value)
        position_value = quantity * current_price
        
        # Check 1: Maximum position size
        if position_value > self.risk_params["max_position_size"]:
            return False, f"Position size {position_value:.2f} exceeds maximum {self.risk_params['max_position_size']:.2f}"
        
        # Check 2: Maximum portfolio risk
        portfolio_risk = position_value / portfolio_value
        if portfolio_risk > self.risk_params["max_portfolio_risk"]:
            return False, f"Portfolio risk {portfolio_risk:.2%} exceeds maximum {self.risk_params['max_portfolio_risk']:.2%}"
        
        # Check 3: Maximum daily loss
        if self.daily_pnl < -self.risk_params["max_daily_loss"] * portfolio_value:
            return False, f"Daily loss {self.daily_pnl:.2f} exceeds maximum {self.risk_params['max_daily_loss'] * portfolio_value:.2f}"
        
        # Check 4: Maximum drawdown
        if self.current_drawdown > self.risk_params["max_drawdown"]:
            return False, f"Current drawdown {self.current_drawdown:.2%} exceeds maximum {self.risk_params['max_drawdown']:.2%}"
        
        # Check 5: Concentration risk (if current positions are provided)
        if current_positions:
            current_position_value = current_positions.get(signal.symbol, 0.0) * current_price
            new_position_value = current_position_value + position_value
            
            # If selling, position value decreases
            if signal.signal_type == SignalType.SELL:
                new_position_value = max(0, current_position_value - position_value)
            
            # Check if new position would exceed 50% of portfolio
            if new_position_value > 0.5 * portfolio_value:
                return False, f"Concentration risk: {signal.symbol} position would be {new_position_value/portfolio_value:.2%} of portfolio"
        
        # All checks passed
        return True, "Signal passes all risk checks"
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              portfolio_value: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            float: Position size in units of the asset
        """
        method = self.risk_params["position_size_method"]
        
        if method == "fixed":
            return self._fixed_position_size(portfolio_value, current_price)
        elif method == "volatility":
            return self._volatility_position_size(signal, current_price, portfolio_value)
        elif method == "kelly":
            return self._kelly_position_size(signal, current_price, portfolio_value)
        else:
            self.logger.warning(f"Unknown position sizing method: {method}, using fixed")
            return self._fixed_position_size(portfolio_value, current_price)
    
    def _fixed_position_size(self, portfolio_value: float, current_price: float) -> float:
        """
        Calculate fixed position size.
        
        Args:
            portfolio_value: Current portfolio value
            current_price: Current market price
            
        Returns:
            float: Position size in units of the asset
        """
        # Use a fixed percentage of portfolio
        position_value = portfolio_value * self.risk_params["max_portfolio_risk"]
        quantity = position_value / current_price
        
        return round(quantity, 6)
    
    def _volatility_position_size(self, signal: Signal, current_price: float, 
                                portfolio_value: float) -> float:
        """
        Calculate position size based on volatility.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            float: Position size in units of the asset
        """
        # This would require historical price data to calculate volatility
        # For now, use a simplified approach based on signal strength
        
        # Adjust position size based on signal strength
        strength_multiplier = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MEDIUM: 0.75,
            SignalStrength.STRONG: 1.0
        }
        
        # Base position size
        base_position = self._fixed_position_size(portfolio_value, current_price)
        
        # Adjust by signal strength
        adjusted_position = base_position * strength_multiplier.get(signal.strength, 0.75)
        
        return round(adjusted_position, 6)
    
    def _kelly_position_size(self, signal: Signal, current_price: float, 
                           portfolio_value: float) -> float:
        """
        Calculate position size using Kelly criterion.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_value: Current portfolio value
            
        Returns:
            float: Position size in units of the asset
        """
        # Kelly criterion: f* = (p*b - q) / b
        # where p = probability of winning, b = net odds, q = probability of losing
        
        # This would require historical win rate and average win/loss ratios
        # For now, use a simplified approach based on signal strength
        
        # Estimate win probability based on signal strength
        win_prob = {
            SignalStrength.WEAK: 0.55,
            SignalStrength.MEDIUM: 0.6,
            SignalStrength.STRONG: 0.65
        }
        
        # Estimate win/loss ratio (average win / average loss)
        win_loss_ratio = 1.5  # Assumed 1.5:1 win/loss ratio
        
        p = win_prob.get(signal.strength, 0.6)
        q = 1 - p
        b = win_loss_ratio
        
        # Calculate Kelly fraction
        kelly_fraction = (p * b - q) / b
        
        # Apply a safety factor (half Kelly)
        kelly_fraction *= 0.5
        
        # Calculate position size
        position_value = portfolio_value * kelly_fraction
        quantity = position_value / current_price
        
        return round(quantity, 6)
    
    def update_risk_state(self, portfolio_value: float, daily_pnl: float = None):
        """
        Update risk state based on current portfolio value.
        
        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily P&L (calculated if None)
        """
        # Update daily P&L
        if daily_pnl is not None:
            self.daily_pnl = daily_pnl
        
        # Update peak portfolio value and drawdown
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        
        # Check for risk violations
        self._check_risk_violations(portfolio_value)
        
        # Update last risk check time
        self.last_risk_check = datetime.now()
    
    def _check_risk_violations(self, portfolio_value: float):
        """
        Check for risk violations and record them.
        
        Args:
            portfolio_value: Current portfolio value
        """
        violations = []
        
        # Check daily loss
        max_daily_loss = self.risk_params["max_daily_loss"] * portfolio_value
        if self.daily_pnl < -max_daily_loss:
            violations.append({
                "type": "daily_loss",
                "value": self.daily_pnl,
                "limit": -max_daily_loss,
                "timestamp": datetime.now()
            })
        
        # Check drawdown
        max_drawdown = self.risk_params["max_drawdown"]
        if self.current_drawdown > max_drawdown:
            violations.append({
                "type": "drawdown",
                "value": self.current_drawdown,
                "limit": max_drawdown,
                "timestamp": datetime.now()
            })
        
        # Add new violations
        self.risk_violations.extend(violations)
        
        # Log violations
        for violation in violations:
            self.logger.warning(f"Risk violation: {violation['type']} - {violation['value']:.2%} exceeds limit {violation['limit']:.2%}")
    
    def reset_daily_risk(self, portfolio_value: float):
        """
        Reset daily risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
        """
        self.daily_pnl = 0.0
        self.daily_start_value = portfolio_value
        self.logger.info("Daily risk metrics reset")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.
        
        Returns:
            Dict: Current risk metrics
        """
        return {
            "daily_pnl": self.daily_pnl,
            "current_drawdown": self.current_drawdown,
            "peak_portfolio_value": self.peak_portfolio_value,
            "daily_start_value": self.daily_start_value,
            "last_risk_check": self.last_risk_check,
            "risk_violations_count": len(self.risk_violations),
            "recent_violations": self.risk_violations[-5:] if self.risk_violations else []
        }
    
    def get_risk_params(self) -> Dict[str, Any]:
        """
        Get current risk parameters.
        
        Returns:
            Dict: Current risk parameters
        """
        return self.risk_params.copy()
    
    def update_risk_param(self, param_name: str, param_value: Any):
        """
        Update a risk parameter.
        
        Args:
            param_name: Name of the parameter to update
            param_value: New value for the parameter
        """
        if param_name in self.risk_params:
            old_value = self.risk_params[param_name]
            self.risk_params[param_name] = param_value
            self.logger.info(f"Updated risk parameter {param_name}: {old_value} -> {param_value}")
        else:
            self.logger.warning(f"Unknown risk parameter: {param_name}")