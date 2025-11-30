"""
Risk management utilities for the quantitative trading framework.

This module provides tools for calculating risk metrics, position sizing,
and implementing risk controls.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    cvar_95: float
    volatility: float
    calmar_ratio: float
    skewness: float
    kurtosis: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


@dataclass
class PositionSize:
    """Container for position sizing information."""
    symbol: str
    price: float
    quantity: float
    notional: float
    weight: float
    risk_contribution: float


class RiskCalculator:
    """Calculator for various risk metrics."""
    
    @staticmethod
    def calculate_returns(equity_curve: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Calculate returns from equity curve.
        
        Args:
            equity_curve: Series/array of equity values
            
        Returns:
            Array of returns
        """
        if isinstance(equity_curve, pd.Series):
            return equity_curve.pct_change().dropna().values
        else:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            return returns[~np.isnan(returns)]
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> Tuple[float, int, int, int]:
        """
        Calculate maximum drawdown and duration.
        
        Args:
            equity_curve: Series/array of equity values
            
        Returns:
            Tuple of (max_drawdown, duration, peak_index, trough_index)
        """
        if isinstance(equity_curve, pd.Series):
            equity_values = equity_curve.values
        else:
            equity_values = equity_curve
        
        if len(equity_values) < 2:
            return 0.0, 0, 0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_values)
        
        # Calculate drawdown
        drawdown = (equity_values - running_max) / running_max
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find peak index (previous maximum)
        peak_idx = np.argmax(equity_values[:max_dd_idx+1])
        
        # Find duration (time to recover from drawdown)
        duration = 0
        if max_dd_idx < len(equity_values) - 1:
            # Find when equity curve recovers to previous peak
            recovery_indices = np.where(equity_values[max_dd_idx+1:] >= equity_values[peak_idx])[0]
            if len(recovery_indices) > 0:
                duration = recovery_indices[0] + 1
            else:
                duration = len(equity_values) - max_dd_idx - 1
        
        return abs(max_dd), duration, peak_idx, max_dd_idx
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% CVaR)
            
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        var = RiskCalculator.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Array of returns
            periods_per_year: Number of periods per year
            
        Returns:
            Annualized volatility
        """
        if len(returns) == 0:
            return 0.0
        
        return np.std(returns) * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, equity_curve: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Array of returns
            equity_curve: Series/array of equity values
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * periods_per_year
        max_dd, _, _, _ = RiskCalculator.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    @staticmethod
    def calculate_skewness(returns: np.ndarray) -> float:
        """
        Calculate skewness of returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Skewness value
        """
        if len(returns) < 3:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        return np.mean(((returns - mean) / std) ** 3)
    
    @staticmethod
    def calculate_kurtosis(returns: np.ndarray) -> float:
        """
        Calculate kurtosis of returns.
        
        Args:
            returns: Array of returns
            
        Returns:
            Kurtosis value
        """
        if len(returns) < 4:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if std == 0:
            return 0.0
        
        return np.mean(((returns - mean) / std) ** 4) - 3  # Excess kurtosis
    
    @staticmethod
    def calculate_beta_alpha(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_rate: float = 0.0) -> Tuple[float, float]:
        """
        Calculate beta and alpha relative to benchmark.
        
        Args:
            portfolio_returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Tuple of (beta, alpha)
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0, 0.0
        
        # Remove NaN values
        valid_indices = ~(np.isnan(portfolio_returns) | np.isnan(benchmark_returns))
        portfolio_returns = portfolio_returns[valid_indices]
        benchmark_returns = benchmark_returns[valid_indices]
        
        if len(portfolio_returns) < 2:
            return 0.0, 0.0
        
        # Calculate beta
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            beta = 0.0
        else:
            beta = covariance / benchmark_variance
        
        # Calculate alpha
        portfolio_mean = np.mean(portfolio_returns)
        benchmark_mean = np.mean(benchmark_returns)
        
        alpha = portfolio_mean - (risk_free_rate + beta * (benchmark_mean - risk_free_rate))
        
        return beta, alpha
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.
        
        Args:
            portfolio_returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            
        Returns:
            Information Ratio
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        # Remove NaN values
        valid_indices = ~(np.isnan(portfolio_returns) | np.isnan(benchmark_returns))
        portfolio_returns = portfolio_returns[valid_indices]
        benchmark_returns = benchmark_returns[valid_indices]
        
        if len(portfolio_returns) < 2:
            return 0.0
        
        # Calculate active return
        active_return = portfolio_returns - benchmark_returns
        
        # Calculate tracking error
        tracking_error = np.std(active_return)
        
        if tracking_error == 0:
            return 0.0
        
        # Calculate Information Ratio
        return np.mean(active_return) / tracking_error
    
    @staticmethod
    def calculate_tracking_error(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Calculate Tracking Error.
        
        Args:
            portfolio_returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            
        Returns:
            Tracking Error
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        # Remove NaN values
        valid_indices = ~(np.isnan(portfolio_returns) | np.isnan(benchmark_returns))
        portfolio_returns = portfolio_returns[valid_indices]
        benchmark_returns = benchmark_returns[valid_indices]
        
        if len(portfolio_returns) < 2:
            return 0.0
        
        # Calculate active return
        active_return = portfolio_returns - benchmark_returns
        
        # Calculate tracking error
        return np.std(active_return)
    
    @staticmethod
    def calculate_risk_metrics(equity_curve: Union[pd.Series, np.ndarray], 
                              benchmark_returns: Optional[np.ndarray] = None,
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            equity_curve: Series/array of equity values
            benchmark_returns: Optional array of benchmark returns
            risk_free_rate: Risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            RiskMetrics object
        """
        returns = RiskCalculator.calculate_returns(equity_curve)
        
        if len(returns) == 0:
            return RiskMetrics(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_duration=0,
                var_95=0.0,
                cvar_95=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                skewness=0.0,
                kurtosis=0.0
            )
        
        # Calculate basic metrics
        sharpe_ratio = RiskCalculator.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino_ratio = RiskCalculator.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        max_dd, duration, _, _ = RiskCalculator.calculate_max_drawdown(equity_curve)
        var_95 = RiskCalculator.calculate_var(returns, 0.05)
        cvar_95 = RiskCalculator.calculate_cvar(returns, 0.05)
        volatility = RiskCalculator.calculate_volatility(returns, periods_per_year)
        calmar_ratio = RiskCalculator.calculate_calmar_ratio(returns, equity_curve, periods_per_year)
        skewness = RiskCalculator.calculate_skewness(returns)
        kurtosis = RiskCalculator.calculate_kurtosis(returns)
        
        # Initialize optional metrics as None
        beta = None
        alpha = None
        information_ratio = None
        tracking_error = None
        
        # Calculate benchmark-relative metrics if benchmark is provided
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            beta, alpha = RiskCalculator.calculate_beta_alpha(returns, benchmark_returns, risk_free_rate)
            information_ratio = RiskCalculator.calculate_information_ratio(returns, benchmark_returns)
            tracking_error = RiskCalculator.calculate_tracking_error(returns, benchmark_returns)
        
        return RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=duration,
            var_95=var_95,
            cvar_95=cvar_95,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )


class PositionSizer:
    """Calculator for position sizing based on risk parameters."""
    
    @staticmethod
    def fixed_dollar_amount(capital: float, amount: float) -> float:
        """
        Calculate position size based on fixed dollar amount.
        
        Args:
            capital: Total capital
            amount: Fixed dollar amount per position
            
        Returns:
            Position size as percentage of capital
        """
        return min(amount / capital, 1.0)
    
    @staticmethod
    def percent_of_capital(capital: float, percent: float) -> float:
        """
        Calculate position size based on percentage of capital.
        
        Args:
            capital: Total capital
            percent: Percentage of capital to allocate
            
        Returns:
            Position size as percentage of capital
        """
        return min(percent / 100, 1.0)
    
    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            win_rate: Win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            
        Returns:
            Position size as percentage of capital
        """
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly_percent = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Limit to reasonable range (0-25%)
        return max(0.0, min(kelly_percent, 0.25))
    
    @staticmethod
    def volatility_based_sizing(returns: np.ndarray, target_volatility: float, 
                               current_volatility: Optional[float] = None,
                               periods_per_year: int = 252) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            returns: Array of returns
            target_volatility: Target annualized volatility
            current_volatility: Current annualized volatility (calculated if None)
            periods_per_year: Number of periods per year
            
        Returns:
            Position size as percentage of capital
        """
        if len(returns) == 0:
            return 0.0
        
        if current_volatility is None:
            current_volatility = RiskCalculator.calculate_volatility(returns, periods_per_year)
        
        if current_volatility == 0:
            return 0.0
        
        # Scale position size to achieve target volatility
        return min(target_volatility / current_volatility, 1.0)
    
    @staticmethod
    def risk_parity(covariance_matrix: np.ndarray, target_volatility: float) -> np.ndarray:
        """
        Calculate position weights using risk parity approach.
        
        Args:
            covariance_matrix: Covariance matrix of asset returns
            target_volatility: Target portfolio volatility
            
        Returns:
            Array of position weights
        """
        n = covariance_matrix.shape[0]
        
        # Initialize weights
        weights = np.ones(n) / n
        
        # Iteratively adjust weights to equalize risk contributions
        for _ in range(100):  # Maximum iterations
            # Calculate portfolio volatility
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate marginal contribution to risk
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_volatility
            
            # Calculate risk contribution
            risk_contrib = weights * marginal_contrib
            
            # Calculate target risk contribution
            target_risk_contrib = target_volatility / n
            
            # Update weights
            new_weights = weights * (target_risk_contrib / risk_contrib)
            
            # Normalize weights
            new_weights = new_weights / np.sum(new_weights)
            
            # Check for convergence
            if np.max(np.abs(new_weights - weights)) < 1e-6:
                break
                
            weights = new_weights
        
        return weights


class RiskController:
    """Implements various risk controls for trading strategies."""
    
    def __init__(self, max_position_size: float = 0.2, max_portfolio_risk: float = 0.02,
                 max_drawdown_limit: float = 0.1, max_daily_loss: float = 0.05):
        """
        Initialize risk controller.
        
        Args:
            max_position_size: Maximum position size as percentage of capital
            max_portfolio_risk: Maximum portfolio risk (VaR) as percentage of capital
            max_drawdown_limit: Maximum allowed drawdown as percentage of capital
            max_daily_loss: Maximum allowed daily loss as percentage of capital
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown_limit = max_drawdown_limit
        self.max_daily_loss = max_daily_loss
        
        # Track daily P&L
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
    
    def check_position_size(self, symbol: str, weight: float) -> Tuple[bool, str]:
        """
        Check if position size is within limits.
        
        Args:
            symbol: Trading symbol
            weight: Position weight as percentage of capital
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if weight > self.max_position_size:
            return False, f"Position size {weight:.2%} exceeds maximum {self.max_position_size:.2%}"
        
        return True, ""
    
    def check_portfolio_risk(self, portfolio_var: float) -> Tuple[bool, str]:
        """
        Check if portfolio risk is within limits.
        
        Args:
            portfolio_var: Portfolio VaR as percentage of capital
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if portfolio_var > self.max_portfolio_risk:
            return False, f"Portfolio VaR {portfolio_var:.2%} exceeds maximum {self.max_portfolio_risk:.2%}"
        
        return True, ""
    
    def check_drawdown_limit(self, current_drawdown: float) -> Tuple[bool, str]:
        """
        Check if current drawdown is within limits.
        
        Args:
            current_drawdown: Current drawdown as percentage of capital
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if current_drawdown > self.max_drawdown_limit:
            return False, f"Current drawdown {current_drawdown:.2%} exceeds maximum {self.max_drawdown_limit:.2%}"
        
        return True, ""
    
    def check_daily_loss_limit(self, pnl: float) -> Tuple[bool, str]:
        """
        Check if daily loss is within limits.
        
        Args:
            pnl: Daily P&L as percentage of capital
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Reset daily P&L if it's a new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss {abs(self.daily_pnl):.2%} exceeds maximum {self.max_daily_loss:.2%}"
        
        return True, ""
    
    def check_all_limits(self, symbol: str, weight: float, portfolio_var: float, 
                        current_drawdown: float, pnl: float) -> Tuple[bool, List[str]]:
        """
        Check all risk limits.
        
        Args:
            symbol: Trading symbol
            weight: Position weight as percentage of capital
            portfolio_var: Portfolio VaR as percentage of capital
            current_drawdown: Current drawdown as percentage of capital
            pnl: Daily P&L as percentage of capital
            
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        violations = []
        
        # Check position size
        allowed, reason = self.check_position_size(symbol, weight)
        if not allowed:
            violations.append(reason)
        
        # Check portfolio risk
        allowed, reason = self.check_portfolio_risk(portfolio_var)
        if not allowed:
            violations.append(reason)
        
        # Check drawdown limit
        allowed, reason = self.check_drawdown_limit(current_drawdown)
        if not allowed:
            violations.append(reason)
        
        # Check daily loss limit
        allowed, reason = self.check_daily_loss_limit(pnl)
        if not allowed:
            violations.append(reason)
        
        return len(violations) == 0, violations


# Legacy classes for backward compatibility with src/utils/risk.py
@dataclass
class RiskConfig:
    max_position_notional_usdt: float
    max_order_notional_usdt: float
    order_percent_balance: float
    # 额外风控项（默认关闭，以保持向后兼容）
    daily_loss_limit_usdt: float = 0.0  # 当天累计亏损超过该值则停止下单；0 表示禁用
    max_intraday_drawdown_pct: float = 0.0  # 当天回撤超过该百分比（0-100）则停止下单；0 表示禁用


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def compute_order_notional(self, free_usdt: float) -> float:
        target = max(0.0, free_usdt * self.cfg.order_percent_balance)
        target = min(target, self.cfg.max_order_notional_usdt)
        return target

    def can_increase_position(self, current_notional: float, add_notional: float) -> bool:
        return (current_notional + add_notional) <= self.cfg.max_position_notional_usdt

    def should_halt_new_orders(
        self,
        daily_realized_pnl_usdt: float,
        starting_equity_usdt: float,
        current_equity_usdt: float,
    ) -> Tuple[bool, str]:
        """
        基于当日亏损上限与当日回撤阈值判断是否应停止新增订单。

        返回: (是否停止, 原因)
        """
        # 1) 当日累计亏损限制
        if self.cfg.daily_loss_limit_usdt and self.cfg.daily_loss_limit_usdt > 0:
            if daily_realized_pnl_usdt <= -abs(self.cfg.daily_loss_limit_usdt):
                return True, f"daily loss limit breached: {daily_realized_pnl_usdt:.2f} USDT"

        # 2) 当日回撤限制（基于权益）
        if self.cfg.max_intraday_drawdown_pct and self.cfg.max_intraday_drawdown_pct > 0:
            if starting_equity_usdt > 0:
                dd = (starting_equity_usdt - current_equity_usdt) / starting_equity_usdt * 100.0
                if dd >= self.cfg.max_intraday_drawdown_pct:
                    return True, f"intraday drawdown {dd:.2f}% >= {self.cfg.max_intraday_drawdown_pct:.2f}%"

        return False, ""