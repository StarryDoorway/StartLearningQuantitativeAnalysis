"""
Strategy performance metrics for the quantitative trading framework.

This module provides functionality for calculating and tracking strategy performance metrics.
"""

import logging
import numpy as np
from typing import Dict, Any
from datetime import datetime


logger = logging.getLogger(__name__)


class StrategyPerformanceManager:
    """
    Manages performance metrics for a trading strategy.
    
    This class calculates and tracks various performance metrics based on trade history.
    """
    
    def __init__(self, strategy_id: str):
        """
        Initialize the performance manager.
        
        Args:
            strategy_id: Strategy identifier
        """
        self.strategy_id = strategy_id
        
        # Performance metrics
        self.performance_metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.{strategy_id}")
    
    def update_metrics(self, trade_history: list) -> None:
        """
        Update performance metrics based on trade history.
        
        Args:
            trade_history: List of trade data
        """
        if not trade_history:
            return
        
        # Calculate total return
        total_pnl = sum(trade.get("pnl", 0) for trade in trade_history)
        self.performance_metrics["total_return"] = total_pnl
        
        # Count trades
        self.performance_metrics["total_trades"] = len(trade_history)
        
        # Count winning and losing trades
        winning_trades = sum(1 for trade in trade_history if trade.get("pnl", 0) > 0)
        losing_trades = sum(1 for trade in trade_history if trade.get("pnl", 0) < 0)
        
        self.performance_metrics["winning_trades"] = winning_trades
        self.performance_metrics["losing_trades"] = losing_trades
        
        # Calculate win rate
        if self.performance_metrics["total_trades"] > 0:
            self.performance_metrics["win_rate"] = winning_trades / self.performance_metrics["total_trades"]
        
        # Calculate profit factor
        total_profit = sum(trade.get("pnl", 0) for trade in trade_history if trade.get("pnl", 0) > 0)
        total_loss = abs(sum(trade.get("pnl", 0) for trade in trade_history if trade.get("pnl", 0) < 0))
        
        if total_loss > 0:
            self.performance_metrics["profit_factor"] = total_profit / total_loss
        
        # Calculate Sharpe ratio (simplified)
        if len(trade_history) > 1:
            returns = [trade.get("pnl", 0) for trade in trade_history]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                self.performance_metrics["sharpe_ratio"] = mean_return / std_return
        
        # Calculate maximum drawdown (simplified)
        cumulative_pnl = np.cumsum([trade.get("pnl", 0) for trade in trade_history])
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (peak - cumulative_pnl) / (peak + 1e-10)  # Avoid division by zero
        self.performance_metrics["max_drawdown"] = np.max(drawdown) * 100  # Convert to percentage
        
        # Calculate annualized return (simplified)
        if len(trade_history) > 1:
            first_trade = trade_history[0].get("timestamp", datetime.now())
            last_trade = trade_history[-1].get("timestamp", datetime.now())
            
            if isinstance(first_trade, str):
                first_trade = datetime.fromisoformat(first_trade)
            if isinstance(last_trade, str):
                last_trade = datetime.fromisoformat(last_trade)
            
            days = (last_trade - first_trade).days
            if days > 0:
                self.performance_metrics["annualized_return"] = (total_pnl / days) * 365
        
        self.logger.debug(f"Updated performance metrics for strategy {self.strategy_id}")
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns:
            Performance metrics
        """
        return self.performance_metrics.copy()
    
    def reset(self) -> None:
        """Reset all performance metrics."""
        self.performance_metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
        self.logger.info(f"Reset performance metrics for strategy {self.strategy_id}")