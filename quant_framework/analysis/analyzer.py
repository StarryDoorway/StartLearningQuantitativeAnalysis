"""
Analysis module for the quantitative trading framework.

This module provides analysis capabilities for strategies, portfolios,
and market data, including performance metrics, risk analysis, and
statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from ..utils.logger import get_logger


class AnalysisType(Enum):
    """Analysis types."""
    PERFORMANCE = "performance"
    RISK = "risk"
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    DRAWDOWN = "drawdown"
    REGIME = "regime"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy or portfolio."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    recovery_factor: float = 0.0
    var_95: float = 0.0  # Value at Risk at 95% confidence
    cvar_95: float = 0.0  # Conditional Value at Risk at 95% confidence
    skewness: float = 0.0
    kurtosis: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "recovery_factor": self.recovery_factor,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "beta": self.beta,
            "alpha": self.alpha,
            "information_ratio": self.information_ratio,
            "tracking_error": self.tracking_error
        }


@dataclass
class TradeAnalysis:
    """Analysis of individual trades."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0  # in days
    avg_winning_trade_duration: float = 0.0
    avg_losing_trade_duration: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_trade: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_trade_duration": self.avg_trade_duration,
            "avg_winning_trade_duration": self.avg_winning_trade_duration,
            "avg_losing_trade_duration": self.avg_losing_trade_duration,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "avg_trade": self.avg_trade,
            "avg_winning_trade": self.avg_winning_trade,
            "avg_losing_trade": self.avg_losing_trade
        }


class Analyzer:
    """
    Analyzer for strategies, portfolios, and market data.
    
    This class provides methods to calculate various performance metrics,
    risk metrics, and statistical analysis for trading strategies and portfolios.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.logger = get_logger(__name__)
    
    def calculate_returns(self, prices: pd.Series, method: str = "simple") -> pd.Series:
        """
        Calculate returns from price series.
        
        Args:
            prices: Series of prices
            method: Method for calculating returns ('simple' or 'log')
            
        Returns:
            Series of returns
        """
        if method == "simple":
            return prices.pct_change().fillna(0)
        elif method == "log":
            return np.log(prices / prices.shift(1)).fillna(0)
        else:
            raise ValueError(f"Unknown return calculation method: {method}")
    
    def calculate_performance_metrics(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics from return series.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            PerformanceMetrics object
        """
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        n_days = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe and Sortino ratios
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Recovery factor
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk and Conditional Value at Risk
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Alpha, beta, information ratio, tracking error (if benchmark provided)
        alpha, beta, information_ratio, tracking_error = 0, 0, 0, 0
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Align returns
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 0:
                # Calculate beta and alpha using regression
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                alpha = (aligned_returns.mean() - risk_free_rate / 252) - beta * (aligned_benchmark.mean() - risk_free_rate / 252)
                alpha = alpha * 252  # Annualize alpha
                
                # Information ratio and tracking error
                excess_returns = aligned_returns - aligned_benchmark
                information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
                tracking_error = excess_returns.std() * np.sqrt(252)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            recovery_factor=recovery_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    def analyze_trades(self, trades: pd.DataFrame) -> TradeAnalysis:
        """
        Analyze a series of trades.
        
        Args:
            trades: DataFrame with columns: 'entry_time', 'exit_time', 'pnl'
            
        Returns:
            TradeAnalysis object
        """
        if trades.empty:
            return TradeAnalysis()
        
        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Trade duration
        trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 86400  # Convert to days
        avg_trade_duration = trades['duration'].mean()
        
        # Winning and losing trade durations
        winning_trades_df = trades[trades['pnl'] > 0]
        losing_trades_df = trades[trades['pnl'] < 0]
        
        avg_winning_trade_duration = winning_trades_df['duration'].mean() if not winning_trades_df.empty else 0
        avg_losing_trade_duration = losing_trades_df['duration'].mean() if not losing_trades_df.empty else 0
        
        # Largest win and loss
        largest_win = trades['pnl'].max()
        largest_loss = trades['pnl'].min()
        
        # Consecutive wins and losses
        trades['is_win'] = trades['pnl'] > 0
        trades['consecutive'] = (trades['is_win'] != trades['is_win'].shift()).cumsum()
        
        # Find longest consecutive wins
        consecutive_wins = trades.groupby(['consecutive', 'is_win']).size().reset_index(name='count')
        consecutive_wins = consecutive_wins[consecutive_wins['is_win'] == True]
        max_consecutive_wins = consecutive_wins['count'].max() if not consecutive_wins.empty else 0
        
        # Find longest consecutive losses
        consecutive_losses = trades.groupby(['consecutive', 'is_win']).size().reset_index(name='count')
        consecutive_losses = consecutive_losses[consecutive_losses['is_win'] == False]
        max_consecutive_losses = consecutive_losses['count'].max() if not consecutive_losses.empty else 0
        
        # Average trade metrics
        avg_trade = trades['pnl'].mean()
        avg_winning_trade = winning_trades_df['pnl'].mean() if not winning_trades_df.empty else 0
        avg_losing_trade = losing_trades_df['pnl'].mean() if not losing_trades_df.empty else 0
        
        return TradeAnalysis(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_trade_duration=avg_trade_duration,
            avg_winning_trade_duration=avg_winning_trade_duration,
            avg_losing_trade_duration=avg_losing_trade_duration,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=max_consecutive_wins,
            consecutive_losses=max_consecutive_losses,
            avg_trade=avg_trade,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade
        )
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple return series.
        
        Args:
            returns_data: DataFrame with columns as different return series
            
        Returns:
            Correlation matrix
        """
        return returns_data.corr()
    
    def calculate_rolling_metrics(
        self, 
        returns: pd.Series, 
        window: int = 252,
        metrics: List[str] = ["sharpe", "sortino", "max_drawdown"]
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        results = pd.DataFrame(index=returns.index)
        
        if "sharpe" in metrics:
            rolling_sharpe = returns.rolling(window).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            results["sharpe"] = rolling_sharpe
        
        if "sortino" in metrics:
            def rolling_sortino(x):
                downside = x[x < 0]
                return x.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
            
            rolling_sortino_series = returns.rolling(window).apply(rolling_sortino)
            results["sortino"] = rolling_sortino_series
        
        if "max_drawdown" in metrics:
            def rolling_max_drawdown(x):
                cumulative = (1 + x).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return drawdown.min()
            
            rolling_drawdown = returns.rolling(window).apply(rolling_max_drawdown)
            results["max_drawdown"] = rolling_drawdown
        
        return results
    
    def detect_regimes(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Detect market regimes based on volatility and returns.
        
        Args:
            returns: Series of returns
            window: Window for calculating statistics
            
        Returns:
            Series with regime labels
        """
        # Calculate rolling statistics
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        # Classify regimes based on mean and volatility
        regimes = pd.Series(index=returns.index, dtype=str)
        
        # High volatility, positive returns: Bull market
        regimes[(rolling_mean > 0) & (rolling_std > rolling_std.quantile(0.6))] = "bull"
        
        # Low volatility, stable returns: Stable market
        regimes[(rolling_mean.abs() < rolling_mean.quantile(0.4)) & (rolling_std < rolling_std.quantile(0.4))] = "stable"
        
        # High volatility, negative returns: Bear market
        regimes[(rolling_mean < 0) & (rolling_std > rolling_std.quantile(0.6))] = "bear"
        
        # Default to transition
        regimes[regimes.isna()] = "transition"
        
        return regimes
    
    def calculate_information_coefficient(
        self, 
        predictions: pd.Series, 
        actuals: pd.Series
    ) -> float:
        """
        Calculate information coefficient (IC) for predictions.
        
        Args:
            predictions: Series of predicted returns
            actuals: Series of actual returns
            
        Returns:
            Information coefficient
        """
        # Align the series
        aligned_pred, aligned_actual = predictions.align(actuals, join='inner')
        
        if len(aligned_pred) < 2:
            return 0.0
        
        # Calculate correlation
        return aligned_pred.corr(aligned_actual)


# Global analyzer instance
_analyzer = None

def get_analyzer() -> Analyzer:
    """Get the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer()
    return _analyzer