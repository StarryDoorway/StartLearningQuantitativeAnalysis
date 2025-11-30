"""
Visualization module for the quantitative trading framework.

This module provides visualization capabilities for strategies, portfolios,
and market data, including performance charts, risk analysis, and
statistical analysis visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

from .analyzer import PerformanceMetrics, TradeAnalysis, get_analyzer
from ..utils.logger import get_logger

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Visualizer for strategies, portfolios, and market data.
    
    This class provides methods to create various charts and visualizations
    for trading strategies and portfolios.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Default DPI for figures
        """
        self.logger = get_logger(__name__)
        self.figsize = figsize
        self.dpi = dpi
        self.analyzer = get_analyzer()
    
    def plot_equity_curve(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot equity curve from returns.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.values, label='Strategy', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    label='Benchmark', linewidth=2, alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown(
        self, 
        returns: pd.Series, 
        title: str = "Drawdown",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot drawdown from returns.
        
        Args:
            returns: Series of returns
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Calculate cumulative returns and drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_returns_heatmap(
        self, 
        returns: pd.Series, 
        title: str = "Monthly Returns Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            returns: Series of returns
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        # Calculate monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table with years as rows and months as columns
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='return')
        
        # Replace month numbers with names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.heatmap(pivot_table, annot=True, fmt=".2%", cmap="RdYlGn", 
                   center=0, ax=ax, cbar_kws={'format': '%.2f%%'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_performance_attribution(
        self, 
        returns: pd.Series, 
        factor_returns: pd.DataFrame,
        title: str = "Performance Attribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance attribution against factors.
        
        Args:
            returns: Series of returns
            factor_returns: DataFrame with factor returns
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        # Align returns
        aligned_returns, aligned_factors = returns.align(factor_returns, join='inner')
        
        # Calculate factor betas using regression
        factor_betas = {}
        for factor in aligned_factors.columns:
            covariance = np.cov(aligned_returns, aligned_factors[factor])[0, 1]
            factor_variance = np.var(aligned_factors[factor])
            beta = covariance / factor_variance if factor_variance > 0 else 0
            factor_betas[factor] = beta
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        factors = list(factor_betas.keys())
        betas = list(factor_betas.values())
        
        bars = ax.bar(factors, betas)
        
        # Color bars based on sign
        for bar, beta in zip(bars, betas):
            if beta >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Factor', fontsize=12)
        ax.set_ylabel('Beta', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, beta in zip(bars, betas):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                   f'{beta:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_rolling_metrics(
        self, 
        returns: pd.Series, 
        window: int = 252,
        title: str = "Rolling Performance Metrics",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        # Calculate rolling metrics
        rolling_metrics = self.analyzer.calculate_rolling_metrics(
            returns, window=window, metrics=["sharpe", "sortino", "max_drawdown"]
        )
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), 
                                 dpi=self.dpi, sharex=True)
        
        # Plot Sharpe ratio
        axes[0].plot(rolling_metrics.index, rolling_metrics["sharpe"], linewidth=2)
        axes[0].set_title('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Sharpe Ratio', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot Sortino ratio
        axes[1].plot(rolling_metrics.index, rolling_metrics["sortino"], linewidth=2, color='green')
        axes[1].set_title('Rolling Sortino Ratio', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Sortino Ratio', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot Max Drawdown
        axes[2].fill_between(rolling_metrics.index, rolling_metrics["max_drawdown"], 0, 
                            color='red', alpha=0.3)
        axes[2].plot(rolling_metrics.index, rolling_metrics["max_drawdown"], 
                    linewidth=1.5, color='red')
        axes[2].set_title('Rolling Max Drawdown', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Drawdown', fontsize=10)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Format x-axis
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(
        self, 
        returns_data: pd.DataFrame, 
        title: str = "Correlation Matrix",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            returns_data: DataFrame with return series
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        # Calculate correlation matrix
        corr_matrix = self.analyzer.calculate_correlation_matrix(returns_data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, ax=ax, vmin=-1, vmax=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_trade_distribution(
        self, 
        trades: pd.DataFrame, 
        title: str = "Trade Distribution",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot trade distribution.
        
        Args:
            trades: DataFrame with trades data
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        if trades.empty or 'pnl' not in trades.columns:
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1] * 1.5), 
                                 dpi=self.dpi)
        
        # Plot PnL distribution
        axes[0, 0].hist(trades['pnl'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('PnL Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('PnL', fontsize=10)
        axes[0, 0].set_ylabel('Frequency', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot trade duration
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            trades['duration'] = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / 86400
            axes[0, 1].hist(trades['duration'], bins=30, alpha=0.7, color='green')
            axes[0, 1].set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Duration (Days)', fontsize=10)
            axes[0, 1].set_ylabel('Frequency', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot cumulative PnL
        trades['cumulative_pnl'] = trades['pnl'].cumsum()
        axes[1, 0].plot(trades.index, trades['cumulative_pnl'], linewidth=2)
        axes[1, 0].set_title('Cumulative PnL', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Trade Number', fontsize=10)
        axes[1, 0].set_ylabel('Cumulative PnL', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot win/loss pie chart
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = len(trades[trades['pnl'] < 0])
        
        if winning_trades > 0 or losing_trades > 0:
            axes[1, 1].pie([winning_trades, losing_trades], 
                          labels=['Winning Trades', 'Losing Trades'],
                          colors=['green', 'red'], autopct='%1.1f%%')
            axes[1, 1].set_title('Win/Loss Ratio', fontsize=12, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_performance_summary(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        title: str = "Performance Summary",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot comprehensive performance summary.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            trades: Optional DataFrame with trades data
            title: Chart title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        # Calculate performance metrics
        metrics = self.analyzer.calculate_performance_metrics(returns, benchmark_returns)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(self.figsize[0] * 1.5, self.figsize[1] * 2), dpi=self.dpi)
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot equity curve
        ax1 = fig.add_subplot(gs[0, :])
        cumulative = (1 + returns).cumprod()
        ax1.plot(cumulative.index, cumulative.values, label='Strategy', linewidth=2)
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax1.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    label='Benchmark', linewidth=2, alpha=0.7)
        
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Cumulative Return', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        # Plot drawdown
        ax2 = fig.add_subplot(gs[1, :])
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1.5)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Drawdown', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        # Plot monthly returns heatmap
        ax3 = fig.add_subplot(gs[2, 0])
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='return')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        sns.heatmap(pivot_table, annot=True, fmt=".1%", cmap="RdYlGn", 
                   center=0, ax=ax3, cbar_kws={'format': '%.0f%%'})
        ax3.set_title('Monthly Returns (%)', fontsize=12, fontweight='bold')
        
        # Plot return distribution
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.hist(returns, bins=30, alpha=0.7, color='blue')
        ax4.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Return', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot performance metrics table
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        
        # Create metrics data
        metrics_data = [
            ['Total Return', f'{metrics.total_return:.2%}'],
            ['Annualized Return', f'{metrics.annualized_return:.2%}'],
            ['Volatility', f'{metrics.volatility:.2%}'],
            ['Sharpe Ratio', f'{metrics.sharpe_ratio:.2f}'],
            ['Sortino Ratio', f'{metrics.sortino_ratio:.2f}'],
            ['Max Drawdown', f'{metrics.max_drawdown:.2%}'],
            ['Calmar Ratio', f'{metrics.calmar_ratio:.2f}'],
            ['Win Rate', f'{metrics.win_rate:.2%}']
        ]
        
        # Add benchmark metrics if available
        if benchmark_returns is not None:
            benchmark_metrics = self.analyzer.calculate_performance_metrics(benchmark_returns)
            metrics_data.extend([
                ['Benchmark Return', f'{benchmark_metrics.total_return:.2%}'],
                ['Alpha', f'{metrics.alpha:.2%}'],
                ['Beta', f'{metrics.beta:.2f}'],
                ['Information Ratio', f'{metrics.information_ratio:.2f}']
            ])
        
        # Create table
        table = ax5.table(cellText=metrics_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax5.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


# Global visualizer instance
_visualizer = None

def get_visualizer(figsize: Tuple[int, int] = (12, 8), dpi: int = 100) -> Visualizer:
    """
    Get the global visualizer instance.
    
    Args:
        figsize: Default figure size
        dpi: Default DPI for figures
        
    Returns:
        Visualizer instance
    """
    global _visualizer
    if _visualizer is None:
        _visualizer = Visualizer(figsize=figsize, dpi=dpi)
    return _visualizer