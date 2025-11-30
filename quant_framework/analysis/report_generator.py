"""
Report generator module for the quantitative trading framework.

This module provides report generation capabilities for strategies and portfolios,
including performance reports, risk analysis, and statistical analysis reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import os
import json

from .analyzer import PerformanceMetrics, TradeAnalysis, get_analyzer
from .visualizer import get_visualizer
from ..utils.logger import get_logger


class ReportGenerator:
    """
    Report generator for strategies and portfolios.
    
    This class provides methods to generate comprehensive reports
    for trading strategies and portfolios.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.logger = get_logger(__name__)
        self.output_dir = output_dir
        self.analyzer = get_analyzer()
        self.visualizer = get_visualizer()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_performance_report(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
        save_plots: bool = True,
        output_format: str = "html"
    ) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            trades: Optional DataFrame with trades data
            strategy_name: Name of the strategy
            save_plots: Whether to save plots
            output_format: Output format ('html', 'json', 'txt')
            
        Returns:
            Path to the generated report
        """
        # Calculate performance metrics
        performance_metrics = self.analyzer.calculate_performance_metrics(returns, benchmark_returns)
        
        # Analyze trades if provided
        trade_analysis = None
        if trades is not None:
            trade_analysis = self.analyzer.analyze_trades(trades)
        
        # Generate report based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_performance_report_{timestamp}.{output_format}"
        filepath = os.path.join(self.output_dir, filename)
        
        if output_format == "html":
            report_content = self._generate_html_report(
                performance_metrics, trade_analysis, strategy_name, returns, benchmark_returns, trades
            )
        elif output_format == "json":
            report_content = self._generate_json_report(
                performance_metrics, trade_analysis, strategy_name
            )
        elif output_format == "txt":
            report_content = self._generate_text_report(
                performance_metrics, trade_analysis, strategy_name
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Write report to file
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        # Save plots if requested
        if save_plots and output_format == "html":
            self._save_performance_plots(returns, benchmark_returns, trades, strategy_name, timestamp)
        
        self.logger.info(f"Generated {output_format} performance report: {filepath}")
        return filepath
    
    def generate_risk_report(
        self, 
        returns: pd.Series, 
        positions: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
        output_format: str = "html"
    ) -> str:
        """
        Generate a risk analysis report.
        
        Args:
            returns: Series of returns
            positions: Optional DataFrame with positions data
            strategy_name: Name of the strategy
            output_format: Output format ('html', 'json', 'txt')
            
        Returns:
            Path to the generated report
        """
        # Calculate risk metrics
        performance_metrics = self.analyzer.calculate_performance_metrics(returns)
        
        # Calculate additional risk metrics
        var_99 = returns.quantile(0.01)
        cvar_99 = returns[returns <= var_99].mean()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(22).std() * np.sqrt(252)  # 22 trading days ~ 1 month
        
        # Generate report based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_risk_report_{timestamp}.{output_format}"
        filepath = os.path.join(self.output_dir, filename)
        
        if output_format == "html":
            report_content = self._generate_risk_html_report(
                performance_metrics, var_99, cvar_99, rolling_vol, strategy_name, returns
            )
        elif output_format == "json":
            risk_data = {
                "strategy_name": strategy_name,
                "var_99": var_99,
                "cvar_99": cvar_99,
                "current_volatility": rolling_vol.iloc[-1] if not rolling_vol.empty else 0,
                "avg_volatility": rolling_vol.mean() if not rolling_vol.empty else 0,
                "max_volatility": rolling_vol.max() if not rolling_vol.empty else 0,
                "performance_metrics": performance_metrics.to_dict()
            }
            report_content = json.dumps(risk_data, indent=4)
        elif output_format == "txt":
            report_content = self._generate_risk_text_report(
                performance_metrics, var_99, cvar_99, rolling_vol, strategy_name
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Write report to file
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        # Save risk plots if HTML format
        if output_format == "html":
            self._save_risk_plots(returns, strategy_name, timestamp)
        
        self.logger.info(f"Generated {output_format} risk report: {filepath}")
        return filepath
    
    def _generate_html_report(
        self, 
        performance_metrics: PerformanceMetrics, 
        trade_analysis: Optional[TradeAnalysis],
        strategy_name: str,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None
    ) -> str:
        """Generate HTML performance report."""
        # Calculate date range
        start_date = returns.index[0].strftime("%Y-%m-%d")
        end_date = returns.index[-1].strftime("%Y-%m-%d")
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{strategy_name} Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin-bottom: 10px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>{strategy_name} Performance Report</h1>
            <p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Period: {start_date} to {end_date}</p>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Return</td><td class="{'positive' if performance_metrics.total_return >= 0 else 'negative'}">{performance_metrics.total_return:.2%}</td></tr>
                <tr><td>Annualized Return</td><td class="{'positive' if performance_metrics.annualized_return >= 0 else 'negative'}">{performance_metrics.annualized_return:.2%}</td></tr>
                <tr><td>Volatility</td><td>{performance_metrics.volatility:.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{performance_metrics.sharpe_ratio:.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{performance_metrics.sortino_ratio:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td class="negative">{performance_metrics.max_drawdown:.2%}</td></tr>
                <tr><td>Calmar Ratio</td><td>{performance_metrics.calmar_ratio:.2f}</td></tr>
                <tr><td>Win Rate</td><td>{performance_metrics.win_rate:.2%}</td></tr>
                <tr><td>Profit Factor</td><td>{performance_metrics.profit_factor:.2f}</td></tr>
                <tr><td>Average Win</td><td class="positive">{performance_metrics.avg_win:.2%}</td></tr>
                <tr><td>Average Loss</td><td class="negative">{performance_metrics.avg_loss:.2%}</td></tr>
                <tr><td>Recovery Factor</td><td>{performance_metrics.recovery_factor:.2f}</td></tr>
                <tr><td>Value at Risk (95%)</td><td class="negative">{performance_metrics.var_95:.2%}</td></tr>
                <tr><td>Conditional Value at Risk (95%)</td><td class="negative">{performance_metrics.cvar_95:.2%}</td></tr>
                <tr><td>Skewness</td><td>{performance_metrics.skewness:.2f}</td></tr>
                <tr><td>Kurtosis</td><td>{performance_metrics.kurtosis:.2f}</td></tr>
        """
        
        # Add benchmark comparison if available
        if benchmark_returns is not None:
            html_content += f"""
                <tr><td>Beta</td><td>{performance_metrics.beta:.2f}</td></tr>
                <tr><td>Alpha</td><td class="{'positive' if performance_metrics.alpha >= 0 else 'negative'}">{performance_metrics.alpha:.2%}</td></tr>
                <tr><td>Information Ratio</td><td>{performance_metrics.information_ratio:.2f}</td></tr>
                <tr><td>Tracking Error</td><td>{performance_metrics.tracking_error:.2%}</td></tr>
            """
        
        html_content += """
            </table>
        """
        
        # Add trade analysis if available
        if trade_analysis is not None:
            html_content += f"""
            <h2>Trade Analysis</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{trade_analysis.total_trades}</td></tr>
                <tr><td>Winning Trades</td><td>{trade_analysis.winning_trades}</td></tr>
                <tr><td>Losing Trades</td><td>{trade_analysis.losing_trades}</td></tr>
                <tr><td>Win Rate</td><td>{trade_analysis.win_rate:.2%}</td></tr>
                <tr><td>Average Trade Duration</td><td>{trade_analysis.avg_trade_duration:.2f} days</td></tr>
                <tr><td>Average Winning Trade Duration</td><td>{trade_analysis.avg_winning_trade_duration:.2f} days</td></tr>
                <tr><td>Average Losing Trade Duration</td><td>{trade_analysis.avg_losing_trade_duration:.2f} days</td></tr>
                <tr><td>Largest Win</td><td class="positive">{trade_analysis.largest_win:.2%}</td></tr>
                <tr><td>Largest Loss</td><td class="negative">{trade_analysis.largest_loss:.2%}</td></tr>
                <tr><td>Consecutive Wins</td><td>{trade_analysis.consecutive_wins}</td></tr>
                <tr><td>Consecutive Losses</td><td>{trade_analysis.consecutive_losses}</td></tr>
                <tr><td>Average Trade</td><td class="{'positive' if trade_analysis.avg_trade >= 0 else 'negative'}">{trade_analysis.avg_trade:.2%}</td></tr>
                <tr><td>Average Winning Trade</td><td class="positive">{trade_analysis.avg_winning_trade:.2%}</td></tr>
                <tr><td>Average Losing Trade</td><td class="negative">{trade_analysis.avg_losing_trade:.2%}</td></tr>
            </table>
            """
        
        # Add plots section
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_content += f"""
            <h2>Performance Charts</h2>
            <div class="plot">
                <h3>Equity Curve</h3>
                <img src="{strategy_name}_equity_curve_{timestamp}.png" alt="Equity Curve">
            </div>
            <div class="plot">
                <h3>Drawdown</h3>
                <img src="{strategy_name}_drawdown_{timestamp}.png" alt="Drawdown">
            </div>
            <div class="plot">
                <h3>Monthly Returns Heatmap</h3>
                <img src="{strategy_name}_monthly_returns_{timestamp}.png" alt="Monthly Returns Heatmap">
            </div>
        """
        
        if benchmark_returns is not None:
            html_content += f"""
            <div class="plot">
                <h3>Performance Attribution</h3>
                <img src="{strategy_name}_performance_attribution_{timestamp}.png" alt="Performance Attribution">
            </div>
            """
        
        if trades is not None:
            html_content += f"""
            <div class="plot">
                <h3>Trade Distribution</h3>
                <img src="{strategy_name}_trade_distribution_{timestamp}.png" alt="Trade Distribution">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_json_report(
        self, 
        performance_metrics: PerformanceMetrics, 
        trade_analysis: Optional[TradeAnalysis],
        strategy_name: str
    ) -> str:
        """Generate JSON performance report."""
        report_data = {
            "strategy_name": strategy_name,
            "report_date": datetime.now().isoformat(),
            "performance_metrics": performance_metrics.to_dict()
        }
        
        if trade_analysis is not None:
            report_data["trade_analysis"] = trade_analysis.to_dict()
        
        return json.dumps(report_data, indent=4)
    
    def _generate_text_report(
        self, 
        performance_metrics: PerformanceMetrics, 
        trade_analysis: Optional[TradeAnalysis],
        strategy_name: str
    ) -> str:
        """Generate text performance report."""
        report_lines = [
            f"{strategy_name} Performance Report",
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Performance Metrics:",
            "-" * 20,
            f"Total Return: {performance_metrics.total_return:.2%}",
            f"Annualized Return: {performance_metrics.annualized_return:.2%}",
            f"Volatility: {performance_metrics.volatility:.2%}",
            f"Sharpe Ratio: {performance_metrics.sharpe_ratio:.2f}",
            f"Sortino Ratio: {performance_metrics.sortino_ratio:.2f}",
            f"Max Drawdown: {performance_metrics.max_drawdown:.2%}",
            f"Calmar Ratio: {performance_metrics.calmar_ratio:.2f}",
            f"Win Rate: {performance_metrics.win_rate:.2%}",
            f"Profit Factor: {performance_metrics.profit_factor:.2f}",
            f"Average Win: {performance_metrics.avg_win:.2%}",
            f"Average Loss: {performance_metrics.avg_loss:.2%}",
            f"Recovery Factor: {performance_metrics.recovery_factor:.2f}",
            f"Value at Risk (95%): {performance_metrics.var_95:.2%}",
            f"Conditional Value at Risk (95%): {performance_metrics.cvar_95:.2%}",
            f"Skewness: {performance_metrics.skewness:.2f}",
            f"Kurtosis: {performance_metrics.kurtosis:.2f}",
            f"Beta: {performance_metrics.beta:.2f}",
            f"Alpha: {performance_metrics.alpha:.2%}",
            f"Information Ratio: {performance_metrics.information_ratio:.2f}",
            f"Tracking Error: {performance_metrics.tracking_error:.2%}"
        ]
        
        if trade_analysis is not None:
            report_lines.extend([
                "",
                "Trade Analysis:",
                "-" * 20,
                f"Total Trades: {trade_analysis.total_trades}",
                f"Winning Trades: {trade_analysis.winning_trades}",
                f"Losing Trades: {trade_analysis.losing_trades}",
                f"Win Rate: {trade_analysis.win_rate:.2%}",
                f"Average Trade Duration: {trade_analysis.avg_trade_duration:.2f} days",
                f"Average Winning Trade Duration: {trade_analysis.avg_winning_trade_duration:.2f} days",
                f"Average Losing Trade Duration: {trade_analysis.avg_losing_trade_duration:.2f} days",
                f"Largest Win: {trade_analysis.largest_win:.2%}",
                f"Largest Loss: {trade_analysis.largest_loss:.2%}",
                f"Consecutive Wins: {trade_analysis.consecutive_wins}",
                f"Consecutive Losses: {trade_analysis.consecutive_losses}",
                f"Average Trade: {trade_analysis.avg_trade:.2%}",
                f"Average Winning Trade: {trade_analysis.avg_winning_trade:.2%}",
                f"Average Losing Trade: {trade_analysis.avg_losing_trade:.2%}"
            ])
        
        return "\n".join(report_lines)
    
    def _generate_risk_html_report(
        self, 
        performance_metrics: PerformanceMetrics,
        var_99: float,
        cvar_99: float,
        rolling_vol: pd.Series,
        strategy_name: str,
        returns: pd.Series
    ) -> str:
        """Generate HTML risk report."""
        # Calculate date range
        start_date = returns.index[0].strftime("%Y-%m-%d")
        end_date = returns.index[-1].strftime("%Y-%m-%d")
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{strategy_name} Risk Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ margin-bottom: 10px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>{strategy_name} Risk Report</h1>
            <p>Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Period: {start_date} to {end_date}</p>
            
            <h2>Risk Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Value at Risk (99%)</td><td class="negative">{var_99:.2%}</td></tr>
                <tr><td>Conditional Value at Risk (99%)</td><td class="negative">{cvar_99:.2%}</td></tr>
                <tr><td>Current Volatility</td><td>{rolling_vol.iloc[-1]:.2%}</td></tr>
                <tr><td>Average Volatility</td><td>{rolling_vol.mean():.2%}</td></tr>
                <tr><td>Maximum Volatility</td><td>{rolling_vol.max():.2%}</td></tr>
                <tr><td>Maximum Drawdown</td><td class="negative">{performance_metrics.max_drawdown:.2%}</td></tr>
                <tr><td>Skewness</td><td>{performance_metrics.skewness:.2f}</td></tr>
                <tr><td>Kurtosis</td><td>{performance_metrics.kurtosis:.2f}</td></tr>
            </table>
            
            <h2>Risk Charts</h2>
        """
        
        # Add plots section
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_content += f"""
            <div class="plot">
                <h3>Rolling Volatility</h3>
                <img src="{strategy_name}_rolling_volatility_{timestamp}.png" alt="Rolling Volatility">
            </div>
            <div class="plot">
                <h3>Return Distribution</h3>
                <img src="{strategy_name}_return_distribution_{timestamp}.png" alt="Return Distribution">
            </div>
        """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_risk_text_report(
        self, 
        performance_metrics: PerformanceMetrics,
        var_99: float,
        cvar_99: float,
        rolling_vol: pd.Series,
        strategy_name: str
    ) -> str:
        """Generate text risk report."""
        report_lines = [
            f"{strategy_name} Risk Report",
            f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Risk Metrics:",
            "-" * 20,
            f"Value at Risk (99%): {var_99:.2%}",
            f"Conditional Value at Risk (99%): {cvar_99:.2%}",
            f"Current Volatility: {rolling_vol.iloc[-1]:.2%}",
            f"Average Volatility: {rolling_vol.mean():.2%}",
            f"Maximum Volatility: {rolling_vol.max():.2%}",
            f"Maximum Drawdown: {performance_metrics.max_drawdown:.2%}",
            f"Skewness: {performance_metrics.skewness:.2f}",
            f"Kurtosis: {performance_metrics.kurtosis:.2f}"
        ]
        
        return "\n".join(report_lines)
    
    def _save_performance_plots(
        self, 
        returns: pd.Series, 
        benchmark_returns: Optional[pd.Series] = None,
        trades: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
        timestamp: str = ""
    ) -> None:
        """Save performance plots."""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Equity curve
        equity_path = os.path.join(self.output_dir, f"{strategy_name}_equity_curve_{timestamp}.png")
        self.visualizer.plot_equity_curve(
            returns, benchmark_returns, save_path=equity_path
        )
        plt.close()
        
        # Drawdown
        drawdown_path = os.path.join(self.output_dir, f"{strategy_name}_drawdown_{timestamp}.png")
        self.visualizer.plot_drawdown(returns, save_path=drawdown_path)
        plt.close()
        
        # Monthly returns heatmap
        monthly_path = os.path.join(self.output_dir, f"{strategy_name}_monthly_returns_{timestamp}.png")
        self.visualizer.plot_monthly_returns_heatmap(returns, save_path=monthly_path)
        plt.close()
        
        # Performance attribution
        if benchmark_returns is not None:
            # Create factor returns (using market returns as a simple factor)
            factor_returns = pd.DataFrame({'market': benchmark_returns})
            
            attribution_path = os.path.join(self.output_dir, f"{strategy_name}_performance_attribution_{timestamp}.png")
            self.visualizer.plot_performance_attribution(
                returns, factor_returns, save_path=attribution_path
            )
            plt.close()
        
        # Trade distribution
        if trades is not None:
            trade_path = os.path.join(self.output_dir, f"{strategy_name}_trade_distribution_{timestamp}.png")
            self.visualizer.plot_trade_distribution(trades, save_path=trade_path)
            plt.close()
    
    def _save_risk_plots(
        self, 
        returns: pd.Series, 
        strategy_name: str,
        timestamp: str = ""
    ) -> None:
        """Save risk plots."""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rolling volatility
        rolling_vol_path = os.path.join(self.output_dir, f"{strategy_name}_rolling_volatility_{timestamp}.png")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rolling_vol = returns.rolling(22).std() * np.sqrt(252)  # 22 trading days ~ 1 month
        ax.plot(rolling_vol.index, rolling_vol.values, linewidth=2)
        ax.set_title('Rolling Volatility (22-day)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Volatility', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(rolling_vol_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Return distribution
        dist_path = os.path.join(self.output_dir, f"{strategy_name}_return_distribution_{timestamp}.png")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(returns, bins=50, alpha=0.7, color='blue')
        ax.set_title('Return Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add VaR lines
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        ax.axvline(x=var_95, color='red', linestyle='--', alpha=0.7, label=f'VaR 95%: {var_95:.2%}')
        ax.axvline(x=var_99, color='darkred', linestyle='--', alpha=0.7, label=f'VaR 99%: {var_99:.2%}')
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(dist_path, dpi=100, bbox_inches='tight')
        plt.close()


# Global report generator instance
_report_generator = None

def get_report_generator(output_dir: str = "reports") -> ReportGenerator:
    """
    Get the global report generator instance.
    
    Args:
        output_dir: Directory to save reports
        
    Returns:
        ReportGenerator instance
    """
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator(output_dir=output_dir)
    return _report_generator