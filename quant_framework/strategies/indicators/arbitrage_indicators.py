"""
Arbitrage indicators module.

This module contains technical indicators and calculations specific to arbitrage strategies,
including spread calculations, hedge ratios, z-scores, and other statistical measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


class ArbitrageIndicators:
    """
    Collection of technical indicators for arbitrage strategies.
    """
    
    @staticmethod
    def calculate_spread_and_hedge_ratio(
        symbol1_data: pd.Series, 
        symbol2_data: pd.Series,
        method: str = "ols"
    ) -> Tuple[pd.Series, float, float, Dict[str, Any]]:
        """
        Calculate spread and hedge ratio between two symbols.
        
        Args:
            symbol1_data: Price series for symbol 1
            symbol2_data: Price series for symbol 2
            method: Method for calculating hedge ratio ("ols", "kalman", "rolling")
            
        Returns:
            Tuple of (spread, hedge_ratio, correlation, additional_info)
        """
        # Ensure data is aligned
        aligned_data = pd.concat([symbol1_data, symbol2_data], axis=1, join='inner')
        aligned_data.columns = ['symbol1', 'symbol2']
        
        y = aligned_data['symbol1'].values
        x = aligned_data['symbol2'].values
        
        # Calculate correlation
        correlation = np.corrcoef(y, x)[0, 1]
        
        # Calculate hedge ratio based on method
        if method == "ols":
            # Ordinary Least Squares using numpy
            # Add a column of ones for the intercept
            x_with_intercept = np.vstack([x, np.ones(len(x))]).T
            
            # Calculate coefficients using least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(x_with_intercept, y, rcond=None)
            hedge_ratio = coeffs[0]
            intercept = coeffs[1]
            
            # Calculate spread
            spread = aligned_data['symbol1'] - hedge_ratio * aligned_data['symbol2'] - intercept
            
            # Calculate R-squared
            y_pred = hedge_ratio * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            additional_info = {
                "intercept": intercept,
                "r_squared": r_squared,
                "method": "ols"
            }
            
        elif method == "kalman":
            # Kalman Filter (simplified implementation)
            # In a full implementation, we would use a proper Kalman filter
            # For now, we'll use a rolling regression as a proxy
            window = min(30, len(x) // 2)
            rolling_model = ArbitrageIndicators._rolling_regression(
                aligned_data['symbol1'], 
                aligned_data['symbol2'], 
                window
            )
            hedge_ratio = rolling_model.iloc[-1]
            intercept = 0  # Simplified
            
            # Calculate spread
            spread = aligned_data['symbol1'] - hedge_ratio * aligned_data['symbol2'] - intercept
            
            additional_info = {
                "intercept": intercept,
                "method": "kalman_simplified",
                "window": window
            }
            
        elif method == "rolling":
            # Rolling regression
            window = min(30, len(x) // 2)
            rolling_model = ArbitrageIndicators._rolling_regression(
                aligned_data['symbol1'], 
                aligned_data['symbol2'], 
                window
            )
            hedge_ratio = rolling_model.iloc[-1]
            intercept = 0  # Simplified
            
            # Calculate spread
            spread = aligned_data['symbol1'] - hedge_ratio * aligned_data['symbol2'] - intercept
            
            additional_info = {
                "intercept": intercept,
                "method": "rolling",
                "window": window
            }
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return spread, hedge_ratio, correlation, additional_info
    
    @staticmethod
    def _rolling_regression(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
        """
        Calculate rolling regression coefficients.
        
        Args:
            y: Dependent variable
            x: Independent variable
            window: Rolling window size
            
        Returns:
            Series of hedge ratios
        """
        # Create a DataFrame with both series
        df = pd.DataFrame({'y': y, 'x': x})
        
        # Calculate rolling hedge ratios
        hedge_ratios = pd.Series(index=y.index, dtype=float)
        
        for i in range(window, len(df)):
            window_y = df['y'].iloc[i-window:i].values
            window_x = df['x'].iloc[i-window:i].values
            
            # Calculate regression using numpy
            # Add a column of ones for the intercept
            x_with_intercept = np.vstack([window_x, np.ones(len(window_x))]).T
            
            # Calculate coefficients using least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(x_with_intercept, window_y, rcond=None)
            hedge_ratios.iloc[i] = coeffs[0]
        
        return hedge_ratios
    
    @staticmethod
    def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate z-score of a spread.
        
        Args:
            spread: Spread series
            window: Rolling window for mean and std calculation
            
        Returns:
            Z-score series
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    @staticmethod
    def calculate_half_life(spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion for a spread.
        
        Args:
            spread: Spread series
            
        Returns:
            Half-life in periods
        """
        # Calculate lagged spread
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align the series
        aligned_data = pd.concat([spread_lag, spread_diff], axis=1, join='inner')
        aligned_data.columns = ['spread_lag', 'spread_diff']
        
        if len(aligned_data) < 10:
            return np.nan
        
        # Calculate regression using numpy
        x = aligned_data['spread_lag'].values
        y = aligned_data['spread_diff'].values
        
        # Add a column of ones for the intercept
        x_with_intercept = np.vstack([x, np.ones(len(x))]).T
        
        # Calculate coefficients using least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(x_with_intercept, y, rcond=None)
        lambda_coeff = coeffs[0]
        
        # Calculate half-life
        # half_life = -ln(2) / lambda, where lambda is the regression coefficient
        if lambda_coeff >= 0:
            return np.nan  # No mean reversion
        
        half_life = -np.log(2) / lambda_coeff
        return half_life
    
    @staticmethod
    def calculate_cointegration(
        symbol1_data: pd.Series, 
        symbol2_data: pd.Series,
        maxlag: int = 1,
        test_method: str = "engle-granger"
    ) -> Dict[str, Any]:
        """
        Test for cointegration between two symbols.
        
        Args:
            symbol1_data: Price series for symbol 1
            symbol2_data: Price series for symbol 2
            maxlag: Maximum lag for the test
            test_method: Method for cointegration test
            
        Returns:
            Dictionary with cointegration test results
        """
        # Ensure data is aligned
        aligned_data = pd.concat([symbol1_data, symbol2_data], axis=1, join='inner')
        aligned_data.columns = ['symbol1', 'symbol2']
        
        if len(aligned_data) < 30:
            return {
                "is_cointegrated": False,
                "p_value": 1.0,
                "test_statistic": 0.0,
                "critical_values": {},
                "error": "Not enough data for cointegration test"
            }
        
        try:
            if test_method == "engle-granger":
                # Engle-Granger test
                from statsmodels.tsa.stattools import coint
                
                result = coint(
                    aligned_data['symbol1'], 
                    aligned_data['symbol2'], 
                    maxlag=maxlag
                )
                
                test_statistic = result[0]
                p_value = result[1]
                critical_values = result[2]
                
                # Determine if cointegrated (typically at 5% significance level)
                is_cointegrated = p_value < 0.05
                
                return {
                    "is_cointegrated": is_cointegrated,
                    "p_value": p_value,
                    "test_statistic": test_statistic,
                    "critical_values": {
                        "1%": critical_values[0],
                        "5%": critical_values[1],
                        "10%": critical_values[2]
                    },
                    "method": "engle-granger"
                }
            
            else:
                return {
                    "is_cointegrated": False,
                    "p_value": 1.0,
                    "test_statistic": 0.0,
                    "critical_values": {},
                    "error": f"Unknown test method: {test_method}"
                }
                
        except Exception as e:
            return {
                "is_cointegrated": False,
                "p_value": 1.0,
                "test_statistic": 0.0,
                "critical_values": {},
                "error": str(e)
            }
    
    @staticmethod
    def calculate_triangular_arbitrage_opportunity(
        symbol1_price: float,
        symbol2_price: float,
        symbol3_price: float,
        fee_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Calculate triangular arbitrage opportunity.
        
        Args:
            symbol1_price: Price of symbol 1
            symbol2_price: Price of symbol 2
            symbol3_price: Price of symbol 3
            fee_rate: Trading fee rate
            
        Returns:
            Dictionary with arbitrage opportunity details
        """
        # Calculate implied product
        implied_product = symbol1_price * symbol2_price * symbol3_price
        
        # Calculate potential profit after fees
        # For triangular arbitrage, we need to make 3 trades, so 3 times the fee
        total_fee_rate = 3 * fee_rate
        
        # Calculate profit percentages
        if implied_product > 1:
            # Arbitrage opportunity in one direction
            profit_pct = implied_product - 1 - total_fee_rate
            direction = "forward"
        else:
            # Arbitrage opportunity in the opposite direction
            profit_pct = 1 - implied_product - total_fee_rate
            direction = "reverse"
        
        # Determine if there's a profitable opportunity
        is_opportunity = profit_pct > 0
        
        return {
            "is_opportunity": is_opportunity,
            "profit_pct": profit_pct,
            "direction": direction,
            "implied_product": implied_product,
            "total_fee_rate": total_fee_rate
        }
    
    @staticmethod
    def calculate_cross_exchange_arbitrage_opportunity(
        exchange1_price: float,
        exchange2_price: float,
        fee_rate: float = 0.001
    ) -> Dict[str, Any]:
        """
        Calculate cross-exchange arbitrage opportunity.
        
        Args:
            exchange1_price: Price on exchange 1
            exchange2_price: Price on exchange 2
            fee_rate: Trading fee rate
            
        Returns:
            Dictionary with arbitrage opportunity details
        """
        # Calculate price difference
        price_diff = abs(exchange1_price - exchange2_price)
        avg_price = (exchange1_price + exchange2_price) / 2
        price_diff_pct = price_diff / avg_price
        
        # Calculate potential profit after fees
        # For cross-exchange arbitrage, we need to make 2 trades, so 2 times the fee
        total_fee_rate = 2 * fee_rate
        
        # Calculate profit percentage
        profit_pct = price_diff_pct - total_fee_rate
        
        # Determine if there's a profitable opportunity
        is_opportunity = profit_pct > 0
        
        # Determine direction
        if exchange1_price < exchange2_price:
            direction = "buy_on_exchange1_sell_on_exchange2"
            buy_exchange = 1
            sell_exchange = 2
            buy_price = exchange1_price
            sell_price = exchange2_price
        else:
            direction = "buy_on_exchange2_sell_on_exchange1"
            buy_exchange = 2
            sell_exchange = 1
            buy_price = exchange2_price
            sell_price = exchange1_price
        
        return {
            "is_opportunity": is_opportunity,
            "profit_pct": profit_pct,
            "direction": direction,
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "price_diff_pct": price_diff_pct,
            "total_fee_rate": total_fee_rate
        }
    
    @staticmethod
    def calculate_volatility(
        price_data: pd.Series,
        window: int = 20,
        method: str = "std"
    ) -> pd.Series:
        """
        Calculate volatility of price data.
        
        Args:
            price_data: Price series
            window: Rolling window size
            method: Method for volatility calculation ("std", "parkinson", "garman_klass")
            
        Returns:
            Volatility series
        """
        if method == "std":
            # Standard deviation of returns
            returns = price_data.pct_change()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            
        elif method == "parkinson":
            # Parkinson estimator (requires high/low data)
            # This is a simplified implementation
            returns = price_data.pct_change()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)
            
        elif method == "garman_klass":
            # Garman-Klass estimator (requires OHLC data)
            # This is a simplified implementation
            returns = price_data.pct_change()
            volatility = returns.rolling(window=window).std() * np.sqrt(252)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return volatility
    
    @staticmethod
    def calculate_all_arbitrage_indicators(
        symbol1_data: pd.Series,
        symbol2_data: pd.Series,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate all arbitrage indicators for a pair of symbols.
        
        Args:
            symbol1_data: Price series for symbol 1
            symbol2_data: Price series for symbol 2
            parameters: Strategy parameters
            
        Returns:
            Dictionary with all calculated indicators
        """
        # Extract parameters
        hedge_ratio_method = parameters.get("hedge_ratio_method", "ols")
        zscore_window = parameters.get("zscore_period", 20)
        volatility_window = parameters.get("volatility_lookback", 20)
        volatility_method = parameters.get("volatility_method", "std")
        
        # Calculate spread and hedge ratio
        spread, hedge_ratio, correlation, hedge_info = ArbitrageIndicators.calculate_spread_and_hedge_ratio(
            symbol1_data, symbol2_data, hedge_ratio_method
        )
        
        # Calculate z-score
        zscore = ArbitrageIndicators.calculate_zscore(spread, zscore_window)
        
        # Calculate half-life
        half_life = ArbitrageIndicators.calculate_half_life(spread)
        
        # Calculate cointegration
        cointegration = ArbitrageIndicators.calculate_cointegration(symbol1_data, symbol2_data)
        
        # Calculate volatilities
        symbol1_volatility = ArbitrageIndicators.calculate_volatility(
            symbol1_data, volatility_window, volatility_method
        )
        symbol2_volatility = ArbitrageIndicators.calculate_volatility(
            symbol2_data, volatility_window, volatility_method
        )
        
        # Get latest values
        latest_zscore = zscore.iloc[-1] if not zscore.empty else np.nan
        latest_spread = spread.iloc[-1] if not spread.empty else np.nan
        latest_symbol1_vol = symbol1_volatility.iloc[-1] if not symbol1_volatility.empty else np.nan
        latest_symbol2_vol = symbol2_volatility.iloc[-1] if not symbol2_volatility.empty else np.nan
        
        return {
            "spread": spread,
            "hedge_ratio": hedge_ratio,
            "correlation": correlation,
            "zscore": zscore,
            "half_life": half_life,
            "cointegration": cointegration,
            "symbol1_volatility": symbol1_volatility,
            "symbol2_volatility": symbol2_volatility,
            "latest_values": {
                "zscore": latest_zscore,
                "spread": latest_spread,
                "symbol1_volatility": latest_symbol1_vol,
                "symbol2_volatility": latest_symbol2_vol
            },
            "hedge_info": hedge_info
        }