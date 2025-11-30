"""
动量策略技术指标计算模块

负责计算动量策略所需的各种技术指标，包括SMA、EMA、MACD、RSI、ADX等。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class MomentumIndicators:
    """动量策略技术指标计算器"""
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        计算简单移动平均线(SMA)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            SMA序列
        """
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        计算指数移动平均线(EMA)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            EMA序列
        """
        return data['close'].ewm(span=period).mean()
    
    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD指标
        
        Args:
            data: 价格数据
            fast: 快速EMA周期
            slow: 慢速EMA周期
            signal: 信号线周期
            
        Returns:
            包含MACD线、信号线和柱状图的字典
        """
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算相对强弱指数(RSI)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            RSI序列
        """
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算平均方向性运动指数(ADX)
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            ADX序列
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = np.where((high - high.shift()) > (low.shift() - low), 
                          np.maximum(high - high.shift(), 0), 0)
        dm_minus = np.where((low.shift() - low) > (high - high.shift()), 
                           np.maximum(low.shift() - low, 0), 0)
        
        # Convert to pandas Series
        dm_plus = pd.Series(dm_plus, index=data.index)
        dm_minus = pd.Series(dm_minus, index=data.index)
        
        # Calculate Smoothed Values
        atr = tr.rolling(window=period).mean()
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_momentum(data: pd.DataFrame, period: int = 10) -> pd.Series:
        """
        计算动量指标
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            动量序列
        """
        return data['close'].pct_change(period)
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        计算波动率指标
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            波动率序列
        """
        returns = data['close'].pct_change()
        return returns.rolling(window=period).std()
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        计算所有技术指标
        
        Args:
            data: 价格数据
            parameters: 策略参数
            
        Returns:
            包含所有指标的字典
        """
        indicators = {}
        
        # SMA indicators
        for period in parameters.get("sma_periods", [10, 20, 50]):
            indicators[f'sma_{period}'] = MomentumIndicators.calculate_sma(data, period)
        
        # EMA indicators
        for period in parameters.get("ema_periods", [12, 26]):
            indicators[f'ema_{period}'] = MomentumIndicators.calculate_ema(data, period)
        
        # MACD indicators
        macd_params = parameters.get("macd_params", {"fast": 12, "slow": 26, "signal": 9})
        macd_indicators = MomentumIndicators.calculate_macd(
            data, 
            macd_params.get("fast", 12), 
            macd_params.get("slow", 26), 
            macd_params.get("signal", 9)
        )
        indicators.update(macd_indicators)
        
        # RSI indicator
        rsi_period = parameters.get("rsi_period", 14)
        indicators['rsi'] = MomentumIndicators.calculate_rsi(data, rsi_period)
        
        # ADX indicator
        adx_period = parameters.get("adx_period", 14)
        indicators['adx'] = MomentumIndicators.calculate_adx(data, adx_period)
        
        # Momentum indicator
        momentum_period = parameters.get("momentum_period", 10)
        indicators['momentum'] = MomentumIndicators.calculate_momentum(data, momentum_period)
        
        # Volatility indicator
        volatility_period = parameters.get("volatility_period", 20)
        indicators['volatility'] = MomentumIndicators.calculate_volatility(data, volatility_period)
        
        return indicators