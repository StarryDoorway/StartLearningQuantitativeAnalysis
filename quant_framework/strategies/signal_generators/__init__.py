"""
Signal generators for trading strategies.

This module contains all the signal generators used by the trading strategies.
"""

from .arbitrage_signal_generator import ArbitrageSignalGenerator
from .mean_reversion_signal_generator import MeanReversionSignalGenerator
from .rsi_divergence_signal_generator import RsiDivergenceSignalGenerator
from .trend_following_signal_generator import TrendFollowingSignalGenerator

__all__ = [
    "ArbitrageSignalGenerator",
    "MeanReversionSignalGenerator",
    "RsiDivergenceSignalGenerator",
    "TrendFollowingSignalGenerator"
]
