"""
Strategies module for the quantitative trading framework.

This module contains all trading strategies that can be used with the framework.
Each strategy inherits from the StrategyBase class and implements the calculate_signals method.
"""

from .base.signal_types import Signal, SignalType, SignalStrength
from .base.strategy_state import StrategyState
from quant_framework.strategies.strategy_base import StrategyBase
from quant_framework.strategies.ema_rsi_strategy import EmaRsiStrategy

__all__ = [
    # Base classes and utilities
    "StrategyBase",
    "Signal",
    "SignalType",
    "SignalStrength",
    "StrategyState",
    
    # Strategies
    "EmaRsiStrategy"
]