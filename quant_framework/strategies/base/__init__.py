"""
Base strategy components for the quantitative trading framework.

This module contains the base classes and utilities that all trading strategies inherit from.
"""

from .signal_types import Signal, SignalType, SignalStrength
from .strategy_state import StrategyState
from .strategy_data_manager import StrategyDataManager
from .strategy_executor import StrategyExecutor
from .strategy_performance_manager import StrategyPerformanceManager

__all__ = [
    # Signal-related
    "Signal",
    "SignalType",
    "SignalStrength",
    
    # Strategy base
    "StrategyState",
    
    # Strategy components
    "StrategyDataManager",
    "StrategyExecutor",
    "StrategyPerformanceManager"
]
