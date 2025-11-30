"""
量化交易框架 (Quantitative Trading Framework)

一个功能完整、模块化设计的Python量化交易框架，支持策略开发、回测、风险管理和实盘交易。
"""

__version__ = "1.0.0"
__author__ = "Quantitative Trading Team"

# 导入核心模块
from . import core
from . import data
from . import strategies
from . import execution
from . import utils

# 导入常用类和函数
from .strategies import StrategyBase, Signal, SignalType
from .core.backtest_engine.backtest_engine import BacktestEngine
from .data import DataManager
from .execution import PortfolioManager, OrderManager
from .core.risk_engine.risk_engine import RiskEngine

__all__ = [
    "core",
    "data",
    "strategies",
    "execution",
    "utils",
    "StrategyBase",
    "Signal",
    "SignalType",
    "BacktestEngine",
    "DataManager",
    "PortfolioManager",
    "OrderManager",
    "RiskEngine",
]