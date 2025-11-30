"""
数据模型定义

定义量化交易系统中需要持久化的数据模型。
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Index, Integer, 
    Numeric, String, Text, UniqueConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from .connection import Base

# 如果使用PostgreSQL，可以启用UUID支持
try:
    import uuid
    UUID_TYPE = UUID(as_uuid=True)
except ImportError:
    UUID_TYPE = String(36)


class Strategy(Base):
    """策略模型"""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    class_name = Column(String(100), nullable=False)
    parameters = Column(Text)  # JSON格式存储参数
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 关系
    backtests = relationship("Backtest", back_populates="strategy")
    positions = relationship("Position", back_populates="strategy")
    trades = relationship("Trade", back_populates="strategy")
    
    def __repr__(self):
        return f"<Strategy(id={self.id}, name='{self.name}')>"


class Backtest(Base):
    """回测模型"""
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(100), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Numeric(20, 8), nullable=False)
    final_capital = Column(Numeric(20, 8))
    total_return = Column(Numeric(10, 6))
    annual_return = Column(Numeric(10, 6))
    max_drawdown = Column(Numeric(10, 6))
    sharpe_ratio = Column(Numeric(10, 6))
    sortino_ratio = Column(Numeric(10, 6))
    win_rate = Column(Numeric(5, 4))
    profit_factor = Column(Numeric(10, 4))
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    status = Column(String(20), default="completed")  # running, completed, failed
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 关系
    strategy = relationship("Strategy", back_populates="backtests")
    trades = relationship("Trade", back_populates="backtest")
    daily_returns = relationship("DailyReturn", back_populates="backtest")
    
    def __repr__(self):
        return f"<Backtest(id={self.id}, name='{self.name}', strategy_id={self.strategy_id})>"


class Symbol(Base):
    """交易标的模型"""
    __tablename__ = "symbols"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True)
    name = Column(String(100))
    asset_class = Column(String(20))  # stock, futures, forex, crypto
    exchange = Column(String(20))
    currency = Column(String(10))
    lot_size = Column(Integer, default=1)
    tick_size = Column(Numeric(10, 8))
    contract_size = Column(Numeric(20, 8))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 关系
    positions = relationship("Position", back_populates="symbol")
    trades = relationship("Trade", back_populates="symbol")
    bars = relationship("Bar", back_populates="symbol")
    
    def __repr__(self):
        return f"<Symbol(id={self.id}, symbol='{self.symbol}')>"


class Position(Base):
    """持仓模型"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    quantity = Column(Numeric(20, 8), nullable=False)
    avg_price = Column(Numeric(20, 8))
    market_value = Column(Numeric(20, 8))
    unrealized_pnl = Column(Numeric(20, 8))
    realized_pnl = Column(Numeric(20, 8))
    side = Column(String(10))  # long, short
    status = Column(String(20), default="open")  # open, closed
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 关系
    strategy = relationship("Strategy", back_populates="positions")
    symbol = relationship("Symbol", back_populates="positions")
    trades = relationship("Trade", back_populates="position")
    
    def __repr__(self):
        return f"<Position(id={self.id}, symbol_id={self.symbol_id}, quantity={self.quantity})>"


class Trade(Base):
    """交易模型"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    backtest_id = Column(Integer, ForeignKey("backtests.id"))
    position_id = Column(Integer, ForeignKey("positions.id"))
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    order_id = Column(String(50))
    trade_id = Column(String(50))
    side = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8), nullable=False)
    commission = Column(Numeric(20, 8), default=0)
    slippage = Column(Numeric(20, 8), default=0)
    pnl = Column(Numeric(20, 8))
    trade_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # 关系
    strategy = relationship("Strategy", back_populates="trades")
    backtest = relationship("Backtest", back_populates="backtest")
    position = relationship("Position", back_populates="trades")
    symbol = relationship("Symbol", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol_id={self.symbol_id}, side='{self.side}', quantity={self.quantity})>"


class Bar(Base):
    """K线数据模型"""
    __tablename__ = "bars"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    datetime = Column(DateTime, nullable=False)
    open_price = Column(Numeric(20, 8), nullable=False)
    high_price = Column(Numeric(20, 8), nullable=False)
    low_price = Column(Numeric(20, 8), nullable=False)
    close_price = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(20, 8), nullable=False)
    open_interest = Column(Numeric(20, 8))
    adjusted_close = Column(Numeric(20, 8))
    created_at = Column(DateTime, default=func.now())
    
    # 关系
    symbol = relationship("Symbol", back_populates="bars")
    
    # 索引
    __table_args__ = (
        Index('ix_bars_symbol_datetime', 'symbol_id', 'datetime'),
        UniqueConstraint('symbol_id', 'datetime', name='uq_bars_symbol_datetime'),
    )
    
    def __repr__(self):
        return f"<Bar(id={self.id}, symbol_id={self.symbol_id}, datetime='{self.datetime}')>"


class DailyReturn(Base):
    """每日收益模型"""
    __tablename__ = "daily_returns"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    portfolio_value = Column(Numeric(20, 8), nullable=False)
    daily_return = Column(Numeric(10, 6))
    cumulative_return = Column(Numeric(10, 6))
    drawdown = Column(Numeric(10, 6))
    created_at = Column(DateTime, default=func.now())
    
    # 关系
    backtest = relationship("Backtest", back_populates="daily_returns")
    
    # 索引
    __table_args__ = (
        Index('ix_daily_returns_backtest_date', 'backtest_id', 'date'),
        UniqueConstraint('backtest_id', 'date', name='uq_daily_returns_backtest_date'),
    )
    
    def __repr__(self):
        return f"<DailyReturn(id={self.id}, backtest_id={self.backtest_id}, date='{self.date}')>"


class RiskMetrics(Base):
    """风险指标模型"""
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    var_95 = Column(Numeric(10, 6))  # 95% VaR
    var_99 = Column(Numeric(10, 6))  # 99% VaR
    cvar_95 = Column(Numeric(10, 6))  # 95% CVaR
    cvar_99 = Column(Numeric(10, 6))  # 99% CVaR
    volatility = Column(Numeric(10, 6))  # 波动率
    beta = Column(Numeric(10, 6))  # Beta系数
    alpha = Column(Numeric(10, 6))  # Alpha系数
    information_ratio = Column(Numeric(10, 6))  # 信息比率
    tracking_error = Column(Numeric(10, 6))  # 跟踪误差
    created_at = Column(DateTime, default=func.now())
    
    # 关系
    backtest = relationship("Backtest")
    
    # 索引
    __table_args__ = (
        Index('ix_risk_metrics_backtest_date', 'backtest_id', 'date'),
        UniqueConstraint('backtest_id', 'date', name='uq_risk_metrics_backtest_date'),
    )
    
    def __repr__(self):
        return f"<RiskMetrics(id={self.id}, backtest_id={self.backtest_id}, date='{self.date}')>"


class PerformanceMetrics(Base):
    """绩效指标模型"""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    total_return = Column(Numeric(10, 6))
    annual_return = Column(Numeric(10, 6))
    max_drawdown = Column(Numeric(10, 6))
    sharpe_ratio = Column(Numeric(10, 6))
    sortino_ratio = Column(Numeric(10, 6))
    calmar_ratio = Column(Numeric(10, 6))
    omega_ratio = Column(Numeric(10, 6))
    win_rate = Column(Numeric(5, 4))
    profit_factor = Column(Numeric(10, 4))
    recovery_factor = Column(Numeric(10, 4))
    var_95 = Column(Numeric(10, 6))
    skewness = Column(Numeric(10, 6))
    kurtosis = Column(Numeric(10, 6))
    created_at = Column(DateTime, default=func.now())
    
    # 关系
    backtest = relationship("Backtest")
    
    # 索引
    __table_args__ = (
        Index('ix_performance_metrics_backtest_date', 'backtest_id', 'date'),
        UniqueConstraint('backtest_id', 'date', name='uq_performance_metrics_backtest_date'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetrics(id={self.id}, backtest_id={self.backtest_id}, date='{self.date}')>"


class StrategyParameter(Base):
    """策略参数模型"""
    __tablename__ = "strategy_parameters"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    backtest_id = Column(Integer, ForeignKey("backtests.id"))
    name = Column(String(50), nullable=False)
    value = Column(String(200), nullable=False)
    data_type = Column(String(20), default="string")  # string, int, float, bool, list, dict
    description = Column(Text)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 关系
    strategy = relationship("Strategy")
    backtest = relationship("Backtest")
    
    # 索引
    __table_args__ = (
        Index('ix_strategy_parameters_strategy_name', 'strategy_id', 'name'),
        UniqueConstraint('strategy_id', 'backtest_id', 'name', name='uq_strategy_parameters_strategy_backtest_name'),
    )
    
    def __repr__(self):
        return f"<StrategyParameter(id={self.id}, strategy_id={self.strategy_id}, name='{self.name}')>"


class Order(Base):
    """订单模型"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), nullable=False, unique=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    backtest_id = Column(Integer, ForeignKey("backtests.id"))
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False)
    side = Column(String(10), nullable=False)  # buy, sell
    order_type = Column(String(20), nullable=False)  # market, limit, stop, stop_limit
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8))
    stop_price = Column(Numeric(20, 8))
    filled_quantity = Column(Numeric(20, 8), default=0)
    avg_fill_price = Column(Numeric(20, 8))
    status = Column(String(20), default="submitted")  # submitted, accepted, partially_filled, filled, cancelled, rejected
    submit_time = Column(DateTime, nullable=False)
    fill_time = Column(DateTime)
    cancel_time = Column(DateTime)
    commission = Column(Numeric(20, 8), default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 关系
    strategy = relationship("Strategy")
    backtest = relationship("Backtest")
    symbol = relationship("Symbol")
    
    def __repr__(self):
        return f"<Order(id={self.id}, order_id='{self.order_id}', symbol_id={self.symbol_id})>"