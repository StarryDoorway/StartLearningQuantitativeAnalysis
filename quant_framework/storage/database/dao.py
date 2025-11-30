"""
数据访问层（DAO）

提供对数据库的CRUD操作，封装数据访问逻辑。
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.orm import Session

from .connection import get_session
from .models import (
    Backtest, Bar, DailyReturn, Order, PerformanceMetrics, Position,
    RiskMetrics, Strategy, StrategyParameter, Symbol, Trade
)


class BaseDAO:
    """基础数据访问对象"""
    
    def __init__(self, model_class):
        self.model_class = model_class
    
    def create(self, **kwargs) -> Any:
        """创建记录"""
        with get_session() as session:
            instance = self.model_class(**kwargs)
            session.add(instance)
            session.flush()
            session.refresh(instance)
            return instance
    
    def get_by_id(self, id: int) -> Optional[Any]:
        """根据ID获取记录"""
        with get_session() as session:
            return session.query(self.model_class).filter(self.model_class.id == id).first()
    
    def get_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Any]:
        """获取所有记录"""
        with get_session() as session:
            query = session.query(self.model_class)
            if offset is not None:
                query = query.offset(offset)
            if limit is not None:
                query = query.limit(limit)
            return query.all()
    
    def update(self, id: int, **kwargs) -> Optional[Any]:
        """更新记录"""
        with get_session() as session:
            instance = session.query(self.model_class).filter(self.model_class.id == id).first()
            if instance:
                for key, value in kwargs.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                session.flush()
                session.refresh(instance)
            return instance
    
    def delete(self, id: int) -> bool:
        """删除记录"""
        with get_session() as session:
            instance = session.query(self.model_class).filter(self.model_class.id == id).first()
            if instance:
                session.delete(instance)
                return True
            return False
    
    def count(self) -> int:
        """获取记录总数"""
        with get_session() as session:
            return session.query(self.model_class).count()


class StrategyDAO(BaseDAO):
    """策略数据访问对象"""
    
    def __init__(self):
        super().__init__(Strategy)
    
    def get_by_name(self, name: str) -> Optional[Strategy]:
        """根据名称获取策略"""
        with get_session() as session:
            return session.query(Strategy).filter(Strategy.name == name).first()
    
    def get_active_strategies(self) -> List[Strategy]:
        """获取所有活跃策略"""
        with get_session() as session:
            return session.query(Strategy).filter(Strategy.is_active == True).all()
    
    def create_strategy(self, name: str, class_name: str, description: str = "", 
                       parameters: Optional[Dict] = None) -> Strategy:
        """创建策略"""
        params_json = json.dumps(parameters) if parameters else None
        return self.create(
            name=name,
            description=description,
            class_name=class_name,
            parameters=params_json
        )
    
    def update_strategy(self, id: int, name: Optional[str] = None, 
                       description: Optional[str] = None, 
                       parameters: Optional[Dict] = None,
                       is_active: Optional[bool] = None) -> Optional[Strategy]:
        """更新策略"""
        update_data = {}
        if name is not None:
            update_data['name'] = name
        if description is not None:
            update_data['description'] = description
        if parameters is not None:
            update_data['parameters'] = json.dumps(parameters)
        if is_active is not None:
            update_data['is_active'] = is_active
        
        return self.update(id, **update_data)
    
    def get_strategy_parameters(self, id: int) -> Optional[Dict]:
        """获取策略参数"""
        strategy = self.get_by_id(id)
        if strategy and strategy.parameters:
            try:
                return json.loads(strategy.parameters)
            except json.JSONDecodeError:
                return None
        return None


class BacktestDAO(BaseDAO):
    """回测数据访问对象"""
    
    def __init__(self):
        super().__init__(Backtest)
    
    def get_by_strategy_id(self, strategy_id: int) -> List[Backtest]:
        """根据策略ID获取回测"""
        with get_session() as session:
            return session.query(Backtest).filter(Backtest.strategy_id == strategy_id).all()
    
    def get_latest_backtest(self, strategy_id: int) -> Optional[Backtest]:
        """获取策略的最新回测"""
        with get_session() as session:
            return session.query(Backtest).filter(
                Backtest.strategy_id == strategy_id
            ).order_by(desc(Backtest.created_at)).first()
    
    def get_backtests_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Backtest]:
        """根据日期范围获取回测"""
        with get_session() as session:
            return session.query(Backtest).filter(
                and_(
                    Backtest.start_date >= start_date,
                    Backtest.end_date <= end_date
                )
            ).all()
    
    def get_performance_summary(self, backtest_id: int) -> Optional[Dict]:
        """获取回测绩效摘要"""
        backtest = self.get_by_id(backtest_id)
        if not backtest:
            return None
        
        return {
            "id": backtest.id,
            "name": backtest.name,
            "strategy_id": backtest.strategy_id,
            "start_date": backtest.start_date,
            "end_date": backtest.end_date,
            "initial_capital": float(backtest.initial_capital),
            "final_capital": float(backtest.final_capital) if backtest.final_capital else None,
            "total_return": float(backtest.total_return) if backtest.total_return else None,
            "annual_return": float(backtest.annual_return) if backtest.annual_return else None,
            "max_drawdown": float(backtest.max_drawdown) if backtest.max_drawdown else None,
            "sharpe_ratio": float(backtest.sharpe_ratio) if backtest.sharpe_ratio else None,
            "sortino_ratio": float(backtest.sortino_ratio) if backtest.sortino_ratio else None,
            "win_rate": float(backtest.win_rate) if backtest.win_rate else None,
            "profit_factor": float(backtest.profit_factor) if backtest.profit_factor else None,
            "total_trades": backtest.total_trades,
            "winning_trades": backtest.winning_trades,
            "losing_trades": backtest.losing_trades,
            "status": backtest.status
        }


class SymbolDAO(BaseDAO):
    """交易标的数据访问对象"""
    
    def __init__(self):
        super().__init__(Symbol)
    
    def get_by_symbol(self, symbol: str) -> Optional[Symbol]:
        """根据代码获取交易标的"""
        with get_session() as session:
            return session.query(Symbol).filter(Symbol.symbol == symbol).first()
    
    def get_by_exchange(self, exchange: str) -> List[Symbol]:
        """根据交易所获取交易标的"""
        with get_session() as session:
            return session.query(Symbol).filter(Symbol.exchange == exchange).all()
    
    def get_by_asset_class(self, asset_class: str) -> List[Symbol]:
        """根据资产类别获取交易标的"""
        with get_session() as session:
            return session.query(Symbol).filter(Symbol.asset_class == asset_class).all()
    
    def get_active_symbols(self) -> List[Symbol]:
        """获取所有活跃交易标的"""
        with get_session() as session:
            return session.query(Symbol).filter(Symbol.is_active == True).all()


class PositionDAO(BaseDAO):
    """持仓数据访问对象"""
    
    def __init__(self):
        super().__init__(Position)
    
    def get_by_strategy_id(self, strategy_id: int) -> List[Position]:
        """根据策略ID获取持仓"""
        with get_session() as session:
            return session.query(Position).filter(Position.strategy_id == strategy_id).all()
    
    def get_by_symbol_id(self, symbol_id: int) -> List[Position]:
        """根据标的ID获取持仓"""
        with get_session() as session:
            return session.query(Position).filter(Position.symbol_id == symbol_id).all()
    
    def get_open_positions(self, strategy_id: Optional[int] = None) -> List[Position]:
        """获取未平仓持仓"""
        with get_session() as session:
            query = session.query(Position).filter(Position.status == "open")
            if strategy_id is not None:
                query = query.filter(Position.strategy_id == strategy_id)
            return query.all()
    
    def get_closed_positions(self, strategy_id: Optional[int] = None) -> List[Position]:
        """获取已平仓持仓"""
        with get_session() as session:
            query = session.query(Position).filter(Position.status == "closed")
            if strategy_id is not None:
                query = query.filter(Position.strategy_id == strategy_id)
            return query.all()


class TradeDAO(BaseDAO):
    """交易数据访问对象"""
    
    def __init__(self):
        super().__init__(Trade)
    
    def get_by_strategy_id(self, strategy_id: int) -> List[Trade]:
        """根据策略ID获取交易"""
        with get_session() as session:
            return session.query(Trade).filter(Trade.strategy_id == strategy_id).all()
    
    def get_by_backtest_id(self, backtest_id: int) -> List[Trade]:
        """根据回测ID获取交易"""
        with get_session() as session:
            return session.query(Trade).filter(Trade.backtest_id == backtest_id).all()
    
    def get_by_symbol_id(self, symbol_id: int) -> List[Trade]:
        """根据标的ID获取交易"""
        with get_session() as session:
            return session.query(Trade).filter(Trade.symbol_id == symbol_id).all()
    
    def get_by_position_id(self, position_id: int) -> List[Trade]:
        """根据持仓ID获取交易"""
        with get_session() as session:
            return session.query(Trade).filter(Trade.position_id == position_id).all()
    
    def get_trades_by_date_range(self, start_date: datetime, end_date: datetime,
                                strategy_id: Optional[int] = None) -> List[Trade]:
        """根据日期范围获取交易"""
        with get_session() as session:
            query = session.query(Trade).filter(
                and_(
                    Trade.trade_time >= start_date,
                    Trade.trade_time <= end_date
                )
            )
            if strategy_id is not None:
                query = query.filter(Trade.strategy_id == strategy_id)
            return query.all()
    
    def get_trade_statistics(self, strategy_id: Optional[int] = None,
                           backtest_id: Optional[int] = None) -> Dict:
        """获取交易统计信息"""
        with get_session() as session:
            query = session.query(Trade)
            if strategy_id is not None:
                query = query.filter(Trade.strategy_id == strategy_id)
            if backtest_id is not None:
                query = query.filter(Trade.backtest_id == backtest_id)
            
            total_trades = query.count()
            if total_trades == 0:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                    "max_profit": 0,
                    "max_loss": 0
                }
            
            winning_trades = query.filter(Trade.pnl > 0).count()
            losing_trades = query.filter(Trade.pnl < 0).count()
            
            pnl_stats = session.query(
                func.sum(Trade.pnl).label("total_pnl"),
                func.avg(Trade.pnl).label("avg_pnl"),
                func.max(Trade.pnl).label("max_profit"),
                func.min(Trade.pnl).label("max_loss")
            ).filter(
                Trade.pnl.isnot(None)
            )
            
            if strategy_id is not None:
                pnl_stats = pnl_stats.filter(Trade.strategy_id == strategy_id)
            if backtest_id is not None:
                pnl_stats = pnl_stats.filter(Trade.backtest_id == backtest_id)
            
            pnl_result = pnl_stats.first()
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
                "total_pnl": float(pnl_result.total_pnl) if pnl_result.total_pnl else 0,
                "avg_pnl": float(pnl_result.avg_pnl) if pnl_result.avg_pnl else 0,
                "max_profit": float(pnl_result.max_profit) if pnl_result.max_profit else 0,
                "max_loss": float(pnl_result.max_loss) if pnl_result.max_loss else 0
            }


class BarDAO(BaseDAO):
    """K线数据访问对象"""
    
    def __init__(self):
        super().__init__(Bar)
    
    def get_by_symbol_id(self, symbol_id: int, limit: Optional[int] = None) -> List[Bar]:
        """根据标的ID获取K线数据"""
        with get_session() as session:
            query = session.query(Bar).filter(Bar.symbol_id == symbol_id).order_by(desc(Bar.datetime))
            if limit is not None:
                query = query.limit(limit)
            return query.all()
    
    def get_by_date_range(self, symbol_id: int, start_date: datetime, 
                         end_date: datetime) -> List[Bar]:
        """根据日期范围获取K线数据"""
        with get_session() as session:
            return session.query(Bar).filter(
                and_(
                    Bar.symbol_id == symbol_id,
                    Bar.datetime >= start_date,
                    Bar.datetime <= end_date
                )
            ).order_by(Bar.datetime).all()
    
    def get_latest_bar(self, symbol_id: int) -> Optional[Bar]:
        """获取最新K线数据"""
        with get_session() as session:
            return session.query(Bar).filter(
                Bar.symbol_id == symbol_id
            ).order_by(desc(Bar.datetime)).first()
    
    def count_bars(self, symbol_id: int) -> int:
        """获取K线数据总数"""
        with get_session() as session:
            return session.query(Bar).filter(Bar.symbol_id == symbol_id).count()
    
    def bulk_insert(self, bars_data: List[Dict]) -> bool:
        """批量插入K线数据"""
        try:
            with get_session() as session:
                session.bulk_insert_mappings(Bar, bars_data)
                return True
        except Exception as e:
            print(f"批量插入K线数据失败: {e}")
            return False


class DailyReturnDAO(BaseDAO):
    """每日收益数据访问对象"""
    
    def __init__(self):
        super().__init__(DailyReturn)
    
    def get_by_backtest_id(self, backtest_id: int) -> List[DailyReturn]:
        """根据回测ID获取每日收益"""
        with get_session() as session:
            return session.query(DailyReturn).filter(
                DailyReturn.backtest_id == backtest_id
            ).order_by(DailyReturn.date).all()
    
    def get_by_date_range(self, backtest_id: int, start_date: datetime, 
                         end_date: datetime) -> List[DailyReturn]:
        """根据日期范围获取每日收益"""
        with get_session() as session:
            return session.query(DailyReturn).filter(
                and_(
                    DailyReturn.backtest_id == backtest_id,
                    DailyReturn.date >= start_date,
                    DailyReturn.date <= end_date
                )
            ).order_by(DailyReturn.date).all()
    
    def get_max_drawdown(self, backtest_id: int) -> Optional[float]:
        """获取最大回撤"""
        with get_session() as session:
            result = session.query(func.max(DailyReturn.drawdown)).filter(
                DailyReturn.backtest_id == backtest_id
            ).first()
            return float(result[0]) if result and result[0] else None


# 创建DAO实例
strategy_dao = StrategyDAO()
backtest_dao = BacktestDAO()
symbol_dao = SymbolDAO()
position_dao = PositionDAO()
trade_dao = TradeDAO()
bar_dao = BarDAO()
daily_return_dao = DailyReturnDAO()