"""
数据仓库服务

提供高级数据访问服务，封装复杂的数据查询和业务逻辑。
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from quant_framework.utils.config_loader import get_config
from .dao import (
    backtest_dao, bar_dao, daily_return_dao, position_dao, 
    strategy_dao, symbol_dao, trade_dao
)
from .models import Backtest, Bar, DailyReturn, Position, Strategy, Symbol, Trade


class DataWarehouse:
    """数据仓库服务
    
    提供高级数据访问服务，封装复杂的数据查询和业务逻辑。
    """
    
    def __init__(self):
        self.config = get_config()
    
    # ==================== 策略相关服务 ====================
    
    def get_strategy_list(self, active_only: bool = True) -> List[Dict]:
        """获取策略列表"""
        strategies = strategy_dao.get_active_strategies() if active_only else strategy_dao.get_all()
        result = []
        for strategy in strategies:
            result.append({
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "class_name": strategy.class_name,
                "parameters": strategy_dao.get_strategy_parameters(strategy.id),
                "is_active": strategy.is_active,
                "created_at": strategy.created_at,
                "updated_at": strategy.updated_at
            })
        return result
    
    def get_strategy_detail(self, strategy_id: int) -> Optional[Dict]:
        """获取策略详情"""
        strategy = strategy_dao.get_by_id(strategy_id)
        if not strategy:
            return None
        
        # 获取最新回测
        latest_backtest = backtest_dao.get_latest_backtest(strategy_id)
        
        # 获取当前持仓
        open_positions = position_dao.get_open_positions(strategy_id)
        
        return {
            "id": strategy.id,
            "name": strategy.name,
            "description": strategy.description,
            "class_name": strategy.class_name,
            "parameters": strategy_dao.get_strategy_parameters(strategy.id),
            "is_active": strategy.is_active,
            "created_at": strategy.created_at,
            "updated_at": strategy.updated_at,
            "latest_backtest": backtest_dao.get_performance_summary(latest_backtest.id) if latest_backtest else None,
            "open_positions_count": len(open_positions),
            "total_positions": position_dao.count()
        }
    
    def create_strategy(self, name: str, class_name: str, description: str = "", 
                       parameters: Optional[Dict] = None) -> Optional[Dict]:
        """创建策略"""
        try:
            strategy = strategy_dao.create_strategy(name, class_name, description, parameters)
            return {
                "id": strategy.id,
                "name": strategy.name,
                "description": strategy.description,
                "class_name": strategy.class_name,
                "parameters": strategy_dao.get_strategy_parameters(strategy.id),
                "is_active": strategy.is_active,
                "created_at": strategy.created_at
            }
        except Exception as e:
            print(f"创建策略失败: {e}")
            return None
    
    def update_strategy(self, strategy_id: int, **kwargs) -> bool:
        """更新策略"""
        try:
            strategy = strategy_dao.update_strategy(strategy_id, **kwargs)
            return strategy is not None
        except Exception as e:
            print(f"更新策略失败: {e}")
            return False
    
    # ==================== 回测相关服务 ====================
    
    def get_backtest_list(self, strategy_id: Optional[int] = None) -> List[Dict]:
        """获取回测列表"""
        if strategy_id:
            backtests = backtest_dao.get_by_strategy_id(strategy_id)
        else:
            backtests = backtest_dao.get_all()
        
        result = []
        for backtest in backtests:
            result.append(backtest_dao.get_performance_summary(backtest.id))
        return result
    
    def get_backtest_detail(self, backtest_id: int) -> Optional[Dict]:
        """获取回测详情"""
        backtest = backtest_dao.get_by_id(backtest_id)
        if not backtest:
            return None
        
        # 获取策略信息
        strategy = strategy_dao.get_by_id(backtest.strategy_id)
        
        # 获取交易统计
        trade_stats = trade_dao.get_trade_statistics(backtest_id=backtest_id)
        
        # 获取每日收益
        daily_returns = daily_return_dao.get_by_backtest_id(backtest_id)
        
        return {
            **backtest_dao.get_performance_summary(backtest_id),
            "strategy_name": strategy.name if strategy else None,
            "trade_statistics": trade_stats,
            "daily_returns_count": len(daily_returns),
            "max_drawdown": daily_return_dao.get_max_drawdown(backtest_id)
        }
    
    def get_backtest_equity_curve(self, backtest_id: int) -> Optional[pd.DataFrame]:
        """获取回测权益曲线"""
        daily_returns = daily_return_dao.get_by_backtest_id(backtest_id)
        if not daily_returns:
            return None
        
        data = []
        for dr in daily_returns:
            data.append({
                "date": dr.date,
                "portfolio_value": float(dr.portfolio_value),
                "daily_return": float(dr.daily_return) if dr.daily_return else 0,
                "cumulative_return": float(dr.cumulative_return) if dr.cumulative_return else 0,
                "drawdown": float(dr.drawdown) if dr.drawdown else 0
            })
        
        df = pd.DataFrame(data)
        df.set_index("date", inplace=True)
        return df
    
    def get_backtest_trades(self, backtest_id: int) -> List[Dict]:
        """获取回测交易记录"""
        trades = trade_dao.get_by_backtest_id(backtest_id)
        result = []
        for trade in trades:
            symbol = symbol_dao.get_by_id(trade.symbol_id)
            result.append({
                "id": trade.id,
                "symbol": symbol.symbol if symbol else None,
                "side": trade.side,
                "quantity": float(trade.quantity),
                "price": float(trade.price),
                "commission": float(trade.commission) if trade.commission else 0,
                "slippage": float(trade.slippage) if trade.slippage else 0,
                "pnl": float(trade.pnl) if trade.pnl else 0,
                "trade_time": trade.trade_time
            })
        return result
    
    # ==================== 交易标的相关服务 ====================
    
    def get_symbol_list(self, asset_class: Optional[str] = None, 
                       exchange: Optional[str] = None,
                       active_only: bool = True) -> List[Dict]:
        """获取交易标的列表"""
        if asset_class:
            symbols = symbol_dao.get_by_asset_class(asset_class)
        elif exchange:
            symbols = symbol_dao.get_by_exchange(exchange)
        elif active_only:
            symbols = symbol_dao.get_active_symbols()
        else:
            symbols = symbol_dao.get_all()
        
        result = []
        for symbol in symbols:
            result.append({
                "id": symbol.id,
                "symbol": symbol.symbol,
                "name": symbol.name,
                "asset_class": symbol.asset_class,
                "exchange": symbol.exchange,
                "currency": symbol.currency,
                "lot_size": symbol.lot_size,
                "tick_size": float(symbol.tick_size) if symbol.tick_size else None,
                "contract_size": float(symbol.contract_size) if symbol.contract_size else None,
                "is_active": symbol.is_active
            })
        return result
    
    def get_symbol_detail(self, symbol_id: int) -> Optional[Dict]:
        """获取交易标的详情"""
        symbol = symbol_dao.get_by_id(symbol_id)
        if not symbol:
            return None
        
        # 获取最新K线
        latest_bar = bar_dao.get_latest_bar(symbol_id)
        
        # 获取当前持仓
        positions = position_dao.get_by_symbol_id(symbol_id)
        open_positions = [p for p in positions if p.status == "open"]
        
        return {
            "id": symbol.id,
            "symbol": symbol.symbol,
            "name": symbol.name,
            "asset_class": symbol.asset_class,
            "exchange": symbol.exchange,
            "currency": symbol.currency,
            "lot_size": symbol.lot_size,
            "tick_size": float(symbol.tick_size) if symbol.tick_size else None,
            "contract_size": float(symbol.contract_size) if symbol.contract_size else None,
            "is_active": symbol.is_active,
            "latest_price": float(latest_bar.close_price) if latest_bar else None,
            "latest_datetime": latest_bar.datetime if latest_bar else None,
            "bar_count": bar_dao.count_bars(symbol_id),
            "open_positions_count": len(open_positions),
            "total_positions": len(positions)
        }
    
    def get_symbol_bars(self, symbol_id: int, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """获取K线数据"""
        if start_date and end_date:
            bars = bar_dao.get_by_date_range(symbol_id, start_date, end_date)
        else:
            bars = bar_dao.get_by_symbol_id(symbol_id, limit)
        
        if not bars:
            return None
        
        data = []
        for bar in bars:
            data.append({
                "datetime": bar.datetime,
                "open": float(bar.open_price),
                "high": float(bar.high_price),
                "low": float(bar.low_price),
                "close": float(bar.close_price),
                "volume": float(bar.volume),
                "open_interest": float(bar.open_interest) if bar.open_interest else 0
            })
        
        df = pd.DataFrame(data)
        df.set_index("datetime", inplace=True)
        return df
    
    # ==================== 持仓相关服务 ====================
    
    def get_position_list(self, strategy_id: Optional[int] = None,
                         status: Optional[str] = None) -> List[Dict]:
        """获取持仓列表"""
        if strategy_id and status:
            if status == "open":
                positions = position_dao.get_open_positions(strategy_id)
            elif status == "closed":
                positions = position_dao.get_closed_positions(strategy_id)
            else:
                positions = position_dao.get_by_strategy_id(strategy_id)
        elif strategy_id:
            positions = position_dao.get_by_strategy_id(strategy_id)
        elif status:
            if status == "open":
                positions = position_dao.get_open_positions()
            elif status == "closed":
                positions = position_dao.get_closed_positions()
            else:
                positions = position_dao.get_all()
        else:
            positions = position_dao.get_all()
        
        result = []
        for position in positions:
            symbol = symbol_dao.get_by_id(position.symbol_id)
            strategy = strategy_dao.get_by_id(position.strategy_id)
            result.append({
                "id": position.id,
                "symbol": symbol.symbol if symbol else None,
                "strategy": strategy.name if strategy else None,
                "quantity": float(position.quantity),
                "avg_price": float(position.avg_price) if position.avg_price else 0,
                "market_value": float(position.market_value) if position.market_value else 0,
                "unrealized_pnl": float(position.unrealized_pnl) if position.unrealized_pnl else 0,
                "realized_pnl": float(position.realized_pnl) if position.realized_pnl else 0,
                "side": position.side,
                "status": position.status,
                "open_time": position.open_time,
                "close_time": position.close_time
            })
        return result
    
    def get_position_detail(self, position_id: int) -> Optional[Dict]:
        """获取持仓详情"""
        position = position_dao.get_by_id(position_id)
        if not position:
            return None
        
        symbol = symbol_dao.get_by_id(position.symbol_id)
        strategy = strategy_dao.get_by_id(position.strategy_id)
        trades = trade_dao.get_by_position_id(position_id)
        
        return {
            "id": position.id,
            "symbol": symbol.symbol if symbol else None,
            "strategy": strategy.name if strategy else None,
            "quantity": float(position.quantity),
            "avg_price": float(position.avg_price) if position.avg_price else 0,
            "market_value": float(position.market_value) if position.market_value else 0,
            "unrealized_pnl": float(position.unrealized_pnl) if position.unrealized_pnl else 0,
            "realized_pnl": float(position.realized_pnl) if position.realized_pnl else 0,
            "side": position.side,
            "status": position.status,
            "open_time": position.open_time,
            "close_time": position.close_time,
            "trades_count": len(trades)
        }
    
    # ==================== 交易相关服务 ====================
    
    def get_trade_list(self, strategy_id: Optional[int] = None,
                      symbol_id: Optional[int] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> List[Dict]:
        """获取交易列表"""
        if start_date and end_date:
            trades = trade_dao.get_trades_by_date_range(start_date, end_date, strategy_id)
        elif strategy_id:
            trades = trade_dao.get_by_strategy_id(strategy_id)
        elif symbol_id:
            trades = trade_dao.get_by_symbol_id(symbol_id)
        else:
            trades = trade_dao.get_all()
        
        result = []
        for trade in trades:
            symbol = symbol_dao.get_by_id(trade.symbol_id)
            strategy = strategy_dao.get_by_id(trade.strategy_id)
            result.append({
                "id": trade.id,
                "symbol": symbol.symbol if symbol else None,
                "strategy": strategy.name if strategy else None,
                "side": trade.side,
                "quantity": float(trade.quantity),
                "price": float(trade.price),
                "commission": float(trade.commission) if trade.commission else 0,
                "slippage": float(trade.slippage) if trade.slippage else 0,
                "pnl": float(trade.pnl) if trade.pnl else 0,
                "trade_time": trade.trade_time
            })
        return result
    
    def get_trade_detail(self, trade_id: int) -> Optional[Dict]:
        """获取交易详情"""
        trade = trade_dao.get_by_id(trade_id)
        if not trade:
            return None
        
        symbol = symbol_dao.get_by_id(trade.symbol_id)
        strategy = strategy_dao.get_by_id(trade.strategy_id)
        position = position_dao.get_by_id(trade.position_id) if trade.position_id else None
        
        return {
            "id": trade.id,
            "symbol": symbol.symbol if symbol else None,
            "strategy": strategy.name if strategy else None,
            "position_id": trade.position_id,
            "position_side": position.side if position else None,
            "order_id": trade.order_id,
            "trade_id": trade.trade_id,
            "side": trade.side,
            "quantity": float(trade.quantity),
            "price": float(trade.price),
            "commission": float(trade.commission) if trade.commission else 0,
            "slippage": float(trade.slippage) if trade.slippage else 0,
            "pnl": float(trade.pnl) if trade.pnl else 0,
            "trade_time": trade.trade_time
        }
    
    def get_trade_statistics(self, strategy_id: Optional[int] = None,
                           backtest_id: Optional[int] = None) -> Dict:
        """获取交易统计信息"""
        return trade_dao.get_trade_statistics(strategy_id, backtest_id)
    
    # ==================== 组合分析服务 ====================
    
    def get_portfolio_summary(self, strategy_id: Optional[int] = None) -> Dict:
        """获取投资组合摘要"""
        # 获取持仓
        positions = position_dao.get_open_positions(strategy_id)
        
        # 计算总市值和总盈亏
        total_market_value = 0
        total_unrealized_pnl = 0
        total_realized_pnl = 0
        
        for position in positions:
            if position.market_value:
                total_market_value += float(position.market_value)
            if position.unrealized_pnl:
                total_unrealized_pnl += float(position.unrealized_pnl)
            if position.realized_pnl:
                total_realized_pnl += float(position.realized_pnl)
        
        # 获取交易统计
        trade_stats = trade_dao.get_trade_statistics(strategy_id)
        
        return {
            "total_positions": len(positions),
            "total_market_value": total_market_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_realized_pnl": total_realized_pnl,
            "total_pnl": total_unrealized_pnl + total_realized_pnl,
            "trade_statistics": trade_stats
        }
    
    def get_sector_allocation(self, strategy_id: Optional[int] = None) -> List[Dict]:
        """获取行业配置"""
        # 这里需要根据实际情况实现，可能需要额外的行业分类数据
        # 简化实现，按资产类别分组
        positions = position_dao.get_open_positions(strategy_id)
        
        sector_map = {}
        for position in positions:
            symbol = symbol_dao.get_by_id(position.symbol_id)
            sector = symbol.asset_class if symbol else "Unknown"
            
            if sector not in sector_map:
                sector_map[sector] = {
                    "sector": sector,
                    "market_value": 0,
                    "count": 0
                }
            
            if position.market_value:
                sector_map[sector]["market_value"] += float(position.market_value)
            sector_map[sector]["count"] += 1
        
        return list(sector_map.values())
    
    def get_performance_attribution(self, backtest_id: int) -> Dict:
        """获取绩效归因"""
        # 简化实现，按标的分组计算收益贡献
        trades = trade_dao.get_by_backtest_id(backtest_id)
        
        symbol_pnl = {}
        total_pnl = 0
        
        for trade in trades:
            if trade.pnl:
                symbol = symbol_dao.get_by_id(trade.symbol_id)
                symbol_name = symbol.symbol if symbol else "Unknown"
                
                if symbol_name not in symbol_pnl:
                    symbol_pnl[symbol_name] = 0
                
                symbol_pnl[symbol_name] += float(trade.pnl)
                total_pnl += float(trade.pnl)
        
        attribution = []
        for symbol, pnl in symbol_pnl.items():
            attribution.append({
                "symbol": symbol,
                "pnl": pnl,
                "contribution": pnl / total_pnl if total_pnl != 0 else 0
            })
        
        return {
            "total_pnl": total_pnl,
            "attribution": attribution
        }


# 全局数据仓库实例
_warehouse = None


def get_data_warehouse() -> DataWarehouse:
    """获取全局数据仓库实例
    
    Returns:
        数据仓库实例
    """
    global _warehouse
    
    if _warehouse is None:
        _warehouse = DataWarehouse()
    
    return _warehouse