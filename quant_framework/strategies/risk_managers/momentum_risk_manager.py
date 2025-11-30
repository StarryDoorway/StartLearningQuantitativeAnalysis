"""
动量策略风险管理模块

负责处理仓位限制、止损止盈、波动率调整等风险管理功能。
"""

import pandas as pd
from typing import Dict, List, Any

from ..base.signal_types import Signal


class MomentumRiskManager:
    """动量策略风险管理器"""
    
    @staticmethod
    def apply_position_limits(signals: List[Signal], current_positions: Dict[str, float], 
                            indicators: Dict[str, Dict[str, pd.Series]], 
                            parameters: Dict[str, Any]) -> List[Signal]:
        """
        应用仓位限制到信号
        
        Args:
            signals: 交易信号列表
            current_positions: 当前持仓
            indicators: 技术指标
            parameters: 策略参数
            
        Returns:
            过滤后的信号列表
        """
        # 检查最大持仓数量
        max_positions = parameters.get("max_positions", 5)
        current_position_count = len([pos for pos in current_positions.values() if pos != 0])
        
        if current_position_count >= max_positions:
            # 只允许出场信号
            return [signal for signal in signals if 
                   (signal.signal_type == SignalType.BUY and current_positions.get(signal.symbol, 0) < 0) or
                   (signal.signal_type == SignalType.SELL and current_positions.get(signal.symbol, 0) > 0)]
        
        # 应用波动率调整到仓位大小
        if parameters.get("volatility_adjustment", True):
            for signal in signals:
                symbol = signal.symbol
                if symbol in indicators and 'volatility' in indicators[symbol]:
                    volatility = indicators[symbol]['volatility'].iloc[-1]
                    # 波动率与仓位大小成反比
                    adjusted_size = parameters.get("position_size", 0.1) / (1 + volatility * 10)
                    signal.quantity = adjusted_size
        
        return signals
    
    @staticmethod
    def check_stop_loss_take_profit(symbol: str, current_price: float, 
                                   entry_prices: Dict[str, float], 
                                   stop_losses: Dict[str, float], 
                                   take_profits: Dict[str, float], 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        检查止损止盈条件
        
        Args:
            symbol: 交易标的
            current_price: 当前价格
            entry_prices: 入场价格
            stop_losses: 止损价格
            take_profits: 止盈价格
            parameters: 策略参数
            
        Returns:
            包含止损止盈检查结果的字典
        """
        result = {
            "trigger_stop_loss": False,
            "trigger_take_profit": False,
            "action": None
        }
        
        # 检查是否有持仓
        if symbol not in entry_prices:
            return result
        
        entry_price = entry_prices[symbol]
        stop_loss_pct = parameters.get("stop_loss_pct", 0.05)
        take_profit_pct = parameters.get("take_profit_pct", 0.15)
        
        # 如果没有预设的止损止盈价格，则计算
        if symbol not in stop_losses:
            stop_losses[symbol] = entry_price * (1 - stop_loss_pct)
        
        if symbol not in take_profits:
            take_profits[symbol] = entry_price * (1 + take_profit_pct)
        
        # 检查止损
        if current_price <= stop_losses[symbol]:
            result["trigger_stop_loss"] = True
            result["action"] = "sell"  # 平多仓
        
        # 检查止盈
        if current_price >= take_profits[symbol]:
            result["trigger_take_profit"] = True
            result["action"] = "sell"  # 平多仓
        
        return result
    
    @staticmethod
    def update_position_data(trade_data: Dict[str, Any], 
                            entry_prices: Dict[str, float], 
                            stop_losses: Dict[str, float], 
                            take_profits: Dict[str, float], 
                            parameters: Dict[str, Any]) -> None:
        """
        更新持仓相关数据
        
        Args:
            trade_data: 交易数据
            entry_prices: 入场价格
            stop_losses: 止损价格
            take_profits: 止盈价格
            parameters: 策略参数
        """
        symbol = trade_data.get("symbol")
        if not symbol:
            return
        
        if trade_data.get("side") == "buy":
            # 更新入场价格
            entry_prices[symbol] = trade_data.get("price", 0)
            
            # 设置止损和止盈
            entry_price = entry_prices[symbol]
            stop_loss_pct = parameters.get("stop_loss_pct", 0.05)
            take_profit_pct = parameters.get("take_profit_pct", 0.15)
            
            stop_losses[symbol] = entry_price * (1 - stop_loss_pct)
            take_profits[symbol] = entry_price * (1 + take_profit_pct)
            
        elif trade_data.get("side") == "sell":
            # 清除持仓相关数据
            if symbol in entry_prices:
                del entry_prices[symbol]
            if symbol in stop_losses:
                del stop_losses[symbol]
            if symbol in take_profits:
                del take_profits[symbol]
    
    @staticmethod
    def calculate_position_size(symbol: str, current_price: float, 
                               portfolio_value: float, indicators: Dict[str, pd.Series], 
                               parameters: Dict[str, Any]) -> float:
        """
        计算建议的仓位大小
        
        Args:
            symbol: 交易标的
            current_price: 当前价格
            portfolio_value: 投资组合价值
            indicators: 技术指标
            parameters: 策略参数
            
        Returns:
            建议的仓位大小
        """
        # 基础仓位大小
        base_position_size = parameters.get("position_size", 0.1)
        
        # 波动率调整
        if parameters.get("volatility_adjustment", True) and 'volatility' in indicators:
            volatility = indicators['volatility'].iloc[-1]
            # 波动率越高，仓位越小
            volatility_adjustment = 1 / (1 + volatility * 10)
            base_position_size *= volatility_adjustment
        
        # 计算股数
        position_value = portfolio_value * base_position_size
        shares = position_value / current_price
        
        return shares