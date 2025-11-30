# -*- coding: utf-8 -*-

"""
交易工具模块，包含价格精度处理、订单限制检查等功能
"""

from __future__ import annotations

import math
from typing import Dict, Any, Tuple


def round_price_amount(market: Dict[str, Any], price: float, amount: float) -> Tuple[float, float]:
    """
    根据市场精度要求调整价格和数量
    
    Args:
        market: 市场信息字典，包含精度信息
        price: 原始价格
        amount: 原始数量
        
    Returns:
        调整后的价格和数量元组
    """
    prec = market.get('precision', {}) or {}
    price_prec = prec.get('price')
    amount_prec = prec.get('amount')

    if price_prec is not None:
        price = float(_round_to_precision(price, price_prec))
    if amount_prec is not None:
        amount = float(_round_to_precision(amount, amount_prec))
    return price, amount


def _round_to_precision(value: float, decimals: int) -> float:
    """
    将数值按指定小数位数进行舍入
    
    Args:
        value: 原始值
        decimals: 小数位数
        
    Returns:
        舍入后的值
    """
    if decimals is None:
        return value
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def satisfies_min_limits(market: Dict[str, Any], price: float, amount: float) -> bool:
    """
    检查订单是否满足最小数量和最小金额限制
    
    Args:
        market: 市场信息字典，包含限制信息
        price: 价格
        amount: 数量
        
    Returns:
        是否满足最小限制
    """
    limits = market.get('limits', {}) or {}
    min_amt = (limits.get('amount') or {}).get('min')
    min_cost = (limits.get('cost') or {}).get('min')

    if min_amt is not None and amount < float(min_amt):
        return False
    if min_cost is not None and (price * amount) < float(min_cost):
        return False
    return True


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float
) -> float:
    """
    基于风险百分比计算仓位大小
    
    Args:
        account_balance: 账户余额
        risk_percent: 风险百分比 (例如: 0.01 表示 1%)
        entry_price: 入场价格
        stop_loss_price: 止损价格
        
    Returns:
        建议的仓位大小
    """
    if stop_loss_price <= 0:
        return 0.0
    
    risk_amount = account_balance * risk_percent
    price_diff = abs(entry_price - stop_loss_price)
    
    if price_diff <= 0:
        return 0.0
    
    position_size = risk_amount / price_diff
    return position_size


def calculate_take_profit_price(
    entry_price: float,
    stop_loss_price: float,
    risk_reward_ratio: float = 2.0
) -> float:
    """
    基于风险回报比计算止盈价格
    
    Args:
        entry_price: 入场价格
        stop_loss_price: 止损价格
        risk_reward_ratio: 风险回报比 (默认为2.0)
        
    Returns:
        止盈价格
    """
    risk = abs(entry_price - stop_loss_price)
    if entry_price > stop_loss_price:
        return entry_price + (risk * risk_reward_ratio)
    else:
        return entry_price - (risk * risk_reward_ratio)