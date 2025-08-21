# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Dict, Any, Tuple


def load_market_info(markets: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    if symbol not in markets:
        raise KeyError(f"Symbol not in markets: {symbol}")
    return markets[symbol]


def round_price_amount(market: Dict[str, Any], price: float, amount: float) -> Tuple[float, float]:
    prec = market.get('precision', {}) or {}
    price_prec = prec.get('price')
    amount_prec = prec.get('amount')

    if price_prec is not None:
        price = float(_round_to_precision(price, price_prec))
    if amount_prec is not None:
        amount = float(_round_to_precision(amount, amount_prec))
    return price, amount


def _round_to_precision(value: float, decimals: int) -> float:
    if decimals is None:
        return value
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def satisfies_min_limits(market: Dict[str, Any], price: float, amount: float) -> bool:
    limits = market.get('limits', {}) or {}
    min_amt = (limits.get('amount') or {}).get('min')
    min_cost = (limits.get('cost') or {}).get('min')

    if min_amt is not None and amount < float(min_amt):
        return False
    if min_cost is not None and (price * amount) < float(min_cost):
        return False
    return True
