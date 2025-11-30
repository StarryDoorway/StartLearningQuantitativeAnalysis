#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
# Ensure project root is on sys.path - going up to project root directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
from typing import Dict, Any

from loguru import logger

from quant_framework.clients.okx_client import OkxClient

# Define the currencies we want to keep with USDT as quote
TARGET_CURRENCIES = ['ETH', 'BTC', 'DOGE', 'TRUMP', 'USDC']
QUOTE_CURRENCY = 'USDT'


def to_export_fields(market: Dict[str, Any]) -> Dict[str, Any]:
    precision = market.get('precision', {}) or {}
    limits = market.get('limits', {}) or {}
    return {
        'symbol': market.get('symbol'),
        'id': market.get('id'),
        'type': market.get('type'),  # spot/swap/future
        'base': market.get('base'),
        'quote': market.get('quote'),
        'contract': market.get('contract'),
        'linear': market.get('linear'),
        'contractSize': market.get('contractSize'),
        'precision': {
            'amount': precision.get('amount'),
            'price': precision.get('price'),
        },
        'limits': {
            'amount': limits.get('amount'),
            'price': limits.get('price'),
            'cost': limits.get('cost'),
        },
    }


def is_target_market(symbol: str) -> bool:
    """Check if the market symbol matches our target currencies with USDT as quote"""
    try:
        base, quote = symbol.split('/')
        return base in TARGET_CURRENCIES and quote == QUOTE_CURRENCY
    except ValueError:
        # If we can't split by '/', it's not a standard pair format
        return False


def main():
    client = OkxClient(public_only=True)
    markets = client.load_markets()
    
    # Filter markets to only include our target currency pairs
    filtered_markets = {
        sym: to_export_fields(m) 
        for sym, m in markets.items() 
        if is_target_market(sym)
    }
    
    out_path = os.path.join('quant_framework/config', 'okx_markets.json')
    os.makedirs('config', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_markets, f, ensure_ascii=False, indent=2)
    
    logger.success(f"Saved {len(filtered_markets)} target markets -> {out_path}")
    
    # Print the filtered markets for verification
    logger.info("Filtered markets:")
    for symbol in sorted(filtered_markets.keys()):
        logger.info(f"  - {symbol}")


if __name__ == '__main__':
    main()
