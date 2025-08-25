#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
from typing import Dict, Any

from loguru import logger

from src.core.okx_client import OkxClient


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


def main():
    client = OkxClient(public_only=True)
    markets = client.load_markets()
    export = {sym: to_export_fields(m) for sym, m in markets.items()}
    out_path = os.path.join('config', 'okx_markets.json')
    os.makedirs('config', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    logger.success(f"Saved {len(export)} markets -> {out_path}")


if __name__ == '__main__':
    main()
