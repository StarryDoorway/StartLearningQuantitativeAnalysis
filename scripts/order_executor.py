#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import json
import time
import uuid
from typing import Any, Dict

import yaml
from loguru import logger

from live.okx_client import OkxClient
from live.risk import RiskManager, RiskConfig
from live.precision import round_price_amount, satisfies_min_limits


def load_markets_json() -> Dict[str, Any]:
    path = os.path.join('config', 'okx_markets.json')
    if not os.path.exists(path):
        raise FileNotFoundError("config/okx_markets.json not found. Run scripts/sync_okx_markets.py first.")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_trading_cfg() -> Dict[str, Any]:
    path = os.path.join('config', 'trading.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError("config/trading.yaml not found.")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Minimal order executor with precision and risk checks')
    parser.add_argument('--side', choices=['buy', 'sell'], required=True)
    parser.add_argument('--type', dest='type_', choices=['market', 'limit'], default='limit')
    parser.add_argument('--price', type=float, default=None, help='Required for limit orders')
    parser.add_argument('--paper', action='store_true', help='Paper mode (no real orders)')
    args = parser.parse_args()

    # Load configs and market meta
    markets = load_markets_json()
    trading = load_trading_cfg()
    symbol = trading['symbol']

    client = OkxClient()
    # balance
    bal = client.fetch_balance()
    usdt_free = float(bal.get('free', {}).get('USDT', 0.0))

    # risk
    rcfg = RiskConfig(
        max_position_notional_usdt=float(trading['max_position_notional_usdt']),
        max_order_notional_usdt=float(trading['max_order_notional_usdt']),
        order_percent_balance=float(trading['order_percent_balance']),
    )
    rman = RiskManager(rcfg)

    # market info
    market = markets.get(symbol)
    if market is None:
        raise KeyError(f"Symbol {symbol} not found in okx_markets.json. Run sync script.")

    # compute order size
    notional = rman.compute_order_notional(usdt_free)
    if notional <= 0:
        logger.error('No free USDT to allocate')
        sys.exit(1)

    # price
    price = args.price
    if args.type_ == 'limit' and (price is None or price <= 0):
        logger.error('Limit order requires --price > 0')
        sys.exit(1)

    # For market orders, approximate price via ticker (fallback to last known close would be better)
    if args.type_ == 'market' and price is None:
        try:
            ticker = client.exchange.fetch_ticker(symbol)
            price = float(ticker['last'])
        except Exception as e:
            logger.warning(f"fetch_ticker failed, fallback to notional/price estimate needed: {e}")
            price = None

    if price is None or price <= 0:
        logger.error('Unable to determine a valid price')
        sys.exit(1)

    amount = notional / price

    # round to precision and check min limits
    price, amount = round_price_amount(market, price, amount)
    if not satisfies_min_limits(market, price, amount):
        logger.error(f"Order fails min limits. price={price}, amount={amount}")
        sys.exit(1)

    # idempotent client order id
    client_oid = f"quant-{uuid.uuid4().hex[:18]}"

    params = {
        'tdMode': trading.get('td_mode', 'cross'),  # cross/isolated
        'clOrdId': client_oid,
    }
    pos_side = (trading.get('pos_side') or '').strip()
    if pos_side:
        params['posSide'] = pos_side  # long/short (hedge mode)
    if trading.get('reduce_only'):
        params['reduceOnly'] = True
    if trading.get('post_only') and args.type_ == 'limit':
        params['postOnly'] = True

    logger.info(f"Placing order: {symbol} {args.side} {args.type_} amount={amount} price={price} paper={args.paper} params={params}")

    try:
        res = client.create_order(symbol=symbol, side=args.side, type_=args.type_, amount=amount, price=price, params=params, dry_run=args.paper)
        logger.success(f"Order result: {res}")
    except Exception as e:
        logger.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
