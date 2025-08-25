# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, Optional

import ccxt
from dotenv import load_dotenv


class OkxClient:
    def __init__(self, public_only: bool = False):
        load_dotenv(os.path.join('config', '.env'))
        http_proxy = os.getenv('HTTP_PROXY') or None
        https_proxy = os.getenv('HTTPS_PROXY') or None

        opts = {
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'},
            'proxies': {'http': http_proxy, 'https': https_proxy} if (http_proxy or https_proxy) else None,
        }

        if public_only:
            self.exchange = ccxt.okx(opts)
            return

        testnet = (os.getenv('OKX_TESTNET', 'true').lower() == 'true')
        passphrase = os.getenv('OKX_PASSPHRASE') or ''

        if testnet:
            print("Using OKX testnet mode")
            api_key = os.getenv('OKX_API_KEY_TEST') or ''
            secret = os.getenv('OKX_SECRET_KEY_TEST') or ''
        else:
            print("Using OKX production mode")
            api_key = os.getenv('OKX_API_KEY') or ''
            secret = os.getenv('OKX_SECRET_KEY') or ''

        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret,
            'password': passphrase,
            **opts,
        })
        if testnet:
            self.exchange.headers = self.exchange.headers or {}
            self.exchange.headers['x-simulated-trading'] = '1'

    def load_markets(self) -> Dict[str, Any]:
        return self.exchange.load_markets()

    def fetch_balance(self) -> Dict[str, Any]:
        return self.exchange.fetch_balance()

    def fetch_positions(self, symbols=None) -> Any:
        if getattr(self.exchange, 'has', {}).get('fetchPositions'):
            return self.exchange.fetch_positions(symbols=symbols)
        return []

    def create_order(self, symbol: str, side: str, type_: str, amount: float, price: Optional[float] = None, params: Optional[Dict[str, Any]] = None, dry_run: bool = True) -> Dict[str, Any]:
        params = params or {}
        if dry_run:
            return {
                'dry_run': True,
                'symbol': symbol,
                'side': side,
                'type': type_,
                'amount': amount,
                'price': price,
                'params': params,
            }
        return self.exchange.create_order(symbol=symbol, type=type_, side=side, amount=amount, price=price, params=params)

    def cancel_order(self, id_: str, symbol: str, params: Optional[Dict[str, Any]] = None) -> Any:
        params = params or {}
        return self.exchange.cancel_order(id_, symbol, params)

    def fetch_open_orders(self, symbol: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Any:
        params = params or {}
        return self.exchange.fetch_open_orders(symbol=symbol, params=params)

    def market_info(self, symbol: str) -> Dict[str, Any]:
        markets = self.exchange.markets or self.exchange.load_markets()
        return markets.get(symbol) or {}
