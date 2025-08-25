#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from loguru import logger

from src.exchanges.okx_client import OkxClient


def public_connectivity_check():
    client_pub = OkxClient(public_only=True)
    markets = client_pub.load_markets()
    logger.info(f"Loaded markets: {len(markets)} symbols")
    try:
        ticker = client_pub.exchange.fetch_ticker('BTC/USDT:USDT')
        last = ticker.get('last')
        logger.info(f"BTC/USDT:USDT last={last}")
    except Exception:
        pass
    logger.success("OKX public connectivity verified.")


def main():
    client = OkxClient()
    try:
        balance = client.fetch_balance()
        logger.info("Balance summary (non-zero):")
        for k, v in balance.get('total', {}).items():
            if v:
                logger.info(f"  {k}: {v}")
        logger.success("OKX private connectivity verified.")
    except Exception as e:
        msg = str(e)
        if '50038' in msg:
            logger.warning("Private endpoint unavailable in demo trading (code 50038). Falling back to public check...")
            public_connectivity_check()
            return
        if 'Authentication' in msg or 'auth' in msg.lower() or '50101' in msg:
            logger.warning("Authentication failed for private endpoint. Check OKX_TESTNET and API keys. Falling back to public check...")
            public_connectivity_check()
            return
        logger.exception(e)


if __name__ == '__main__':
    main()
