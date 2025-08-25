#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from loguru import logger

from src.exchanges.okx_client import OkxClient


def main():
    client = OkxClient()
    try:
        balance = client.fetch_balance()
        logger.info("Balance summary (non-zero):")
        for k, v in balance.get('total', {}).items():
            if v:
                logger.info(f"  {k}: {v}")
        positions = client.fetch_positions()
        if positions:
            logger.info(f"Positions: {len(positions)} entries")
        else:
            logger.info("Positions: none or not supported via unified fetch")
        logger.success("OKX connectivity verified.")
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    main()
