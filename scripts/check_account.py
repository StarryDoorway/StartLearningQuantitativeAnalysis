#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from loguru import logger

from live.okx_client import OkxClient


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
