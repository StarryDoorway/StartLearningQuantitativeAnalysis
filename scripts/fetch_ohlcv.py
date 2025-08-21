#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Optional

import ccxt
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import yaml


def load_settings(settings_path: str) -> dict:
    if not os.path.exists(settings_path):
        return {}
    with open(settings_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def timeframe_to_millis(tf: str) -> int:
    units = {
        's': 1,
        'm': 60,
        'h': 60 * 60,
        'd': 24 * 60 * 60,
    }
    num = int(''.join([c for c in tf if c.isdigit()]))
    unit = ''.join([c for c in tf if c.isalpha()]).lower()
    if unit not in units:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return num * units[unit] * 1000


def parse_date(s: str) -> int:
    # returns milliseconds since epoch (UTC)
    try:
        # allow YYYY-MM-DD
        if len(s) == 10 and s[4] == '-' and s[7] == '-':
            return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)
        # allow ISO8601
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception as e:
        raise ValueError(f"Invalid date format: {s}") from e


def symbol_to_slug(symbol: str) -> str:
    # e.g., BTC/USDT:USDT -> btc-usdt-usdt
    slug = symbol.lower().replace('/', '-').replace(':', '-')
    return slug


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def init_okx() -> ccxt.okx:
    load_dotenv(os.path.join('config', '.env'))
    api_key = os.getenv('OKX_API_KEY')
    secret = os.getenv('OKX_SECRET_KEY')
    passphrase = os.getenv('OKX_PASSPHRASE')
    testnet = os.getenv('OKX_TESTNET', 'true').lower() == 'true'

    http_proxy = os.getenv('HTTP_PROXY') or None
    https_proxy = os.getenv('HTTPS_PROXY') or None

    exchange = ccxt.okx({
        'apiKey': api_key or '',
        'secret': secret or '',
        'password': passphrase or '',
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',  # 永续合约
        },
        'proxies': {
            'http': http_proxy,
            'https': https_proxy,
        } if (http_proxy or https_proxy) else None,
    })
    # 模拟盘开关（仅对私有/交易接口生效；公共行情也可保留此头部）
    if testnet:
        exchange.headers = exchange.headers or {}
        exchange.headers['x-simulated-trading'] = '1'
    return exchange


def fetch_ohlcv_all(
    exchange: ccxt.okx,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int],
    limit: int,
) -> List[List[float]]:
    all_rows: List[List[float]] = []
    current_since = since_ms
    step_ms = timeframe_to_millis(timeframe)
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=limit)
        if not candles:
            break
        all_rows.extend(candles)
        last_ts = candles[-1][0]
        # 终止条件
        if until_ms is not None and last_ts + step_ms >= until_ms:
            break
        # 防止死循环
        if last_ts <= current_since:
            break
        current_since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000.0)
    return all_rows


def load_existing_parquet(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        logger.warning(f"Failed to read existing parquet {path}: {e}")
        return None


def merge_candles(df_old: Optional[pd.DataFrame], candles: List[List[float]]) -> pd.DataFrame:
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_new = pd.DataFrame(candles, columns=cols)
    if df_old is None or df_old.empty:
        df = df_new
    else:
        df = pd.concat([df_old[cols], df_new], ignore_index=True)
    # 去重、排序
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df


def determine_start_ts(existing_df: Optional[pd.DataFrame], timeframe: str, default_since_ms: int) -> int:
    if existing_df is None or existing_df.empty:
        return default_since_ms
    step_ms = timeframe_to_millis(timeframe)
    last_ts = int(existing_df['timestamp'].iloc[-1])
    return last_ts + step_ms


def main():
    parser = argparse.ArgumentParser(description='Fetch OKX OHLCV to Parquet with resume')
    parser.add_argument('--symbols', nargs='+', help='Symbols like BTC/USDT:USDT ETH/USDT:USDT')
    parser.add_argument('--timeframes', nargs='+', default=None, help='e.g., 1m 5m 1h')
    parser.add_argument('--since', type=str, default=None, help='Start time (YYYY-MM-DD or ISO8601). Defaults to settings.yaml')
    parser.add_argument('--until', type=str, default=None, help='End time (YYYY-MM-DD or ISO8601). Defaults to now')
    parser.add_argument('--base-dir', type=str, default=None, help='Base data dir, default from settings.yaml')
    parser.add_argument('--limit', type=int, default=None, help='Max candles per request')

    args = parser.parse_args()

    settings = load_settings(os.path.join('config', 'settings.yaml'))
    base_dir = args.base_dir or settings.get('base_dir', 'data/raw')
    symbols = args.symbols or settings.get('symbols', ['BTC/USDT:USDT'])
    timeframes = args.timeframes or settings.get('timeframes', ['5m'])
    limit = args.limit or int(settings.get('max_candles_per_request', 100))

    if args.since:
        since_ms = parse_date(args.since)
    else:
        default_start = settings.get('start_time', '2024-01-01T00:00:00Z')
        since_ms = parse_date(default_start)

    until_ms = parse_date(args.until) if args.until else int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    exchange = init_okx()
    logger.info(f"Using base_dir={base_dir}, symbols={symbols}, timeframes={timeframes}, limit={limit}")

    for symbol in symbols:
        slug = symbol_to_slug(symbol)
        symbol_dir = os.path.join(base_dir, slug)
        ensure_dir(symbol_dir)
        for tf in timeframes:
            parquet_path = os.path.join(symbol_dir, f"{tf}.parquet")
            existing_df = load_existing_parquet(parquet_path)
            start_ms = determine_start_ts(existing_df, tf, since_ms)
            if existing_df is not None and not existing_df.empty:
                logger.info(f"Resuming {symbol} {tf} from {datetime.utcfromtimestamp(start_ms/1000).isoformat()} (existing up to {datetime.utcfromtimestamp(int(existing_df['timestamp'].iloc[-1])/1000).isoformat()})")
            else:
                logger.info(f"Fetching {symbol} {tf} from {datetime.utcfromtimestamp(start_ms/1000).isoformat()} (fresh)")

            rows = fetch_ohlcv_all(exchange, symbol, tf, start_ms, until_ms, limit)
            if not rows:
                logger.info(f"No new candles for {symbol} {tf}")
                continue

            df_merged = merge_candles(existing_df, rows)
            # 保存为 parquet
            df_merged.to_parquet(parquet_path, index=False)
            logger.success(f"Saved {len(df_merged)} rows -> {parquet_path}")

    logger.info("All done.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
