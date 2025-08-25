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
from dataclasses import dataclass
from typing import Any, Dict

import ccxt
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

from src.exchanges.okx_client import OkxClient
from src.utils.risk import RiskManager, RiskConfig
from src.utils.precision import round_price_amount, satisfies_min_limits


STATE_DIR = os.path.join('live', 'state')


@dataclass
class StrategyParams:
    fast_ema: int = 20
    slow_ema: int = 50
    rsi_period: int = 14
    rsi_entry: float = 52.0
    rsi_exit: float = 48.0


def load_markets_json() -> Dict[str, Any]:
    path = os.path.join('config', 'okx_markets.json')
    if not os.path.exists(path):
        raise FileNotFoundError("config/okx_markets.json not found. Run scripts/sync_okx_markets.py first.")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_trading_cfg() -> Dict[str, Any]:
    import yaml
    path = os.path.join('config', 'trading.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError("config/trading.yaml not found.")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def state_path(symbol_slug: str, timeframe: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    return os.path.join(STATE_DIR, f"{symbol_slug}_{timeframe}.json")


def read_state(symbol_slug: str, timeframe: str) -> Dict[str, Any]:
    path = state_path(symbol_slug, timeframe)
    if not os.path.exists(path):
        return {'in_position': False, 'last_amount': 0.0}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_state(symbol_slug: str, timeframe: str, state: Dict[str, Any]) -> None:
    path = state_path(symbol_slug, timeframe)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def fetch_ohlcv(exchange: ccxt.okx, symbol: str, timeframe: str, limit: int = 250) -> pd.DataFrame:
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df


def compute_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=params.fast_ema, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=params.slow_ema, adjust=False).mean()
    # RSI implementation
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / params.rsi_period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / params.rsi_period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    df['cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
    df['cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
    return df


def symbol_to_slug(symbol: str) -> str:
    return symbol.lower().replace('/', '-').replace(':', '-')


def run_once(symbol: str, timeframe: str, paper: bool, params: StrategyParams) -> None:
    load_dotenv(os.path.join('config', '.env'))
    client = OkxClient()
    exchange = client.exchange

    df = compute_indicators(fetch_ohlcv(exchange, symbol, timeframe), params)
    if df.empty or len(df) < max(params.slow_ema, params.rsi_period) + 2:
        logger.warning('Not enough candles to evaluate.')
        return

    last = df.iloc[-1]
    markets = load_markets_json()
    market = markets.get(symbol)
    if market is None:
        raise KeyError(f"Symbol {symbol} not in okx_markets.json. Run sync script.")

    trading = load_trading_cfg()
    rcfg = RiskConfig(
        max_position_notional_usdt=float(trading['max_position_notional_usdt']),
        max_order_notional_usdt=float(trading['max_order_notional_usdt']),
        order_percent_balance=float(trading['order_percent_balance']),
    )
    rman = RiskManager(rcfg)

    slug = symbol_to_slug(symbol)
    st = read_state(slug, timeframe)

    bal = client.fetch_balance()
    usdt_free = float(bal.get('free', {}).get('USDT', 0.0))

    want_buy = (not st.get('in_position')) and bool(last['cross_up']) and float(last['rsi']) >= params.rsi_entry
    want_exit = st.get('in_position') and (bool(last['cross_down']) or float(last['rsi']) <= params.rsi_exit)

    if not want_buy and not want_exit:
        logger.info(f"No action. RSI={last['rsi']:.2f} fast={last['ema_fast']:.2f} slow={last['ema_slow']:.2f}")
        return

    price = float(last['close'])

    if want_buy:
        notional = rman.compute_order_notional(usdt_free)
        if notional <= 0:
            logger.warning('Skip buy: no free USDT.')
            return
        amount = notional / price
        price, amount = round_price_amount(market, price, amount)
        if not satisfies_min_limits(market, price, amount):
            logger.warning('Skip buy: does not satisfy min limits.')
            return
        params_order = {
            'tdMode': trading.get('td_mode', 'cross'),
            'clOrdId': f"runner-{int(time.time())}",
        }
        pos_side = (trading.get('pos_side') or '').strip()
        if pos_side:
            params_order['posSide'] = pos_side
        logger.info(f"BUY {symbol} market amount={amount} price_ref={price} paper={paper}")
        res = OkxClient().create_order(symbol, 'buy', 'market', amount, None, params=params_order, dry_run=paper)
        logger.success(f"Order result: {res}")
        st['in_position'] = True
        st['last_amount'] = amount
        write_state(slug, timeframe, st)
        return

    if want_exit:
        amount = float(st.get('last_amount') or 0.0)
        if amount <= 0:
            logger.warning('Skip exit: no amount in state.')
            st['in_position'] = False
            write_state(slug, timeframe, st)
            return
        params_order = {
            'tdMode': trading.get('td_mode', 'cross'),
            'clOrdId': f"runner-{int(time.time())}",
            'reduceOnly': True,
        }
        pos_side = (trading.get('pos_side') or '').strip()
        if pos_side:
            params_order['posSide'] = pos_side
        logger.info(f"SELL (reduce) {symbol} market amount={amount} price_ref={price} paper={paper}")
        res = OkxClient().create_order(symbol, 'sell', 'market', amount, None, params=params_order, dry_run=paper)
        logger.success(f"Order result: {res}")
        st['in_position'] = False
        st['last_amount'] = 0.0
        write_state(slug, timeframe, st)
        return


def main():
    parser = argparse.ArgumentParser(description='Live EMA+RSI runner (polling)')
    parser.add_argument('--symbol', type=str, default=None, help='OKX symbol, e.g., BTC/USDT:USDT')
    parser.add_argument('--timeframe', type=str, default='5m')
    parser.add_argument('--paper', action='store_true')
    parser.add_argument('--loop', action='store_true', help='Loop forever')
    parser.add_argument('--interval-seconds', type=int, default=30)
    args = parser.parse_args()

    trading = load_trading_cfg()
    symbol = args.symbol or trading.get('symbol', 'BTC/USDT:USDT')

    params = StrategyParams()

    if args.loop:
        while True:
            try:
                run_once(symbol, args.timeframe, args.paper, params)
            except Exception as e:
                logger.exception(e)
            time.sleep(max(5, int(args.interval_seconds)))
    else:
        run_once(symbol, args.timeframe, args.paper, params)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.exception(e)
        sys.exit(1)
