## OKX 加密货币量化机器人（入门实战骨架）

### 1. 环境准备
- Python 3.10+
- 建议创建虚拟环境：
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 OKX 密钥（支持模拟盘/真盘切换）
- 新建 `config/.env` 并填写：`OKX_TESTNET`、`OKX_API_KEY[_TEST]`、`OKX_SECRET_KEY[_TEST]`、`OKX_PASSPHRASE`（详见文件内注释）

### 3. 同步市场元数据（无密钥）
```bash
python -m scripts.sync_okx_markets
# 或：python scripts/sync_okx_markets.py
```

### 4. 拉取历史 K 线（CCXT 公共接口）
```bash
python -m scripts.fetch_ohlcv --symbols BTC/USDT:USDT --timeframes 5m --since 2024-01-01
```

### 5. 回测（Backtrader）
```bash
python -m scripts.run_backtest --symbol-slug btc-usdt-usdt --timeframe 5m --cash 10000 --commission 0.0005 --plot
```

### 6. 纸交易/实盘执行（统一用 src 模块）
```bash
# 纸交易
python -m scripts.order_executor --side buy --type limit --price 30000 --paper
# 真盘（谨慎）
python -m scripts.order_executor --side buy --type limit --price 30000
```

### 7. 实时轮询 Runner（EMA+RSI）
```bash
python live/runner_ema_rsi.py --paper --timeframe 5m --loop --interval-seconds 30
```

### 8. 项目结构（核心逻辑集中在 src）
```
src/
  ├─ exchanges/okx_client.py     # 统一 OKX 客户端（testnet/真盘自动选择）
  └─ utils/
       ├─ precision.py           # 精度与最小下单量校验
       └─ risk.py                # 基础风控
scripts/
  ├─ __init__.py
  ├─ sync_okx_markets.py         # 公共接口获取市场元数据（无密钥）
  ├─ fetch_ohlcv.py              # OHLCV 抓取（公共接口）
  ├─ run_backtest.py             # Backtrader 回测
  └─ order_executor.py           # 纸/实盘执行
live/
  ├─ runner_ema_rsi.py           # 简易实时策略执行（轮询）
  └─ state/
strategies/
  └─ ema_rsi_backtrader.py
freqtrade/
  ├─ config_okx_example.json
  └─ strategies/EmaRsiFtStrategy.py
config/
  ├─ settings.yaml
  ├─ trading.yaml
  └─ okx_markets.json
```
