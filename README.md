## OKX 加密货币量化机器人（入门实战骨架）

### 1. 环境准备
- Python 3.10+
- 建议创建虚拟环境：
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 OKX 密钥（支持模拟盘/真盘切换）
- 新建 `config/.env` 并填写以下变量：
```
# 切换环境：模拟盘 true；真盘 false
OKX_TESTNET=true

# 模拟盘（Demo Trading）API（若使用模拟盘）
OKX_API_KEY_TEST=
OKX_SECRET_KEY_TEST=

# 真盘 API（若使用真盘）
OKX_API_KEY=
OKX_SECRET_KEY=

# 两个环境共用或分别设置的口令
OKX_PASSPHRASE=

# 可选代理
HTTP_PROXY=
HTTPS_PROXY=
```
- 客户端会根据 `OKX_TESTNET` 自动选择对应的 Key，并打印当前模式（testnet/production）。
- 市场元数据同步脚本使用公共接口，无需密钥。

### 3. 同步交易所市场元数据（精度/最小下单量等）
```bash
python scripts/sync_okx_markets.py
# 输出：config/okx_markets.json
```

### 4. 拉取历史 K 线数据（CCXT）
```bash
python scripts/fetch_ohlcv.py \
  --symbols BTC/USDT:USDT ETH/USDT:USDT \
  --timeframes 5m 1h \
  --since 2024-01-01
```

### 5. 回测（Backtrader）
```bash
python scripts/run_backtest.py --symbol-slug btc-usdt-usdt --timeframe 5m --cash 10000 --commission 0.0005 --plot
```

### 6. 纸交易/实盘执行器（最小可用）
```bash
python scripts/order_executor.py --side buy --type limit --price 30000 --paper
```

### 7. 实时轮询 Runner（EMA+RSI 示例）
```bash
python live/runner_ema_rsi.py --paper --timeframe 5m --loop --interval-seconds 30
```

### 8. Freqtrade 模板（OKX）
见 `freqtrade/` 目录（dry-run 默认）。

### 9. 目录结构（节选）
```
config/
  ├─ settings.yaml
  ├─ okx_markets.json
  └─ trading.yaml
scripts/
  ├─ fetch_ohlcv.py
  ├─ run_backtest.py
  ├─ sync_okx_markets.py
  ├─ check_account.py
  └─ order_executor.py
src/
  ├─ exchanges/okx_client.py
  └─ utils/{precision.py,risk.py}
live/
  ├─ runner_ema_rsi.py
  └─ state/
```
