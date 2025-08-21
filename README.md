## OKX 加密货币量化机器人（入门实战骨架）

### 1. 环境准备
- Python 3.10+
- 建议创建虚拟环境：
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 OKX 密钥（可先用模拟盘）
- 新建 `config/.env` 并填写以下变量：
```
OKX_API_KEY=
OKX_SECRET_KEY=
OKX_PASSPHRASE=
OKX_TESTNET=true  # 模拟盘：true；真盘：false
HTTP_PROXY=
HTTPS_PROXY=
```

- 验证连通性（只读，不下单）：
```bash
python scripts/check_account.py
```

### 3. 同步交易所市场元数据（精度/最小下单量等）
```bash
python scripts/sync_okx_markets.py
# 输出：config/okx_markets.json
```

### 4. 拉取历史 K 线数据（CCXT）
```bash
# 例：拉取 BTC/USDT:USDT 与 ETH/USDT:USDT 的 5m 和 1h 数据，自 2024-01-01 起
python scripts/fetch_ohlcv.py \
  --symbols BTC/USDT:USDT ETH/USDT:USDT \
  --timeframes 5m 1h \
  --since 2024-01-01
```
- 数据文件保存在 `data/raw/{symbol_slug}/{timeframe}.parquet`（例如 `btc-usdt-usdt/5m.parquet`）。
- 支持断点续传：重复运行会自动从上次最后一根 K 线继续。

### 5. 回测（Backtrader）
```bash
# 例：对 BTC/USDT:USDT 的 5m 数据做回测
python scripts/run_backtest.py \
  --symbol-slug btc-usdt-usdt \
  --timeframe 5m \
  --cash 10000 \
  --commission 0.0005 --plot
```
- 策略位于 `strategies/ema_rsi_backtrader.py`，为 EMA 交叉 + RSI 过滤 + ATR 止损示例。
- 回测输出含收益、回撤等分析信息；图表会保存至 `backtests/`。

### 6. 纸交易/实盘执行器（最小可用）
- 配置交易执行参数：`config/trading.yaml`
- 先创建/更新市场元数据：`python scripts/sync_okx_markets.py`
- 纸交易（不下真实单，做全流程校验）：
```bash
python scripts/order_executor.py --side buy --type limit --price 30000 --paper
```
- 真盘（危险，请先小额）：
```bash
# 确认 config/.env 中 OKX_TESTNET=false 且使用真盘API。务必小仓位灰度。
python scripts/order_executor.py --side buy --type limit --price 30000
```
- 逻辑要点：
  - 读取 `config/okx_markets.json` 做价格/数量精度和最小下单量校验
  - 幂等下单：使用 `clOrdId`（client OID）
  - 风控：按 `trading.yaml` 限制单笔和总仓位名义价值

### 7. 实时轮询 Runner（EMA+RSI 示例）
```bash
# 单次评估与下单（paper）
python live/runner_ema_rsi.py --paper --timeframe 5m
# 循环轮询，每 30s 评估一次
python live/runner_ema_rsi.py --paper --timeframe 5m --loop --interval-seconds 30
```

### 8. Freqtrade 模板（OKX）
- 配置示例：`freqtrade/config_okx_example.json`（默认 dry-run）
- 策略：`freqtrade/strategies/EmaRsiFtStrategy.py`
- 常用命令（需安装 freqtrade：`pip install freqtrade`）：
```bash
# 下载数据（由 freqtrade 使用 ccxt 下载）
freqtrade download-data --config freqtrade/config_okx_example.json --timeframe 5m

# 回测
freqtrade backtesting --config freqtrade/config_okx_example.json -s EmaRsiFtStrategy

# 超参优化（示例空间：roi/stoploss）
freqtrade hyperopt --config freqtrade/config_okx_example.json -s EmaRsiFtStrategy --spaces roi stoploss

# 纸交易
freqtrade trade --config freqtrade/config_okx_example.json -s EmaRsiFtStrategy --dry-run
```

### 9. 目录结构（节选）
```
config/
  ├─ settings.yaml
  ├─ okx_markets.json
  └─ trading.yaml
scripts/
  ├─ fetch_ohlcv.py          # 历史行情拉取
  ├─ run_backtest.py         # Backtrader 回测
  ├─ sync_okx_markets.py     # 交易所元数据
  ├─ check_account.py        # 账户与连通性检查
  └─ order_executor.py       # 纸交易/实盘执行（最小可用）
live/
  ├─ okx_client.py
  ├─ precision.py
  ├─ risk.py
  ├─ runner_ema_rsi.py
  └─ state/
freqtrade/
  ├─ config_okx_example.json
  └─ strategies/EmaRsiFtStrategy.py
strategies/
  └─ ema_rsi_backtrader.py
data/
  ├─ raw/
  └─ processed/
backtests/
notebooks/
```

### 下一步
- 将回测策略信号接入 `order_executor.py`（策略 → 交易信号 → 下单），完善风控（止损、减少频繁交易、日损阈值）
- 使用 Freqtrade 进行纸交易与超参优化，并逐步灰度到真盘
