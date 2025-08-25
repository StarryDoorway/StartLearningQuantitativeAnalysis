## OKX 加密货币量化机器人（团队规范版）

### 1. 环境准备
- Python 3.10+
- 建议创建虚拟环境：
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置（支持模拟盘/真盘切换）
- 新建并编辑 `src/config/.env`：
```
# 切换环境：模拟盘 true；真盘 false
OKX_TESTNET=true

# 模拟盘（Demo Trading）API（使用模拟盘时填写）
OKX_API_KEY_TEST=
OKX_SECRET_KEY_TEST=

# 真盘 API（使用真盘时填写）
OKX_API_KEY=
OKX_SECRET_KEY=

# 两环境共用或分别设置
OKX_PASSPHRASE=

# 可选代理
HTTP_PROXY=
HTTPS_PROXY=
```
- 其它默认配置：
  - 数据采集：`src/config/settings.yaml`
  - 交易/风控：`src/config/trading.yaml`

### 3. 同步市场元数据（公共接口，无需密钥）
```bash
python -m src.scripts.sync_okx_markets
# 生成：src/config/okx_markets.json
```

### 4. 拉取历史 K 线（CCXT 公共接口）
```bash
python -m src.scripts.fetch_ohlcv \
  --symbols BTC/USDT:USDT ETH/USDT:USDT \
  --timeframes 5m 1h \
  --since 2024-01-01
# 输出到：src/data/raw/<symbol-slug>/<timeframe>.parquet
```

### 5. 回测（Backtrader）
```bash
python -m src.scripts.run_backtest \
  --symbol-slug btc-usdt-usdt \
  --timeframe 5m \
  --cash 10000 \
  --commission 0.0005 \
  --plot
# 图表在项目根的 backtests/（如未自动创建，可自行创建）
```

### 6. 账户检查（优先私有，失败回退公共）
```bash
python -m src.scripts.check_account
# 真盘/可用模拟盘会显示余额；若模拟盘限制或鉴权问题，会提示并回退公共检查
```

### 7. 下单执行（纸/真一体，统一风控与精度校验）
- 配置交易参数：`src/config/trading.yaml`
- 纸交易（不真实下单）：
```bash
python -m src.scripts.order_executor --side buy --type limit --price 30000 --paper
```
- 真盘（谨慎小额灰度，确认 OKX_TESTNET=false 且真盘 API）：
```bash
python -m src.scripts.order_executor --side buy --type limit --price 30000
```

### 8. 目录结构（团队化）
```
src/
  config/
    .env                    # 密钥与环境
    okx_markets.json        # 市场元数据（脚本生成）
    settings.yaml           # 数据采集默认配置
    trading.yaml            # 执行风控配置
  core/
    okx_client.py           # 统一 OKX 客户端（testnet/真盘自动选择）
  utils/
    precision.py            # 精度与最小下单量校验
    risk.py                 # 基础风控
  scripts/
    __init__.py
    sync_okx_markets.py     # 公共接口获取市场元数据
    fetch_ohlcv.py          # 历史 K 线抓取（公共接口）
    run_backtest.py         # Backtrader 回测
    check_account.py        # 账户/连通性检查（私有优先，失败回退公共）
    order_executor.py       # 纸/真执行器（风控+精度校验+幂等 clOrdId）
  strategies/
    ema_rsi_backtrader.py   # 示例策略（EMA+RSI+ATR）
  data/
    raw/                    # 原始 K 线保存目录（parquet）
    processed/              # 后续特征或清洗数据
```

### 9. 常见问题
- 50101：密钥环境不匹配（真/模拟盘错置）。检查 `OKX_TESTNET` 与使用的 API Key 是否对应。
- 50038：模拟盘禁用部分私有接口。账户检查会自动回退公共检查。


