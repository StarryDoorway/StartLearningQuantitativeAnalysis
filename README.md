# 量化交易框架 (Quantitative Trading Framework)

一个功能完整、模块化设计的Python量化交易框架，支持策略开发、回测、风险管理和实盘交易。

## 框架特性

### 核心功能
- **模块化架构**: 清晰的模块分离，易于扩展和维护
- **策略开发**: 提供策略基类，简化策略开发流程
- **回测引擎**: 高效的事件驱动回测引擎，支持多种评估指标
- **风险管理**: 全面的风险控制工具，包括仓位管理、止损止盈等
- **数据管理**: 支持多种数据源，提供数据清洗和验证功能
- **执行引擎**: 模拟交易执行，支持多种订单类型
- **OKX集成**: 完整的OKX交易所API集成，支持模拟盘和实盘交易

### 技术特点
- **高性能**: 基于NumPy和Pandas优化，支持向量化计算
- **可扩展**: 插件式架构，易于添加新功能
- **类型安全**: 使用Python类型注解，提高代码质量
- **文档完善**: 详细的API文档和使用示例

## 项目结构

```
quant_framework/
├── analysis/               # 分析模块
│   ├── __init__.py
│   ├── analyzer.py         # 性能分析器
│   ├── report_generator.py # 报告生成器
│   └── visualizer.py       # 结果可视化
├── clients/              # 交易所接口
│   ├── __init__.py
│   └── okx_client.py       # OKX客户端
│   ├── okx_markets.json    # OKX市场信息
│   └── .env                # 密钥配置
├── config/                 # 配置文件
│   ├── __init__.py
│   ├── config.yaml         # 项目配置文件
│   ├── trading_config.yaml # 交易参数配置文件
├── core/                   # 核心模块
│   ├── __init__.py
│   ├── common.py           # 通用功能
│   ├── event_bus.py        # 事件总线
│   ├── backtest_engine/    # 回测引擎
│   ├── data_engine/        # 数据引擎
│   ├── live_engine/        # 实盘引擎
│   └── risk_engine/        # 风险引擎
├── data/                   # 数据管理
│   ├── __init__.py
│   ├── data_manager.py     # 数据管理器
│   ├── data_processor.py   # 数据处理器
│   ├── data_sources.py     # 数据源
│   ├── features/           # 特征数据
│   ├── processed/          # 处理后数据
│   └── raw/                # 原始数据
├── execution/              # 执行引擎
│   ├── __init__.py
│   ├── brokers/            # 经纪商接口
│   ├── execution_engine.py  # 执行引擎
│   ├── order_manager/      # 订单管理
│   ├── order_manager.py    # 订单管理器
│   └── portfolio_manager.py # 投资组合管理
├── persistence/            # 数据持久化
│   ├── __init__.py
│   ├── connection.py       # 数据库连接
│   ├── dao.py              # 数据访问对象
│   ├── models.py           # 数据模型
│   └── warehouse.py       # 数据仓库
├── scripts/                # 脚本工具
│   ├── __init__.py
│   ├── check_account.py    # 账户检查
│   ├── fetch_ohlcv.py      # 获取OHLCV数据
│   ├── order_executor.py   # 订单执行
│   ├── run_backtest.py     # 运行回测
│   └── sync_okx_markets.py # 同步OKX市场数据
├── storage/                # 存储模块
│   ├── __init__.py
│   ├── cache/              # 缓存
│   └── database/           # 数据库
├── strategies/             # 策略模块
│   ├── __init__.py
│   ├── strategy_base.py    # 策略基类
│   ├── strategy_data_manager.py # 策略数据管理
│   ├── strategy_executor.py # 策略执行器
│   ├── strategy_performance_manager.py # 策略性能管理
│   ├── strategy_risk_manager.py # 策略风险管理
│   ├── strategy_state.py   # 策略状态
│   ├── signal_generators/  # 信号生成器
│   ├── indicators/         # 策略指标
│   ├── arbitrage_strategy.py # 套利策略
│   ├── ema_rsi_strategy.py # EMA RSI策略
│   ├── mean_reversion_strategy.py # 均值回归策略
│   ├── momentum_strategy.py # 动量策略
│   ├── rsi_divergence_strategy.py # RSI背离策略
│   └── trend_following_strategy.py # 趋势跟踪策略
├── tests/                  # 测试模块
│   └── __init__.py
└── utils/                  # 工具模块
    ├── __init__.py
    ├── config_loader.py    # 配置加载器
    ├── data_validation.py  # 数据验证
    ├── indicators.py       # 技术指标
    ├── risk_utils.py       # 风险工具
    └── trading_utils.py    # 交易工具
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

#### 1. 数据获取

```bash
# 获取OHLCV数据
python quant_framework/scripts/fetch_ohlcv.py \
  --symbols BTC/USDT:USDT ETH/USDT:USDT \
  --timeframes 5m 15m 1h 4h 1d \
  --since 2024-01-01
```

#### 2. 回测

```bash
# 运行回测
python quant_framework/scripts/run_backtest.py \
  --symbol-slug btc-usdt-usdt \
  --timeframe 15m \
  --cash 10000 \
  --commission 0.0005 \
  --plot
```

#### 2.1 回测工具使用

#### 基本回测命令

```bash
# 使用 Backtrader 回测引擎
python quant_framework/scripts/run_backtest.py --symbol-slug btc-usdt-usdt --timeframe 1h --cash 10000 --commission 0.0005 --plot

# 使用简化回测引擎
python quant_framework/scripts/run_backtest_simple_engine.py --symbol-slug btc-usdt-usdt --timeframe 1h --cash 10000 --commission 0.0005 --plot

# 使用自定义回测引擎（需要配置文件支持）
python quant_framework/scripts/run_backtest_custom_engine.py --symbol-slug btc-usdt-usdt --timeframe 1h --cash 10000 --commission 0.0005 --plot
```

#### 批量回测示例

```bash
# 对多个交易对和时间框架进行回测
for symbol in btc-usdt-usdt eth-usdt-usdt; do
  for tf in 15m 1h; do
    python quant_framework/scripts/run_backtest.py --symbol-slug $symbol --timeframe $tf --cash 10000 --commission 0.0005 --plot
  done
done
```

#### 回测结果查看

```bash
# 查看所有回测结果
python backtest_viewer.py

# 查看特定回测图表
python backtest_viewer.py --view btc-usdt-usdt_1h
```

#### 回测引擎对比

项目提供了三种不同的回测实现方式，详细对比请参考 [回测引擎对比指南](BACKTEST_COMPARISON.md)。

### 2.2 回测引擎对比

本项目提供了三种不同的回测实现方式：

1. **Backtrader 回测引擎**：使用成熟的 Backtrader 框架，适合作为基准测试
2. **自定义回测引擎**：使用项目自定义的事件驱动回测引擎，支持多种回测模式
3. **简化回测引擎**：轻量级实现，不依赖配置文件，适合快速验证策略

详细对比和使用指南请参考 [回测引擎对比文档](BACKTEST_COMPARISON.md)。

#### 3. 策略开发

```python
from quant_framework.strategies import StrategyBase, Signal, SignalType
from quant_framework.utils import sma, ema

class CustomStrategy(StrategyBase):
    def __init__(self, name, symbols, fast_period=10, slow_period=20):
        super().__init__(name, symbols)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def _initialize(self):
        """初始化策略参数和指标"""
        for symbol in self.symbols:
            self.indicators[symbol] = {
                'fast_ma': [],
                'slow_ma': []
            }
    
    def calculate_signals(self, data):
        """计算交易信号"""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in data:
                continue
                
            df = data[symbol]
            if len(df) < self.slow_period:
                continue
            
            # 计算指标
            fast_ma = ema(df['close'], self.fast_period)
            slow_ma = ema(df['close'], self.slow_period)
            
            # 更新指标
            self.indicators[symbol]['fast_ma'].append(fast_ma.iloc[-1])
            self.indicators[symbol]['slow_ma'].append(slow_ma.iloc[-1])
            
            # 生成信号
            if len(self.indicators[symbol]['fast_ma']) < 2:
                continue
                
            fast_prev = self.indicators[symbol]['fast_ma'][-2]
            fast_curr = self.indicators[symbol]['fast_ma'][-1]
            slow_prev = self.indicators[symbol]['slow_ma'][-2]
            slow_curr = self.indicators[symbol]['slow_ma'][-1]
            
            # 金叉买入
            if fast_prev <= slow_prev and fast_curr > slow_curr:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=df['close'].iloc[-1],
                    timestamp=df.index[-1],
                    strength=0.7
                )
                signals.append(signal)
            
            # 死叉卖出
            elif fast_prev >= slow_prev and fast_curr < slow_curr:
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=df['close'].iloc[-1],
                    timestamp=df.index[-1],
                    strength=0.7
                )
                signals.append(signal)
        
        return signals
```

## OKX 加密货币量化交易

### 1. 环境准备

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置

复制并编辑配置文件：

```bash
cp quant_framework/clients/.env.example quant_framework/clients/.env
```
编辑 `quant_framework/clients/.env`：

### 3. 同步市场数据

```bash
python quant_framework/scripts/sync_okx_markets.py
```

### 4. 账户检查

```bash
python quant_framework/scripts/check_account.py
```

### 5. 订单执行

#### 5.1 模拟交易

模拟交易允许您在不使用真实资金的情况下测试策略：

```bash
# 基本模拟交易
python quant_framework/scripts/order_executor.py --side buy --type market --symbol BTC/USDT:USDT --amount 0.001 --paper

# 模拟限价单
python quant_framework/scripts/order_executor.py --side sell --type limit --symbol ETH/USDT:USDT --amount 0.1 --price 2000 --paper

# 模拟止损单
python quant_framework/scripts/order_executor.py --side sell --type stop --symbol BTC/USDT:USDT --amount 0.001 --price 25000 --paper
```

#### 5.2 模拟盘交易详细操作

1. **运行策略回测**

```bash
# 使用历史数据回测
python quant_framework/scripts/run_backtest.py \
  --symbol-slug btc-usdt-usdt \
  --timeframe 1h \
  --cash 10000 \
  --commission 0.0005 \
  --plot
```



#### 5.4 模拟盘与实盘的差异

1. **执行延迟**：模拟盘没有真实的网络延迟和交易所处理时间
2. **市场影响**：模拟盘不考虑订单对市场价格的影响
3. **流动性限制**：模拟盘假设无限流动性，不考虑订单簿深度
4. **滑点模型**：模拟盘使用简化的滑点模型，可能与实际情况有差异

#### 5.5 从模拟盘过渡到实盘

当您在模拟盘上验证了策略的有效性后，可以按照以下步骤过渡到实盘：

1. 逐步减小仓位大小，从最小交易量开始
2. 设置严格的风险控制，如每日最大损失限制
3. 密切监控实际执行情况与模拟结果的差异
4. 根据实际市场情况调整策略参数

### 5.6 实盘交易

```bash
# 实盘交易（请谨慎操作）
python quant_framework/scripts/order_executor.py --side buy --type market --symbol BTC/USDT:USDT --amount 0.001

# 实盘限价单
python quant_framework/scripts/order_executor.py --side sell --type limit --symbol ETH/USDT:USDT --amount 0.1 --price 2000
```

## 技术指标

框架提供了丰富的技术指标计算函数：

```python
from quant_framework.utils import sma, ema, rsi, macd, bollinger_bands, stochastic, atr

# 计算移动平均线
sma_value = sma(price_data, period=20)
ema_value = ema(price_data, period=20)

# 计算RSI
rsi_value = rsi(price_data, period=14)

# 计算MACD
macd_line, signal_line, histogram = macd(price_data)
```

## 风险管理

框架提供了全面的风险管理工具：

```python
from quant_framework.utils import RiskCalculator, PositionSizer, RiskController

# 计算风险指标
risk_metrics = RiskCalculator.calculate_risk_metrics(equity_curve)
print(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {risk_metrics.max_drawdown:.2%}")

# 仓位管理
position_size = PositionSizer.kelly_criterion(win_rate=0.6, avg_win=100, avg_loss=50)

# 风险控制
risk_controller = RiskController(
    max_position_size=0.2,
    max_portfolio_risk=0.02,
    max_drawdown_limit=0.1
)

is_allowed, violations = risk_controller.check_all_limits(
    symbol='BTC/USDT:USDT',
    weight=0.25,
    portfolio_var=0.03,
    current_drawdown=0.12,
    pnl=-0.06
)
```

## 数据验证

框架提供了强大的数据验证功能：

```python
from quant_framework.utils import DataValidator, DataCleaner

# 验证数据
validator = DataValidator()
validation_result = validator.validate_ohlcv_data(data)

if not validation_result.is_valid:
    print("Data validation failed:")
    for error in validation_result.errors:
        print(f"  - {error}")

# 清洗数据
cleaner = DataCleaner()
clean_data = cleaner.handle_missing_values(data, method="forward_fill")
clean_data = cleaner.handle_outliers(clean_data, method="clip")
```

## 性能分析

框架提供了详细的性能分析工具：

```python
from quant_framework.analysis import get_analyzer

# 分析回测结果
analyzer = get_analyzer()
performance_metrics = analyzer.calculate_performance_metrics(equity_curve)

print(f"Total Return: {performance_metrics.total_return:.2%}")
print(f"Annualized Return: {performance_metrics.annualized_return:.2%}")
print(f"Sharpe Ratio: {performance_metrics.sharpe_ratio:.2f}")
print(f"Sortino Ratio: {performance_metrics.sortino_ratio:.2f}")
print(f"Max Drawdown: {performance_metrics.max_drawdown:.2%}")
```



## 常见问题

### 1. 网络连接问题

如果遇到网络超时错误，可以尝试：

1. 检查网络连接
2. 使用代理（在配置文件中设置）
3. 增加重试次数和超时时间

### 2. API限制

OKX API有请求频率限制，建议：

1. 控制请求频率
2. 使用缓存减少重复请求
3. 实现指数退避重试机制

### 3. 数据质量问题

如果遇到数据质量问题，可以使用框架提供的数据验证和清洗工具。

## 贡献指南

欢迎贡献代码！
## 联系方式
如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送邮件至 [964389650@qq.com]

## 更新日志
### v2.0.0 (2024-01-01)
- 重构项目架构，采用更模块化的设计
- 添加OKX交易所完整集成
- 增强风险管理功能
- 优化性能和可扩展性
- 添加更多技术指标和策略示例

### v1.0.0 (2023-01-01)
- 初始版本发布
- 实现核心框架功能
- 添加示例策略和文档