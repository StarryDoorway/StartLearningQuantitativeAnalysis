import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. 下载数据
data = yf.download("AAPL", start="2022-01-01", end="2023-01-01")

# 2. 计算短期/长期均线
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA20"] = data["Close"].rolling(window=20).mean()

# 3. 生成交易信号
data["Signal"] = 0
data.loc[data["MA5"] > data["MA20"], "Signal"] = 1   # 短期均线在上 → 买入
data.loc[data["MA5"] < data["MA20"], "Signal"] = -1  # 短期均线在下 → 卖出

# 4. 标记买入和卖出点
data["Buy_Signal"] = (data["Signal"] == 1) & (data["Signal"].shift(1) != 1)
data["Sell_Signal"] = (data["Signal"] == -1) & (data["Signal"].shift(1) != -1)

# 5. 策略收益回测
data["Daily_Return"] = data["Close"].pct_change()
data["Strategy_Return"] = data["Signal"].shift(1) * data["Daily_Return"]  # 用前一天信号决定今天仓位

cumulative_strategy = (1 + data["Strategy_Return"].dropna()).cumprod()
cumulative_buyhold = (1 + data["Daily_Return"].dropna()).cumprod()

# 6. 画图：股价 + 均线 + 买卖点
plt.figure(figsize=(14,7))
plt.plot(data["Close"], label="Price", alpha=0.6)
plt.plot(data["MA5"], label="MA5", alpha=0.8)
plt.plot(data["MA20"], label="MA20", alpha=0.8)

# 买入点：绿色圆点
plt.scatter(data.index[data["Buy_Signal"]],
            data["Close"][data["Buy_Signal"]],
            color="green", s=60, edgecolors='black', label="Buy Signal", alpha=0.8, zorder=5)

# 卖出点：红色圆点
plt.scatter(data.index[data["Sell_Signal"]],
            data["Close"][data["Sell_Signal"]],
            color="red", s=60, edgecolors='black', label="Sell Signal", alpha=0.8, zorder=5)

plt.title("AAPL Moving Average Crossover Strategy")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# 7. 策略 vs 买入持有表现
plt.figure(figsize=(14,7))
plt.plot(cumulative_strategy, label="Strategy", color="blue")
plt.plot(cumulative_buyhold, label="Buy & Hold", color="orange")
plt.title("Cumulative Returns: Strategy vs Buy & Hold")
plt.legend()
plt.show()
