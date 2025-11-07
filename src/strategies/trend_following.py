# -*- coding: utf-8 -*-

import backtrader as bt
import backtrader.indicators as btind


class TrendFollowingStrategy(bt.Strategy):
    params = dict(
        # EMA 参数
        ema_fast=20,
        ema_slow=50,
        # 布林带参数
        bollinger_period=20,
        bollinger_dev=2.0,
        # MACD 参数
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        # 通道突破参数
        channel_length=20,
        # 信号组合方式: 'any', 'all', 'ema_bollinger', 'ema_macd', 'bollinger_macd'
        signal_combination='any',
        # 风险管理参数
        risk_per_trade=0.01,  # 每笔交易风险资金比例
        stop_loss_type='atr',  # 'atr', 'fixed_percent', 'trailing'
        atr_period=14,
        atr_mult=2.0,
        fixed_stop_percent=0.02,
        trailing_stop_percent=0.03,
        # 确认阈值，减少假信号
        confirm_threshold=1,  # 信号持续确认的周期数
    )

    def __init__(self):
        # 初始化指标
        # 1. EMA指标
        self.ema_fast = btind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = btind.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_crossover = btind.CrossOver(self.ema_fast, self.ema_slow)

        # 2. 布林带指标
        self.bollinger = btind.BollingerBands(
            self.data.close,
            period=self.p.bollinger_period,
            devfactor=self.p.bollinger_dev
        )

        # 3. MACD指标
        self.macd = btind.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )

        # 4. 通道突破指标
        self.highest = btind.Highest(self.data.high, period=self.p.channel_length)
        self.lowest = btind.Lowest(self.data.low, period=self.p.channel_length)

        # 风险管理指标
        self.atr = btind.ATR(self.data, period=self.p.atr_period)

        # 交易状态
        self.entry_price = None
        self.stop_price = None
        self.entry_date = None

        # 信号计数器
        self.buy_signal_count = 0
        self.sell_signal_count = 0

    def next(self):
        if not self.position:
            # 计算买入信号
            buy_signal = self._calculate_buy_signal()
            
            # 信号确认机制
            if buy_signal:
                self.buy_signal_count += 1
            else:
                self.buy_signal_count = 0

            # 当信号持续指定周期后才执行买入
            if self.buy_signal_count >= self.p.confirm_threshold:
                # 计算仓位大小
                size = self._calculate_position_size()
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = self.data.close[0]
                    self.entry_date = len(self)
                    self.stop_price = self._calculate_stop_loss(self.entry_price)
                    self.buy_signal_count = 0  # 重置计数器
        else:
            # 检查卖出信号
            sell_signal = self._calculate_sell_signal()
            price = self.data.close[0]
            
            # 信号确认机制
            if sell_signal:
                self.sell_signal_count += 1
            else:
                self.sell_signal_count = 0

            # 当信号持续指定周期后才执行卖出
            sell_signal_confirmed = self.sell_signal_count >= self.p.confirm_threshold
            
            # 检查止损条件
            stop_hit = False
            if self.stop_price is not None and price <= self.stop_price:
                stop_hit = True
                self.sell_signal_count = 0  # 重置计数器
                
            # 更新追踪止损
            if self.p.stop_loss_type == 'trailing' and price > self.entry_price:
                new_stop = price * (1 - self.p.trailing_stop_percent)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

            # 执行卖出
            if sell_signal_confirmed or stop_hit:
                self.close()
                self.entry_price = None
                self.stop_price = None
                self.entry_date = None
                self.sell_signal_count = 0

    def _calculate_buy_signal(self):
        # EMA买入信号
        ema_buy = self.ema_crossover[0] > 0
        
        # 布林带上轨突破信号
        bollinger_buy = self.data.close[0] > self.bollinger.top[0]
        
        # MACD买入信号
        macd_buy = self.macd.lines.macd[0] > self.macd.lines.signal[0]
        
        # 通道上轨突破信号
        channel_buy = self.data.close[0] > self.highest[-1]
        
        # 根据组合方式返回最终信号
        if self.p.signal_combination == 'any':
            return ema_buy or bollinger_buy or macd_buy or channel_buy
        elif self.p.signal_combination == 'all':
            return ema_buy and bollinger_buy and macd_buy and channel_buy
        elif self.p.signal_combination == 'ema_bollinger':
            return ema_buy and bollinger_buy
        elif self.p.signal_combination == 'ema_macd':
            return ema_buy and macd_buy
        elif self.p.signal_combination == 'bollinger_macd':
            return bollinger_buy and macd_buy
        else:
            return ema_buy

    def _calculate_sell_signal(self):
        # EMA卖出信号
        ema_sell = self.ema_crossover[0] < 0
        
        # 布林带下轨突破信号
        bollinger_sell = self.data.close[0] < self.bollinger.bot[0]
        
        # MACD卖出信号
        macd_sell = self.macd.lines.macd[0] < self.macd.lines.signal[0]
        
        # 通道下轨突破信号
        channel_sell = self.data.close[0] < self.lowest[-1]
        
        # 根据组合方式返回最终信号
        if self.p.signal_combination == 'any':
            return ema_sell or bollinger_sell or macd_sell or channel_sell
        elif self.p.signal_combination == 'all':
            return ema_sell and bollinger_sell and macd_sell and channel_sell
        elif self.p.signal_combination == 'ema_bollinger':
            return ema_sell and bollinger_sell
        elif self.p.signal_combination == 'ema_macd':
            return ema_sell and macd_sell
        elif self.p.signal_combination == 'bollinger_macd':
            return bollinger_sell and macd_sell
        else:
            return ema_sell

    def _calculate_position_size(self):
        # 基于风险计算仓位大小
        cash = self.broker.getcash()
        risk_amount = cash * self.p.risk_per_trade
        
        if self.p.stop_loss_type == 'atr' and len(self.atr) > 0:
            atr_value = float(self.atr[0])
            if atr_value > 0:
                # 基于ATR的仓位计算
                risk_per_unit = atr_value * self.p.atr_mult
                size = risk_amount / risk_per_unit
                return size
        
        # 固定百分比止损的仓位计算
        elif self.p.stop_loss_type == 'fixed_percent' or self.p.stop_loss_type == 'trailing':
            price = self.data.close[0]
            risk_per_unit = price * self.p.fixed_stop_percent
            if risk_per_unit > 0:
                size = risk_amount / risk_per_unit
                return size
        
        # 默认仓位：使用可用资金的95%
        price = self.data.close[0]
        return cash * 0.95 / price

    def _calculate_stop_loss(self, entry_price):
        # 计算止损价格
        if self.p.stop_loss_type == 'atr' and len(self.atr) > 0:
            atr_value = float(self.atr[0])
            return entry_price - atr_value * self.p.atr_mult
        elif self.p.stop_loss_type == 'fixed_percent':
            return entry_price * (1 - self.p.fixed_percent)
        elif self.p.stop_loss_type == 'trailing':
            return entry_price * (1 - self.p.trailing_stop_percent)
        else:
            # 默认止损：入场价格的2%
            return entry_price * 0.98

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已接受，不需要处理
            return

        # 检查订单是否已完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}')
                self.log(f'买入成本: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}')
            else:
                self.log(f'卖出执行: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}')
                self.log(f'卖出收入: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单状态: {order.getstatusname()}')

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'交易结束, 利润: {trade.pnl:.2f}, 净利润: {trade.pnlcomm:.2f}')

    def log(self, txt, dt=None):
        '''记录交易日志'''        
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')