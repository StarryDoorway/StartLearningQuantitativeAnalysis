# -*- coding: utf-8 -*-

import backtrader as bt
import backtrader.indicators as btind
from loguru import logger


class RsiDivergenceStrategy(bt.Strategy):
    """
    RSI背离策略
    
    核心逻辑：
    1. 当RSI进入超卖区间(<30)后继续下降，但价格没有明显下降甚至上升时，买入
    2. 当RSI进入超买区间(>70)后继续上升，但价格没有明显上升甚至下降时，卖出
    
    这种背离现象表明价格动能可能即将反转。
    """
    
    params = dict(
        # RSI参数
        rsi_period=14,          # RSI周期，14是标准设置，适用于大多数时间框架
        rsi_overbought=70,       # 超买阈值，70是标准设置
        rsi_oversold=30,         # 超卖阈值，30是标准设置
        
        # 背离检测参数
        divergence_lookback=5,   # 背离检测回看K线数，5根K线足够检测短期背离
        price_change_threshold=0.005,  # 价格变化阈值，0.5%的变化被认为是"明显"变化
        rsi_change_threshold=2.0,  # RSI变化阈值，2点的变化被认为是"明显"变化
        
        # 风险管理参数
        risk_per_trade=0.02,     # 每笔交易风险比例，2%是常见的风险控制水平
        stop_loss_type='atr',    # 止损类型，ATR止损能更好地适应市场波动
        atr_period=14,           # ATR周期，与RSI周期保持一致
        atr_mult=2.0,            # ATR倍数，2倍ATR提供了合理的止损距离
        take_profit_mult=3.0,    # 盈利倍数，3:1的风险回报比是常见的目标
        
        # 信号确认参数
        confirm_bars=1,          # 信号确认K线数，1根K线确认即可，避免错过机会
        min_trade_bars=20,       # 最少交易K线数，确保有足够数据计算指标
    )
    
    def __init__(self):
        # RSI指标
        self.rsi = btind.RSI(self.data.close, period=self.p.rsi_period)
        
        # ATR指标，用于止损和仓位计算
        self.atr = btind.ATR(self.data, period=self.p.atr_period)
        
        # 价格平滑指标，用于识别价格趋势
        self.sma = btind.SMA(self.data.close, period=20)
        
        # 交易状态变量
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.take_profit_price = None
        
        # 信号确认计数器
        self.buy_signal_count = 0
        self.sell_signal_count = 0
        
        # 背离检测相关变量
        self.last_rsi_low = None
        self.last_price_low = None
        self.last_rsi_high = None
        self.last_price_high = None
        
    def next(self):
        # 确保有足够的数据
        if len(self) < self.p.min_trade_bars:
            return
            
        # 更新背离检测的极值点
        self._update_extremes()
        
        # 如果有未完成的订单，不执行新交易
        if self.order:
            return
            
        # 如果没有持仓，检查买入信号
        if not self.position:
            self._check_buy_signal()
        else:
            # 如果有持仓，检查卖出信号或止损
            self._check_exit_conditions()
    
    def _update_extremes(self):
        """更新RSI和价格的极值点，用于背离检测"""
        current_rsi = self.rsi[0]
        current_price = self.data.close[0]
        
        # 更新RSI和价格的低点
        if self.last_rsi_low is None or current_rsi < self.last_rsi_low:
            self.last_rsi_low = current_rsi
            self.last_price_low = current_price
            
        # 更新RSI和价格的高点
        if self.last_rsi_high is None or current_rsi > self.last_rsi_high:
            self.last_rsi_high = current_rsi
            self.last_price_high = current_price
    
    def _check_buy_signal(self):
        """检查买入信号（RSI超卖背离）"""
        # RSI进入超卖区间
        if self.rsi[0] < self.p.rsi_oversold:
            # 检查是否有足够的背离数据
            if self.last_rsi_low is None or self.last_price_low is None:
                return
                
            # 获取过去几根K线的RSI和价格数据
            past_rsi = [self.rsi[-i] for i in range(1, self.p.divergence_lookback+1) if len(self.rsi) > i]
            past_prices = [self.data.close[-i] for i in range(1, self.p.divergence_lookback+1) if len(self) > i]
            
            if not past_rsi or not past_prices:
                return
                
            # 检查RSI是否仍在下降（动能不减）
            rsi_decreasing = all(past_rsi[i] > past_rsi[i+1] for i in range(len(past_rsi)-1))
            rsi_still_decreasing = past_rsi[-1] > self.rsi[0]
            
            # 检查价格是否没有明显下降（甚至上升）
            price_change = (self.data.close[0] - past_prices[-1]) / past_prices[-1]
            price_not_decreasing = price_change > -self.p.price_change_threshold
            
            # 检查RSI变化是否足够明显
            rsi_change = past_rsi[0] - self.rsi[0]
            rsi_significant_change = rsi_change > self.p.rsi_change_threshold
            
            # 确认背离买入信号
            if (rsi_decreasing and rsi_still_decreasing and 
                price_not_decreasing and rsi_significant_change):
                self.buy_signal_count += 1
            else:
                self.buy_signal_count = 0
                
            # 当信号持续指定周期后执行买入
            if self.buy_signal_count >= self.p.confirm_bars:
                self._execute_buy()
                self.buy_signal_count = 0
        else:
            self.buy_signal_count = 0
    
    def _check_sell_signal(self):
        """检查卖出信号（RSI超买背离）"""
        # RSI进入超买区间
        if self.rsi[0] > self.p.rsi_overbought:
            # 检查是否有足够的背离数据
            if self.last_rsi_high is None or self.last_price_high is None:
                return
                
            # 获取过去几根K线的RSI和价格数据
            past_rsi = [self.rsi[-i] for i in range(1, self.p.divergence_lookback+1) if len(self.rsi) > i]
            past_prices = [self.data.close[-i] for i in range(1, self.p.divergence_lookback+1) if len(self) > i]
            
            if not past_rsi or not past_prices:
                return
                
            # 检查RSI是否仍在上升（动能不减）
            rsi_increasing = all(past_rsi[i] < past_rsi[i+1] for i in range(len(past_rsi)-1))
            rsi_still_increasing = past_rsi[-1] < self.rsi[0]
            
            # 检查价格是否没有明显上升（甚至下降）
            price_change = (self.data.close[0] - past_prices[-1]) / past_prices[-1]
            price_not_increasing = price_change < self.p.price_change_threshold
            
            # 检查RSI变化是否足够明显
            rsi_change = self.rsi[0] - past_rsi[0]
            rsi_significant_change = rsi_change > self.p.rsi_change_threshold
            
            # 确认背离卖出信号
            if (rsi_increasing and rsi_still_increasing and 
                price_not_increasing and rsi_significant_change):
                self.sell_signal_count += 1
            else:
                self.sell_signal_count = 0
                
            # 当信号持续指定周期后执行卖出
            if self.sell_signal_count >= self.p.confirm_bars:
                self._execute_sell()
                self.sell_signal_count = 0
        else:
            self.sell_signal_count = 0
    
    def _check_exit_conditions(self):
        """检查退出条件（止损、止盈或卖出信号）"""
        current_price = self.data.close[0]
        
        # 检查止损
        if self.stop_price is not None and current_price <= self.stop_price:
            self.order = self.close()
            return
            
        # 检查止盈
        if self.take_profit_price is not None and current_price >= self.take_profit_price:
            self.order = self.close()
            return
            
        # 检查卖出信号
        self._check_sell_signal()
    
    def _execute_buy(self):
        """执行买入操作"""
        # 计算仓位大小
        size = self._calculate_position_size()
        if size <= 0:
            return
            
        # 执行买入
        self.order = self.buy(size=size)
        
        # 记录入场价格
        self.entry_price = self.data.close[0]
        
        # 计算止损和止盈价格
        self._calculate_stop_and_profit()
    
    def _execute_sell(self):
        """执行卖出操作（如果有持仓）"""
        if self.position:
            self.order = self.close()
    
    def _calculate_position_size(self):
        """基于风险计算仓位大小"""
        cash = self.broker.getcash()
        risk_amount = cash * self.p.risk_per_trade
        
        if self.p.stop_loss_type == 'atr' and len(self.atr) > 0:
            atr_value = float(self.atr[0])
            if atr_value > 0:
                # 基于ATR的仓位计算
                risk_per_unit = atr_value * self.p.atr_mult
                size = risk_amount / risk_per_unit
                return size
        
        # 默认使用固定百分比计算
        price = self.data.close[0]
        size = cash * self.p.risk_per_trade / price
        return size
    
    def _calculate_stop_and_profit(self):
        """计算止损和止盈价格"""
        if self.entry_price is None:
            return
            
        if self.p.stop_loss_type == 'atr' and len(self.atr) > 0:
            atr_value = float(self.atr[0])
            if atr_value > 0:
                # 基于ATR的止损
                self.stop_price = self.entry_price - (atr_value * self.p.atr_mult)
                # 基于ATR的止盈（风险回报比）
                self.take_profit_price = self.entry_price + (atr_value * self.p.atr_mult * self.p.take_profit_mult)
        else:
            # 固定百分比止损
            self.stop_price = self.entry_price * (1 - self.p.risk_per_trade)
            # 固定百分比止盈
            self.take_profit_price = self.entry_price * (1 + self.p.risk_per_trade * self.p.take_profit_mult)
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行: 价格 {order.executed.price:.2f}, 成本 {order.executed.value:.2f}, '
                         f'手续费 {order.executed.comm:.2f}, 数量 {order.executed.size:.2f}')
            elif order.issell():
                self.log(f'卖出执行: 价格 {order.executed.price:.2f}, 成本 {order.executed.value:.2f}, '
                         f'手续费 {order.executed.comm:.2f}, 数量 {order.executed.size:.2f}')
                         
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单取消/保证金不足/被拒绝: {order.getstatusname()}')
            
        self.order = None
    
    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return
            
        self.log(f'交易闭合: 毛利润 {trade.pnl:.2f}, 净利润 {trade.pnlcomm:.2f}')
    
    def log(self, txt, dt=None):
        """记录日志"""
        try:
            dt = dt or self.datas[0].datetime.date(0)
            logger.info(f'{dt.isoformat()} {txt}')
        except Exception as e:
            logger.info(f'日志记录错误: {e}, 内容: {txt}')
        
    def stop(self):
        """回测结束时调用"""
        self.log(f'回测结束 - 最终资产: {self.broker.getvalue():.2f}')