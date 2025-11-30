"""
套利策略信号生成器

包含统计套利、三角套利和跨交易所套利的信号生成方法。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from .arbitrage_indicators import ArbitrageIndicators


class ArbitrageSignalGenerator:
    """
    套利策略信号生成器
    
    负责根据套利指标生成各种类型的套利交易信号。
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化信号生成器
        
        Args:
            params: 信号生成参数
        """
        self.params = params or {}
        
        # 默认参数
        self.default_params = {
            'z_entry_threshold': 2.0,    # Z-score入场阈值
            'z_exit_threshold': 0.5,     # Z-score出场阈值
            'min_correlation': 0.7,      # 最小相关性
            'min_spread': 0.001,         # 最小价差比例
            'max_spread': 0.1,           # 最大价差比例
            'min_profit_ratio': 0.002,   # 最小利润比例
            'triangular_threshold': 0.01, # 三角套利阈值
            'cross_exchange_threshold': 0.005, # 跨交易所套利阈值
            'signal_strength_factor': 1.0, # 信号强度因子
            'confidence_threshold': 0.6   # 置信度阈值
        }
        
        # 合并参数
        for key, value in self.default_params.items():
            if key not in self.params:
                self.params[key] = value
    
    def generate_statistical_arbitrage_signals(
        self, 
        pair_data: Dict[str, pd.DataFrame],
        spreads: Dict[str, pd.Series],
        zscores: Dict[str, pd.Series],
        hedge_ratios: Dict[str, float],
        correlations: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        生成统计套利信号
        
        Args:
            pair_data: 交易对数据
            spreads: 价差数据
            zscores: Z-score数据
            hedge_ratios: 对冲比率
            correlations: 相关性
            
        Returns:
            交易信号字典
        """
        signals = {}
        
        for pair, spread in spreads.items():
            if pair not in zscores or pair not in correlations:
                continue
                
            # 检查相关性
            if correlations[pair] < self.params['min_correlation']:
                continue
                
            # 获取当前Z-score
            current_z = zscores[pair].iloc[-1]
            prev_z = zscores[pair].iloc[-2] if len(zscores[pair]) > 1 else current_z
            
            # 解析交易对
            symbol1, symbol2 = pair.split('/')
            
            # 生成信号
            signal = self._generate_pair_signals(
                symbol1, symbol2, current_z, prev_z, 
                hedge_ratios.get(pair, 1.0), correlations[pair]
            )
            
            if signal:
                signals[pair] = signal
                
        return signals
    
    def generate_triangular_arbitrage_signals(
        self,
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        生成三角套利信号
        
        Args:
            price_data: 价格数据
            
        Returns:
            三角套利信号字典
        """
        signals = {}
        
        # 获取所有可用符号
        symbols = list(price_data.keys())
        
        # 检查所有可能的三元组
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                for k in range(j+1, len(symbols)):
                    symbol_a, symbol_b, symbol_c = symbols[i], symbols[j], symbols[k]
                    
                    # 检查是否有足够的数据
                    if (symbol_a not in price_data or 
                        symbol_b not in price_data or 
                        symbol_c not in price_data):
                        continue
                        
                    # 获取最新价格
                    price_a = price_data[symbol_a]['close'].iloc[-1]
                    price_b = price_data[symbol_b]['close'].iloc[-1]
                    price_c = price_data[symbol_c]['close'].iloc[-1]
                    
                    # 计算三角套利机会
                    opportunity = ArbitrageIndicators.calculate_triangular_arbitrage_opportunity(
                        price_a, price_b, price_c
                    )
                    
                    # 检查是否满足阈值
                    if opportunity['profit_pct'] > self.params['min_profit_ratio']:
                        # 生成信号
                        signal = self._generate_triangular_signals(
                            symbol_a, symbol_b, symbol_c, opportunity
                        )
                        
                        if signal:
                            triangle_key = f"{symbol_a}/{symbol_b}/{symbol_c}"
                            signals[triangle_key] = signal
                            
        return signals
    
    def generate_cross_exchange_arbitrage_signals(
        self,
        exchange_data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        生成跨交易所套利信号
        
        Args:
            exchange_data: 交易所数据 {exchange: {symbol: data}}
            
        Returns:
            跨交易所套利信号字典
        """
        signals = {}
        
        # 获取所有交易所和符号
        exchanges = list(exchange_data.keys())
        
        # 检查所有交易所对
        for i in range(len(exchanges)):
            for j in range(i+1, len(exchanges)):
                exchange_a, exchange_b = exchanges[i], exchanges[j]
                
                # 获取共同符号
                symbols_a = set(exchange_data[exchange_a].keys())
                symbols_b = set(exchange_data[exchange_b].keys())
                common_symbols = symbols_a.intersection(symbols_b)
                
                # 检查每个共同符号
                for symbol in common_symbols:
                    # 获取价格数据
                    data_a = exchange_data[exchange_a][symbol]
                    data_b = exchange_data[exchange_b][symbol]
                    
                    # 获取最新价格
                    price_a = data_a['close'].iloc[-1]
                    price_b = data_b['close'].iloc[-1]
                    
                    # 计算套利机会
                    opportunity = ArbitrageIndicators.calculate_cross_exchange_arbitrage_opportunity(
                        price_a, price_b
                    )
                    
                    # 检查是否满足阈值
                    if opportunity['profit_pct'] > self.params['min_profit_ratio']:
                        # 生成信号
                        signal = self._generate_cross_exchange_signals(
                            exchange_a, exchange_b, symbol, opportunity
                        )
                        
                        if signal:
                            signal_key = f"{exchange_a}:{exchange_b}:{symbol}"
                            signals[signal_key] = signal
                            
        return signals
    
    def _generate_pair_signals(
        self,
        symbol1: str,
        symbol2: str,
        current_z: float,
        prev_z: float,
        hedge_ratio: float,
        correlation: float
    ) -> Optional[Dict[str, Any]]:
        """
        生成交易对信号
        
        Args:
            symbol1: 第一个符号
            symbol2: 第二个符号
            current_z: 当前Z-score
            prev_z: 前一个Z-score
            hedge_ratio: 对冲比率
            correlation: 相关性
            
        Returns:
            交易信号或None
        """
        # 入场阈值
        entry_threshold = self.params['z_entry_threshold']
        exit_threshold = self.params['z_exit_threshold']
        
        # 初始化信号
        signal = None
        
        # 检查入场条件
        if current_z > entry_threshold and prev_z <= entry_threshold:
            # 做空价差 (卖symbol1，买symbol2)
            signal = {
                'type': 'spread_short',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'action1': 'sell',
                'action2': 'buy',
                'hedge_ratio': hedge_ratio,
                'z_score': current_z,
                'correlation': correlation,
                'strength': min(abs(current_z) / entry_threshold, 2.0) * self.params['signal_strength_factor'],
                'confidence': min(correlation, 0.9),
                'reason': f'Z-score ({current_z:.2f}) exceeded entry threshold ({entry_threshold})',
                'timestamp': datetime.now(),
                'metadata': {
                    'pair': f"{symbol1}/{symbol2}",
                    'z_threshold': entry_threshold,
                    'signal_type': 'statistical_arbitrage'
                }
            }
            
        elif current_z < -entry_threshold and prev_z >= -entry_threshold:
            # 做多价差 (买symbol1，卖symbol2)
            signal = {
                'type': 'spread_long',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'action1': 'buy',
                'action2': 'sell',
                'hedge_ratio': hedge_ratio,
                'z_score': current_z,
                'correlation': correlation,
                'strength': min(abs(current_z) / entry_threshold, 2.0) * self.params['signal_strength_factor'],
                'confidence': min(correlation, 0.9),
                'reason': f'Z-score ({current_z:.2f}) below negative entry threshold (-{entry_threshold})',
                'timestamp': datetime.now(),
                'metadata': {
                    'pair': f"{symbol1}/{symbol2}",
                    'z_threshold': entry_threshold,
                    'signal_type': 'statistical_arbitrage'
                }
            }
            
        # 检查出场条件
        elif abs(current_z) < exit_threshold and abs(prev_z) >= exit_threshold:
            # 平仓
            signal = {
                'type': 'close_position',
                'symbol1': symbol1,
                'symbol2': symbol2,
                'action1': 'close',
                'action2': 'close',
                'hedge_ratio': hedge_ratio,
                'z_score': current_z,
                'correlation': correlation,
                'strength': 1.0,
                'confidence': min(correlation, 0.9),
                'reason': f'Z-score ({current_z:.2f}) returned within exit threshold ({exit_threshold})',
                'timestamp': datetime.now(),
                'metadata': {
                    'pair': f"{symbol1}/{symbol2}",
                    'z_threshold': exit_threshold,
                    'signal_type': 'statistical_arbitrage'
                }
            }
            
        # 检查置信度
        if signal and signal['confidence'] < self.params['confidence_threshold']:
            return None
            
        return signal
    
    def _generate_triangular_signals(
        self,
        symbol_a: str,
        symbol_b: str,
        symbol_c: str,
        opportunity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        生成三角套利信号
        
        Args:
            symbol_a: 符号A
            symbol_b: 符号B
            symbol_c: 符号C
            opportunity: 套利机会
            
        Returns:
            三角套利信号或None
        """
        # 检查是否满足阈值
        if opportunity['profit_pct'] < self.params['min_profit_ratio']:
            return None
            
        # 确定套利方向
        direction = opportunity['direction']
        
        # 构建信号
        signal = {
            'type': 'triangular_arbitrage',
            'symbols': [symbol_a, symbol_b, symbol_c],
            'direction': direction,
            'profit_ratio': opportunity['profit_pct'],
            'strength': min(opportunity['profit_pct'] / self.params['min_profit_ratio'], 2.0) * self.params['signal_strength_factor'],
            'confidence': 0.8,  # 三角套利的置信度通常较高
            'reason': f"Triangular arbitrage opportunity with {opportunity['profit_pct']:.4f} profit ratio",
            'timestamp': datetime.now(),
            'metadata': {
                'triangle': f"{symbol_a}/{symbol_b}/{symbol_c}",
                'signal_type': 'triangular_arbitrage',
                'price_product': opportunity['implied_product']
            }
        }
        
        # 检查置信度
        if signal['confidence'] < self.params['confidence_threshold']:
            return None
            
        return signal
    
    def _generate_cross_exchange_signals(
        self,
        exchange_a: str,
        exchange_b: str,
        symbol: str,
        opportunity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        生成跨交易所套利信号
        
        Args:
            exchange_a: 交易所A
            exchange_b: 交易所B
            symbol: 交易符号
            opportunity: 套利机会
            
        Returns:
            跨交易所套利信号或None
        """
        # 检查是否满足阈值
        if opportunity['profit_pct'] < self.params['min_profit_ratio']:
            return None
            
        # 确定套利方向
        direction = opportunity['direction']
        
        # 构建信号
        signal = {
            'type': 'cross_exchange_arbitrage',
            'symbol': symbol,
            'clients': [exchange_a, exchange_b],
            'direction': direction,
            'price_a': opportunity['buy_price'],
            'price_b': opportunity['sell_price'],
            'price_diff': opportunity['price_diff_pct'],
            'profit_ratio': opportunity['profit_pct'],
            'strength': min(opportunity['profit_pct'] / self.params['min_profit_ratio'], 2.0) * self.params['signal_strength_factor'],
            'confidence': 0.7,  # 跨交易所套利的置信度中等
            'reason': f"Cross-exchange arbitrage opportunity with {opportunity['profit_pct']:.4f} profit ratio",
            'timestamp': datetime.now(),
            'metadata': {
                'signal_type': 'cross_exchange_arbitrage',
                'exchange_pair': f"{exchange_a}/{exchange_b}"
            }
        }
        
        # 检查置信度
        if signal['confidence'] < self.params['confidence_threshold']:
            return None
            
        return signal
    
    def combine_signals(
        self,
        statistical_signals: Dict[str, Dict[str, Any]],
        triangular_signals: Dict[str, Dict[str, Any]],
        cross_exchange_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        合并所有类型的套利信号
        
        Args:
            statistical_signals: 统计套利信号
            triangular_signals: 三角套利信号
            cross_exchange_signals: 跨交易所套利信号
            
        Returns:
            合并后的信号字典
        """
        all_signals = {}
        
        # 添加统计套利信号
        all_signals.update(statistical_signals)
        
        # 添加三角套利信号
        all_signals.update(triangular_signals)
        
        # 添加跨交易所套利信号
        all_signals.update(cross_exchange_signals)
        
        # 按强度排序
        sorted_signals = dict(sorted(
            all_signals.items(), 
            key=lambda x: x[1].get('strength', 0), 
            reverse=True
        ))
        
        return sorted_signals
    
    def filter_signals_by_strength(
        self,
        signals: Dict[str, Dict[str, Any]],
        min_strength: float = 0.5
    ) -> Dict[str, Dict[str, Any]]:
        """
        根据信号强度过滤信号
        
        Args:
            signals: 信号字典
            min_strength: 最小强度阈值
            
        Returns:
            过滤后的信号字典
        """
        filtered_signals = {}
        
        for key, signal in signals.items():
            if signal.get('strength', 0) >= min_strength:
                filtered_signals[key] = signal
                
        return filtered_signals
    
    def get_top_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        top_n: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        获取前N个最强信号
        
        Args:
            signals: 信号字典
            top_n: 返回的信号数量
            
        Returns:
            前N个信号字典
        """
        # 按强度排序
        sorted_signals = sorted(
            signals.items(), 
            key=lambda x: x[1].get('strength', 0), 
            reverse=True
        )
        
        # 取前N个
        top_signals = dict(sorted_signals[:top_n])
        
        return top_signals