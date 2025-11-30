"""
分析与可视化模块

该模块提供了量化交易策略分析、绩效评估和可视化功能。

主要组件:
- analyzer: 分析器，提供各种分析功能
- visualizer: 可视化器，提供各种图表绘制功能

使用示例:
    from quant_framework.analysis import get_analyzer, get_visualizer
    
    analyzer = get_analyzer()
    visualizer = get_visualizer()
    
    # 计算绩效指标
    metrics = analyzer.calculate_performance_metrics(returns)
    
    # 绘制权益曲线
    visualizer.plot_equity_curve(equity_curve)
"""

from .analyzer import (
    AnalysisType,
    PerformanceMetrics,
    TradeAnalysis,
    Analyzer,
    get_analyzer
)

from .visualizer import (
    Visualizer,
    get_visualizer
)

__all__ = [
    # 分析器
    "AnalysisType",
    "PerformanceMetrics", 
    "TradeAnalysis",
    "Analyzer",
    "get_analyzer",
    
    # 可视化器
    "Visualizer",
    "get_visualizer"
]