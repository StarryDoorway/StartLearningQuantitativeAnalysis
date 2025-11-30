"""
Strategy state and performance metrics for the quantitative trading framework.

This module provides data structures for tracking strategy state and performance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any


@dataclass
class StrategyState:
    """
    State of a strategy.
    
    Attributes:
        strategy_id: Strategy ID
        is_running: Whether the strategy is running
        is_live_mode: Whether the strategy is in live mode
        last_update: Last update timestamp
        current_positions: Current positions
        performance_metrics: Performance metrics
        parameters: Strategy parameters
        metadata: Additional metadata
    """
    strategy_id: str
    is_running: bool
    is_live_mode: bool
    last_update: datetime
    current_positions: Dict[str, float]
    performance_metrics: Dict[str, float]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "is_running": self.is_running,
            "is_live_mode": self.is_live_mode,
            "last_update": self.last_update,
            "current_positions": self.current_positions,
            "performance_metrics": self.performance_metrics,
            "parameters": self.parameters,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyState':
        """Create from dictionary."""
        return cls(
            strategy_id=data["strategy_id"],
            is_running=data["is_running"],
            is_live_mode=data["is_live_mode"],
            last_update=data["last_update"],
            current_positions=data["current_positions"],
            performance_metrics=data["performance_metrics"],
            parameters=data["parameters"],
            metadata=data["metadata"]
        )