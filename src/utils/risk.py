# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    max_position_notional_usdt: float
    max_order_notional_usdt: float
    order_percent_balance: float


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def compute_order_notional(self, free_usdt: float) -> float:
        target = max(0.0, free_usdt * self.cfg.order_percent_balance)
        target = min(target, self.cfg.max_order_notional_usdt)
        return target

    def can_increase_position(self, current_notional: float, add_notional: float) -> bool:
        return (current_notional + add_notional) <= self.cfg.max_position_notional_usdt
