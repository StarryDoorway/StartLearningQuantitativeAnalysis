# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RiskConfig:
    max_position_notional_usdt: float
    max_order_notional_usdt: float
    order_percent_balance: float
    # 额外风控项（默认关闭，以保持向后兼容）
    daily_loss_limit_usdt: float = 0.0  # 当天累计亏损超过该值则停止下单；0 表示禁用
    max_intraday_drawdown_pct: float = 0.0  # 当天回撤超过该百分比（0-100）则停止下单；0 表示禁用


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def compute_order_notional(self, free_usdt: float) -> float:
        target = max(0.0, free_usdt * self.cfg.order_percent_balance)
        target = min(target, self.cfg.max_order_notional_usdt)
        return target

    def can_increase_position(self, current_notional: float, add_notional: float) -> bool:
        return (current_notional + add_notional) <= self.cfg.max_position_notional_usdt

    def should_halt_new_orders(
        self,
        daily_realized_pnl_usdt: float,
        starting_equity_usdt: float,
        current_equity_usdt: float,
    ) -> Tuple[bool, str]:
        """
        基于当日亏损上限与当日回撤阈值判断是否应停止新增订单。

        返回: (是否停止, 原因)
        """
        # 1) 当日累计亏损限制
        if self.cfg.daily_loss_limit_usdt and self.cfg.daily_loss_limit_usdt > 0:
            if daily_realized_pnl_usdt <= -abs(self.cfg.daily_loss_limit_usdt):
                return True, f"daily loss limit breached: {daily_realized_pnl_usdt:.2f} USDT"

        # 2) 当日回撤限制（基于权益）
        if self.cfg.max_intraday_drawdown_pct and self.cfg.max_intraday_drawdown_pct > 0:
            if starting_equity_usdt > 0:
                dd = (starting_equity_usdt - current_equity_usdt) / starting_equity_usdt * 100.0
                if dd >= self.cfg.max_intraday_drawdown_pct:
                    return True, f"intraday drawdown {dd:.2f}% >= {self.cfg.max_intraday_drawdown_pct:.2f}%"

        return False, ""
