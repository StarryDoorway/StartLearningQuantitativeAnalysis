"""
Event bus module for the quantitative trading framework.

This module provides a publish-subscribe pattern for communication between
different components of the trading system.
"""

import threading
import time
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of different event types in the system."""
    
    # Market data events
    MARKET_DATA = "market_data"
    TICK_DATA = "tick_data"
    BAR_DATA = "bar_data"
    
    # Trading events
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_UPDATE = "order_update"
    ORDER_UPDATED = "order_updated"
    ORDER_EXECUTED = "order_executed"
    ORDER_CANCELED = "order_canceled"
    TRADE = "trade"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # Portfolio events
    PORTFOLIO_UPDATE = "portfolio_update"
    
    # Strategy events
    STRATEGY_SIGNAL = "strategy_signal"
    STRATEGY_START = "strategy_start"
    STRATEGY_STOP = "strategy_stop"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM = "system"
    ERROR = "error"
    WARNING = "warning"
    
    # Risk events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    RISK_LIMIT_RESTORED = "risk_limit_restored"


@dataclass
class Event:
    """
    Base event class for all events in the system.
    
    Attributes:
        event_type: Type of the event
        timestamp: Event creation timestamp
        data: Event-specific data
        source: Source of the event (module name)
        priority: Event priority (0=highest, higher number=lower priority)
    """
    event_type: EventType
    timestamp: float
    data: Dict[str, Any]
    source: str
    priority: int = 0
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.timestamp is None:
            self.timestamp = time.time()


class EventBus:
    """
    Event bus implementation for publish-subscribe pattern.
    
    This class manages event subscriptions and publications, allowing
    different components to communicate without direct dependencies.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_queue: List[Event] = []
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread = None
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Function to handle the event
        """
        with self._lock:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed to {event_type.value} with handler {handler.__name__}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler function to remove
        """
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"Unsubscribed from {event_type.value} with handler {handler.__name__}")
    
    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        with self._lock:
            self._event_queue.append(event)
            logger.debug(f"Published event {event.event_type.value} from {event.source}")
    
    def start(self) -> None:
        """Start the event bus worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self._worker_thread.start()
        logger.info("Event bus started")
    
    def stop(self) -> None:
        """Stop the event bus worker thread."""
        if not self._running:
            return
        
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Event bus stopped")
    
    def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            events_to_process = []
            
            with self._lock:
                if self._event_queue:
                    # Sort events by priority (lower number = higher priority)
                    self._event_queue.sort(key=lambda e: e.priority)
                    events_to_process = self._event_queue.copy()
                    self._event_queue.clear()
            
            for event in events_to_process:
                self._handle_event(event)
            
            # Small delay to prevent busy waiting
            time.sleep(0.001)
    
    def _handle_event(self, event: Event) -> None:
        """
        Handle a single event by notifying all subscribers.
        
        Args:
            event: Event to handle
        """
        handlers = self._subscribers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler {handler.__name__}: {str(e)}")


# Global event bus instance
_event_bus = None


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.
    
    Returns:
        EventBus instance
    """
    global _event_bus
    
    if _event_bus is None:
        _event_bus = EventBus()
    
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus instance."""
    global _event_bus
    
    if _event_bus is not None:
        _event_bus.stop()
    
    _event_bus = None


def publish_event(event_type: EventType, data: Dict[str, Any], 
                 source: str = "unknown", priority: int = 0) -> None:
    """
    Convenience function to publish an event.
    
    Args:
        event_type: Type of event to publish
        data: Event data
        source: Source of the event
        priority: Event priority
    """
    event = Event(
        event_type=event_type,
        timestamp=time.time(),
        data=data,
        source=source,
        priority=priority
    )
    
    get_event_bus().publish(event)


def subscribe_to_event(event_type: EventType, handler: Callable[[Event], None]) -> None:
    """
    Convenience function to subscribe to an event type.
    
    Args:
        event_type: Type of event to subscribe to
        handler: Function to handle the event
    """
    get_event_bus().subscribe(event_type, handler)