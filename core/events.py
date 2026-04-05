"""
Event-driven architecture for CLASDE.

Provides pub/sub event bus for decoupled communication between components.
"""

import time
import logging
from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Campaign lifecycle events."""
    
    # Campaign lifecycle
    CAMPAIGN_STARTED = "campaign_started"
    CAMPAIGN_PAUSED = "campaign_paused"
    CAMPAIGN_RESUMED = "campaign_resumed"
    CAMPAIGN_COMPLETED = "campaign_completed"
    CAMPAIGN_FAILED = "campaign_failed"
    
    # Iteration events
    ITERATION_STARTED = "iteration_started"
    ITERATION_COMPLETED = "iteration_completed"
    
    # Agent events
    IDEA_PROPOSED = "idea_proposed"
    IDEA_CRITIQUED = "idea_critiqued"
    IDEA_REJECTED = "idea_rejected"
    IDEA_ACCEPTED = "idea_accepted"
    
    # Knowledge transformation
    HYPOTHESIS_FORMULATED = "hypothesis_formulated"
    EXPERIMENT_DESIGNED = "experiment_designed"
    EXPERIMENT_SUBMITTED = "experiment_submitted"
    EXPERIMENT_RUNNING = "experiment_running"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    
    # Analysis events
    INSIGHT_GENERATED = "insight_generated"
    PATTERN_DISCOVERED = "pattern_discovered"
    
    # Memory events
    MEMORY_UPDATED = "memory_updated"
    CHECKPOINT_SAVED = "checkpoint_saved"
    
    # Error events
    ERROR_OCCURRED = "error_occurred"
    WARNING_RAISED = "warning_raised"


@dataclass
class Event:
    """
    Event data structure.
    
    Attributes:
        type: Event type identifier
        data: Event payload
        timestamp: Unix timestamp of event occurrence
        source: Component that generated the event
    """
    type: EventType
    data: Dict[str, Any]
    timestamp: float
    source: str = "system"
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class EventBus:
    """
    Publish/Subscribe event dispatcher.
    
    Enables decoupled communication between components via events.
    Subscribers register callbacks for specific event types.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000  # Prevent unbounded growth
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """
        Register a callback for an event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to {event_type}: {callback.__name__}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """Remove a callback from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb != callback
            ]
    
    def publish(self, event: Event) -> None:
        """
        Notify all subscribers of an event.
        
        Args:
            event: Event to publish
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify subscribers
        for callback in self._subscribers.get(event.type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    f"Event handler failed for {event.type}: {callback.__name__}",
                    exc_info=True
                )
                # Don't let handler failures break the event bus
    
    def get_history(self, event_type: EventType = None, limit: int = 100) -> List[Event]:
        """
        Retrieve event history.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()


# Default global event bus
_global_event_bus: EventBus = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def reset_event_bus() -> None:
    """Reset global event bus (mainly for testing)."""
    global _global_event_bus
    _global_event_bus = EventBus()
