"""
Event-driven architecture implementation for the email classifier.
Provides a simple event bus for decoupled communication between components.
"""

from typing import Dict, List, Callable, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """
    Simple event bus implementation using the Observer pattern.
    Allows components to subscribe to events and publish notifications
    without direct coupling.
    """
    
    def __init__(self):
        """Initialize the event bus with empty subscriber lists."""
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: List[Dict[str, Any]] = []
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to a specific event type.
        
        Args:
            event_type: The type of event to subscribe to
            callback: Function to call when event is published
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        
        self._subscribers[event_type].append(callback)
        logger.debug(f"Subscribed {callback.__name__} to event '{event_type}'")
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from a specific event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: Function to remove from subscribers
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed {callback.__name__} from event '{event_type}'")
            except ValueError:
                logger.warning(f"Callback {callback.__name__} not found in subscribers for '{event_type}'")
    
    def publish(self, event_type: str, data: Any = None) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: The type of event to publish
            data: Optional data to pass to subscribers
        """
        event_record = {
            'type': event_type,
            'data': data,
            'timestamp': None  # Could add timestamp if needed
        }
        self._event_history.append(event_record)
        
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback {callback.__name__} for event '{event_type}': {e}")
        
        logger.debug(f"Published event '{event_type}' to {len(self._subscribers[event_type])} subscribers")
    
    def get_subscribers(self, event_type: str) -> List[Callable]:
        """
        Get all subscribers for a specific event type.
        
        Args:
            event_type: The event type to get subscribers for
            
        Returns:
            List of callback functions
        """
        return self._subscribers.get(event_type, []).copy()
    
    def get_event_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all published events.
        
        Returns:
            List of event records
        """
        return self._event_history.copy()
    
    def clear_history(self) -> None:
        """Clear the event history."""
        self._event_history.clear()
        logger.debug("Event history cleared")


# Example usage and common event types
class EventTypes:
    """Common event types used in the email classifier."""
    MODEL_TRAINED = "model_trained"
    CLASSIFICATION_COMPLETE = "classification_complete"
    DATA_PREPROCESSED = "data_preprocessed"
    EVALUATION_COMPLETE = "evaluation_complete"
    ERROR_OCCURRED = "error_occurred"
