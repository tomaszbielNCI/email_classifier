"""
Event-driven architecture components for the email classifier.
"""

from .event_bus import EventBus, EventTypes
from .run_all import start_event_system, get_event_system, stop_event_system

__all__ = ['EventBus', 'EventTypes', 'start_event_system', 'get_event_system', 'stop_event_system']
