"""
Event-driven system runner for the email classifier.
Integrates event bus with classification strategies for real-time processing.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from .event_bus import EventBus, EventTypes
from ..core.strategies.chained_strategy import ChainedMultiLabelStrategy
from ..core.strategies.hierarchical_strategy import HierarchicalMultiLabelStrategy
from ..core.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class EventDrivenSystem:
    """
    Event-driven system that orchestrates email classification through events.
    Integrates with different strategies and provides real-time processing capabilities.
    """
    
    def __init__(self, strategy: str = "chained"):
        """
        Initialize the event-driven system.
        
        Args:
            strategy: Classification strategy to use ("chained" or "hierarchical")
        """
        self.event_bus = EventBus()
        self.strategy_name = strategy
        self.strategy = None
        self.model_factory = ModelFactory()
        self.is_initialized = False
        
        # Subscribe to events
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers for the system."""
        self.event_bus.subscribe(EventTypes.DATA_PREPROCESSED, self._on_data_preprocessed)
        self.event_bus.subscribe(EventTypes.MODEL_TRAINED, self._on_model_trained)
        self.event_bus.subscribe(EventTypes.CLASSIFICATION_COMPLETE, self._on_classification_complete)
        self.event_bus.subscribe(EventTypes.ERROR_OCCURRED, self._on_error_occurred)
    
    def initialize_strategy(self) -> bool:
        """
        Initialize the classification strategy.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.strategy_name == "chained":
                self.strategy = ChainedMultiLabelStrategy(self.model_factory)
            elif self.strategy_name == "hierarchical":
                self.strategy = HierarchicalMultiLabelStrategy(self.model_factory)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy_name}")
            
            self.is_initialized = True
            logger.info(f"Initialized {self.strategy_name} strategy")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            self.event_bus.publish(EventTypes.ERROR_OCCURRED, {
                'error': str(e),
                'context': 'strategy_initialization'
            })
            return False
    
    def train_models(self, X: pd.DataFrame, y2: pd.Series, y3: pd.Series, y4: pd.Series) -> Dict[str, Any]:
        """
        Train models using the selected strategy.
        
        Args:
            X: Training features
            y2: Type 2 labels
            y3: Type 3 labels
            y4: Type 4 labels
            
        Returns:
            Training results
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_strategy() first.")
        
        try:
            logger.info(f"Training models with {self.strategy_name} strategy")
            self.event_bus.publish(EventTypes.DATA_PREPROCESSED, {
                'samples': len(X),
                'features': X.shape[1]
            })
            
            results = self.strategy.train_models(X, y2, y3, y4)
            
            self.event_bus.publish(EventTypes.MODEL_TRAINED, {
                'strategy': self.strategy_name,
                'results': results
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.event_bus.publish(EventTypes.ERROR_OCCURRED, {
                'error': str(e),
                'context': 'model_training'
            })
            raise
    
    def classify_emails(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify emails using the trained strategy.
        
        Args:
            X: Features for classification
            
        Returns:
            Classification results
        """
        if not self.is_initialized or not self.strategy.is_trained:
            raise RuntimeError("Models not trained. Call train_models() first.")
        
        try:
            logger.info(f"Classifying {len(X)} emails with {self.strategy_name} strategy")
            
            predictions = self.strategy.predict(X)
            
            self.event_bus.publish(EventTypes.CLASSIFICATION_COMPLETE, {
                'strategy': self.strategy_name,
                'samples': len(X),
                'predictions': predictions
            })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            self.event_bus.publish(EventTypes.ERROR_OCCURRED, {
                'error': str(e),
                'context': 'classification'
            })
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y2_test: pd.Series, 
                       y3_test: pd.Series, y4_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate trained models.
        
        Args:
            X_test: Test features
            y2_test: Test Type 2 labels
            y3_test: Test Type 3 labels
            y4_test: Test Type 4 labels
            
        Returns:
            Evaluation results
        """
        if not self.is_initialized or not self.strategy.is_trained:
            raise RuntimeError("Models not trained. Call train_models() first.")
        
        try:
            logger.info(f"Evaluating {self.strategy_name} strategy")
            
            results = self.strategy.evaluate(X_test, y2_test, y3_test, y4_test)
            
            self.event_bus.publish(EventTypes.EVALUATION_COMPLETE, {
                'strategy': self.strategy_name,
                'results': results
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.event_bus.publish(EventTypes.ERROR_OCCURRED, {
                'error': str(e),
                'context': 'evaluation'
            })
            raise
    
    def get_event_history(self) -> list:
        """Get the history of all events."""
        return self.event_bus.get_event_history()
    
    def _on_data_preprocessed(self, data: Dict[str, Any]):
        """Handle data preprocessing events."""
        logger.info(f"Data preprocessed: {data}")
    
    def _on_model_trained(self, data: Dict[str, Any]):
        """Handle model training events."""
        logger.info(f"Models trained: {data}")
    
    def _on_classification_complete(self, data: Dict[str, Any]):
        """Handle classification completion events."""
        logger.info(f"Classification completed: {data}")
    
    def _on_error_occurred(self, data: Dict[str, Any]):
        """Handle error events."""
        logger.error(f"Error occurred: {data}")


# Global system instance
_system_instance: Optional[EventDrivenSystem] = None


def start_event_system(strategy: str = "chained") -> EventDrivenSystem:
    """
    Start the event-driven system for email classification.
    
    Args:
        strategy: Classification strategy to use ("chained" or "hierarchical")
        
    Returns:
        Initialized EventDrivenSystem instance
        
    Example:
        >>> system = start_event_system(strategy="chained")
        >>> # Train models
        >>> results = system.train_models(X_train, y2_train, y3_train, y4_train)
        >>> # Classify emails
        >>> predictions = system.classify_emails(X_test)
    """
    global _system_instance
    
    if _system_instance is not None:
        logger.warning("Event system already running. Returning existing instance.")
        return _system_instance
    
    logger.info(f"Starting event-driven system with {strategy} strategy")
    
    _system_instance = EventDrivenSystem(strategy)
    
    if not _system_instance.initialize_strategy():
        raise RuntimeError(f"Failed to initialize event system with strategy: {strategy}")
    
    logger.info("Event-driven system started successfully")
    return _system_instance


def get_event_system() -> Optional[EventDrivenSystem]:
    """
    Get the current event system instance.
    
    Returns:
        Current EventDrivenSystem instance or None if not started
    """
    return _system_instance


def stop_event_system():
    """Stop the event-driven system."""
    global _system_instance
    
    if _system_instance is not None:
        logger.info("Stopping event-driven system")
        _system_instance = None
    else:
        logger.warning("Event system not running")
