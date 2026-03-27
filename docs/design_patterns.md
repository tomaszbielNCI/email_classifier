# Design Patterns

This document outlines the key design patterns implemented in the Email Classifier project.

## 🏗️ Architectural Patterns

### Strategy Pattern
The email classifier uses the Strategy pattern to implement different classification algorithms:

- **Location**: `src/core/models/`
- **Purpose**: Allows switching between different classification strategies (Naive Bayes, SVM, etc.)
- **Benefits**: Easy to add new classifiers without modifying existing code

### Factory Pattern
Used for creating model instances and preprocessing components:

- **Location**: `src/core/models/` and `src/core/preprocessing/`
- **Purpose**: Centralized object creation with consistent initialization
- **Benefits**: Decouples client code from concrete class implementations

### Observer Pattern (Event-Driven Architecture)
Implemented through the event bus system:

- **Location**: `src/event_driven/event_bus.py`
- **Purpose**: Enables loose coupling between components through event notifications
- **Benefits**: Scalable architecture where components can react to system events

## 🔄 Behavioral Patterns

### Chain of Responsibility
Used in the preprocessing pipeline:

- **Location**: `src/core/preprocessing/`
- **Purpose**: Each preprocessing step handles specific transformations and passes data to the next step
- **Benefits**: Flexible pipeline that can be easily reconfigured

### Template Method
Used in evaluation and model training:

- **Location**: `src/core/evaluation/`
- **Purpose**: Defines skeleton algorithms while allowing subclasses to override specific steps
- **Benefits**: Consistent evaluation process across different model types

## 📊 Structural Patterns

### Facade
The main pipeline provides a simplified interface:

- **Location**: `scripts/run_pipeline.py`
- **Purpose**: Hides complexity of the underlying system
- **Benefits**: Easy-to-use interface for running end-to-end classification

### Decorator
Used for adding functionality to models:

- **Location**: `src/core/models/`
- **Purpose**: Adds logging, caching, or validation to existing models
- **Benefits**: Extends functionality without modifying core model logic

## 🔧 Implementation Details

### Configuration Management
- **Pattern**: Builder Pattern
- **Location**: Configuration files and loading mechanisms
- **Purpose**: Flexible configuration assembly

### Data Access
- **Pattern**: Repository Pattern (if implemented)
- **Location**: Data loading components
- **Purpose**: Abstraction layer over data storage

## 🎯 Benefits of These Patterns

1. **Maintainability**: Clear separation of concerns
2. **Extensibility**: Easy to add new features or modify existing ones
3. **Testability**: Components can be tested in isolation
4. **Reusability**: Patterns promote code reuse across the project

## 📝 Usage Examples

### Adding a New Classifier
```python
# Using Factory Pattern
from src.core.models.model_factory import ModelFactory

# Create a specific classifier type
trainer = ModelFactory.create_trainer("random_forest", random_state=42)
```

### Setting Up Event Handling
```python
# Using Observer Pattern
from src.event_driven.event_bus import EventBus

# Define event handler function
def handle_training_complete(training_data):
    """Handle completion of model training."""
    print(f"Training completed with accuracy: {training_data.get('accuracy', 'N/A')}")
    # Additional processing logic here

event_bus = EventBus()
event_bus.subscribe("model_trained", handle_training_complete)
```

### Creating a Preprocessing Pipeline
```python
# Using Pipeline Pattern
from src.core.preprocessing.pipeline import EmailClassificationPipeline

pipeline = EmailClassificationPipeline()
# Pipeline handles preprocessing steps internally through configuration
# Run full pipeline with data
results = pipeline.run_full_pipeline("data/AppGallery.csv")
```

---

*This document is continuously updated as the project evolves.*
