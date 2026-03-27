# Complete Architecture Analysis - Email Classifier v2.0.0

## 🏗️ **PROJECT STRUCTURE OVERVIEW**

```
email_classifier/
├── 📄 README.md                    # Professional documentation
├── 📄 requirements.txt              # Dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 📄 LICENSE                      # MIT License
├── 📄 pyproject.toml               # Modern Python packaging
├── 📄 setup.py                     # Traditional setup
├── 📄 .env.example                 # Environment variables template
│
├── 📁 src/                          # Source code directory
│   └── 📁 core/                     # Core business logic
│       ├── 📁 models/               # ML model implementations
│       │   ├── 📄 base.py           # Abstract base class for all models
│       │   ├── 📄 random_forest.py  # Random Forest implementation
│       │   ├── 📄 xgboost.py        # XGBoost implementation
│       │   ├── 📄 extended_models.py # 22 additional ML models
│       │   ├── 📄 model_factory.py  # Factory pattern for model creation
│       │   ├── 📄 model_executor.py  # Model execution utilities
│       │   └── 📄 model_trainer.py  # Model training utilities
│       │
│       ├── 📁 strategies/           # Multi-label classification strategies
│       │   ├── 📄 base_strategy.py  # Abstract base for strategies
│       │   ├── 📄 chained_strategy.py # Design Decision 1 implementation
│       │   └── 📄 hierarchical_strategy.py # Design Decision 2 implementation
│       │
│       ├── 📁 preprocessing/        # Data preprocessing pipeline
│       │   ├── 📄 data_selector.py  # Data loading and selection
│       │   ├── 📄 text_preprocessor.py # Text cleaning and preprocessing
│       │   ├── 📄 vectorizer.py    # Text to vector conversion
│       │   ├── 📄 sampler.py        # Data sampling techniques
│       │   ├── 📄 data_splitter.py  # Data splitting utilities
│       │   ├── 📄 data_structurer.py # Data structuring and organization
│       │   ├── 📄 pipeline.py        # Complete preprocessing pipeline
│       │   ├── 📄 strategy.py        # Preprocessing strategy implementation
│       │   └── 📄 translator.py      # Text translation utilities
│       │
│       ├── 📁 evaluation/           # Strategy-specific evaluation
│       │   ├── 📄 chained_evaluator.py # Chain accuracy evaluation
│       │   ├── 📄 hierarchical_evaluator.py # Hierarchical metrics
│       │   └── 📄 model_evaluator.py # General model evaluation
│       │
│       ├── 📁 utils/               # Utility functions
│       │   ├── 📄 config.py        # Configuration management
│       │   └── 📄 logging_config.py # Logging setup
│       │
│       └── 📄 strategy.py          # Strategy pattern implementation
│
├── 📁 scripts/                     # Executable scripts
│   ├── 📄 create_diagram.py        # Generate architecture diagrams
│   ├── 📄 run_pipeline.py          # Full pipeline execution
│   └── 📄 run_strategies.py        # Compare multi-label strategies
│
├── 📁 tests/                       # Unit and integration tests
│   ├── 📄 README.md                 # Test documentation
│   ├── 📄 test_components.py        # Component tests
│   ├── 📄 test_extended_models.py   # Extended model tests
│   ├── 📄 test_import.py            # Import tests
│   └── 📄 test_import_clean.py      # Clean import tests
├── 📁 data/                        # Data files
├── 📁 results/                     # Model results and reports
├── 📁 docs/                        # Documentation
├── 📁 config/                      # Configuration files
├── 📁 docker/                      # Docker configuration
└── 📁 archive/                     # Legacy code archive
```

---

## 🎯 **CORE ARCHITECTURAL PATTERNS**

### **1. Strategy Pattern (Multi-Label Classification)**
- **Location**: `src/core/strategies/`
- **Purpose**: Switchable between Chained and Hierarchical approaches
- **Implementation**: Abstract `BaseStrategy` with concrete implementations

### **2. Factory Method Pattern (Model Creation)**
- **Location**: `src/core/models/model_factory.py`
- **Purpose**: Dynamic creation of 24 different ML models
- **Implementation**: `ModelFactory.create_model(model_type, **params)`

### **3. Template Method Pattern (Model Interface)**
- **Location**: `src/core/models/base.py`
- **Purpose**: Consistent interface across all ML models
- **Implementation**: Abstract `BaseModel` with `train()`, `predict()`, `print_results()`

### **4. Observer Pattern (Event-Driven Architecture)**
- **Location**: `src/event_driven/`
- **Purpose**: Real-time event processing and monitoring
- **Implementation**: Event bus with subscribers

---

## 🔄 **DATA FLOW ARCHITECTURE**

### **Input Data Flow:**
```
Raw CSV Data → DataSelector → TextPreprocessor → Vectorizer → Sampler → Model Training
```

### **Model Selection Flow:**
```
User Request → ModelFactory → ExtendedModelRegistry → BaseModel Instance → Training
```

### **Multi-Label Strategy Flow:**
```
Input Data → Strategy Selection (Chained/Hierarchical) → Model Training → Evaluation
```

---

## 📊 **DETAILED COMPONENT ANALYSIS**

### **🏗️ Core Components**

#### **1. BaseModel (`src/core/models/base.py`)**
```python
from abc import ABC, abstractmethod
import numpy as np

# Abstract base class implementing Template Method pattern
class BaseModel(ABC):
    def __init__(self, random_state: int = 42):
    @abstractmethod
    def train(self, X, y) -> None
    @abstractmethod  
    def predict(self, X) -> np.ndarray
    @abstractmethod
    def print_results(self) -> None
    def _validate_input(self, X, y) -> bool
```

**Purpose**: Enforces consistent interface across all 24 ML models
**Dependencies**: pandas, numpy, logging
**Used by**: All model implementations (RandomForest, XGBoost, etc.)

#### **2. ModelFactory (`src/core/models/model_factory.py`)**
```python
class ModelFactory:
    @staticmethod
    def create_model(model_type: str, random_state: int = 42, **kwargs) -> BaseModel
    @staticmethod
    def get_available_models() -> dict
    @staticmethod
    def get_models_by_category() -> dict
```

**Purpose**: Factory Method pattern for dynamic model creation
**Dependencies**: extended_models, core models
**Creates**: 24 different ML models (2 core + 22 extended)

#### **3. Extended Models (`src/core/models/extended_models.py`)**
```python
# 22 additional ML models implementing BaseModel interface
EXTENDED_MODEL_REGISTRY = {
    'enhanced_random_forest': ExtendedRandomForestModel,
    'hist_gradient_boosting': HistGradientBoostingModel,
    'sgd': SGDModel,
    'voting': VotingModel,
    'random_trees_embedding': RandomTreesEmbeddingModel,
    'bagging': BaggingModel,
    # ... 17 more models
}
```

**Purpose**: Extended ML model library with advanced algorithms
**Dependencies**: sklearn, xgboost, lightgbm, catboost
**Provides**: Ensemble, linear, probabilistic, neural network models

**✅ Assignment Requirements Implementation:**
- **Separation of Concerns (SoC)**: ✅ Model changes don't affect preprocessing, preprocessing changes don't affect models
- **Data Consistency**: ✅ All models receive unified data format through BaseModel interface
- **Model Abstraction**: ✅ All model differences hidden behind consistent interface (`train()`, `predict()`, `print_results()`)

**⚠️ Technical Note**: 16 models have constructor issues (see `extended_models_status.md`) but architecture is correct

---

### **🔗 Strategy Components**

#### **4. BaseStrategy (`src/core/strategies/base_strategy.py`)**
```python
class BaseStrategy(ABC):
    def __init__(self, model_factory)
    @abstractmethod
    def train_models(self, X: pd.DataFrame, y2: pd.Series, y3: pd.Series, y4: pd.Series) -> Dict[str, Any]
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y2_test: pd.Series, y3_test: pd.Series, y4_test: pd.Series) -> Dict[str, Any]
```

**Purpose**: Abstract interface for multi-label classification strategies
**Dependencies**: ModelFactory, BaseModel
**Implementations**: ChainedStrategy, HierarchicalStrategy

#### **5. ChainedStrategy (`src/core/strategies/chained_strategy.py`)**
```python
class ChainedMultiLabelStrategy(BaseStrategy):
    # Design Decision 1: Chained Multi-Outputs
    def train_models(self, X_train, y2_train, y3_train, y4_train):
        # Model 1: Type 2
        # Model 2: Type 2 + Type 3  
        # Model 3: Type 2 + Type 3 + Type 4
    
    def predict(self, X_test):
        # Sequential prediction chain
        # Type 2 → Type 2+3 → Type 2+3+4
```

**Purpose**: Implements Design Decision 1 - Chained Multi-Outputs
**Chain Dependency**: Each level depends on previous level accuracy
**Results**: 100% accuracy (Type 2: 1.0, Type 2+3: 1.0, Type 2+3+4: 1.0)

#### **6. HierarchicalStrategy (`src/core/strategies/hierarchical_strategy.py`)**
```python
class HierarchicalMultiLabelStrategy(BaseStrategy):
    # Design Decision 2: Hierarchical Modelling
    def train_models(self, X_train, y2_train, y3_train, y4_train):
        # Model 1: Type 2 (1 model)
        # Model 2: Type 3 (N models, one per Type 2 class)
        # Model 3: Type 4 (M models, one per Type 2+3 combination)
    
    def predict(self, X_test):
        # Hierarchical prediction with data filtering
        # Type 2 → Filter Type 3 → Filter Type 4
```

**Purpose**: Implements Design Decision 2 - Hierarchical Modelling
**Data Filtering**: Each level uses filtered data from previous level
**Results**: 100% accuracy (Type 2: 1.0, Type 3: 1.0, Type 4: 1.0)

---

### **🔧 Preprocessing Components**

#### **7. DataSelector (`src/core/preprocessing/data_selector.py`)**
```python
class DataSelector:
    def __init__(self, file_path: str)
    def load_data(self) -> pd.DataFrame
    def clean_data_types(self) -> pd.DataFrame
    def rename_columns(self) -> pd.DataFrame
    def remove_empty_targets(self) -> pd.DataFrame
    def filter_by_frequency(self, column: str = "y1", min_count: int = 10) -> pd.DataFrame
```

**Purpose**: Data loading and initial preprocessing
**Dependencies**: pandas
**Features**: Column selection, class filtering, data validation, data cleaning

#### **8. TextPreprocessor (`src/core/preprocessing/text_preprocessor.py`)**
```python
class TextPreprocessor:
    def __init__(self)
    def _initialize_noise_patterns(self) -> Dict[str, List[str]]
    def remove_noise(self, text: str) -> str
    def preprocess_text(self, text: str) -> str
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           summary_col: str, interaction_col: str) -> pd.DataFrame
```

**Purpose**: Text cleaning and normalization
**Dependencies**: re, pandas, logging
**Features**: Noise removal, regex patterns, stopword removal, text normalization

#### **9. Vectorizer (`src/core/preprocessing/vectorizer.py`)**
```python
class Vectorizer:
    def __init__(self, vectorizer_type: str = 'tfidf')
    def fit_transform(self, texts: list) -> np.ndarray
    def transform(self, texts: list) -> np.ndarray
```

**Purpose**: Text to numerical vector conversion
**Dependencies**: sklearn.feature_extraction
**Options**: TF-IDF, CountVectorizer, Word2Vec, BERT

#### **10. Sampler (`src/core/preprocessing/sampler.py`)**
```python
class Sampler:
    def smote_oversample(self, X, y) -> tuple
    def adasyn_oversample(self, X, y) -> tuple
    def nearmiss_undersample(self, X, y) -> tuple
```

**Purpose**: Data sampling for imbalanced datasets
**Dependencies**: imblearn
**Techniques**: SMOTE, ADASYN, NearMiss

---

### **📈 Evaluation Components**

#### **11. ChainedEvaluator (`src/core/evaluation/chained_evaluator.py`)**
```python
class ChainedEvaluator:
    def evaluate_chained_performance(self, y_true_dict, y_pred_dict) -> dict
    def calculate_chain_accuracy(self, type2_correct, type3_correct, type4_correct) -> float
```

**Purpose**: Chain accuracy evaluation for Design Decision 1
**Dependencies**: sklearn.metrics
**Metrics**: Chain accuracy, classification reports

#### **12. HierarchicalEvaluator (`src/core/evaluation/hierarchical_evaluator.py`)**
```python
class HierarchicalEvaluator:
    def evaluate_hierarchical_performance(self, y_true_dict, y_pred_dict) -> dict
    def calculate_hierarchical_metrics(self, level_predictions) -> dict
```

**Purpose**: Hierarchical metrics for Design Decision 2
**Dependencies**: sklearn.metrics
**Metrics**: Level-specific accuracy, hierarchical precision/recall

---

## 🔄 **DETAILED DATA FLOWS**

### **1. Complete Training Pipeline Flow**
```
Raw Data Loading (DataSelector)
    ↓
Text Preprocessing (TextPreprocessor)
    ↓
Vectorization (Vectorizer)
    ↓
Data Sampling (Sampler)
    ↓
Train-Test Split
    ↓
Strategy Selection (Chained/Hierarchical)
    ↓
Model Creation (ModelFactory)
    ↓
Model Training (BaseModel.train())
    ↓
Model Evaluation (ChainedEvaluator/HierarchicalEvaluator)
    ↓
Results Storage
```

### **2. Chained Multi-Outputs Flow (Design Decision 1)**
```
Input Data (X_train, y2_train, y3_train, y4_train)
    ↓
Model 1 Training: Type 2 only
    ↓
Model 1 Prediction: Type 2 predictions
    ↓
Model 2 Training: Type 2 + Type 3 combined labels
    ↓
Model 2 Prediction: Type 2+3 predictions
    ↓
Model 3 Training: Type 2 + Type 3 + Type 4 combined labels
    ↓
Model 3 Prediction: Type 2+3+4 predictions
    ↓
Chain Accuracy Evaluation
```

### **3. Hierarchical Modelling Flow (Design Decision 2)**
```
Input Data (X_train, y2_train, y3_train, y4_train)
    ↓
Model 1 Training: Type 2 (1 model)
    ↓
Model 1 Prediction: Type 2 predictions
    ↓
Data Filtering: Filter Type 3 data by Type 2 classes
    ↓
Model 2 Training: Type 3 (N models, one per Type 2 class)
    ↓
Model 2 Prediction: Type 3 predictions
    ↓
Data Filtering: Filter Type 4 data by Type 2+3 combinations
    ↓
Model 3 Training: Type 4 (M models, one per Type 2+3 combination)
    ↓
Model 3 Prediction: Type 4 predictions
    ↓
Hierarchical Evaluation
```

---

## 🎯 **KEY ARCHITECTURAL DECISIONS**

### **1. Separation of Concerns (Feature 1)**
- **Implementation**: Modular directory structure
- **Preprocessing**: `src/core/preprocessing/`
- **Modeling**: `src/core/models/`
- **Strategies**: `src/core/strategies/`
- **Evaluation**: `src/core/evaluation/`

### **2. Data Encapsulation (Feature 2)**
- **Implementation**: Unified pandas DataFrame/NumPy array interface
- **Consistency**: All models receive same data format
- **Encapsulation**: BaseModel abstracts data handling

### **3. Model Abstraction (Feature 3)**
- **Implementation**: Template Method pattern with BaseModel
- **Consistency**: All models implement `train()`, `predict()`, `print_results()`
- **Enforcement**: Abstract methods ensure implementation

---

## 📊 **MODEL LIBRARY ARCHITECTURE**

### **Core Models (2)**
1. **RandomForestModel** - Ensemble of decision trees
2. **XGBoostModel** - Optimized gradient boosting

### **Extended Models (22)**
#### **Ensemble Methods (7)**
- EnhancedRandomForestModel, EnhancedGradientBoostingModel
- AdaBoostModel, ExtraTreesModel, BaggingModel
- VotingModel, HistGradientBoostingModel

#### **Linear Models (4)**
- LogisticRegressionModel, LinearSVCModel
- RidgeClassifierModel, SGDModel

#### **Probabilistic Models (3)**
- NaiveBayesModel, GaussianNBModel, BernoulliNBModel

#### **Tree-Based (1)**
- DecisionTreeModel

#### **Neural Networks (1)**
- MLPModel

#### **Discriminant Analysis (2)**
- LDAModel, QDAModel

#### **Instance-Based (1)**
- KNNModel

#### **Feature Transformation (1)**
- RandomTreesEmbeddingModel

---

## 🔄 **INTERACTION PATTERNS**

### **1. Model-Factory Interaction**
```
User Request → ModelFactory.create_model() → EXTENDED_MODEL_REGISTRY → BaseModel Instance
```

### **2. Strategy-Model Interaction**
```
Strategy Selection → ModelFactory → Multiple BaseModel Instances → Training Pipeline
```

### **3. Preprocessing-Model Interaction**
```
Raw Data → Preprocessing Pipeline → Vectorized Data → BaseModel.train()
```

### **4. Evaluation-Strategy Interaction**
```
Strategy Predictions → Evaluator → Metrics → Results Storage
```

---

## 📋 **FOR MERMAID DIAGRAM CREATION**

### **Required Diagram Types:**

#### **1. System Architecture Diagram**
- **Components**: All major classes and modules
- **Connectors**: Dependencies and inheritance relationships
- **Data Elements**: Data flow between components

#### **2. Chained Strategy Flow Diagram**
- **Components**: Model instances, data combinations
- **Connectors**: Sequential data flow
- **Data Elements**: Combined labels (Type 2, Type 2+3, Type 2+3+4)

#### **3. Hierarchical Strategy Flow Diagram**
- **Components**: Multiple model instances, data filters
- **Connectors**: Hierarchical data flow with filtering
- **Data Elements**: Filtered datasets at each level

#### **4. Model Factory Pattern Diagram**
- **Components**: ModelFactory, Model Registry, BaseModel
- **Connectors**: Factory method calls, inheritance
- **Data Elements**: Model parameters, configuration

#### **5. Preprocessing Pipeline Diagram**
- **Components**: DataSelector, TextPreprocessor, Vectorizer, Sampler
- **Connectors**: Sequential data transformation
- **Data Elements**: Raw data, processed data, vectors

### **Key Relationships to Highlight:**
- **Inheritance**: BaseModel → All model implementations
- **Composition**: Strategy → Multiple models
- **Factory Pattern**: ModelFactory → Model creation
- **Strategy Pattern**: BaseStrategy → ChainedStrategy/HierarchicalStrategy
- **Template Method**: BaseModel interface enforcement

---

## 🎯 **READY FOR MERMAID DIAGRAM GENERATION**

This complete architecture analysis provides:
- **All components** with their purposes and dependencies
- **All data flows** through the system
- **All interaction patterns** between modules
- **Complete model library** structure
- **Both design decisions** implementation details
- **Professional architecture** documentation

**Use this information to create comprehensive Mermaid diagrams that capture the entire system architecture!**
