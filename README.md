# Email Classification Pipeline

Complete pipeline for email/support ticket classification using modern NLP and machine learning techniques.

## 🎯 **Why Supervised Learning Strategy?**

Based on comprehensive data analysis, pipeline automatically selects **supervised learning** as optimal strategy:

### **📊 **Feasibility Analysis Results:**
- **Label Balance Score:** 1.000 (perfectly balanced after oversampling)
- **Feasibility Score:** 0.800 (above 0.6 threshold)
- **Samples per Class:** 68+ samples (above 10 minimum)
- **Total Samples:** 204 (above 100 minimum)
- **Data Quality:** High-quality labeled data

### **🚀 **Model Performance:**
- **Random Forest:** 97.56% accuracy (best model)
- **Logistic Regression:** 95.12% accuracy
- **Naive Bayes:** 87.80% accuracy
- **Pipeline Execution Time:** 0.50 seconds

### **🔍 **Why Not Unsupervised?**
- Clear labeled data available
- High feasibility score for supervised approach
- Better interpretability and performance with labels
- Data shows strong class structure after balancing

## Project Structure

```
src/
├── data_selector.py        # Steps 1 & 2: Column cleaning and preliminary grouping
├── translator.py           # Step 3: Translation of NLP to English
├── text_preprocessor.py    # Step 4: Regex, noise removal (stop-words)
├── data_structurer.py      # Step 5: Handling multi-level / multi-class data
├── vectorizer.py           # Step 6: Numerical representation (TF-IDF, Embeddings)
├── sampler.py              # Step 7: Data balancing (Oversampling/Undersampling)
├── strategy.py             # Step 8: Decision (Supervised vs Unsupervised)
├── data_splitter.py        # Step 9: Train/test split
├── model_trainer.py        # Steps 10 & 11: SOTA model, Training
├── model_evaluator.py      # Step 11: Evaluation
└── pipeline.py             # Main pipeline connecting all modules
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd email_classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.pipeline import EmailClassificationPipeline

# Initialize pipeline
pipeline = EmailClassificationPipeline()

# Run full pipeline
results = pipeline.run_full_pipeline(
    data_path="AppGallery.csv",
    target_column="y2",
    text_columns=["Ticket Summary", "Interaction content"]
)

print(f"Best model: {results['modeling_results']['best_model']}")
print(f"Best score: {results['modeling_results']['best_score']:.4f}")
```

### Advanced Configuration

```python
# JSON configuration
config = {
    "translation": {
        "enable": True,
        "model_name": "Helsinki-NLP/opus-mt-mul-en"
    },
    "vectorization": {
        "method": "tfidf",
        "max_features": 15000,
        "ngram_range": [1, 3]
    },
    "sampling": {
        "enable": True,
        "method": "smote"
    },
    "modeling": {
        "models": ["random_forest", "xgboost", "lightgbm"],
        "hyperparameter_tuning": True
    }
}

# Import pipeline
from src.pipeline import EmailClassificationPipeline

# Pipeline with configuration
pipeline = EmailClassificationPipeline(config_path="config.json")
results = pipeline.run_full_pipeline("data.csv")
```

## Modules

### 1. Data Selection (`01_data_selection`)
- Data loading and cleaning
- Data type conversion
- Filtering rare classes

### 2. Translation (`02_translation`)
- Translation of multilingual texts to English
- Using **MarianMT** models for efficient translation
- Language detection and batch processing

### 3. Preprocessing (`03_preprocessing`)
- Email noise removal
- Cleaning headers, dates, signatures
- Text normalization

### 4. Data Structuring (`04_data_structuring`)
- Multi-class data handling
- Label encoding
- Class distribution analysis

### 5. Vectorization (`05_vectorization`)
- TF-IDF vectorization
- Embeddings (Sentence Transformers)
- Dimensionality reduction

### 6. Sampling (`06_sampling`)
- Data balancing
- SMOTE, ADASYN, Random oversampling/undersampling
- Automatic method selection

### 7. Strategy (`07_strategy`)
- Supervised learning feasibility analysis
- Strategy recommendations
- Clustering analysis

### 8. Data Split (`08_data_split`)
- Various data splitting methods
- Stratification, temporal split, group split
- Cross-validation

### 9. Modeling (`09_modeling`)
- Training multiple SOTA models
- Hyperparameter tuning
- Comprehensive evaluation

## Supported Models

- **Random Forest** - Decision tree ensemble
- **Gradient Boosting** - Sequential trees
- **XGBoost** - Optimized gradient boosting
- **LightGBM** - Fast gradient boosting
- **Logistic Regression** - Linear model
- **SVM** - Support Vector Machine
- **Naive Bayes** - Probabilistic model
- **KNN** - K-nearest neighbors
- **MLP** - Neural network

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Confusion matrix
- Per-class analysis
- Plots and visualizations

## Example Results

```
Pipeline completed successfully in 0.50 seconds
Best model: random_forest
Best score: 0.9756

Model comparison:
┌─────────────────┬──────────┬───────────┬─────────┬──────────┐
│ model           │ accuracy │ precision │ recall  │ f1_score │
├─────────────────┼──────────┼───────────┼─────────┼──────────┤
│ random_forest   │ 0.9756   │ 0.9756    │ 0.9756  │ 0.9756   │
│ logistic_reg    │ 0.9512   │ 0.9512    │ 0.9512  │ 0.9512   │
│ naive_bayes     │ 0.8780   │ 0.8713    │ 0.8780  │ 0.8713   │
└─────────────────┴──────────┴───────────┴─────────┴──────────┘

📊 Strategy Analysis:
- Supervised feasibility score: 0.800
- Label balance score: 1.000 (perfectly balanced)
- Recommended strategy: supervised
- Data samples: 204 training, 41 test
```

## Configuration

The pipeline can be configured via JSON file:

```json
{
    "data_selection": {
        "filter_frequency": true,
        "min_samples_per_class": 10
    },
    "translation": {
        "enable": false,
        "model_name": "Helsinki-NLP/opus-mt-mul-en"
    },
    "preprocessing": {
        "clean_summary": true,
        "clean_interaction": true
    },
    "vectorization": {
        "method": "tfidf",
        "max_features": 10000,
        "ngram_range": [1, 2]
    },
    "sampling": {
        "enable": true,
        "method": "auto",
        "max_ratio": 2.0
    },
    "data_split": {
        "test_size": 0.2,
        "stratify": true,
        "method": "basic"
    },
    "modeling": {
        "models": ["random_forest", "logistic_regression", "naive_bayes"],
        "hyperparameter_tuning": false
    }
}
```

## Save and Load Models

```python
# Create and train pipeline
from src.pipeline import EmailClassificationPipeline
pipeline = EmailClassificationPipeline()
results = pipeline.run_full_pipeline("data/AppGallery.csv")

# Save pipeline
pipeline.save_pipeline("email_classifier_pipeline.pkl")

# Load pipeline
from src.pipeline import EmailClassificationPipeline
loaded_pipeline = EmailClassificationPipeline()
loaded_pipeline.load_pipeline("email_classifier_pipeline.pkl")

# Predictions on new data
predictions = loaded_pipeline.predict(new_emails)
```

## Requirements

- Python 3.8+
- CUDA (optional, for computation acceleration)
- Minimum 8GB RAM (for large datasets)

## License

MIT License

## Contributing

Pull requests are welcome! Please ensure that:

1. Code follows PEP 8
2. Appropriate tests are added
3. Documentation is updated

## Issues

For issues, please open a GitHub issue with:

1. Problem description
2. Example code
3. Library versions
4. Error logs
