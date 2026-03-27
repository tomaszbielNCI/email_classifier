"""
Extended Models - Additional model implementations for comprehensive model library
"""

from .base import BaseModel
import pandas as pd
import numpy as np
from typing import Union, Optional, Any
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, RandomTreesEmbedding
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

logger = logging.getLogger(__name__)


class ExtendedRandomForestModel(BaseModel):
    """Enhanced Random Forest with additional parameters"""
    
    def __init__(self, random_state: int = 42, **params):
        super().__init__(random_state=random_state)
        self.params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'n_jobs': -1
        }
        self.params.update(params)
        self._initialize_model()
    
    def _initialize_model(self):
        self.model = RandomForestClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Random Forest model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Extended Random Forest trained successfully")
            
            # Log feature importance if available
            if hasattr(X, 'columns'):
                feature_importance = dict(zip(X.columns, self.model.feature_importances_))
                logging.info(f"Feature importance: {feature_importance}")
            else:
                logging.info(f"Feature importance: {self.model.feature_importances_}")
                
        except Exception as e:
            logging.error(f"Error training Extended Random Forest: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nExtended Random Forest Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of estimators: {self.model.n_estimators}")
            print(f"Feature importance available: {hasattr(self.model, 'feature_importances_')}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Enhanced Random Forest',
            'type': 'ensemble',
            'description': 'Random Forest with enhanced parameters and feature importance',
            'parameters': self.params
        }


class ExtendedGradientBoostingModel(BaseModel):
    """Enhanced Gradient Boosting with additional parameters"""
    
    def __init__(self, **params):
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'max_features': None,
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = GradientBoostingClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Gradient Boosting model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Extended Gradient Boosting trained successfully")
            
            # Log feature importance if available
            if hasattr(X, 'columns'):
                feature_importance = dict(zip(X.columns, self.model.feature_importances_))
                logging.info(f"Feature importance: {feature_importance}")
            else:
                logging.info(f"Feature importance: {self.model.feature_importances_}")
                
        except Exception as e:
            logging.error(f"Error training Extended Gradient Boosting: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nExtended Gradient Boosting Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of estimators: {self.model.n_estimators}")
            print(f"Learning rate: {self.model.learning_rate}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Enhanced Gradient Boosting',
            'type': 'ensemble',
            'description': 'Gradient Boosting with enhanced parameters and learning rate',
            'parameters': self.params
        }


class LogisticRegressionModel(BaseModel):
    """Logistic Regression with regularization options"""
    
    def __init__(self, **params):
        default_params = {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'liblinear',
            'max_iter': 1000,
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = LogisticRegression(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Logistic Regression model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Logistic Regression trained successfully")
            
            # Log coefficients if available
            if hasattr(X, 'columns'):
                coefficients = dict(zip(X.columns, self.model.coef_[0]))
                logging.info(f"Coefficients: {coefficients}")
            else:
                logging.info(f"Coefficients shape: {self.model.coef_.shape}")
                
        except Exception as e:
            logging.error(f"Error training Logistic Regression: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nLogistic Regression Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Regularization: {self.model.penalty}")
            print(f"C parameter: {self.model.C}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Logistic Regression',
            'type': 'linear',
            'description': 'Linear model with L2 regularization',
            'parameters': self.params
        }


class SVMModel(BaseModel):
    """Support Vector Machine with kernel options"""
    
    def __init__(self, **params):
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = SVC(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the SVM model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("SVM trained successfully")
            
            # Log support vectors
            logging.info(f"Number of support vectors: {len(self.model.support_vectors_)}")
                
        except Exception as e:
            logging.error(f"Error training SVM: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nSVM Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Kernel: {self.model.kernel}")
            print(f"C parameter: {self.model.C}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Support Vector Machine',
            'type': 'kernel',
            'description': 'SVM with RBF kernel and probability estimates',
            'parameters': self.params
        }


class NaiveBayesModel(BaseModel):
    """Multinomial Naive Bayes for text classification"""
    
    def __init__(self, **params):
        default_params = {
            'alpha': 1.0,
            'fit_prior': True
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = MultinomialNB(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Naive Bayes model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Naive Bayes trained successfully")
            
            # Log class priors
            logging.info(f"Class priors: {self.model.class_log_prior_}")
                
        except Exception as e:
            logging.error(f"Error training Naive Bayes: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nNaive Bayes Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Alpha: {self.model.alpha}")
            print(f"Number of classes: {len(self.model.classes_)}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Multinomial Naive Bayes',
            'type': 'probabilistic',
            'description': 'Naive Bayes suitable for text classification',
            'parameters': self.params
        }


class KNNModel(BaseModel):
    """K-Nearest Neighbors with distance metrics"""
    
    def __init__(self, **params):
        default_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'metric': 'minkowski'
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = KNeighborsClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the KNN model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("KNN trained successfully")
                
        except Exception as e:
            logging.error(f"Error training KNN: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nKNN Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of neighbors: {self.model.n_neighbors}")
            print(f"Weights: {self.model.weights}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'K-Nearest Neighbors',
            'type': 'instance-based',
            'description': 'KNN with configurable distance metrics',
            'parameters': self.params
        }


class MLPModel(BaseModel):
    """Multi-layer Perceptron Neural Network"""
    
    def __init__(self, **params):
        default_params = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = MLPClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the MLP model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("MLP trained successfully")
            
            # Log training info
            logging.info(f"Number of iterations: {self.model.n_iter_}")
            logging.info(f"Loss: {self.model.loss_}")
                
        except Exception as e:
            logging.error(f"Error training MLP: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nMLP Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Hidden layers: {self.model.hidden_layer_sizes}")
            print(f"Number of iterations: {self.model.n_iter_}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Multi-layer Perceptron',
            'type': 'neural_network',
            'description': 'Neural network with configurable layers',
            'parameters': self.params
        }


class DecisionTreeModel(BaseModel):
    """Decision Tree with pruning options"""
    
    def __init__(self, **params):
        default_params = {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = DecisionTreeClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Decision Tree model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Decision Tree trained successfully")
            
            # Log tree info
            logging.info(f"Tree depth: {self.model.get_depth()}")
            logging.info(f"Number of leaves: {self.model.get_n_leaves()}")
                
        except Exception as e:
            logging.error(f"Error training Decision Tree: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nDecision Tree Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Tree depth: {self.model.get_depth()}")
            print(f"Number of leaves: {self.model.get_n_leaves()}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Decision Tree',
            'type': 'tree',
            'description': 'Single decision tree with pruning options',
            'parameters': self.params
        }


class AdaBoostModel(BaseModel):
    """AdaBoost ensemble method"""
    
    def __init__(self, **params):
        default_params = {
            'n_estimators': 50,
            'learning_rate': 1.0,
            'algorithm': 'SAMME.R',
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = AdaBoostClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the AdaBoost model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("AdaBoost trained successfully")
                
        except Exception as e:
            logging.error(f"Error training AdaBoost: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nAdaBoost Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of estimators: {self.model.n_estimators}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'AdaBoost',
            'type': 'ensemble',
            'description': 'Adaptive boosting ensemble method',
            'parameters': self.params
        }


class ExtraTreesModel(BaseModel):
    """Extremely Randomized Trees ensemble"""
    
    def __init__(self, **params):
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': False,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = ExtraTreesClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Extra Trees model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Extra Trees trained successfully")
            
            # Log feature importance if available
            if hasattr(X, 'columns'):
                feature_importance = dict(zip(X.columns, self.model.feature_importances_))
                logging.info(f"Feature importance: {feature_importance}")
            else:
                logging.info(f"Feature importance: {self.model.feature_importances_}")
                
        except Exception as e:
            logging.error(f"Error training Extra Trees: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nExtra Trees Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of estimators: {self.model.n_estimators}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Extra Trees',
            'type': 'ensemble',
            'description': 'Extremely randomized trees ensemble',
            'parameters': self.params
        }


class LinearSVCModel(BaseModel):
    """Linear Support Vector Machine"""
    
    def __init__(self, **params):
        default_params = {
            'penalty': 'l2',
            'loss': 'squared_hinge',
            'dual': True,
            'tol': 1e-4,
            'C': 1.0,
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = LinearSVC(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Linear SVM model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Linear SVM trained successfully")
                
        except Exception as e:
            logging.error(f"Error training Linear SVM: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nLinear SVM Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"C parameter: {self.model.C}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Linear SVM',
            'type': 'linear',
            'description': 'Linear Support Vector Machine',
            'parameters': self.params
        }


class RidgeClassifierModel(BaseModel):
    """Ridge Classifier for linear classification"""
    
    def __init__(self, **params):
        default_params = {
            'alpha': 1.0,
            'solver': 'auto',
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = RidgeClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Ridge Classifier model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Ridge Classifier trained successfully")
                
        except Exception as e:
            logging.error(f"Error training Ridge Classifier: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nRidge Classifier Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Alpha: {self.model.alpha}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Ridge Classifier',
            'type': 'linear',
            'description': 'Linear classifier with L2 regularization',
            'parameters': self.params
        }


class SGDClassifierModel(BaseModel):
    """Stochastic Gradient Descent Classifier"""
    
    def __init__(self, **params):
        default_params = {
            'loss': 'hinge',
            'penalty': 'l2',
            'alpha': 0.0001,
            'max_iter': 1000,
            'tol': 1e-3,
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = SGDClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the SGD Classifier model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("SGD Classifier trained successfully")
                
        except Exception as e:
            logging.error(f"Error training SGD Classifier: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nSGD Classifier Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Loss function: {self.model.loss}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'SGD Classifier',
            'type': 'linear',
            'description': 'Stochastic Gradient Descent classifier',
            'parameters': self.params
        }


class GaussianNBModel(BaseModel):
    """Gaussian Naive Bayes for continuous features"""
    
    def __init__(self, **params):
        default_params = {
            'priors': None,
            'var_smoothing': 1e-9
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = GaussianNB(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Gaussian NB model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Gaussian NB trained successfully")
                
        except Exception as e:
            logging.error(f"Error training Gaussian NB: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nGaussian NB Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of classes: {len(self.model.classes_)}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Gaussian Naive Bayes',
            'type': 'probabilistic',
            'description': 'Naive Bayes for continuous features',
            'parameters': self.params
        }


class BernoulliNBModel(BaseModel):
    """Bernoulli Naive Bayes for binary features"""
    
    def __init__(self, **params):
        default_params = {
            'alpha': 1.0,
            'binarize': 0.0,
            'fit_prior': True
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = BernoulliNB(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Bernoulli NB model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Bernoulli NB trained successfully")
                
        except Exception as e:
            logging.error(f"Error training Bernoulli NB: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nBernoulli NB Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of classes: {len(self.model.classes_)}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Bernoulli Naive Bayes',
            'type': 'probabilistic',
            'description': 'Naive Bayes for binary features',
            'parameters': self.params
        }


class LDAModel(BaseModel):
    """Linear Discriminant Analysis"""
    
    def __init__(self, **params):
        default_params = {
            'solver': 'svd',
            'shrinkage': None,
            'priors': None,
            'n_components': None
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = LinearDiscriminantAnalysis(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the LDA model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("LDA trained successfully")
                
        except Exception as e:
            logging.error(f"Error training LDA: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nLDA Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Solver: {self.model.solver}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Linear Discriminant Analysis',
            'type': 'linear',
            'description': 'Linear discriminant analysis classifier',
            'parameters': self.params
        }


class QDAModel(BaseModel):
    """Quadratic Discriminant Analysis"""
    
    def __init__(self, **params):
        default_params = {
            'priors': None,
            'reg_param': 0.0,
            'store_covariance': False,
            'tol': 1e-4
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = QuadraticDiscriminantAnalysis(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the QDA model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("QDA trained successfully")
                
        except Exception as e:
            logging.error(f"Error training QDA: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nQDA Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Regularization parameter: {self.model.reg_param}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Quadratic Discriminant Analysis',
            'type': 'linear',
            'description': 'Quadratic discriminant analysis classifier',
            'parameters': self.params
        }


class BaggingModel(BaseModel):
    """Bagging Classifier ensemble"""
    
    def __init__(self, **params):
        default_params = {
            'n_estimators': 10,
            'max_samples': 1.0,
            'max_features': 1.0,
            'bootstrap': True,
            'bootstrap_features': False,
            'random_state': 42
        }
        default_params.update(params)
        super().__init__(**default_params)
    
    def _initialize_model(self):
        self.model = BaggingClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Bagging model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Bagging Classifier trained successfully")
                
        except Exception as e:
            logging.error(f"Error training Bagging Classifier: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nBagging Classifier Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of estimators: {self.model.n_estimators}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Bagging Classifier',
            'type': 'ensemble',
            'description': 'Bagging ensemble method',
            'parameters': self.params
        }


class HistGradientBoostingModel(BaseModel):
    """Histogram-based Gradient Boosting"""
    
    def __init__(self, random_state: int = 42, **params):
        super().__init__(random_state=random_state)
        self.params = {
            'max_iter': 100,
            'max_depth': None,
            'learning_rate': 0.1,
            'l2_regularization': 0.0,
            'random_state': random_state
        }
        self.params.update(params)
        self._initialize_model()
    
    def _initialize_model(self):
        self.model = HistGradientBoostingClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Histogram Gradient Boosting model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Histogram Gradient Boosting trained successfully")
                
        except Exception as e:
            logging.error(f"Error training Histogram Gradient Boosting: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nHistogram Gradient Boosting Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of iterations: {self.model.n_iter_}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Histogram Gradient Boosting',
            'type': 'ensemble',
            'description': 'Histogram-based gradient boosting for faster training',
            'parameters': self.params
        }


class SGDModel(BaseModel):
    """Stochastic Gradient Descent Classifier"""
    
    def __init__(self, random_state: int = 42, **params):
        super().__init__(random_state=random_state)
        self.params = {
            'loss': 'hinge',
            'penalty': 'l2',
            'alpha': 0.0001,
            'max_iter': 1000,
            'tol': 1e-3,
            'random_state': random_state
        }
        self.params.update(params)
        self._initialize_model()
    
    def _initialize_model(self):
        self.model = SGDClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the SGD model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("SGD Classifier trained successfully")
                
        except Exception as e:
            logging.error(f"Error training SGD Classifier: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nSGD Classifier Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Loss function: {self.model.loss}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'SGD Classifier',
            'type': 'linear',
            'description': 'Stochastic Gradient Descent classifier',
            'parameters': self.params
        }


class VotingModel(BaseModel):
    """Voting Classifier Ensemble"""
    
    def __init__(self, random_state: int = 42, **params):
        super().__init__(random_state=random_state)
        self.params = {
            'estimators': [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=random_state)),
                ('lr', LogisticRegression(random_state=random_state)),
                ('svc', SVC(probability=True, random_state=random_state))
            ],
            'voting': 'soft',
            'weights': None
        }
        self.params.update(params)
        self._initialize_model()
    
    def _initialize_model(self):
        self.model = VotingClassifier(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Voting model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Voting Classifier trained successfully")
                
        except Exception as e:
            logging.error(f"Error training Voting Classifier: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nVoting Classifier Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Number of estimators: {len(self.model.estimators_)}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Voting Classifier',
            'type': 'ensemble',
            'description': 'Voting ensemble combining multiple classifiers',
            'parameters': self.params
        }


class RandomTreesEmbeddingModel(BaseModel):
    """Random Trees Embedding for unsupervised feature transformation"""
    
    def __init__(self, random_state: int = 42, **params):
        super().__init__(random_state=random_state)
        self.params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': random_state
        }
        self.params.update(params)
        self._initialize_model()
    
    def _initialize_model(self):
        self.model = RandomTreesEmbedding(**self.params)
    
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> None:
        """Train the Random Trees Embedding model"""
        if not self._validate_input(X, y):
            return
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            logging.info("Random Trees Embedding trained successfully")
                
        except Exception as e:
            logging.error(f"Error training Random Trees Embedding: {str(e)}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.transform(X)
    
    def print_results(self) -> None:
        """Print model information"""
        print(f"\nRandom Trees Embedding Model Results:")
        print(f"Trained: {self.is_trained}")
        print(f"Parameters: {self.params}")
        if self.is_trained:
            print(f"Output dimension: {self.model.n_estimators}")
    
    def get_model_info(self) -> dict:
        return {
            'name': 'Random Trees Embedding',
            'type': 'embedding',
            'description': 'Unsupervised feature transformation using random trees',
            'parameters': self.params
        }


# Extended model registry
EXTENDED_MODEL_REGISTRY = {
    'enhanced_random_forest': ExtendedRandomForestModel,
    'enhanced_gradient_boosting': ExtendedGradientBoostingModel,
    'logistic_regression': LogisticRegressionModel,
    'svm': SVMModel,
    'naive_bayes': NaiveBayesModel,
    'knn': KNNModel,
    'mlp': MLPModel,
    'decision_tree': DecisionTreeModel,
    'adaboost': AdaBoostModel,
    'extra_trees': ExtraTreesModel,
    'linear_svc': LinearSVCModel,
    'ridge_classifier': RidgeClassifierModel,
    'sgd_classifier': SGDClassifierModel,
    'gaussian_nb': GaussianNBModel,
    'bernoulli_nb': BernoulliNBModel,
    'lda': LDAModel,
    'qda': QDAModel,
    'bagging': BaggingModel,
    'hist_gradient_boosting': HistGradientBoostingModel,
    'sgd': SGDModel,
    'voting': VotingModel,
    'random_trees_embedding': RandomTreesEmbeddingModel
}


def get_extended_model_info() -> dict:
    """Get information about all extended models"""
    info = {}
    for name, model_class in EXTENDED_MODEL_REGISTRY.items():
        try:
            model = model_class(random_state=42)
            info[name] = model.get_model_info()
        except Exception as e:
            # Fallback if model creation fails
            info[name] = {
                'name': name.replace('_', ' ').title(),
                'type': 'unknown',
                'description': f'Extended model (error getting info: {str(e)})',
                'parameters': [],
                'category': 'extended'
            }
    return info


def list_extended_models() -> list:
    """List all available extended models"""
    return list(EXTENDED_MODEL_REGISTRY.keys())
