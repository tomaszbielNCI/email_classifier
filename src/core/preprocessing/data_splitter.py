import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import (
    train_test_split, StratifiedShuffleSplit, StratifiedKFold,
    cross_val_score, GridSearchCV
)
from sklearn.preprocessing import LabelEncoder
import logging


class DataSplitter:
    """
    Step 9: Train/test split
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.split_info = {}
        
    def basic_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], 
                Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """Basic train/test split"""
        
        stratify_param = y if stratify and len(np.unique(y)) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        self.split_info['basic'] = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_percentage': test_size,
            'stratified': stratify,
            'train_class_distribution': self._get_class_distribution(y_train),
            'test_class_distribution': self._get_class_distribution(y_test)
        }
        
        logging.info(f"Basic split: {len(X_train)} training, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test
    
    def stratified_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
        n_splits: int = 5
    ) -> List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray], 
                     Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]]:
        """Stratified split with cross-validation"""
        
        if len(np.unique(y)) < 2:
            raise ValueError("Stratification requires at least 2 classes")
        
        sss = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=self.random_state
        )
        
        splits = []
        
        for i, (train_idx, test_idx) in enumerate(sss.split(X, y)):
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
            
            if isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]
            
            splits.append((X_train, X_test, y_train, y_test))
            
            logging.info(f"Split {i+1}/{n_splits}: {len(X_train)} training, {len(X_test)} test")
        
        self.split_info['stratified'] = {
            'n_splits': n_splits,
            'test_size': test_size,
            'splits_created': len(splits)
        }
        
        return splits
    
    def temporal_split(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        feature_columns: List[str],
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Temporal split (for time series data)"""
        
        if date_column not in df.columns:
            raise ValueError(f"Column '{date_column}' does not exist")
        
        # Sort by date
        df_sorted = df.sort_values(date_column).reset_index(drop=True)
        
        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_validation = int(n_total * validation_size)
        n_train = n_total - n_test - n_validation
        
        # Temporal split
        train_df = df_sorted.iloc[:n_train]
        validation_df = df_sorted.iloc[n_train:n_train + n_validation] if n_validation > 0 else None
        test_df = df_sorted.iloc[n_train + n_validation:]
        
        self.split_info['temporal'] = {
            'train_size': len(train_df),
            'validation_size': len(validation_df) if validation_df is not None else 0,
            'test_size': len(test_df),
            'date_range': {
                'train': (train_df[date_column].min(), train_df[date_column].max()),
                'validation': (validation_df[date_column].min(), validation_df[date_column].max()) if validation_df is not None else None,
                'test': (test_df[date_column].min(), test_df[date_column].max())
            }
        }
        
        logging.info(f"Temporal split: {len(train_df)} training, {len(validation_df) if validation_df is not None else 0} validation, {len(test_df)} test")
        
        if validation_df is not None:
            return train_df, validation_df, test_df
        else:
            return train_df, test_df
    
    def group_based_split(
        self,
        df: pd.DataFrame,
        group_column: str,
        target_column: str,
        feature_columns: List[str],
        test_size: float = 0.2,
        max_groups_per_split: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Group-based split (e.g., customers, products)"""
        
        if group_column not in df.columns:
            raise ValueError(f"Column '{group_column}' does not exist")
        
        # Unique groups
        unique_groups = df[group_column].unique()
        np.random.shuffle(unique_groups)  # Random order
        
        if max_groups_per_split and len(unique_groups) > max_groups_per_split:
            # Limit number of groups
            unique_groups = unique_groups[:max_groups_per_split]
        
        # Split groups
        n_groups = len(unique_groups)
        n_test_groups = int(n_groups * test_size)
        
        test_groups = unique_groups[:n_test_groups]
        train_groups = unique_groups[n_test_groups:]
        
        # Split DataFrame
        train_df = df[df[group_column].isin(train_groups)]
        test_df = df[df[group_column].isin(test_groups)]
        
        self.split_info['group_based'] = {
            'train_groups': len(train_groups),
            'test_groups': len(test_groups),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'group_column': group_column
        }
        
        logging.info(f"Group split: {len(train_groups)} training groups, {len(test_groups)} test groups")
        
        return train_df, test_df
    
    def cross_validation_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv_folds: int = 5,
        stratified: bool = True
    ) -> List[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]]:
        """Split for cross-validation"""
        
        if stratified and len(np.unique(y)) > 1:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        splits = []
        
        for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            
            if isinstance(y, pd.Series):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            splits.append((X_train, X_val, y_train, y_val))
            
            logging.info(f"CV Fold {i+1}/{cv_folds}: {len(X_train)} training, {len(X_val)} validation")
        
        self.split_info['cross_validation'] = {
            'cv_folds': cv_folds,
            'stratified': stratified,
            'folds_created': len(splits)
        }
        
        return splits
    
    def nested_cross_validation_split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        outer_folds: int = 5,
        inner_folds: int = 3
    ) -> Dict[str, List]:
        """Nested cross-validation for hyperparameter optimization"""
        
        # Outer CV loop
        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=self.random_state)
        
        nested_splits = {}
        
        for i, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
            # Outer split
            if isinstance(X, pd.DataFrame):
                X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
            else:
                X_outer_train, X_outer_test = X[outer_train_idx], X[outer_test_idx]
            
            if isinstance(y, pd.Series):
                y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]
            else:
                y_outer_train, y_outer_test = y[outer_train_idx], y[outer_test_idx]
            
            # Inner CV loop for hyperparameter optimization
            inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=self.random_state + i)
            inner_splits = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_outer_train, y_outer_train):
                if isinstance(X_outer_train, pd.DataFrame):
                    X_inner_train, X_inner_val = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_val_idx]
                else:
                    X_inner_train, X_inner_val = X_outer_train[inner_train_idx], X_outer_train[inner_val_idx]
                
                if isinstance(y_outer_train, pd.Series):
                    y_inner_train, y_inner_val = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]
                else:
                    y_inner_train, y_inner_val = y_outer_train[inner_train_idx], y_outer_train[inner_val_idx]
                
                inner_splits.append((X_inner_train, X_inner_val, y_inner_train, y_inner_val))
            
            nested_splits[f'fold_{i+1}'] = {
                'outer_split': (X_outer_train, X_outer_test, y_outer_train, y_outer_test),
                'inner_splits': inner_splits
            }
        
        self.split_info['nested_cv'] = {
            'outer_folds': outer_folds,
            'inner_folds': inner_folds,
            'total_inner_splits': outer_folds * inner_folds
        }
        
        return nested_splits
    
    def _get_class_distribution(self, y: Union[pd.Series, np.ndarray]) -> Dict:
        """Calculate class distribution"""
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        return y.value_counts().to_dict()
    
    def get_split_summary(self) -> Dict[str, Any]:
        """Returns summary of all splits"""
        return self.split_info
    
    def validate_split_quality(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Validate data split quality"""
        
        validation_results = {
            'size_ratio': len(X_test) / len(X_train),
            'feature_drift': {},
            'label_drift': {},
            'recommendations': []
        }
        
        # Check size proportions
        if validation_results['size_ratio'] < 0.15:
            validation_results['recommendations'].append("Test set is too small (< 15%)")
        elif validation_results['size_ratio'] > 0.4:
            validation_results['recommendations'].append("Test set is too large (> 40%)")
        
        # Check label drift
        train_dist = self._get_class_distribution(y_train)
        test_dist = self._get_class_distribution(y_test)
        
        for class_label in set(train_dist.keys()) | set(test_dist.keys()):
            train_pct = train_dist.get(class_label, 0) / sum(train_dist.values())
            test_pct = test_dist.get(class_label, 0) / sum(test_dist.values())
            drift = abs(train_pct - test_pct)
            
            validation_results['label_drift'][class_label] = drift
            
            if drift > 0.1:
                validation_results['recommendations'].append(
                    f"Significant drift for class {class_label}: {drift:.3f}"
                )
        
        # Check feature drift (for DataFrame)
        if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
            for col in X_train.columns:
                if X_train[col].dtype in ['int64', 'float64']:
                    train_mean = X_train[col].mean()
                    test_mean = X_test[col].mean()
                    
                    if train_mean != 0:
                        drift = abs(train_mean - test_mean) / abs(train_mean)
                        validation_results['feature_drift'][col] = drift
                        
                        if drift > 0.2:
                            validation_results['recommendations'].append(
                                f"Significant feature drift for {col}: {drift:.3f}"
                            )
        
        return validation_results


if __name__ == "__main__":
    # Usage example
    import pandas as pd
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_classes=3, n_features=10, random_state=42)
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
    df['group'] = np.random.choice(['A', 'B', 'C'], len(df))
    
    splitter = DataSplitter()
    
    # Basic split
    X_train, X_test, y_train, y_test = splitter.basic_split(X, y)
    print(f"Basic split: {X_train.shape[0]} training, {X_test.shape[0]} test")
    
    # Temporal split
    train_df, test_df = splitter.temporal_split(df, 'date', 'target', [f'feature_{i}' for i in range(10)], validation_size=0)
    print(f"Temporal split: {len(train_df)} training, {len(test_df)} test")
    
    # Group split
    train_df, test_df = splitter.group_based_split(df, 'group', 'target', [f'feature_{i}' for i in range(10)])
    print(f"Group split: {len(train_df)} training, {len(test_df)} test")
    
    # Validate split quality
    validation = splitter.validate_split_quality(X_train, X_test, y_train, y_test)
    print(f"\nSplit validation: {len(validation['recommendations'])} recommendations")
    
    # Summary
    summary = splitter.get_split_summary()
    print(f"\nPerformed {len(summary)} split types")
