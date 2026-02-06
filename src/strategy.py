import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import logging


class StrategySelector:
    """
    Step 8: Decision (Supervised vs Unsupervised)
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.strategy_results = {}
        
    def analyze_data_characteristics(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Analyze data characteristics"""
        
        characteristics = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'has_labels': y is not None,
            'label_quality': {},
            'data_sparsity': self._calculate_sparsity(X),
            'feature_variance': np.var(X, axis=0).mean() if hasattr(X, 'shape') else 0
        }
        
        if y is not None:
            # Label quality analysis
            unique_labels = len(np.unique(y))
            label_balance = self._calculate_label_balance(y)
            
            characteristics['label_quality'] = {
                'n_unique_labels': unique_labels,
                'label_balance_score': label_balance,
                'is_multiclass': unique_labels > 2,
                'is_imbalanced': label_balance < 0.7
            }
        
        return characteristics
    
    def _calculate_sparsity(self, X: Union[pd.DataFrame, np.ndarray]) -> float:
        """Calculate data sparsity"""
        if hasattr(X, 'toarray'):  # Sparse matrix
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)
        
        zero_elements = np.sum(X_dense == 0)
        total_elements = X_dense.size
        
        return zero_elements / total_elements
    
    def _calculate_label_balance(self, y: Union[pd.Series, np.ndarray]) -> float:
        """Calculate label balance score (0-1, where 1 = perfectly balanced)"""
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        value_counts = y.value_counts()
        normalized_counts = value_counts / value_counts.sum()
        
        # Entropy as balance measure
        entropy = -np.sum(normalized_counts * np.log2(normalized_counts + 1e-10))
        max_entropy = np.log2(len(value_counts))
        
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def evaluate_supervised_feasibility(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Evaluate supervised learning feasibility"""
        
        characteristics = self.analyze_data_characteristics(X, y)
        label_quality = characteristics['label_quality']
        
        feasibility_score = 0
        reasons = []
        
        # Check samples per class
        min_samples_per_class = min(pd.Series(y).value_counts())
        if min_samples_per_class >= 10:
            feasibility_score += 0.3
        else:
            reasons.append(f"Too few samples per class (min: {min_samples_per_class})")
        
        # Check class balance
        balance_score = label_quality['label_balance_score']
        logging.info(f"Label balance score: {balance_score:.3f}")
        if balance_score >= 0.7:
            feasibility_score += 0.2
        else:
            reasons.append("Class is imbalanced")
        
        # Check sample count
        if characteristics['n_samples'] >= 100:
            feasibility_score += 0.2
        else:
            reasons.append("Too few samples overall")
        
        # Check feature count
        if characteristics['n_features'] <= characteristics['n_samples'] / 10:
            feasibility_score += 0.2
        else:
            reasons.append("Too many features compared to samples")
        
        # Check label quality
        if not label_quality['is_imbalanced']:
            feasibility_score += 0.1
        
        recommendation = "supervised" if feasibility_score >= 0.6 else "unsupervised"
        
        logging.info(f"Supervised feasibility score: {feasibility_score:.3f}")
        logging.info(f"Recommendation: {recommendation}")
        
        return {
            'feasibility_score': feasibility_score,
            'recommendation': recommendation,
            'reasons': reasons,
            'characteristics': characteristics
        }
    
    def perform_clustering_analysis(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        max_clusters: int = 10
    ) -> Dict[str, Any]:
        """Perform clustering analysis"""
        
        results = {}
        
        # Dimensionality reduction for visualization
        if X.shape[1] > 50:
            pca = PCA(n_components=min(50, X.shape[1]), random_state=self.random_state)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
        
        # K-means clustering for different cluster counts
        k_range = range(2, min(max_clusters + 1, X.shape[0]))
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_reduced)
            
            if len(np.unique(cluster_labels)) > 1:  # Require at least 2 clusters
                silhouette_avg = silhouette_score(X_reduced, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            else:
                silhouette_scores.append(0)
        
        # Find optimal number of clusters
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
            best_silhouette = max(silhouette_scores)
        else:
            optimal_k = 2
            best_silhouette = 0
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_reduced)
        n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        results = {
            'optimal_k': optimal_k,
            'silhouette_scores': dict(zip(k_range, silhouette_scores)),
            'best_silhouette_score': best_silhouette,
            'dbscan_clusters': n_dbscan_clusters,
            'dbscan_noise_points': list(dbscan_labels).count(-1),
            'clustering_feasible': best_silhouette > 0.3
        }
        
        return results
    
    def detect_anomalies(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """Detect anomalies in data"""
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.random_state
        )
        iso_labels = iso_forest.fit_predict(X)
        iso_anomalies = (iso_labels == -1).sum()
        
        # One-Class SVM
        oc_svm = OneClassSVM(nu=contamination)
        svm_labels = oc_svm.fit_predict(X)
        svm_anomalies = (svm_labels == -1).sum()
        
        return {
            'isolation_forest_anomalies': iso_anomalies,
            'one_class_svm_anomalies': svm_anomalies,
            'anomaly_ratio': iso_anomalies / len(X),
            'has_significant_anomalies': iso_anomalies > len(X) * 0.05
        }
    
    def recommend_strategy(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        domain_knowledge: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Recommend machine learning strategy"""
        
        # Analyze data characteristics
        characteristics = self.analyze_data_characteristics(X, y)
        
        recommendation = {
            'primary_strategy': None,
            'confidence': 0,
            'reasoning': [],
            'alternative_strategies': [],
            'preprocessing_recommendations': []
        }
        
        if y is not None:
            # We have labels - evaluate supervised learning
            supervised_eval = self.evaluate_supervised_feasibility(X, y)
            
            if supervised_eval['feasibility_score'] >= 0.6:
                recommendation['primary_strategy'] = 'supervised'
                recommendation['confidence'] = supervised_eval['feasibility_score']
                recommendation['reasoning'].append("Labels are of good quality")
                
                if supervised_eval['characteristics']['label_quality']['is_imbalanced']:
                    recommendation['preprocessing_recommendations'].append("Consider data balancing")
            else:
                recommendation['primary_strategy'] = 'unsupervised'
                recommendation['confidence'] = 1 - supervised_eval['feasibility_score']
                recommendation['reasoning'].extend(supervised_eval['reasons'])
                recommendation['alternative_strategies'].append('semi_supervised')
        else:
            # No labels - unsupervised learning
            clustering_analysis = self.perform_clustering_analysis(X)
            anomaly_analysis = self.detect_anomalies(X)
            
            if clustering_analysis['clustering_feasible']:
                recommendation['primary_strategy'] = 'clustering'
                recommendation['confidence'] = clustering_analysis['best_silhouette_score']
                recommendation['reasoning'].append(f"Data shows clustering structure (k={clustering_analysis['optimal_k']})")
            else:
                recommendation['primary_strategy'] = 'anomaly_detection'
                recommendation['confidence'] = 0.5
                recommendation['reasoning'].append("No clear clustering structure")
            
            if anomaly_analysis['has_significant_anomalies']:
                recommendation['alternative_strategies'].append('anomaly_detection')
        
        # Add preprocessing recommendations
        if characteristics['data_sparsity'] > 0.8:
            recommendation['preprocessing_recommendations'].append("Data is very sparse - consider dimensionality reduction")
        
        if characteristics['feature_variance'] < 0.01:
            recommendation['preprocessing_recommendations'].append("Low feature variance - consider feature selection")
        
        # Consider domain knowledge
        if domain_knowledge:
            if domain_knowledge.get('requires_interpretability', False):
                recommendation['reasoning'].append("Model interpretability required")
            
            if domain_knowledge.get('known_classes', 0) > 0:
                recommendation['reasoning'].append(f"Known number of classes: {domain_knowledge['known_classes']}")
        
        return recommendation
    
    def generate_strategy_report(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        domain_knowledge: Optional[Dict] = None
    ) -> str:
        """Generate strategy recommendation report"""
        
        recommendation = self.recommend_strategy(X, y, domain_knowledge)
        
        report = f"""
# Machine Learning Strategy Recommendation Report

## Data Analysis
- Number of samples: {X.shape[0]}
- Number of features: {X.shape[1]}
- Labels available: {'Yes' if y is not None else 'No'}

## Recommended Strategy
**Primary strategy:** {recommendation['primary_strategy']}
**Recommendation confidence:** {recommendation['confidence']:.2f}

## Justification
"""
        
        for reason in recommendation['reasoning']:
            report += f"- {reason}\n"
        
        if recommendation['alternative_strategies']:
            report += "\n## Alternative Strategies\n"
            for alt in recommendation['alternative_strategies']:
                report += f"- {alt}\n"
        
        if recommendation['preprocessing_recommendations']:
            report += "\n## Preprocessing Recommendations\n"
            for prep in recommendation['preprocessing_recommendations']:
                report += f"- {prep}\n"
        
        return report


if __name__ == "__main__":
    # Usage example
    from sklearn.datasets import make_classification, make_blobs
    import numpy as np
    
    # Example 1: Supervised data
    X_supervised, y_supervised = make_classification(
        n_samples=1000,
        n_classes=3,
        n_features=20,
        weights=[0.6, 0.3, 0.1],
        random_state=42
    )
    
    # Example 2: Unsupervised data
    X_unsupervised, _ = make_blobs(
        n_samples=500,
        centers=4,
        n_features=10,
        random_state=42
    )
    
    strategy = StrategySelector()
    
    # Supervised data analysis
    print("=== Supervised Data Analysis ===")
    rec_supervised = strategy.recommend_strategy(X_supervised, y_supervised)
    print(f"Recommendation: {rec_supervised['primary_strategy']}")
    print(f"Confidence: {rec_supervised['confidence']:.2f}")
    print(f"Justification: {rec_supervised['reasoning']}")
    
    # Unsupervised data analysis
    print("\n=== Unsupervised Data Analysis ===")
    rec_unsupervised = strategy.recommend_strategy(X_unsupervised)
    print(f"Recommendation: {rec_unsupervised['primary_strategy']}")
    print(f"Confidence: {rec_unsupervised['confidence']:.2f}")
    print(f"Justification: {rec_unsupervised['reasoning']}")
    
    # Generate report
    report = strategy.generate_strategy_report(X_supervised, y_supervised)
    print(report)
