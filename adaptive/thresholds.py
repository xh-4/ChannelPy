"""
Advanced threshold learning and adaptation strategies

Provides multiple methods for learning optimal thresholds from data
"""
from typing import Optional, Tuple, List, Callable, Dict
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize_scalar
from sklearn.metrics import mutual_info_score


class BaseThresholdLearner(ABC):
    """
    Abstract base class for threshold learning
    """
    
    def __init__(self):
        self.threshold_i = None
        self.threshold_q = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Learn optimal thresholds
        
        Parameters
        ----------
        X : array-like
            Feature values
        y : array-like, optional
            Target labels (for supervised methods)
            
        Returns
        -------
        self : BaseThresholdLearner
        """
        pass
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Return learned thresholds"""
        if not self.is_fitted:
            raise RuntimeError("Thresholds not fitted. Call fit() first.")
        return self.threshold_i, self.threshold_q


class StatisticalThresholds(BaseThresholdLearner):
    """
    Statistical threshold learning using percentiles
    
    Examples
    --------
    >>> learner = StatisticalThresholds(percentile_i=50, percentile_q=75)
    >>> learner.fit(data)
    >>> threshold_i, threshold_q = learner.get_thresholds()
    """
    
    def __init__(self, percentile_i: float = 50.0, percentile_q: float = 75.0):
        """
        Parameters
        ----------
        percentile_i : float
            Percentile for i-bit threshold (0-100)
        percentile_q : float
            Percentile for q-bit threshold (0-100)
        """
        super().__init__()
        self.percentile_i = percentile_i
        self.percentile_q = percentile_q
    
    def fit(self, X, y=None):
        """Learn thresholds from data distribution"""
        X = np.asarray(X)
        
        self.threshold_i = np.percentile(X, self.percentile_i)
        self.threshold_q = np.percentile(X, self.percentile_q)
        
        self.is_fitted = True
        return self


class SupervisedThresholds(BaseThresholdLearner):
    """
    Learn thresholds to maximize classification performance
    
    Finds thresholds that best separate classes
    
    Examples
    --------
    >>> learner = SupervisedThresholds(metric='mutual_info')
    >>> learner.fit(features, labels)
    >>> threshold_i, threshold_q = learner.get_thresholds()
    """
    
    def __init__(self, metric: str = 'mutual_info', n_candidates: int = 20):
        """
        Parameters
        ----------
        metric : str
            Optimization metric:
            - 'mutual_info': Mutual information
            - 'correlation': Pearson correlation
            - 'accuracy': Classification accuracy
        n_candidates : int
            Number of threshold candidates to try
        """
        super().__init__()
        self.metric = metric
        self.n_candidates = n_candidates
    
    def fit(self, X, y):
        """Learn thresholds by optimizing classification metric"""
        if y is None:
            raise ValueError("Supervised learning requires labels (y)")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Generate candidate thresholds
        candidates = np.percentile(
            X, 
            np.linspace(10, 90, self.n_candidates)
        )
        
        # Find optimal i-threshold
        best_score_i = -np.inf
        best_threshold_i = candidates[0]
        
        for threshold in candidates:
            bit_values = (X > threshold).astype(int)
            score = self._compute_score(bit_values, y)
            
            if score > best_score_i:
                best_score_i = score
                best_threshold_i = threshold
        
        self.threshold_i = best_threshold_i
        
        # Find optimal q-threshold (among values > threshold_i)
        X_filtered = X[X > self.threshold_i]
        y_filtered = y[X > self.threshold_i]
        
        if len(X_filtered) > 10:
            candidates_q = np.percentile(
                X_filtered,
                np.linspace(10, 90, self.n_candidates)
            )
            
            best_score_q = -np.inf
            best_threshold_q = candidates_q[0]
            
            for threshold in candidates_q:
                bit_values = (X_filtered > threshold).astype(int)
                score = self._compute_score(bit_values, y_filtered)
                
                if score > best_score_q:
                    best_score_q = score
                    best_threshold_q = threshold
            
            self.threshold_q = best_threshold_q
        else:
            # Not enough data, use percentile
            self.threshold_q = self.threshold_i * 1.5
        
        self.is_fitted = True
        return self
    
    def _compute_score(self, predictions, targets):
        """Compute optimization metric"""
        if self.metric == 'mutual_info':
            return mutual_info_score(targets, predictions)
        
        elif self.metric == 'correlation':
            return np.corrcoef(predictions, targets)[0, 1]
        
        elif self.metric == 'accuracy':
            # Assume binary classification
            return np.mean(predictions == targets)
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")


class DomainThresholds(BaseThresholdLearner):
    """
    Domain-specific threshold rules
    
    Uses expert knowledge about the domain
    
    Examples
    --------
    >>> learner = DomainThresholds(domain='trading')
    >>> learner.set_rule('rsi', threshold_i=50, threshold_q=70)
    >>> learner.fit(rsi_values)
    """
    
    def __init__(self, domain: str = 'generic'):
        """
        Parameters
        ----------
        domain : str
            Domain context for default rules
        """
        super().__init__()
        self.domain = domain
        self.rules = self._get_domain_rules()
    
    def _get_domain_rules(self) -> Dict:
        """Get default rules for domain"""
        if self.domain == 'trading':
            return {
                'rsi': {'threshold_i': 50, 'threshold_q': 70},
                'volume': {'threshold_i': 1.0, 'threshold_q': 1.5},  # Relative to average
                'price': {'threshold_i': 0.0, 'threshold_q': 0.02}  # % change
            }
        elif self.domain == 'medical':
            return {
                'temperature': {'threshold_i': 37.5, 'threshold_q': 38.5},
                'blood_pressure': {'threshold_i': 140, 'threshold_q': 160},
                'heart_rate': {'threshold_i': 100, 'threshold_q': 120}
            }
        else:
            return {}
    
    def set_rule(self, feature_name: str, threshold_i: float, threshold_q: float):
        """Set threshold rule for specific feature"""
        self.rules[feature_name] = {
            'threshold_i': threshold_i,
            'threshold_q': threshold_q
        }
    
    def fit(self, X, y=None, feature_name: Optional[str] = None):
        """Apply domain rules"""
        if feature_name and feature_name in self.rules:
            rule = self.rules[feature_name]
            self.threshold_i = rule['threshold_i']
            self.threshold_q = rule['threshold_q']
        else:
            # Fallback to statistical
            statistical = StatisticalThresholds()
            statistical.fit(X, y)
            self.threshold_i = statistical.threshold_i
            self.threshold_q = statistical.threshold_q
        
        self.is_fitted = True
        return self


class OptimalThresholds(BaseThresholdLearner):
    """
    Find optimal thresholds using numerical optimization
    
    Minimizes custom objective function
    
    Examples
    --------
    >>> def custom_loss(threshold, X, y):
    ...     predictions = (X > threshold).astype(int)
    ...     return -accuracy_score(y, predictions)
    >>> 
    >>> learner = OptimalThresholds(objective=custom_loss)
    >>> learner.fit(X, y)
    """
    
    def __init__(self, objective: Optional[Callable] = None):
        """
        Parameters
        ----------
        objective : callable, optional
            Custom objective function(threshold, X, y) -> score
            If None, uses default (maximize separation)
        """
        super().__init__()
        self.objective = objective or self._default_objective
    
    def fit(self, X, y=None):
        """Find optimal thresholds via optimization"""
        X = np.asarray(X)
        
        x_min, x_max = X.min(), X.max()
        
        # Optimize i-threshold
        result_i = minimize_scalar(
            lambda t: -self.objective(t, X, y),
            bounds=(x_min, x_max),
            method='bounded'
        )
        self.threshold_i = result_i.x
        
        # Optimize q-threshold (constrained to be > threshold_i)
        result_q = minimize_scalar(
            lambda t: -self.objective(t, X, y),
            bounds=(self.threshold_i, x_max),
            method='bounded'
        )
        self.threshold_q = result_q.x
        
        self.is_fitted = True
        return self
    
    def _default_objective(self, threshold, X, y):
        """Default objective: maximize class separation"""
        bits = (X > threshold).astype(int)
        
        if y is not None:
            # With labels: maximize correlation
            return np.corrcoef(bits, y)[0, 1]
        else:
            # Without labels: maximize entropy (balanced split)
            p = np.mean(bits)
            entropy = -p * np.log(p + 1e-10) - (1-p) * np.log(1-p + 1e-10)
            return entropy


class MultiFeatureThresholdLearner:
    """
    Learn thresholds for multiple features simultaneously
    
    Examples
    --------
    >>> learner = MultiFeatureThresholdLearner()
    >>> learner.add_feature('feature_1', StatisticalThresholds())
    >>> learner.add_feature('feature_2', SupervisedThresholds())
    >>> learner.fit(X_dict, y)
    """
    
    def __init__(self):
        self.feature_learners = {}  # feature_name -> learner
        self.is_fitted = False
    
    def add_feature(self, feature_name: str, learner: BaseThresholdLearner):
        """Add threshold learner for specific feature"""
        self.feature_learners[feature_name] = learner
    
    def fit(self, X_dict: Dict[str, np.ndarray], y=None):
        """
        Fit all feature learners
        
        Parameters
        ----------
        X_dict : dict
            Dictionary of feature_name -> feature_values
        y : array-like, optional
            Target labels
        """
        for feature_name, learner in self.feature_learners.items():
            if feature_name in X_dict:
                learner.fit(X_dict[feature_name], y)
        
        self.is_fitted = True
        return self
    
    def get_thresholds(self, feature_name: str) -> Tuple[float, float]:
        """Get thresholds for specific feature"""
        if not self.is_fitted:
            raise RuntimeError("Learner not fitted. Call fit() first.")
        
        if feature_name not in self.feature_learners:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        return self.feature_learners[feature_name].get_thresholds()
    
    def get_all_thresholds(self) -> Dict[str, Tuple[float, float]]:
        """Get thresholds for all features"""
        return {
            name: learner.get_thresholds()
            for name, learner in self.feature_learners.items()
        }


class ThresholdStabilityAnalyzer:
    """
    Analyze stability of learned thresholds
    
    Uses bootstrap resampling to assess threshold variance
    
    Examples
    --------
    >>> analyzer = ThresholdStabilityAnalyzer(n_bootstrap=100)
    >>> stability = analyzer.analyze(X, y, StatisticalThresholds())
    >>> print(f"Threshold CI: {stability['confidence_interval']}")
    """
    
    def __init__(self, n_bootstrap: int = 100, confidence: float = 0.95):
        """
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples
        confidence : float
            Confidence level for intervals
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
    
    def analyze(self, X, y, learner: BaseThresholdLearner) -> Dict:
        """
        Analyze threshold stability
        
        Returns
        -------
        stability : dict
            Stability metrics including confidence intervals
        """
        X = np.asarray(X)
        n = len(X)
        
        thresholds_i = []
        thresholds_q = []
        
        # Bootstrap resampling
        for _ in range(self.n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            X_boot = X[indices]
            y_boot = y[indices] if y is not None else None
            
            # Fit learner
            learner_copy = type(learner)()
            try:
                learner_copy.fit(X_boot, y_boot)
                thresh_i, thresh_q = learner_copy.get_thresholds()
                thresholds_i.append(thresh_i)
                thresholds_q.append(thresh_q)
            except:
                # Skip if fitting fails
                continue
        
        # Compute statistics
        alpha = 1 - self.confidence
        
        return {
            'mean_threshold_i': np.mean(thresholds_i),
            'std_threshold_i': np.std(thresholds_i),
            'ci_threshold_i': np.percentile(thresholds_i, [alpha/2*100, (1-alpha/2)*100]),
            'mean_threshold_q': np.mean(thresholds_q),
            'std_threshold_q': np.std(thresholds_q),
            'ci_threshold_q': np.percentile(thresholds_q, [alpha/2*100, (1-alpha/2)*100]),
            'stability_score': 1.0 / (1.0 + np.std(thresholds_i) + np.std(thresholds_q))
        }




