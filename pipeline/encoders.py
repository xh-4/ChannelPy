"""
Encoders: Features → Channel States

Stage 2 of the pipeline: convert numerical features to channel states
"""

from typing import Optional, Callable, List, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..adaptive.streaming import StreamingAdaptiveThreshold
from ..adaptive.topology_adaptive import TopologyAdaptiveThreshold
from ..adaptive.thresholds import ThresholdLearner


class Encoder(ABC):
    """
    Base class for encoders
    
    All encoders convert features to channel states
    """
    
    @abstractmethod
    def fit(self, X, y=None):
        """Learn encoding parameters"""
        pass
    
    @abstractmethod
    def encode(self, value):
        """Encode single value to State"""
        pass
    
    def __call__(self, X):
        """Make encoder callable"""
        X = np.asarray(X)
        
        if X.ndim == 0 or (X.ndim == 1 and len(X) == 1):
            # Single value
            return self.encode(float(X))
        else:
            # Array of values
            return self.encode_array(X)
    
    def encode_array(self, X):
        """Encode array of values"""
        X = np.asarray(X)
        
        if X.ndim == 1:
            # 1D array
            i_bits = np.array([self.encode(x).i for x in X], dtype=np.int8)
            q_bits = np.array([self.encode(x).q for x in X], dtype=np.int8)
            return StateArray(i=i_bits, q=q_bits)
        else:
            # 2D array - encode first column only (extend for multi-feature)
            return self.encode_array(X[:, 0])


class ThresholdEncoder(Encoder):
    """
    Simple threshold-based encoder
    
    i-bit: value > threshold_i
    q-bit: value > threshold_q
    
    Examples
    --------
    >>> encoder = ThresholdEncoder(threshold_i=0.5, threshold_q=0.75)
    >>> state = encoder.encode(0.8)  # Returns ψ
    >>> states = encoder(np.array([0.3, 0.6, 0.9]))
    """
    
    def __init__(self, threshold_i: float = 0.5, threshold_q: float = 0.75):
        self.threshold_i = threshold_i
        self.threshold_q = threshold_q
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """No learning needed for fixed thresholds"""
        self.is_fitted = True
        return self
    
    def encode(self, value: float) -> State:
        """Encode single value"""
        return State(
            i=int(value > self.threshold_i),
            q=int(value > self.threshold_q)
        )


class LearnedThresholdEncoder(Encoder):
    """
    Learn optimal thresholds from data
    
    Methods:
    - 'percentile': Use percentiles (e.g., 50th, 75th)
    - 'supervised': Optimize for classification (requires labels)
    - 'kmeans': Use k-means clustering
    
    Examples
    --------
    >>> encoder = LearnedThresholdEncoder(method='percentile')
    >>> encoder.fit(X_train)
    >>> states = encoder(X_test)
    """
    
    def __init__(
        self, 
        method: str = 'percentile',
        percentile_i: float = 50,
        percentile_q: float = 75
    ):
        """
        Parameters
        ----------
        method : str
            Learning method
        percentile_i : float
            Percentile for i-threshold (if method='percentile')
        percentile_q : float
            Percentile for q-threshold (if method='percentile')
        """
        self.method = method
        self.percentile_i = percentile_i
        self.percentile_q = percentile_q
        self.threshold_i = None
        self.threshold_q = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Learn thresholds"""
        X = np.asarray(X)
        
        if self.method == 'percentile':
            self.threshold_i = np.percentile(X, self.percentile_i)
            self.threshold_q = np.percentile(X, self.percentile_q)
        
        elif self.method == 'supervised':
            if y is None:
                raise ValueError("Method 'supervised' requires labels")
            
            y = np.asarray(y)
            self._fit_supervised(X, y)
        
        elif self.method == 'kmeans':
            self._fit_kmeans(X)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def _fit_supervised(self, X, y):
        """Find thresholds that maximize separation"""
        # Try different threshold candidates
        candidates = np.percentile(X, np.linspace(10, 90, 9))
        
        best_i = candidates[0]
        best_q = candidates[-1]
        best_score = -np.inf
        
        # Search for optimal i-threshold
        for threshold in candidates:
            bit_values = (X > threshold).astype(int)
            
            # Correlation with labels
            if len(np.unique(y)) > 1:
                score = np.abs(np.corrcoef(bit_values, y)[0, 1])
                
                if score > best_score:
                    best_score = score
                    best_i = threshold
        
        self.threshold_i = best_i
        
        # Search for optimal q-threshold (above i-threshold)
        above_i = X > self.threshold_i
        if np.any(above_i):
            X_above = X[above_i]
            y_above = y[above_i]
            
            candidates_q = np.percentile(X_above, np.linspace(10, 90, 9))
            best_score = -np.inf
            
            for threshold in candidates_q:
                bit_values = (X > threshold).astype(int)
                
                if len(np.unique(y)) > 1:
                    score = np.abs(np.corrcoef(bit_values, y)[0, 1])
                    
                    if score > best_score:
                        best_score = score
                        best_q = threshold
            
            self.threshold_q = best_q
        else:
            self.threshold_q = self.threshold_i * 1.5
    
    def _fit_kmeans(self, X):
        """Use k-means to find natural breakpoints"""
        try:
            from sklearn.cluster import KMeans
            
            # Cluster into 3 groups
            kmeans = KMeans(n_clusters=3, random_state=42)
            X_reshaped = X.reshape(-1, 1)
            kmeans.fit(X_reshaped)
            
            # Sort centroids
            centroids = np.sort(kmeans.cluster_centers_.flatten())
            
            # Use midpoints between centroids as thresholds
            if len(centroids) >= 2:
                self.threshold_i = (centroids[0] + centroids[1]) / 2
                if len(centroids) >= 3:
                    self.threshold_q = (centroids[1] + centroids[2]) / 2
                else:
                    self.threshold_q = centroids[1]
            else:
                # Fallback to percentiles
                self.threshold_i = np.percentile(X, 50)
                self.threshold_q = np.percentile(X, 75)
        
        except ImportError:
            # Fallback if sklearn not available
            self.threshold_i = np.percentile(X, 33)
            self.threshold_q = np.percentile(X, 67)
    
    def encode(self, value: float) -> State:
        """Encode single value"""
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        return State(
            i=int(value > self.threshold_i),
            q=int(value > self.threshold_q)
        )


class AdaptiveEncoder(Encoder):
    """
    Encoder with streaming adaptive thresholds
    
    Thresholds automatically adjust as data evolves
    
    Examples
    --------
    >>> encoder = AdaptiveEncoder(window_size=1000)
    >>> encoder.fit(X_train)
    >>> 
    >>> # In streaming context
    >>> for value in stream:
    ...     state = encoder.encode(value)
    ...     # Thresholds adapt automatically
    """
    
    def __init__(
        self, 
        window_size: int = 1000,
        adaptation_rate: float = 0.01,
        use_topology: bool = False
    ):
        """
        Parameters
        ----------
        window_size : int
            Size of sliding window
        adaptation_rate : float
            Rate of adaptation (0-1)
        use_topology : bool
            Use topology-aware adaptation
        """
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.use_topology = use_topology
        
        if use_topology:
            self.threshold_tracker = TopologyAdaptiveThreshold(
                window_size=window_size,
                adaptation_rate=adaptation_rate
            )
        else:
            self.threshold_tracker = StreamingAdaptiveThreshold(
                window_size=window_size,
                adaptation_rate=adaptation_rate
            )
        
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Initialize with training data"""
        X = np.asarray(X).flatten()
        
        # Initialize tracker with training data
        for value in X:
            self.threshold_tracker.update(value)
        
        self.is_fitted = True
        return self
    
    def encode(self, value: float) -> State:
        """Encode value and update thresholds"""
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        # Update tracker (adapts thresholds)
        self.threshold_tracker.update(value)
        
        # Encode with current thresholds
        return self.threshold_tracker.encode(value)
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds"""
        if hasattr(self.threshold_tracker, 'get_thresholds'):
            return self.threshold_tracker.get_thresholds()
        else:
            return self.threshold_tracker.get_stats()


class DualFeatureEncoder(Encoder):
    """
    Encode from two separate features
    
    One feature for i-bit, another for q-bit
    
    Examples
    --------
    >>> encoder = DualFeatureEncoder()
    >>> encoder.fit(feature_i, feature_q)
    >>> states = encoder(test_feature_i, test_feature_q)
    """
    
    def __init__(self):
        self.encoder_i = LearnedThresholdEncoder(method='percentile', percentile_i=50)
        self.encoder_q = LearnedThresholdEncoder(method='percentile', percentile_q=50)
        self.is_fitted = False
    
    def fit(self, X_i, X_q=None, y=None):
        """
        Fit encoders for both features
        
        Parameters
        ----------
        X_i : array-like
            Feature for i-bit
        X_q : array-like, optional
            Feature for q-bit (if None, uses X_i)
        y : array-like, optional
            Labels
        """
        if X_q is None:
            X_q = X_i
        
        self.encoder_i.fit(X_i, y)
        self.encoder_q.fit(X_q, y)
        
        self.is_fitted = True
        return self
    
    def encode(self, value_i: float, value_q: Optional[float] = None) -> State:
        """Encode from two values"""
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        if value_q is None:
            value_q = value_i
        
        i_bit = self.encoder_i.encode(value_i).i
        q_bit = self.encoder_q.encode(value_q).i  # Use i-bit from q encoder
        
        return State(i=i_bit, q=q_bit)
    
    def __call__(self, X_i, X_q=None):
        """Encode arrays"""
        if X_q is None:
            X_q = X_i
        
        X_i = np.asarray(X_i).flatten()
        X_q = np.asarray(X_q).flatten()
        
        i_bits = np.array([self.encoder_i.encode(x).i for x in X_i], dtype=np.int8)
        q_bits = np.array([self.encoder_q.encode(x).i for x in X_q], dtype=np.int8)
        
        return StateArray(i=i_bits, q=q_bits)


class MultiFeatureEncoder(Encoder):
    """
    Encode multiple features to single state
    
    Aggregates evidence from multiple features
    
    Examples
    --------
    >>> encoder = MultiFeatureEncoder(aggregation='majority_vote')
    >>> encoder.fit(X_train)  # X_train has multiple columns
    >>> states = encoder(X_test)
    """
    
    def __init__(self, aggregation: str = 'majority_vote'):
        """
        Parameters
        ----------
        aggregation : str
            How to aggregate multiple features:
            - 'majority_vote': Majority vote for each bit
            - 'all': All features must agree (conservative)
            - 'any': Any feature triggers (liberal)
        """
        self.aggregation = aggregation
        self.feature_encoders = []
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit encoder for each feature"""
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        self.feature_encoders = []
        
        for i in range(n_features):
            encoder = LearnedThresholdEncoder()
            encoder.fit(X[:, i], y)
            self.feature_encoders.append(encoder)
        
        self.is_fitted = True
        return self
    
    def encode(self, values) -> State:
        """Encode feature vector"""
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        values = np.asarray(values).flatten()
        
        # Encode each feature
        states = [enc.encode(val) for enc, val in zip(self.feature_encoders, values)]
        
        # Aggregate
        i_bits = [s.i for s in states]
        q_bits = [s.q for s in states]
        
        if self.aggregation == 'majority_vote':
            i = int(sum(i_bits) > len(i_bits) / 2)
            q = int(sum(q_bits) > len(q_bits) / 2)
        
        elif self.aggregation == 'all':
            i = int(all(i_bits))
            q = int(all(q_bits))
        
        elif self.aggregation == 'any':
            i = int(any(i_bits))
            q = int(any(q_bits))
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return State(i=i, q=q)
    
    def encode_array(self, X):
        """Encode 2D array"""
        X = np.asarray(X)
        
        if X.ndim == 1:
            return self.encode(X)
        
        states = [self.encode(X[i, :]) for i in range(len(X))]
        i_bits = np.array([s.i for s in states], dtype=np.int8)
        q_bits = np.array([s.q for s in states], dtype=np.int8)
        
        return StateArray(i=i_bits, q=q_bits)


class TopologyAwareEncoder(Encoder):
    """
    Encoder that uses distributional topology
    
    Automatically adapts to multimodal, skewed, or clustered distributions
    
    Examples
    --------
    >>> encoder = TopologyAwareEncoder()
    >>> encoder.fit(X_train)
    >>> states = encoder(X_test)
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.threshold_tracker = TopologyAdaptiveThreshold(
            window_size=window_size
        )
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Initialize with training data"""
        X = np.asarray(X).flatten()
        
        for value in X:
            self.threshold_tracker.update(value)
        
        self.is_fitted = True
        return self
    
    def encode(self, value: float) -> State:
        """Encode with topology-aware thresholds"""
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        self.threshold_tracker.update(value)
        return self.threshold_tracker.encode(value)
    
    def get_topology(self):
        """Get current topology features"""
        return self.threshold_tracker.get_topology()