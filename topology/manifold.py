"""
Manifold structure detection and analysis

Detects and analyzes low-dimensional manifold structure in data,
enabling manifold-aware encoding and threshold adaptation.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from dataclasses import dataclass
import warnings


@dataclass
class ManifoldFeatures:
    """
    Characteristics of data manifold structure
    
    Attributes
    ----------
    intrinsic_dimension : float
        Estimated intrinsic dimension of data manifold
    ambient_dimension : int
        Dimension of embedding space
    local_density : np.ndarray
        Local density at each point
    curvature : float
        Average manifold curvature
    is_manifold : bool
        Whether data lies on low-dimensional manifold
    confidence : float
        Confidence in manifold detection
    """
    intrinsic_dimension: float
    ambient_dimension: int
    local_density: Optional[np.ndarray] = None
    curvature: float = 0.0
    is_manifold: bool = False
    confidence: float = 0.0
    
    def dimension_ratio(self) -> float:
        """Ratio of intrinsic to ambient dimension"""
        return self.intrinsic_dimension / self.ambient_dimension
    
    def __repr__(self) -> str:
        return (f"ManifoldFeatures(intrinsic_dim={self.intrinsic_dimension:.2f}, "
                f"ambient_dim={self.ambient_dimension}, "
                f"is_manifold={self.is_manifold})")


class ManifoldAnalyzer:
    """
    Analyze manifold structure in data
    
    Detects whether data lies on low-dimensional manifold
    and characterizes its properties.
    
    Examples
    --------
    >>> analyzer = ManifoldAnalyzer()
    >>> data = np.random.randn(1000, 10)  # 10D ambient space
    >>> features = analyzer.analyze(data)
    >>> print(f"Intrinsic dimension: {features.intrinsic_dimension:.2f}")
    >>> print(f"Lies on manifold: {features.is_manifold}")
    """
    
    def __init__(
        self,
        method: str = 'pca',
        n_neighbors: int = 10
    ):
        """
        Parameters
        ----------
        method : str
            Dimension estimation method:
            - 'pca': Principal Component Analysis
            - 'mle': Maximum Likelihood Estimation
            - 'correlation': Correlation dimension
            - 'all': Try all methods and take median
        n_neighbors : int
            Number of neighbors for local analysis
        """
        self.method = method
        self.n_neighbors = n_neighbors
    
    def analyze(self, data: np.ndarray) -> ManifoldFeatures:
        """
        Analyze manifold structure of data
        
        Parameters
        ----------
        data : np.ndarray
            Data to analyze, shape (n_samples, n_features)
            
        Returns
        -------
        features : ManifoldFeatures
            Detected manifold characteristics
        """
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples, n_features = data.shape
        
        if n_samples < 10:
            warnings.warn("Too few samples for reliable manifold analysis")
            return ManifoldFeatures(
                intrinsic_dimension=n_features,
                ambient_dimension=n_features,
                is_manifold=False,
                confidence=0.0
            )
        
        # Estimate intrinsic dimension
        intrinsic_dim = self._estimate_dimension(data)
        
        # Compute local density
        local_density = self._compute_local_density(data)
        
        # Estimate curvature
        curvature = self._estimate_curvature(data)
        
        # Determine if data lies on manifold
        # Heuristic: intrinsic_dim significantly less than ambient_dim
        is_manifold = intrinsic_dim < 0.7 * n_features
        
        # Confidence based on variance explained
        confidence = self._compute_confidence(data, intrinsic_dim)
        
        return ManifoldFeatures(
            intrinsic_dimension=intrinsic_dim,
            ambient_dimension=n_features,
            local_density=local_density,
            curvature=curvature,
            is_manifold=is_manifold,
            confidence=confidence
        )
    
    def _estimate_dimension(self, data: np.ndarray) -> float:
        """
        Estimate intrinsic dimension
        
        Returns fractional dimension estimate
        """
        if self.method == 'all':
            # Try multiple methods and take median
            estimates = []
            
            try:
                estimates.append(self._estimate_pca(data))
            except:
                pass
            
            try:
                estimates.append(self._estimate_mle(data))
            except:
                pass
            
            try:
                estimates.append(self._estimate_correlation(data))
            except:
                pass
            
            if estimates:
                return np.median(estimates)
            else:
                return data.shape[1]  # Fallback
        
        elif self.method == 'pca':
            return self._estimate_pca(data)
        
        elif self.method == 'mle':
            return self._estimate_mle(data)
        
        elif self.method == 'correlation':
            return self._estimate_correlation(data)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _estimate_pca(self, data: np.ndarray) -> float:
        """
        Estimate dimension via PCA variance
        
        Count components needed for 95% variance
        """
        from sklearn.decomposition import PCA
        
        pca = PCA()
        pca.fit(data)
        
        # Cumulative variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        
        # Number of components for 95% variance
        dim = np.argmax(cumsum >= 0.95) + 1
        
        return float(dim)
    
    def _estimate_mle(self, data: np.ndarray) -> float:
        """
        Maximum Likelihood Estimation of intrinsic dimension
        
        Based on local neighborhood analysis
        """
        from scipy.spatial import cKDTree
        
        tree = cKDTree(data)
        
        # Query k nearest neighbors
        k = min(self.n_neighbors, len(data) - 1)
        distances, _ = tree.query(data, k=k+1)
        
        # Remove self (distance 0)
        distances = distances[:, 1:]
        
        # MLE estimator
        # d̂ = (k-1) / Σ log(r_k / r_i)
        
        dimensions = []
        for i in range(len(data)):
            r_k = distances[i, -1]  # k-th nearest neighbor distance
            r_i = distances[i, :-1]  # distances to first k-1 neighbors
            
            # Avoid log(0)
            r_i = r_i[r_i > 0]
            if len(r_i) > 0 and r_k > 0:
                log_sum = np.sum(np.log(r_k / r_i))
                if log_sum > 0:
                    dim_estimate = (len(r_i)) / log_sum
                    dimensions.append(dim_estimate)
        
        if dimensions:
            # Take median to be robust to outliers
            return float(np.median(dimensions))
        else:
            return data.shape[1]
    
    def _estimate_correlation(self, data: np.ndarray) -> float:
        """
        Correlation dimension estimation
        
        Based on scaling of point pairs with distance
        """
        from scipy.spatial.distance import pdist
        
        # Sample if data is large
        if len(data) > 1000:
            indices = np.random.choice(len(data), 1000, replace=False)
            sample = data[indices]
        else:
            sample = data
        
        # Compute pairwise distances
        distances = pdist(sample)
        
        # Range of radii
        radii = np.logspace(
            np.log10(np.min(distances[distances > 0])),
            np.log10(np.max(distances)),
            20
        )
        
        # Count pairs within each radius
        counts = []
        for r in radii:
            count = np.sum(distances <= r)
            counts.append(count)
        
        counts = np.array(counts)
        
        # Fit log-log relationship: log(C(r)) ~ d * log(r)
        # where d is correlation dimension
        
        # Avoid zeros
        valid = (counts > 0) & (radii > 0)
        if np.sum(valid) > 5:
            log_r = np.log(radii[valid])
            log_c = np.log(counts[valid])
            
            # Linear regression
            slope, _ = np.polyfit(log_r, log_c, 1)
            
            return float(slope)
        else:
            return data.shape[1]
    
    def _compute_local_density(self, data: np.ndarray) -> np.ndarray:
        """
        Compute local density at each point
        
        Returns array of density estimates
        """
        from scipy.spatial import cKDTree
        
        tree = cKDTree(data)
        
        # Query k nearest neighbors
        k = min(self.n_neighbors, len(data) - 1)
        distances, _ = tree.query(data, k=k+1)
        
        # Remove self
        distances = distances[:, 1:]
        
        # Density = k / (volume of k-ball)
        # For simplicity, use inverse of average distance
        avg_distances = np.mean(distances, axis=1)
        
        # Avoid division by zero
        densities = 1.0 / (avg_distances + 1e-10)
        
        return densities
    
    def _estimate_curvature(self, data: np.ndarray) -> float:
        """
        Estimate average manifold curvature
        
        Uses local PCA to estimate curvature at each point
        """
        from scipy.spatial import cKDTree
        from sklearn.decomposition import PCA
        
        tree = cKDTree(data)
        
        # Query local neighborhoods
        k = min(self.n_neighbors, len(data) - 1)
        _, indices = tree.query(data, k=k+1)
        
        curvatures = []
        
        for i in range(len(data)):
            # Local neighborhood
            neighbors = data[indices[i]]
            
            if len(neighbors) < 5:
                continue
            
            # Fit local PCA
            pca = PCA()
            pca.fit(neighbors)
            
            # Curvature estimate: ratio of smallest to largest eigenvalue
            eigenvalues = pca.explained_variance_
            
            if len(eigenvalues) > 1 and eigenvalues[0] > 0:
                curvature = eigenvalues[-1] / eigenvalues[0]
                curvatures.append(curvature)
        
        if curvatures:
            return float(np.mean(curvatures))
        else:
            return 0.0
    
    def _compute_confidence(
        self, 
        data: np.ndarray, 
        intrinsic_dim: float
    ) -> float:
        """
        Compute confidence in dimension estimate
        
        Based on variance explained by top components
        """
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=min(int(np.ceil(intrinsic_dim)), data.shape[1]))
        pca.fit(data)
        
        # Variance explained by estimated dimensions
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        return float(variance_explained)


class ManifoldAwareEncoder:
    """
    Channel encoder aware of manifold structure
    
    Adapts encoding strategy based on local manifold properties
    
    Examples
    --------
    >>> encoder = ManifoldAwareEncoder()
    >>> encoder.fit(training_data)
    >>> states = encoder.encode(test_data)
    """
    
    def __init__(
        self,
        n_regions: int = 5,
        method: str = 'density'
    ):
        """
        Parameters
        ----------
        n_regions : int
            Number of manifold regions to identify
        method : str
            Regionalization method:
            - 'density': Based on local density
            - 'cluster': Based on clustering
            - 'adaptive': Density-adaptive thresholds
        """
        self.n_regions = n_regions
        self.method = method
        
        self.analyzer = ManifoldAnalyzer()
        self.region_thresholds = {}
        self.is_fitted = False
    
    def fit(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Fit encoder to data manifold
        
        Parameters
        ----------
        data : np.ndarray
            Training data
        labels : np.ndarray, optional
            Target labels for supervised threshold learning
        """
        from ..core.state import State
        
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Analyze manifold
        features = self.analyzer.analyze(data)
        self.manifold_features = features
        
        # Partition into regions
        if self.method == 'density':
            self._partition_by_density(data)
        elif self.method == 'cluster':
            self._partition_by_clustering(data)
        elif self.method == 'adaptive':
            self._partition_adaptive(data, labels)
        
        self.is_fitted = True
        return self
    
    def _partition_by_density(self, data: np.ndarray):
        """Partition manifold by density quantiles"""
        densities = self.manifold_features.local_density
        
        # Quantile-based regions
        quantiles = np.linspace(0, 100, self.n_regions + 1)
        density_thresholds = np.percentile(densities, quantiles)
        
        # Learn threshold for each region
        for i in range(self.n_regions):
            # Points in this density region
            mask = (densities >= density_thresholds[i]) & \
                   (densities < density_thresholds[i+1])
            
            if np.sum(mask) > 10:
                region_data = data[mask]
                
                # Compute thresholds for this region
                threshold_i = np.median(region_data)
                threshold_q = np.percentile(region_data, 75)
            else:
                # Fallback
                threshold_i = np.median(data)
                threshold_q = np.percentile(data, 75)
            
            self.region_thresholds[i] = {
                'density_min': density_thresholds[i],
                'density_max': density_thresholds[i+1],
                'threshold_i': threshold_i,
                'threshold_q': threshold_q
            }
    
    def _partition_by_clustering(self, data: np.ndarray):
        """Partition manifold by clustering"""
        from sklearn.cluster import KMeans
        
        # Cluster in manifold space
        kmeans = KMeans(n_clusters=self.n_regions, random_state=42)
        labels = kmeans.fit_predict(data)
        
        # Learn threshold for each cluster
        for i in range(self.n_regions):
            mask = labels == i
            
            if np.sum(mask) > 10:
                region_data = data[mask]
                
                threshold_i = np.median(region_data)
                threshold_q = np.percentile(region_data, 75)
            else:
                threshold_i = np.median(data)
                threshold_q = np.percentile(data, 75)
            
            self.region_thresholds[i] = {
                'cluster_center': kmeans.cluster_centers_[i],
                'threshold_i': threshold_i,
                'threshold_q': threshold_q
            }
    
    def _partition_adaptive(
        self, 
        data: np.ndarray, 
        labels: Optional[np.ndarray]
    ):
        """Adaptive partitioning using both density and labels"""
        # Combine density and supervised information
        densities = self.manifold_features.local_density
        
        if labels is not None:
            # Weight by label correlation
            from scipy.stats import spearmanr
            
            for i in range(self.n_regions):
                # Density quantile
                q_low = i / self.n_regions * 100
                q_high = (i + 1) / self.n_regions * 100
                
                density_low = np.percentile(densities, q_low)
                density_high = np.percentile(densities, q_high)
                
                mask = (densities >= density_low) & (densities < density_high)
                
                if np.sum(mask) > 10:
                    region_data = data[mask]
                    region_labels = labels[mask]
                    
                    # Find threshold that best separates labels
                    candidates = np.percentile(region_data, [25, 50, 75])
                    best_thresh = candidates[1]  # Default to median
                    best_corr = 0
                    
                    for thresh in candidates:
                        bits = (region_data > thresh).astype(int)
                        corr, _ = spearmanr(bits, region_labels)
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_thresh = thresh
                    
                    threshold_i = best_thresh
                    threshold_q = np.percentile(region_data, 75)
                else:
                    threshold_i = np.median(data)
                    threshold_q = np.percentile(data, 75)
                
                self.region_thresholds[i] = {
                    'density_min': density_low,
                    'density_max': density_high,
                    'threshold_i': threshold_i,
                    'threshold_q': threshold_q
                }
        else:
            # Fallback to density partitioning
            self._partition_by_density(data)
    
    def encode(self, data: np.ndarray) -> 'StateArray':
        """
        Encode data using manifold-aware thresholds
        
        Parameters
        ----------
        data : np.ndarray
            Data to encode
            
        Returns
        -------
        states : StateArray
            Encoded states
        """
        from ..core.state import StateArray
        
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Compute local densities for new data
        # (Use training manifold as reference)
        # For simplicity, use global thresholds
        # TODO: Implement proper local density computation for test data
        
        # For now, use median region
        region_id = self.n_regions // 2
        thresholds = self.region_thresholds[region_id]
        
        # Encode
        i_bits = (data.flatten() > thresholds['threshold_i']).astype(np.int8)
        q_bits = (data.flatten() > thresholds['threshold_q']).astype(np.int8)
        
        return StateArray(i=i_bits, q=q_bits)


def estimate_manifold_dimension(
    data: np.ndarray,
    method: str = 'all'
) -> float:
    """
    Convenience function for dimension estimation
    
    Parameters
    ----------
    data : np.ndarray
        Data to analyze
    method : str
        Estimation method
        
    Returns
    -------
    dimension : float
        Estimated intrinsic dimension
        
    Examples
    --------
    >>> data = np.random.randn(1000, 50)  # 50D ambient
    >>> dim = estimate_manifold_dimension(data)
    >>> print(f"Intrinsic dimension: {dim:.1f}")
    """
    analyzer = ManifoldAnalyzer(method=method)
    features = analyzer.analyze(data)
    return features.intrinsic_dimension


def detect_manifold_structure(
    data: np.ndarray
) -> Tuple[bool, float]:
    """
    Detect if data lies on low-dimensional manifold
    
    Parameters
    ----------
    data : np.ndarray
        Data to analyze
        
    Returns
    -------
    is_manifold : bool
        Whether data lies on manifold
    confidence : float
        Confidence in detection
        
    Examples
    --------
    >>> data = make_swiss_roll()
    >>> is_manifold, confidence = detect_manifold_structure(data)
    >>> print(f"Manifold: {is_manifold} (confidence: {confidence:.2f})")
    """
    analyzer = ManifoldAnalyzer()
    features = analyzer.analyze(data)
    
    return features.is_manifold, features.confidence