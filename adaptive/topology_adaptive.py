"""
Topology-aware adaptive thresholding

This is the KEY INNOVATION of ChannelPy: using topological features of data
distributions to inform threshold adaptation.

Rather than blindly using statistical thresholds, we analyze the shape
(topology) of the data distribution and adapt accordingly:

- Multimodal distributions → threshold between modes
- Clustered distributions → gap-based thresholds  
- Skewed distributions → asymmetric thresholds
- Heavy-tailed distributions → robust percentile thresholds

This makes thresholds robust to distribution shifts while remaining interpretable.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import warnings

from ..core.state import State, StateArray
from .streaming import StreamingAdaptiveThreshold
from .scoring import FeatureScorer


@dataclass
class TopologyFeatures:
    """
    Topological features of a data distribution
    
    Attributes
    ----------
    modality : int
        Number of modes (peaks) in distribution
    skewness : float
        Distribution skewness
    kurtosis : float
        Distribution kurtosis (tail heaviness)
    gaps : List[Tuple[float, float]]
        Significant gaps in the data
    local_maxima : List[float]
        Locations of local maxima (modes)
    density_variance : float
        Variance of local density (clustered vs uniform)
    connected_components : int
        Number of separated clusters
    """
    modality: int = 1
    skewness: float = 0.0
    kurtosis: float = 0.0
    gaps: List[Tuple[float, float]] = None
    local_maxima: List[float] = None
    density_variance: float = 0.0
    connected_components: int = 1
    
    def __post_init__(self):
        if self.gaps is None:
            self.gaps = []
        if self.local_maxima is None:
            self.local_maxima = []


class TopologyAnalyzer:
    """
    Analyze topological features of data distributions
    
    Examples
    --------
    >>> analyzer = TopologyAnalyzer()
    >>> data = np.random.randn(1000)
    >>> features = analyzer.analyze(data)
    >>> print(f"Modality: {features.modality}")
    >>> print(f"Skewness: {features.skewness:.3f}")
    """
    
    def __init__(self, bandwidth: float = 0.1):
        """
        Parameters
        ----------
        bandwidth : float
            Bandwidth for kernel density estimation
        """
        self.bandwidth = bandwidth
    
    def analyze(self, data: np.ndarray) -> TopologyFeatures:
        """
        Compute topological features of data
        
        Parameters
        ----------
        data : np.ndarray
            Data to analyze
            
        Returns
        -------
        features : TopologyFeatures
            Computed topological features
        """
        data = np.asarray(data).flatten()
        
        if len(data) < 10:
            warnings.warn("Too few samples for reliable topology analysis")
            return TopologyFeatures()
        
        features = TopologyFeatures()
        
        # Basic statistics
        try:
            from scipy import stats
            features.skewness = stats.skew(data)
            features.kurtosis = stats.kurtosis(data)
        except ImportError:
            # Fallback if scipy not available
            features.skewness = self._compute_skewness(data)
            features.kurtosis = self._compute_kurtosis(data)
        
        # Modality and local maxima
        features.modality, features.local_maxima = self._detect_modality(data)
        
        # Gaps
        features.gaps = self._find_gaps(data)
        
        # Density variance
        features.density_variance = self._compute_density_variance(data)
        
        # Connected components
        features.connected_components = self._count_components(data)
        
        return features
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness (fallback implementation)"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis (fallback implementation)"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _detect_modality(self, data: np.ndarray) -> Tuple[int, List[float]]:
        """
        Detect number of modes and their locations
        
        Returns
        -------
        num_modes : int
            Number of detected modes
        mode_locations : List[float]
            Locations of modes
        """
        try:
            from scipy.stats import gaussian_kde
            from scipy.signal import find_peaks
            
            # Kernel density estimation
            kde = gaussian_kde(data, bw_method=self.bandwidth)
            
            # Evaluate on grid
            x_grid = np.linspace(data.min(), data.max(), 1000)
            density = kde(x_grid)
            
            # Find peaks
            peaks, properties = find_peaks(
                density, 
                prominence=0.05 * density.max(),
                distance=50
            )
            
            num_modes = len(peaks)
            mode_locations = x_grid[peaks].tolist()
            
            return max(num_modes, 1), mode_locations
            
        except ImportError:
            # Fallback: simple histogram-based detection
            hist, bin_edges = np.histogram(data, bins=50)
            
            # Find local maxima in histogram
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    if hist[i] > 0.05 * hist.max():  # Significance threshold
                        peaks.append(bin_edges[i])
            
            return max(len(peaks), 1), peaks
    
    def _find_gaps(self, data: np.ndarray) -> List[Tuple[float, float]]:
        """
        Find significant gaps in data
        
        Returns
        -------
        gaps : List[Tuple[float, float]]
            List of (gap_start, gap_end) tuples
        """
        # Sort data
        sorted_data = np.sort(data)
        
        # Compute gaps between consecutive points
        gaps_sizes = np.diff(sorted_data)
        
        # Significant gaps: larger than 3 * median gap
        median_gap = np.median(gaps_sizes)
        significant_threshold = 3 * median_gap
        
        gaps = []
        for i, gap_size in enumerate(gaps_sizes):
            if gap_size > significant_threshold:
                gap_start = sorted_data[i]
                gap_end = sorted_data[i + 1]
                gaps.append((gap_start, gap_end))
        
        return gaps
    
    def _compute_density_variance(self, data: np.ndarray) -> float:
        """
        Compute variance of local density
        
        High variance indicates clustered data
        Low variance indicates uniform data
        """
        # Compute local density at each point using k-nearest neighbors
        k = min(10, len(data) // 10)
        
        if k < 2:
            return 0.0
        
        densities = []
        for point in data:
            # Distance to k-th nearest neighbor
            distances = np.abs(data - point)
            distances.sort()
            kth_distance = distances[k]
            
            # Density = k / (2 * kth_distance)
            # Avoid division by zero
            if kth_distance > 0:
                density = k / (2 * kth_distance)
            else:
                density = 0
            
            densities.append(density)
        
        return np.var(densities)
    
    def _count_components(self, data: np.ndarray) -> int:
        """
        Count number of separated clusters
        
        Uses density threshold to identify separated components
        """
        if len(data) < 10:
            return 1
        
        try:
            from scipy.stats import gaussian_kde
            
            kde = gaussian_kde(data, bw_method=self.bandwidth)
            x_grid = np.linspace(data.min(), data.max(), 1000)
            density = kde(x_grid)
            
            # Threshold at 10% of max density
            threshold = 0.1 * density.max()
            
            # Count connected regions above threshold
            above_threshold = density > threshold
            
            # Count transitions from False to True
            components = 0
            in_component = False
            
            for above in above_threshold:
                if above and not in_component:
                    components += 1
                    in_component = True
                elif not above:
                    in_component = False
            
            return max(components, 1)
            
        except ImportError:
            # Fallback: use gap-based heuristic
            return len(self._find_gaps(data)) + 1


class TopologyAdaptiveThreshold:
    """
    Adaptive thresholds that respond to distributional topology
    
    This is the core innovation: thresholds adapt not just to mean/variance,
    but to the actual shape of the data distribution.
    
    Examples
    --------
    >>> threshold = TopologyAdaptiveThreshold(window_size=1000)
    >>> 
    >>> for value in data_stream:
    ...     threshold.update(value)
    ...     state = threshold.encode(value)
    ...     
    ...     # Check if topology changed
    ...     if threshold.topology_changed():
    ...         print(f"Distribution topology changed: {threshold.get_topology()}")
    ...     
    ...     process(state)
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        adaptation_rate: float = 0.01,
        topology_update_interval: int = 100,
        feature_scorer: Optional[FeatureScorer] = None
    ):
        """
        Parameters
        ----------
        window_size : int
            Size of sliding window for analysis
        adaptation_rate : float
            Rate of threshold adaptation (0 = no adaptation, 1 = instant)
        topology_update_interval : int
            How often to recompute topology (computational cost)
        feature_scorer : FeatureScorer, optional
            Scorer for multi-dimensional feature evaluation
        """
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.topology_update_interval = topology_update_interval
        
        # Components
        self.topology_analyzer = TopologyAnalyzer()
        self.streaming_threshold = StreamingAdaptiveThreshold(
            window_size=window_size,
            adaptation_rate=adaptation_rate
        )
        self.feature_scorer = feature_scorer
        
        # State
        self.window = []
        self.update_counter = 0
        self.current_topology = TopologyFeatures()
        self.previous_topology = TopologyFeatures()
        
        # Thresholds
        self.threshold_i = 0.0
        self.threshold_q = 0.0
        
        # History for analysis
        self.topology_history = []
    
    def update(self, value: float):
        """
        Update with new value
        
        Parameters
        ----------
        value : float
            New data value
        """
        # Update streaming threshold (always)
        self.streaming_threshold.update(value)
        
        # Update window
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        
        # Update counter
        self.update_counter += 1
        
        # Periodically recompute topology
        if self.update_counter % self.topology_update_interval == 0:
            self._update_topology()
            self._update_thresholds()
    
    def _update_topology(self):
        """Recompute topological features"""
        if len(self.window) < 50:
            return  # Need enough data
        
        # Save previous topology
        self.previous_topology = self.current_topology
        
        # Compute new topology
        window_array = np.array(self.window)
        self.current_topology = self.topology_analyzer.analyze(window_array)
        
        # Record history
        self.topology_history.append({
            'update_count': self.update_counter,
            'topology': self.current_topology
        })
    
    def _update_thresholds(self):
        """
        Update thresholds based on topology
        
        This is where the magic happens!
        """
        window_array = np.array(self.window)
        topology = self.current_topology
        
        # Choose threshold strategy based on topology
        
        # Strategy 1: Multimodal distribution
        if topology.modality > 1 and len(topology.local_maxima) >= 2:
            threshold_i, threshold_q = self._threshold_for_multimodal(
                window_array, topology
            )
        
        # Strategy 2: Heavy-tailed distribution
        elif topology.kurtosis > 3:
            threshold_i, threshold_q = self._threshold_for_heavy_tailed(
                window_array, topology
            )
        
        # Strategy 3: Skewed distribution
        elif abs(topology.skewness) > 1:
            threshold_i, threshold_q = self._threshold_for_skewed(
                window_array, topology
            )
        
        # Strategy 4: Clustered distribution (high density variance
        elif topology.density_variance > np.var(window_array):
            threshold_i, threshold_q = self._threshold_for_clustered(
                window_array, topology
            )
        
        # Strategy 5: Normal-ish distribution (default)
        else:
            threshold_i, threshold_q = self._threshold_for_normal(
                window_array, topology
            )
        
        # Smooth update
        self.threshold_i = (
            (1 - self.adaptation_rate) * self.threshold_i +
            self.adaptation_rate * threshold_i
        )
        self.threshold_q = (
            (1 - self.adaptation_rate) * self.threshold_q +
            self.adaptation_rate * threshold_q
        )
    
    def _threshold_for_multimodal(
        self, 
        data: np.ndarray, 
        topology: TopologyFeatures
    ) -> Tuple[float, float]:
        """
        Threshold for multimodal distribution
        
        Place threshold between modes (in the valley)
        """
        modes = sorted(topology.local_maxima)
        
        # Threshold i: between first and second mode
        if len(modes) >= 2:
            # Find valley between first two modes
            mode1, mode2 = modes[0], modes[1]
            search_region = data[(data >= mode1) & (data <= mode2)]
            
            if len(search_region) > 0:
                # Use median of region as threshold
                threshold_i = np.median(search_region)
            else:
                threshold_i = (mode1 + mode2) / 2
        else:
            threshold_i = np.median(data)
        
        # Threshold q: at second mode or above
        if len(modes) >= 2:
            threshold_q = modes[1]
        else:
            threshold_q = np.percentile(data, 75)
        
        return threshold_i, threshold_q
    
    def _threshold_for_heavy_tailed(
        self, 
        data: np.ndarray, 
        topology: TopologyFeatures
    ) -> Tuple[float, float]:
        """
        Threshold for heavy-tailed distribution
        
        Use robust percentiles to avoid outliers
        """
        # Use interquartile range instead of mean/std
        q25 = np.percentile(data, 25)
        q50 = np.percentile(data, 50)
        q75 = np.percentile(data, 75)
        iqr = q75 - q25
        
        threshold_i = q50  # Median
        threshold_q = q50 + iqr  # Median + IQR
        
        return threshold_i, threshold_q
    
    def _threshold_for_skewed(
        self, 
        data: np.ndarray, 
        topology: TopologyFeatures
    ) -> Tuple[float, float]:
        """
        Threshold for skewed distribution
        
        Adjust asymmetrically based on skew direction
        """
        median = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        
        if topology.skewness > 0:
            # Right-skewed: use median + larger offset for q
            threshold_i = median
            threshold_q = median + 1.5 * iqr
        else:
            # Left-skewed: use median - larger offset for i
            threshold_i = median - 0.5 * iqr
            threshold_q = median + 0.5 * iqr
        
        return threshold_i, threshold_q
    
    def _threshold_for_clustered(
        self, 
        data: np.ndarray, 
        topology: TopologyFeatures
    ) -> Tuple[float, float]:
        """
        Threshold for clustered distribution
        
        Use gaps to separate clusters
        """
        if topology.gaps:
            # Use largest gap
            largest_gap = max(topology.gaps, key=lambda g: g[1] - g[0])
            threshold_i = (largest_gap[0] + largest_gap[1]) / 2
            
            # Threshold q: above the gap
            threshold_q = largest_gap[1]
        else:
            # Fallback to percentiles
            threshold_i = np.percentile(data, 50)
            threshold_q = np.percentile(data, 75)
        
        return threshold_i, threshold_q
    
    def _threshold_for_normal(
        self, 
        data: np.ndarray, 
        topology: TopologyFeatures
    ) -> Tuple[float, float]:
        """
        Threshold for normal-ish distribution
        
        Use standard statistical approach
        """
        mean = np.mean(data)
        std = np.std(data)
        
        threshold_i = mean
        threshold_q = mean + 0.5 * std
        
        return threshold_i, threshold_q
    
    def encode(self, value: float) -> State:
        """
        Encode value using topology-aware thresholds
        
        Parameters
        ----------
        value : float
            Value to encode
            
        Returns
        -------
        state : State
            Encoded channel state
        """
        return State(
            i=int(value > self.threshold_i),
            q=int(value > self.threshold_q)
        )
    
    def topology_changed(self, sensitivity: float = 0.5) -> bool:
        """
        Check if topology has changed significantly
        
        Parameters
        ----------
        sensitivity : float
            Sensitivity to change (0 = insensitive, 1 = very sensitive)
            
        Returns
        -------
        changed : bool
            True if significant topology change detected
        """
        if self.previous_topology is None:
            return False
        
        curr = self.current_topology
        prev = self.previous_topology
        
        # Check modality change
        if curr.modality != prev.modality:
            return True
        
        # Check skewness change
        skew_change = abs(curr.skewness - prev.skewness)
        if skew_change > sensitivity:
            return True
        
        # Check kurtosis change
        kurt_change = abs(curr.kurtosis - prev.kurtosis)
        if kurt_change > sensitivity * 3:
            return True
        
        return False
    
    def get_topology(self) -> TopologyFeatures:
        """Get current topological features"""
        return self.current_topology
    
    def get_thresholds(self) -> Dict[str, Any]:
        """
        Get current thresholds with explanation
        
        Returns
        -------
        info : Dict
            Dictionary with thresholds and topology info
        """
        return {
            'threshold_i': self.threshold_i,
            'threshold_q': self.threshold_q,
            'topology': {
                'modality': self.current_topology.modality,
                'skewness': self.current_topology.skewness,
                'kurtosis': self.current_topology.kurtosis,
                'num_gaps': len(self.current_topology.gaps),
                'density_variance': self.current_topology.density_variance
            },
            'n_samples': len(self.window),
            'adaptation_strategy': self._get_current_strategy()
        }
    
    def _get_current_strategy(self) -> str:
        """Get name of current adaptation strategy"""
        topology = self.current_topology
        
        if topology.modality > 1:
            return "multimodal"
        elif topology.kurtosis > 3:
            return "heavy_tailed"
        elif abs(topology.skewness) > 1:
            return "skewed"
        elif topology.density_variance > np.var(self.window):
            return "clustered"
        else:
            return "normal"
    
    def plot_topology_and_thresholds(self):
        """
        Plot current data distribution with thresholds
        
        Requires matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde
        except ImportError:
            raise ImportError("Requires matplotlib and scipy for plotting")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        data = np.array(self.window)
        
        # Plot 1: Histogram with thresholds
        ax1.hist(data, bins=50, density=True, alpha=0.5, label='Data')
        
        # KDE
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min(), data.max(), 1000)
        density = kde(x_grid)
        ax1.plot(x_grid, density, 'k-', linewidth=2, label='Density')
        
        # Thresholds
        ax1.axvline(self.threshold_i, color='orange', linestyle='--', 
                    linewidth=2, label=f'Threshold i = {self.threshold_i:.3f}')
        ax1.axvline(self.threshold_q, color='red', linestyle='--', 
                    linewidth=2, label=f'Threshold q = {self.threshold_q:.3f}')
        
        # Modes
        for mode in self.current_topology.local_maxima:
            ax1.axvline(mode, color='green', linestyle=':', alpha=0.5)
        
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution (Strategy: {self._get_current_strategy()})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Topology features over time
        if len(self.topology_history) > 1:
            updates = [h['update_count'] for h in self.topology_history]
            modalities = [h['topology'].modality for h in self.topology_history]
            skewnesses = [h['topology'].skewness for h in self.topology_history]
            
            ax2_twin = ax2.twinx()
            
            ax2.plot(updates, modalities, 'b-', marker='o', label='Modality')
            ax2.set_ylabel('Modality', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            ax2_twin.plot(updates, skewnesses, 'r-', marker='s', label='Skewness')
            ax2_twin.set_ylabel('Skewness', color='r')
            ax2_twin.tick_params(axis='y', labelcolor='r')
            
            ax2.set_xlabel('Update Count')
            ax2.set_title('Topology Evolution')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig




