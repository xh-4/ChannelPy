"""
Betti number computation and topological feature extraction

Betti numbers are topological invariants that count holes of different dimensions:
- β₀: Number of connected components
- β₁: Number of 1-dimensional holes (loops/cycles)
- β₂: Number of 2-dimensional voids (cavities)

For channel algebra, these numbers reveal structural features in data distributions
that inform encoding strategies.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import warnings


@dataclass
class BettiNumbers:
    """
    Container for Betti numbers at different homological dimensions
    
    Attributes
    ----------
    beta_0 : int
        Number of connected components
    beta_1 : int
        Number of 1-dimensional holes (loops)
    beta_2 : int
        Number of 2-dimensional voids (cavities)
    dimensions : int
        Maximum homological dimension computed
    filtration_values : List[float]
        Filtration values used in computation
    """
    beta_0: int
    beta_1: int = 0
    beta_2: int = 0
    dimensions: int = 1
    filtration_values: Optional[List[float]] = None
    
    def __repr__(self) -> str:
        return f"BettiNumbers(β₀={self.beta_0}, β₁={self.beta_1}, β₂={self.beta_2})"
    
    def __str__(self) -> str:
        return f"β₀={self.beta_0}, β₁={self.beta_1}, β₂={self.beta_2}"
    
    def euler_characteristic(self) -> int:
        """
        Compute Euler characteristic
        
        χ = β₀ - β₁ + β₂
        
        Returns
        -------
        chi : int
            Euler characteristic
        """
        return self.beta_0 - self.beta_1 + self.beta_2
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return {
            'beta_0': self.beta_0,
            'beta_1': self.beta_1,
            'beta_2': self.beta_2,
            'euler_characteristic': self.euler_characteristic()
        }


class BettiComputer:
    """
    Compute Betti numbers from point cloud data
    
    Uses persistent homology via ripser library for rigorous computation,
    with fallback heuristics if ripser not available.
    
    Examples
    --------
    >>> computer = BettiComputer()
    >>> data = np.random.randn(100, 2)
    >>> betti = computer.compute(data)
    >>> print(f"Connected components: {betti.beta_0}")
    >>> print(f"Loops: {betti.beta_1}")
    """
    
    def __init__(
        self,
        max_dimension: int = 2,
        distance_threshold: Optional[float] = None,
        method: str = 'ripser'
    ):
        """
        Parameters
        ----------
        max_dimension : int
            Maximum homological dimension to compute
        distance_threshold : float, optional
            Maximum distance for filtration. If None, use data-driven value
        method : str
            Computation method: 'ripser' (rigorous) or 'heuristic' (fallback)
        """
        self.max_dimension = max_dimension
        self.distance_threshold = distance_threshold
        self.method = method
        
        # Check if ripser available
        self._has_ripser = self._check_ripser()
        if method == 'ripser' and not self._has_ripser:
            warnings.warn("ripser not available, falling back to heuristic method")
            self.method = 'heuristic'
    
    def _check_ripser(self) -> bool:
        """Check if ripser is available"""
        try:
            import ripser
            return True
        except ImportError:
            return False
    
    def compute(
        self, 
        data: np.ndarray,
        threshold: Optional[float] = None
    ) -> BettiNumbers:
        """
        Compute Betti numbers from point cloud
        
        Parameters
        ----------
        data : np.ndarray
            Point cloud data. Shape: (n_points, n_features) or (n_points,) for 1D
        threshold : float, optional
            Distance threshold for filtration
            
        Returns
        -------
        betti : BettiNumbers
            Computed Betti numbers
        """
        # Ensure 2D array
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Determine threshold
        if threshold is None:
            threshold = self.distance_threshold
        if threshold is None:
            threshold = self._auto_threshold(data)
        
        # Compute using selected method
        if self.method == 'ripser' and self._has_ripser:
            return self._compute_ripser(data, threshold)
        else:
            return self._compute_heuristic(data, threshold)
    
    def _auto_threshold(self, data: np.ndarray) -> float:
        """
        Automatically determine distance threshold
        
        Uses median pairwise distance as heuristic
        """
        # Sample if data is large
        if len(data) > 1000:
            indices = np.random.choice(len(data), 1000, replace=False)
            sample = data[indices]
        else:
            sample = data
        
        # Compute pairwise distances (sample)
        from scipy.spatial.distance import pdist
        distances = pdist(sample)
        
        # Use median distance
        threshold = np.median(distances)
        
        return threshold
    
    def _compute_ripser(
        self, 
        data: np.ndarray, 
        threshold: float
    ) -> BettiNumbers:
        """
        Compute Betti numbers using ripser (rigorous)
        
        Uses Vietoris-Rips complex and persistent homology
        """
        try:
            from ripser import ripser as ripser_compute
        except ImportError:
            raise ImportError("ripser required for rigorous Betti computation")
        
        # Compute persistent homology
        result = ripser_compute(
            data,
            maxdim=self.max_dimension,
            thresh=threshold
        )
        
        # Extract persistence diagrams
        diagrams = result['dgms']
        
        # Count features at the given threshold
        beta_0 = self._count_features_at_threshold(diagrams[0], threshold)
        beta_1 = 0
        beta_2 = 0
        
        if len(diagrams) > 1:
            beta_1 = self._count_features_at_threshold(diagrams[1], threshold)
        
        if len(diagrams) > 2:
            beta_2 = self._count_features_at_threshold(diagrams[2], threshold)
        
        return BettiNumbers(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=beta_2,
            dimensions=self.max_dimension,
            filtration_values=[threshold]
        )
    
    def _count_features_at_threshold(
        self, 
        diagram: np.ndarray, 
        threshold: float
    ) -> int:
        """
        Count features alive at given threshold
        
        A feature is alive if birth <= threshold < death
        """
        count = 0
        for birth, death in diagram:
            if birth <= threshold < death:
                count += 1
            # Special case: infinite death (persists forever)
            if np.isinf(death) and birth <= threshold:
                count += 1
        
        return count
    
    def _compute_heuristic(
        self, 
        data: np.ndarray, 
        threshold: float
    ) -> BettiNumbers:
        """
        Compute Betti numbers using heuristic methods
        
        Less rigorous but doesn't require ripser
        """
        # β₀: Connected components via clustering
        beta_0 = self._heuristic_components(data, threshold)
        
        # β₁: Loops via density topology (simplified)
        beta_1 = self._heuristic_loops(data, threshold)
        
        return BettiNumbers(
            beta_0=beta_0,
            beta_1=beta_1,
            beta_2=0,  # β₂ requires more sophisticated computation
            dimensions=1,
            filtration_values=[threshold]
        )
    
    def _heuristic_components(
        self, 
        data: np.ndarray, 
        threshold: float
    ) -> int:
        """
        Estimate connected components using single-linkage clustering
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        
        if len(data) < 2:
            return len(data)
        
        # Compute linkage
        try:
            distances = pdist(data)
            Z = linkage(distances, method='single')
            
            # Cut at threshold
            labels = fcluster(Z, threshold, criterion='distance')
            
            # Number of clusters = β₀
            return len(np.unique(labels))
        except:
            # Fallback: assume all connected
            return 1
    
    def _heuristic_loops(
        self, 
        data: np.ndarray, 
        threshold: float
    ) -> int:
        """
        Heuristic loop detection
        
        Looks for cycles in neighborhood graph
        This is a simplified approximation
        """
        if data.shape[1] != 2:
            # Only meaningful for 2D embeddings
            return 0
        
        # Build k-nearest neighbor graph
        from scipy.spatial import cKDTree
        
        tree = cKDTree(data)
        
        # Find neighbors within threshold
        edges = set()
        for i, point in enumerate(data):
            neighbors = tree.query_ball_point(point, threshold)
            for j in neighbors:
                if i != j:
                    edge = tuple(sorted([i, j]))
                    edges.add(edge)
        
        # Simple cycle detection (Euler characteristic for graph)
        # For planar graph: β₁ ≈ |E| - |V| + 1
        V = len(data)
        E = len(edges)
        
        # Estimate loops
        loops = max(0, E - V + 1)
        
        return loops


class PersistentBettiTracker:
    """
    Track Betti numbers over multiple filtration values
    
    Useful for understanding how topology changes with scale
    
    Examples
    --------
    >>> tracker = PersistentBettiTracker()
    >>> data = np.random.randn(100, 2)
    >>> persistence = tracker.track(data, num_thresholds=10)
    >>> tracker.plot_betti_curves()
    """
    
    def __init__(self, max_dimension: int = 2):
        """
        Parameters
        ----------
        max_dimension : int
            Maximum homological dimension
        """
        self.max_dimension = max_dimension
        self.computer = BettiComputer(max_dimension=max_dimension)
        self.history: List[Tuple[float, BettiNumbers]] = []
    
    def track(
        self,
        data: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        num_thresholds: int = 20
    ) -> List[Tuple[float, BettiNumbers]]:
        """
        Compute Betti numbers across multiple thresholds
        
        Parameters
        ----------
        data : np.ndarray
            Point cloud data
        thresholds : np.ndarray, optional
            Specific thresholds to use
        num_thresholds : int
            Number of thresholds to sample (if thresholds not provided)
            
        Returns
        -------
        persistence : List[Tuple[float, BettiNumbers]]
            List of (threshold, betti_numbers) tuples
        """
        # Determine thresholds
        if thresholds is None:
            # Auto-generate thresholds
            max_threshold = self.computer._auto_threshold(data) * 2
            thresholds = np.linspace(0, max_threshold, num_thresholds)
        
        # Compute Betti numbers at each threshold
        self.history = []
        for threshold in thresholds:
            betti = self.computer.compute(data, threshold=threshold)
            self.history.append((threshold, betti))
        
        return self.history
    
    def get_betti_curves(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Extract Betti number curves from history
        
        Returns
        -------
        curves : Dict[str, Tuple[np.ndarray, np.ndarray]]
            Dictionary with 'beta_0', 'beta_1', 'beta_2' keys
            Each value is (thresholds, betti_values)
        """
        if not self.history:
            return {}
        
        thresholds = np.array([h[0] for h in self.history])
        beta_0 = np.array([h[1].beta_0 for h in self.history])
        beta_1 = np.array([h[1].beta_1 for h in self.history])
        beta_2 = np.array([h[1].beta_2 for h in self.history])
        
        return {
            'beta_0': (thresholds, beta_0),
            'beta_1': (thresholds, beta_1),
            'beta_2': (thresholds, beta_2)
        }
    
    def plot_betti_curves(self, title: str = "Betti Number Evolution"):
        """
        Plot Betti numbers vs filtration threshold
        
        Shows how topology changes with scale
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        curves = self.get_betti_curves()
        
        if not curves:
            raise ValueError("No history to plot. Run track() first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each Betti number
        for name, (thresholds, values) in curves.items():
            label = name.replace('_', '₀' if name.endswith('0') else 
                                       '₁' if name.endswith('1') else '₂')
            ax.plot(thresholds, values, marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Filtration Threshold (Distance)')
        ax.set_ylabel('Betti Number')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


def compute_betti_for_states(
    states: 'StateArray',
    embed_method: str = 'coordinate'
) -> BettiNumbers:
    """
    Compute Betti numbers for sequence of channel states
    
    Embeds states in space and computes topological features
    
    Parameters
    ----------
    states : StateArray
        Sequence of channel states
    embed_method : str
        Embedding method:
        - 'coordinate': Use (i, q) coordinates
        - 'complex': Use complex representation i + iq
        - 'integer': Use integer encoding 0-3
        
    Returns
    -------
    betti : BettiNumbers
        Topological features of state sequence
        
    Examples
    --------
    >>> from channelpy.core import StateArray
    >>> states = StateArray.from_bits(i=[1,0,1,1,0], q=[1,1,0,1,0])
    >>> betti = compute_betti_for_states(states)
    >>> print(f"State sequence has {betti.beta_0} components")
    """
    # Embed states in space
    if embed_method == 'coordinate':
        # Use (i, q) coordinates
        data = np.column_stack([states.i, states.q])
    
    elif embed_method == 'complex':
        # Use complex plane: i + iq
        data = states.i.astype(float).reshape(-1, 1)
        data = data + 1j * states.q.astype(float).reshape(-1, 1)
        # Convert to 2D real representation
        data = np.column_stack([data.real, data.imag])
    
    elif embed_method == 'integer':
        # Use integer encoding
        data = states.to_ints().reshape(-1, 1)
    
    else:
        raise ValueError(f"Unknown embed_method: {embed_method}")
    
    # Compute Betti numbers
    computer = BettiComputer(max_dimension=1)  # 1D sufficient for state sequences
    betti = computer.compute(data)
    
    return betti


def betti_signature(data: np.ndarray) -> str:
    """
    Compute topological signature string for data
    
    Useful for comparing topological structures
    
    Parameters
    ----------
    data : np.ndarray
        Point cloud data
        
    Returns
    -------
    signature : str
        String like "β₀=2-β₁=1-β₂=0"
        
    Examples
    --------
    >>> data = make_two_clusters()
    >>> sig = betti_signature(data)
    >>> print(sig)  # "β₀=2-β₁=0-β₂=0"
    """
    computer = BettiComputer()
    betti = computer.compute(data)
    
    return f"β₀={betti.beta_0}-β₁={betti.beta_1}-β₂={betti.beta_2}"


# ============================================================================
# Utility Functions
# ============================================================================

def detect_clusters_via_betti(
    data: np.ndarray,
    threshold_range: Optional[Tuple[float, float]] = None,
    num_thresholds: int = 20
) -> int:
    """
    Detect number of clusters using persistent β₀
    
    The number of connected components (β₀) at an appropriate scale
    reveals the cluster structure
    
    Parameters
    ----------
    data : np.ndarray
        Data to analyze
    threshold_range : Tuple[float, float], optional
        (min, max) threshold range
    num_thresholds : int
        Number of thresholds to sample
        
    Returns
    -------
    num_clusters : int
        Estimated number of clusters
        
    Examples
    --------
    >>> data = make_three_clusters()
    >>> num_clusters = detect_clusters_via_betti(data)
    >>> print(f"Detected {num_clusters} clusters")
    """
    tracker = PersistentBettiTracker()
    
    # Determine threshold range
    if threshold_range is None:
        max_thresh = tracker.computer._auto_threshold(data) * 2
        thresholds = np.linspace(0, max_thresh, num_thresholds)
    else:
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    
    # Track β₀
    tracker.track(data, thresholds=thresholds)
    curves = tracker.get_betti_curves()
    
    _, beta_0_values = curves['beta_0']
    
    # Most stable β₀ value (mode)
    from scipy.stats import mode
    result = mode(beta_0_values, keepdims=True)
    num_clusters = int(result.mode[0])
    
    return num_clusters


def detect_loops_via_betti(
    data: np.ndarray,
    threshold: Optional[float] = None
) -> int:
    """
    Detect number of loops/cycles using β₁
    
    Parameters
    ----------
    data : np.ndarray
        Data to analyze (should be 2D for meaningful loop detection)
    threshold : float, optional
        Distance threshold
        
    Returns
    -------
    num_loops : int
        Number of detected loops
        
    Examples
    --------
    >>> data = make_circle()
    >>> num_loops = detect_loops_via_betti(data)
    >>> print(f"Detected {num_loops} loop(s)")
    """
    computer = BettiComputer(max_dimension=1)
    betti = computer.compute(data, threshold=threshold)
    
    return betti.beta_1


def topological_complexity(data: np.ndarray) -> float:
    """
    Measure topological complexity of data
    
    Combines all Betti numbers into single complexity score
    Higher complexity = more intricate topology
    
    Parameters
    ----------
    data : np.ndarray
        Data to analyze
        
    Returns
    -------
    complexity : float
        Topological complexity score
        
    Examples
    --------
    >>> simple_data = np.random.randn(100, 2)
    >>> complex_data = make_swiss_roll()
    >>> print(f"Simple: {topological_complexity(simple_data):.2f}")
    >>> print(f"Complex: {topological_complexity(complex_data):.2f}")
    """
    computer = BettiComputer()
    betti = computer.compute(data)
    
    # Weighted sum of Betti numbers
    # β₀ contributes least (just connectivity)
    # β₁ contributes more (loops add complexity)
    # β₂ contributes most (voids are most complex)
    
    complexity = (
        1.0 * betti.beta_0 +
        2.0 * betti.beta_1 +
        3.0 * betti.beta_2
    )
    
    return complexity