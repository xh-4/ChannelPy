"""
Persistent Homology for Topological Data Analysis

Persistent homology tracks topological features (connected components, holes, voids)
across multiple scales, revealing intrinsic structure in data.

Key concepts:
- **Birth**: Scale at which a feature appears
- **Death**: Scale at which a feature disappears  
- **Persistence**: Lifespan of a feature (death - birth)
- **Betti numbers**: Count of features at each dimension

This module provides:
- Persistence diagram computation
- Betti number extraction
- Persistence statistics (entropy, distances)
- Visualization tools
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import warnings


@dataclass
class PersistenceDiagram:
    """
    Persistence diagram for a single homology dimension
    
    Attributes
    ----------
    dimension : int
        Homology dimension (0=components, 1=holes, 2=voids)
    births : np.ndarray
        Birth times of features
    deaths : np.ndarray
        Death times of features
    persistence : np.ndarray
        Persistence values (death - birth)
    """
    dimension: int
    births: np.ndarray
    deaths: np.ndarray
    
    def __post_init__(self):
        self.births = np.asarray(self.births)
        self.deaths = np.asarray(self.deaths)
        
        # Replace infinite deaths with large value for computation
        finite_deaths = self.deaths.copy()
        finite_deaths[np.isinf(finite_deaths)] = 2 * np.max(self.births[np.isfinite(self.births)])
        
        self.persistence = finite_deaths - self.births
    
    @property
    def n_features(self) -> int:
        """Number of features in diagram"""
        return len(self.births)
    
    def filter_by_persistence(self, threshold: float) -> 'PersistenceDiagram':
        """
        Filter features by persistence threshold
        
        Parameters
        ----------
        threshold : float
            Minimum persistence to keep
            
        Returns
        -------
        filtered : PersistenceDiagram
            Filtered diagram
        """
        mask = self.persistence >= threshold
        return PersistenceDiagram(
            dimension=self.dimension,
            births=self.births[mask],
            deaths=self.deaths[mask]
        )
    
    def top_k_features(self, k: int) -> 'PersistenceDiagram':
        """
        Get top k most persistent features
        
        Parameters
        ----------
        k : int
            Number of features to keep
            
        Returns
        -------
        top_k : PersistenceDiagram
            Diagram with only top k features
        """
        if k >= len(self.births):
            return self
        
        # Sort by persistence
        idx = np.argsort(self.persistence)[::-1][:k]
        
        return PersistenceDiagram(
            dimension=self.dimension,
            births=self.births[idx],
            deaths=self.deaths[idx]
        )
    
    def __len__(self):
        return self.n_features


def compute_persistence_diagram(
    data: np.ndarray,
    max_dimension: int = 2,
    max_edge_length: Optional[float] = None
) -> Dict[int, PersistenceDiagram]:
    """
    Compute persistence diagrams using Vietoris-Rips complex
    
    Parameters
    ----------
    data : np.ndarray
        Point cloud data (n_samples, n_features)
    max_dimension : int
        Maximum homology dimension to compute
    max_edge_length : float, optional
        Maximum edge length in Rips complex (defaults to diameter)
        
    Returns
    -------
    diagrams : Dict[int, PersistenceDiagram]
        Persistence diagrams for each dimension
        
    Examples
    --------
    >>> data = np.random.randn(100, 2)
    >>> diagrams = compute_persistence_diagram(data)
    >>> print(f"0-dim features: {len(diagrams[0])}")
    >>> print(f"1-dim features: {len(diagrams[1])}")
    """
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError(
            "Requires ripser. Install with: pip install ripser"
        )
    
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Compute persistence
    result = ripser(
        data, 
        maxdim=max_dimension,
        thresh=max_edge_length if max_edge_length else np.inf
    )
    
    # Extract diagrams
    diagrams = {}
    for dim in range(max_dimension + 1):
        dgm = result['dgms'][dim]
        
        if len(dgm) > 0:
            births = dgm[:, 0]
            deaths = dgm[:, 1]
            
            diagrams[dim] = PersistenceDiagram(
                dimension=dim,
                births=births,
                deaths=deaths
            )
        else:
            diagrams[dim] = PersistenceDiagram(
                dimension=dim,
                births=np.array([]),
                deaths=np.array([])
            )
    
    return diagrams


def compute_betti_numbers(
    diagrams: Dict[int, PersistenceDiagram],
    threshold: float
) -> Dict[int, int]:
    """
    Compute Betti numbers at given threshold
    
    Parameters
    ----------
    diagrams : Dict[int, PersistenceDiagram]
        Persistence diagrams
    threshold : float
        Filtration value at which to compute Betti numbers
        
    Returns
    -------
    betti : Dict[int, int]
        Betti numbers for each dimension
        
    Examples
    --------
    >>> diagrams = compute_persistence_diagram(data)
    >>> betti = compute_betti_numbers(diagrams, threshold=0.5)
    >>> print(f"β₀ = {betti[0]}, β₁ = {betti[1]}")
    """
    betti = {}
    
    for dim, diagram in diagrams.items():
        # Count features alive at threshold
        alive = (diagram.births <= threshold) & (diagram.deaths > threshold)
        betti[dim] = np.sum(alive)
    
    return betti


def persistence_entropy(diagram: PersistenceDiagram) -> float:
    """
    Compute entropy of persistence diagram
    
    Higher entropy indicates more complex topology
    
    Parameters
    ----------
    diagram : PersistenceDiagram
        Persistence diagram
        
    Returns
    -------
    entropy : float
        Persistence entropy
        
    References
    ----------
    Rucco, M., et al. (2016). Characterisation of the idiotypic immune 
    network through persistent entropy.
    """
    if len(diagram) == 0:
        return 0.0
    
    # Normalize persistences to probabilities
    persistences = diagram.persistence[np.isfinite(diagram.persistence)]
    
    if len(persistences) == 0:
        return 0.0
    
    total = np.sum(persistences)
    if total == 0:
        return 0.0
    
    probs = persistences / total
    
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return entropy


def bottleneck_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram
) -> float:
    """
    Compute bottleneck distance between persistence diagrams
    
    The bottleneck distance is the infimum over all bijections of the
    maximum distance between matched points.
    
    Parameters
    ----------
    diagram1, diagram2 : PersistenceDiagram
        Persistence diagrams to compare
        
    Returns
    -------
    distance : float
        Bottleneck distance
    """
    try:
        from persim import bottleneck
    except ImportError:
        raise ImportError(
            "Requires persim. Install with: pip install persim"
        )
    
    if diagram1.dimension != diagram2.dimension:
        raise ValueError("Diagrams must have same dimension")
    
    # Prepare diagrams
    dgm1 = np.column_stack([diagram1.births, diagram1.deaths])
    dgm2 = np.column_stack([diagram2.births, diagram2.deaths])
    
    # Compute distance
    distance = bottleneck(dgm1, dgm2)
    
    return distance


def wasserstein_distance(
    diagram1: PersistenceDiagram,
    diagram2: PersistenceDiagram,
    order: int = 1
) -> float:
    """
    Compute Wasserstein distance between persistence diagrams
    
    Parameters
    ----------
    diagram1, diagram2 : PersistenceDiagram
        Persistence diagrams to compare
    order : int
        Order of Wasserstein distance (typically 1 or 2)
        
    Returns
    -------
    distance : float
        Wasserstein distance
    """
    try:
        from persim import wasserstein
    except ImportError:
        raise ImportError(
            "Requires persim. Install with: pip install persim"
        )
    
    if diagram1.dimension != diagram2.dimension:
        raise ValueError("Diagrams must have same dimension")
    
    # Prepare diagrams
    dgm1 = np.column_stack([diagram1.births, diagram1.deaths])
    dgm2 = np.column_stack([diagram2.births, diagram2.deaths])
    
    # Compute distance
    distance = wasserstein(dgm1, dgm2, order=order)
    
    return distance


def plot_persistence_diagram(
    diagrams: Dict[int, PersistenceDiagram],
    title: str = "Persistence Diagram",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot persistence diagrams
    
    Parameters
    ----------
    diagrams : Dict[int, PersistenceDiagram]
        Persistence diagrams to plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Requires matplotlib for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['C0', 'C1', 'C2', 'C3']
    markers = ['o', 's', '^', 'D']
    
    max_death = 0
    
    for dim, diagram in diagrams.items():
        if len(diagram) == 0:
            continue
        
        # Prepare points
        births = diagram.births
        deaths = diagram.deaths
        
        # Handle infinite deaths
        finite_mask = np.isfinite(deaths)
        
        if np.any(finite_mask):
            max_death = max(max_death, np.max(deaths[finite_mask]))
        
        # Plot finite points
        if np.any(finite_mask):
            ax.scatter(
                births[finite_mask],
                deaths[finite_mask],
                c=colors[dim % len(colors)],
                marker=markers[dim % len(markers)],
                s=50,
                alpha=0.6,
                label=f'H_{dim}'
            )
        
        # Plot infinite points
        if np.any(~finite_mask):
            inf_births = births[~finite_mask]
            # Place at top of plot
            inf_deaths = np.full_like(inf_births, max_death * 1.1)
            
            ax.scatter(
                inf_births,
                inf_deaths,
                c=colors[dim % len(colors)],
                marker=markers[dim % len(markers)],
                s=50,
                alpha=0.3,
                edgecolors='black',
                linewidths=2
            )
    
    # Diagonal line
    if max_death > 0:
        ax.plot([0, max_death], [0, max_death], 'k--', alpha=0.5, label='Diagonal')
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def plot_barcode(
    diagrams: Dict[int, PersistenceDiagram],
    title: str = "Persistence Barcode",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Plot persistence barcode
    
    Parameters
    ----------
    diagrams : Dict[int, PersistenceDiagram]
        Persistence diagrams to plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Requires matplotlib for plotting")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['C0', 'C1', 'C2', 'C3']
    y_pos = 0
    
    for dim in sorted(diagrams.keys()):
        diagram = diagrams[dim]
        
        if len(diagram) == 0:
            continue
        
        # Sort by birth time
        idx = np.argsort(diagram.births)
        births = diagram.births[idx]
        deaths = diagram.deaths[idx]
        
        for birth, death in zip(births, deaths):
            if np.isinf(death):
                death = births.max() * 1.2
                linestyle = ':'
            else:
                linestyle = '-'
            
            ax.plot(
                [birth, death],
                [y_pos, y_pos],
                color=colors[dim % len(colors)],
                linestyle=linestyle,
                linewidth=2,
                alpha=0.7
            )
            y_pos += 1
        
        # Add dimension label
        ax.text(
            -0.1, y_pos - len(diagram) / 2,
            f'H_{dim}',
            ha='right',
            va='center',
            fontsize=12,
            fontweight='bold'
        )
        
        y_pos += 5  # Gap between dimensions
    
    ax.set_xlabel('Filtration value')
    ax.set_ylabel('Features')
    ax.set_title(title)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig