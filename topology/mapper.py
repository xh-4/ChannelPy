"""
Mapper Algorithm for Topological Data Analysis

The Mapper algorithm creates a compressed topological representation of 
high-dimensional data by:

1. Projecting data to lower dimensions (filter function)
2. Covering the projection with overlapping intervals
3. Clustering data within each cover element
4. Building a graph showing cluster relationships

This reveals the "shape" of data and is particularly useful for:
- Visualizing high-dimensional state spaces
- Understanding decision boundaries
- Detecting subpopulations and outliers
- Exploring feature relationships

References
----------
Singh, G., Mémoli, F., & Carlsson, G. (2007). 
Topological Methods for the Analysis of High Dimensional Data Sets and 
3D Object Recognition. Eurographics Symposium on Point-Based Graphics.
"""

from typing import Callable, Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import warnings


@dataclass
class CoverScheme:
    """
    Configuration for covering the filter space
    
    Attributes
    ----------
    n_intervals : int
        Number of intervals for each dimension
    overlap : float
        Overlap fraction between adjacent intervals (0-1)
    filter_function : Callable, optional
        Function to project data (default: PCA to 1D)
    """
    n_intervals: int = 10
    overlap: float = 0.3
    filter_function: Optional[Callable] = None
    
    def __post_init__(self):
        if not 0 <= self.overlap < 1:
            raise ValueError("Overlap must be in [0, 1)")
        if self.n_intervals < 2:
            raise ValueError("Need at least 2 intervals")


@dataclass  
class MapperNode:
    """
    Node in the Mapper graph
    
    Attributes
    ----------
    id : int
        Unique node identifier
    members : List[int]
        Indices of data points in this node
    centroid : np.ndarray
        Centroid of member points
    filter_values : np.ndarray
        Filter values of member points
    size : int
        Number of members
    """
    id: int
    members: List[int]
    centroid: np.ndarray
    filter_values: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        return len(self.members)
    
    def __repr__(self):
        return f"MapperNode(id={self.id}, size={self.size})"


@dataclass
class MapperEdge:
    """
    Edge in the Mapper graph
    
    Attributes
    ----------
    source : int
        Source node ID
    target : int
        Target node ID
    weight : float
        Number of shared points
    """
    source: int
    target: int
    weight: float
    
    def __repr__(self):
        return f"MapperEdge({self.source} → {self.target}, weight={self.weight})"


class MapperGraph:
    """
    Mapper graph for topological data visualization
    
    The Mapper algorithm creates a graph that captures the topological
    structure of high-dimensional data.
    
    Examples
    --------
    >>> # Basic usage
    >>> mapper = MapperGraph(data)
    >>> mapper.build(n_intervals=15, overlap=0.4)
    >>> mapper.plot()
    >>> 
    >>> # Custom filter function
    >>> def custom_filter(X):
    ...     return X[:, 0] + X[:, 1]  # Sum of first two features
    >>> 
    >>> mapper = MapperGraph(data)
    >>> mapper.build(filter_function=custom_filter)
    >>> 
    >>> # Color by labels
    >>> mapper.plot(color_by=labels)
    >>> 
    >>> # Analyze structure
    >>> print(f"Nodes: {len(mapper.nodes)}")
    >>> print(f"Edges: {len(mapper.edges)}")
    >>> print(f"Components: {mapper.n_components()}")
    """
    
    def __init__(
        self, 
        data: np.ndarray,
        distance_metric: str = 'euclidean'
    ):
        """
        Initialize Mapper
        
        Parameters
        ----------
        data : np.ndarray
            Data matrix (n_samples, n_features)
        distance_metric : str
            Distance metric for clustering
        """
        self.data = np.asarray(data)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        
        self.distance_metric = distance_metric
        
        # Graph components
        self.nodes: Dict[int, MapperNode] = {}
        self.edges: List[MapperEdge] = []
        self.node_counter = 0
        
        # Build info
        self.filter_values = None
        self.cover_scheme = None
        self.is_built = False
    
    def build(
        self,
        n_intervals: int = 10,
        overlap: float = 0.3,
        filter_function: Optional[Callable] = None,
        clustering_method: str = 'single_linkage',
        min_cluster_size: int = 3
    ):
        """
        Build the Mapper graph
        
        Parameters
        ----------
        n_intervals : int
            Number of intervals in cover
        overlap : float
            Overlap fraction between intervals
        filter_function : Callable, optional
            Custom filter function. If None, uses PCA to 1D
        clustering_method : str
            Clustering method: 'single_linkage', 'dbscan', 'kmeans'
        min_cluster_size : int
            Minimum cluster size to keep
        
        Returns
        -------
        self : MapperGraph
            Returns self for chaining
        """
        # Reset graph
        self.nodes = {}
        self.edges = []
        self.node_counter = 0
        
        # Step 1: Apply filter function
        if filter_function is None:
            filter_function = self._default_filter
        
        self.filter_values = filter_function(self.data)
        if self.filter_values.ndim == 1:
            self.filter_values = self.filter_values.reshape(-1, 1)
        
        # Step 2: Create cover
        self.cover_scheme = CoverScheme(
            n_intervals=n_intervals,
            overlap=overlap,
            filter_function=filter_function
        )
        
        cover_elements = self._create_cover()
        
        # Step 3: Cluster within each cover element
        node_to_cover = {}  # Track which cover element each node came from
        
        for cover_idx, cover_indices in enumerate(cover_elements):
            if len(cover_indices) < min_cluster_size:
                continue
            
            # Get data in this cover element
            cover_data = self.data[cover_indices]
            
            # Cluster
            clusters = self._cluster_data(
                cover_data, 
                method=clustering_method,
                min_size=min_cluster_size
            )
            
            # Create nodes for each cluster
            for cluster_label in np.unique(clusters):
                if cluster_label == -1:  # Noise in DBSCAN
                    continue
                
                # Get members of this cluster
                cluster_mask = clusters == cluster_label
                cluster_indices = cover_indices[cluster_mask]
                
                if len(cluster_indices) < min_cluster_size:
                    continue
                
                # Create node
                node = MapperNode(
                    id=self.node_counter,
                    members=cluster_indices.tolist(),
                    centroid=np.mean(self.data[cluster_indices], axis=0),
                    filter_values=self.filter_values[cluster_indices],
                    metadata={'cover_element': cover_idx}
                )
                
                self.nodes[self.node_counter] = node
                node_to_cover[self.node_counter] = cover_idx
                self.node_counter += 1
        
        # Step 4: Create edges between overlapping nodes
        self._create_edges()
        
        self.is_built = True
        return self
    
    def _default_filter(self, X: np.ndarray) -> np.ndarray:
        """
        Default filter: PCA projection to 1D
        
        Parameters
        ----------
        X : np.ndarray
            Data matrix
            
        Returns
        -------
        projection : np.ndarray
            1D projection
        """
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            return pca.fit_transform(X).flatten()
        except ImportError:
            # Fallback: use first principal component manually
            X_centered = X - np.mean(X, axis=0)
            cov = np.cov(X_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            pc1 = eigenvectors[:, -1]  # Largest eigenvalue
            return X_centered @ pc1
    
    def _create_cover(self) -> List[np.ndarray]:
        """
        Create overlapping cover of filter space
        
        Returns
        -------
        cover_elements : List[np.ndarray]
            List of arrays containing indices for each cover element
        """
        n_dims = self.filter_values.shape[1]
        
        if n_dims > 1:
            warnings.warn(
                f"Filter has {n_dims} dimensions. "
                "Using only first dimension for cover."
            )
        
        # Use first dimension only
        filter_1d = self.filter_values[:, 0]
        
        # Compute interval bounds
        f_min, f_max = filter_1d.min(), filter_1d.max()
        
        n_intervals = self.cover_scheme.n_intervals
        overlap = self.cover_scheme.overlap
        
        # Interval length with overlap
        interval_length = (f_max - f_min) / (n_intervals - overlap * (n_intervals - 1))
        step = interval_length * (1 - overlap)
        
        cover_elements = []
        
        for i in range(n_intervals):
            # Interval bounds
            lower = f_min + i * step
            upper = lower + interval_length
            
            # Find points in interval
            mask = (filter_1d >= lower) & (filter_1d <= upper)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                cover_elements.append(indices)
        
        return cover_elements
    
    def _cluster_data(
        self, 
        data: np.ndarray, 
        method: str = 'single_linkage',
        min_size: int = 3
    ) -> np.ndarray:
        """
        Cluster data within a cover element
        
        Parameters
        ----------
        data : np.ndarray
            Data to cluster
        method : str
            Clustering method
        min_size : int
            Minimum cluster size
            
        Returns
        -------
        labels : np.ndarray
            Cluster labels for each point
        """
        if len(data) < min_size:
            return np.zeros(len(data), dtype=int)
        
        if method == 'single_linkage':
            return self._single_linkage_clustering(data, min_size)
        
        elif method == 'dbscan':
            return self._dbscan_clustering(data, min_size)
        
        elif method == 'kmeans':
            return self._kmeans_clustering(data, min_size)
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _single_linkage_clustering(
        self, 
        data: np.ndarray, 
        min_size: int
    ) -> np.ndarray:
        """Single linkage hierarchical clustering"""
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist
            
            # Compute distances
            distances = pdist(data, metric=self.distance_metric)
            
            # Hierarchical clustering
            Z = linkage(distances, method='single')
            
            # Cut tree to get reasonable number of clusters
            max_clusters = max(2, len(data) // min_size)
            labels = fcluster(Z, max_clusters, criterion='maxclust') - 1
            
            return labels
            
        except ImportError:
            # Fallback: simple connected components
            return self._connected_components_clustering(data, min_size)
    
    def _dbscan_clustering(
        self, 
        data: np.ndarray, 
        min_size: int
    ) -> np.ndarray:
        """DBSCAN clustering"""
        try:
            from sklearn.cluster import DBSCAN
            
            # Estimate eps from data
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min_size).fit(data)
            distances, _ = nbrs.kneighbors(data)
            eps = np.median(distances[:, -1])
            
            clusterer = DBSCAN(
                eps=eps, 
                min_samples=min_size,
                metric=self.distance_metric
            )
            
            labels = clusterer.fit_predict(data)
            return labels
            
        except ImportError:
            return self._connected_components_clustering(data, min_size)
    
    def _kmeans_clustering(
        self, 
        data: np.ndarray, 
        min_size: int
    ) -> np.ndarray:
        """K-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            n_clusters = max(1, len(data) // min_size)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(data)
            
            return labels
            
        except ImportError:
            # Fallback
            return np.zeros(len(data), dtype=int)
    
    def _connected_components_clustering(
        self, 
        data: np.ndarray, 
        min_size: int
    ) -> np.ndarray:
        """
        Fallback clustering: simple connected components
        """
        n = len(data)
        labels = -np.ones(n, dtype=int)
        current_label = 0
        
        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(data, data, metric=self.distance_metric)
        
        # Threshold for connectivity
        threshold = np.percentile(distances, 10)
        
        # Build connected components
        visited = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if visited[i]:
                continue
            
            # BFS to find connected component
            component = []
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                if visited[current]:
                    continue
                
                visited[current] = True
                component.append(current)
                
                # Find neighbors
                neighbors = np.where(distances[current] < threshold)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        queue.append(neighbor)
            
            # Assign label if large enough
            if len(component) >= min_size:
                for idx in component:
                    labels[idx] = current_label
                current_label += 1
        
        return labels
    
    def _create_edges(self):
        """
        Create edges between nodes with shared members
        """
        node_ids = list(self.nodes.keys())
        
        for i, node_id1 in enumerate(node_ids):
            for node_id2 in node_ids[i+1:]:
                node1 = self.nodes[node_id1]
                node2 = self.nodes[node_id2]
                
                # Find shared members
                shared = set(node1.members) & set(node2.members)
                
                if shared:
                    edge = MapperEdge(
                        source=node_id1,
                        target=node_id2,
                        weight=len(shared)
                    )
                    self.edges.append(edge)
    
    def n_components(self) -> int:
        """
        Count connected components in graph
        
        Returns
        -------
        n : int
            Number of connected components
        """
        if not self.is_built:
            raise RuntimeError("Graph not built. Call build() first.")
        
        if not self.nodes:
            return 0
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for edge in self.edges:
            adjacency[edge.source].append(edge.target)
            adjacency[edge.target].append(edge.source)
        
        # Count components via DFS
        visited = set()
        n_components = 0
        
        for node_id in self.nodes:
            if node_id in visited:
                continue
            
            # DFS
            stack = [node_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            n_components += 1
        
        return n_components
    
    def get_node_values(self, feature_idx: int = 0) -> np.ndarray:
        """
        Get feature values for each node (mean of members)
        
        Parameters
        ----------
        feature_idx : int
            Feature index to extract
            
        Returns
        -------
        values : np.ndarray
            Mean feature value for each node
        """
        if not self.is_built:
            raise RuntimeError("Graph not built. Call build() first.")
        
        values = np.zeros(len(self.nodes))
        for node_id, node in self.nodes.items():
            member_values = self.data[node.members, feature_idx]
            values[node_id] = np.mean(member_values)
        
        return values
    
    def plot(
        self,
        color_by: Optional[np.ndarray] = None,
        node_size_scale: float = 50.0,
        layout: str = 'spring',
        figsize: Tuple[int, int] = (12, 8),
        title: str = "Mapper Graph"
    ):
        """
        Plot the Mapper graph
        
        Parameters
        ----------
        color_by : np.ndarray, optional
            Values to color nodes by (e.g., labels, feature values)
            If None, colors by node size
        node_size_scale : float
            Scaling factor for node sizes
        layout : str
            Graph layout: 'spring', 'kamada_kawai', 'circular'
        figsize : Tuple[int, int]
            Figure size
        title : str
            Plot title
        
        Returns
        -------
        fig : matplotlib.Figure
            Figure object
        """
        if not self.is_built:
            raise RuntimeError("Graph not built. Call build() first.")
        
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            raise ImportError("Requires matplotlib and networkx for plotting")
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, size=node.size)
        
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Determine node colors
        if color_by is None:
            # Color by size
            node_colors = [self.nodes[node_id].size for node_id in G.nodes()]
            cmap = 'viridis'
            label = 'Node size'
        else:
            # Color by provided values (mean over members)
            node_colors = []
            for node_id in G.nodes():
                member_values = color_by[self.nodes[node_id].members]
                node_colors.append(np.mean(member_values))
            cmap = 'coolwarm'
            label = 'Mean value'
        
        # Node sizes
        node_sizes = [self.nodes[node_id].size * node_size_scale for node_id in G.nodes()]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            alpha=0.3,
            width=[edge.weight / 5 for edge in self.edges],
            ax=ax
        )
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=cmap,
            alpha=0.8,
            ax=ax
        )
        
        # Colorbar
        plt.colorbar(nodes, label=label, ax=ax)
        
        # Title and formatting
        ax.set_title(f"{title}\n{len(self.nodes)} nodes, {len(self.edges)} edges, {self.n_components()} components")
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the graph
        
        Returns
        -------
        summary : Dict
            Summary statistics
        """
        if not self.is_built:
            raise RuntimeError("Graph not built. Call build() first.")
        
        # Node statistics
        node_sizes = [node.size for node in self.nodes.values()]
        
        # Edge statistics
        edge_weights = [edge.weight for edge in self.edges]
        
        return {
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'n_components': self.n_components(),
            'avg_node_size': np.mean(node_sizes),
            'max_node_size': np.max(node_sizes) if node_sizes else 0,
            'min_node_size': np.min(node_sizes) if node_sizes else 0,
            'avg_edge_weight': np.mean(edge_weights) if edge_weights else 0,
            'cover_intervals': self.cover_scheme.n_intervals,
            'cover_overlap': self.cover_scheme.overlap
        }


# Convenience function
def build_mapper_graph(
    data: np.ndarray,
    n_intervals: int = 10,
    overlap: float = 0.3,
    filter_function: Optional[Callable] = None,
    **kwargs
) -> MapperGraph:
    """
    Build Mapper graph in one call
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix
    n_intervals : int
        Number of cover intervals
    overlap : float
        Overlap fraction
    filter_function : Callable, optional
        Custom filter function
    **kwargs
        Additional arguments for build()
        
    Returns
    -------
    mapper : MapperGraph
        Built Mapper graph
        
    Examples
    --------
    >>> mapper = build_mapper_graph(data, n_intervals=15, overlap=0.4)
    >>> mapper.plot()
    """
    mapper = MapperGraph(data)
    mapper.build(
        n_intervals=n_intervals,
        overlap=overlap,
        filter_function=filter_function,
        **kwargs
    )
    return mapper


def plot_mapper_graph(
    data: np.ndarray,
    n_intervals: int = 10,
    overlap: float = 0.3,
    color_by: Optional[np.ndarray] = None,
    **kwargs
):
    """
    Build and plot Mapper graph in one call
    
    Parameters
    ----------
    data : np.ndarray
        Data matrix
    n_intervals : int
        Number of cover intervals
    overlap : float
        Overlap fraction
    color_by : np.ndarray, optional
        Values to color nodes by
    **kwargs
        Additional arguments for plot()
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object
        
    Examples
    --------
    >>> fig = plot_mapper_graph(data, color_by=labels)
    >>> plt.show()
    """
    mapper = build_mapper_graph(data, n_intervals, overlap)
    return mapper.plot(color_by=color_by, **kwargs)