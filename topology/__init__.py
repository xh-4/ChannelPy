"""
Topological Data Analysis for Channel Algebra

This module provides tools for analyzing the topological structure of
data distributions and state spaces:

- **Persistence**: Persistent homology for detecting features at multiple scales
- **Betti Numbers**: Counting topological features (components, holes, voids)
- **Cobordism**: State transition topology and path analysis
- **Manifold**: Detecting and analyzing manifold structure in data
- **Mapper**: Visualization of high-dimensional data topology

These tools enable:
- Topology-aware threshold adaptation
- Understanding state space structure  
- Detecting regime changes via topological features
- Visualizing complex decision boundaries

Examples
--------
>>> from channelpy.topology import compute_persistence, compute_betti_numbers
>>> from channelpy.topology import Cobordism, MapperGraph
>>> 
>>> # Analyze data topology
>>> persistence = compute_persistence(data)
>>> betti = compute_betti_numbers(persistence)
>>> print(f"Components: {betti[0]}, Holes: {betti[1]}")
>>> 
>>> # Visualize with Mapper
>>> mapper = MapperGraph(data)
>>> mapper.build(cover_resolution=10, overlap=0.3)
>>> mapper.plot()
>>> 
>>> # Analyze state transitions
>>> cobordism = Cobordism(source_state, target_state)
>>> path = cobordism.optimal_path()
"""

from .persistence import (
    compute_persistence_diagram,
    compute_betti_numbers,
    persistence_entropy,
    bottleneck_distance,
    wasserstein_distance,
    PersistenceDiagram,
    plot_persistence_diagram,
    plot_barcode
)

from .betti import (
    BettiNumbers,
    compute_betti_from_data,
    compute_betti_sequence,
    euler_characteristic
)

from .cobordism import (
    Cobordism,
    StateTransitionGraph,
    enumerate_paths,
    optimal_path,
    path_risk,
    transition_cost
)

from .manifold import (
    ManifoldAnalyzer,
    estimate_intrinsic_dimension,
    compute_local_density,
    detect_manifold_structure,
    tangent_space_pca
)

from .mapper import (
    MapperGraph,
    CoverScheme,
    build_mapper_graph,
    plot_mapper_graph
)

__all__ = [
    # Persistence
    'compute_persistence_diagram',
    'compute_betti_numbers',
    'persistence_entropy',
    'bottleneck_distance',
    'wasserstein_distance',
    'PersistenceDiagram',
    'plot_persistence_diagram',
    'plot_barcode',
    
    # Betti
    'BettiNumbers',
    'compute_betti_from_data',
    'compute_betti_sequence',
    'euler_characteristic',
    
    # Cobordism
    'Cobordism',
    'StateTransitionGraph',
    'enumerate_paths',
    'optimal_path',
    'path_risk',
    'transition_cost',
    
    # Manifold
    'ManifoldAnalyzer',
    'estimate_intrinsic_dimension',
    'compute_local_density',
    'detect_manifold_structure',
    'tangent_space_pca',
    
    # Mapper
    'MapperGraph',
    'CoverScheme',
    'build_mapper_graph',
    'plot_mapper_graph',
]

# Version info
__version__ = '0.1.0'