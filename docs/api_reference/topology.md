# Topology API Reference

The topology module provides tools for analyzing topological features of data distributions and state transition spaces. This enables topology-aware threshold adaptation and deeper understanding of data structure.

## Overview
```python
from channelpy.topology import (
    # Persistence analysis
    compute_persistence_diagram,
    compute_betti_numbers,
    persistence_entropy,
    
    # Topology analysis
    TopologyAnalyzer,
    TopologyFeatures,
    
    # State transition topology
    Cobordism,
    StateTransitionGraph,
    
    # Visualization
    plot_persistence_diagram,
    plot_state_transition_graph
)
```

---

## Core Classes

### `TopologyAnalyzer`

Analyzes topological features of data distributions.
```python
class TopologyAnalyzer:
    """
    Analyze topological features of data distributions
    
    Computes:
    - Modality (number of peaks)
    - Skewness and kurtosis
    - Density variance
    - Gaps and clusters
    - Connected components
    
    Examples
    --------
    >>> analyzer = TopologyAnalyzer(bandwidth=0.1)
    >>> features = analyzer.analyze(data)
    >>> print(f"Distribution has {features.modality} modes")
    >>> print(f"Skewness: {features.skewness:.3f}")
    """
```

#### Constructor
```python
def __init__(self, bandwidth: float = 0.1)
```

**Parameters:**
- `bandwidth` (float): Bandwidth for kernel density estimation. Controls smoothness of density estimate. Smaller values detect finer structure. Default: 0.1

#### Methods

##### `analyze(data)`

Compute all topological features of data.
```python
def analyze(self, data: np.ndarray) -> TopologyFeatures
```

**Parameters:**
- `data` (np.ndarray): 1D array of data values

**Returns:**
- `TopologyFeatures`: Object containing all computed features

**Example:**
```python
import numpy as np
from channelpy.topology import TopologyAnalyzer

# Generate bimodal data
data = np.concatenate([
    np.random.normal(0, 1, 500),
    np.random.normal(5, 1, 500)
])

analyzer = TopologyAnalyzer()
features = analyzer.analyze(data)

print(f"Modality: {features.modality}")  # Should be 2
print(f"Gaps: {features.gaps}")  # Should show gap around 2.5
```

---

### `TopologyFeatures`

Data class containing topological features of a distribution.
```python
@dataclass
class TopologyFeatures:
    """
    Topological features of a data distribution
    
    Attributes
    ----------
    modality : int
        Number of modes (peaks) in distribution
    skewness : float
        Distribution skewness (asymmetry)
    kurtosis : float
        Distribution kurtosis (tail heaviness)
    gaps : List[Tuple[float, float]]
        Significant gaps: [(start, end), ...]
    local_maxima : List[float]
        Locations of density peaks
    density_variance : float
        Variance of local density
    connected_components : int
        Number of separated clusters
    """
```

**Attributes:**

- `modality` (int): Number of modes detected
  - 1 = unimodal
  - 2+ = multimodal
  - Higher values indicate more complex structure

- `skewness` (float): Measure of asymmetry
  - 0 = symmetric
  - \> 0 = right-skewed (long right tail)
  - < 0 = left-skewed (long left tail)

- `kurtosis` (float): Measure of tail heaviness
  - 0 = normal distribution
  - \> 0 = heavy-tailed
  - < 0 = light-tailed

- `gaps` (List[Tuple[float, float]]): Significant gaps between data clusters
  - Each tuple is (gap_start, gap_end)
  - Useful for cluster separation

- `local_maxima` (List[float]): X-coordinates of density peaks
  - One per mode
  - Sorted by location

- `density_variance` (float): How much density varies across space
  - High value → clustered data
  - Low value → uniform data

- `connected_components` (int): Number of separated clusters
  - Determined by density thresholding
  - >= modality

**Example:**
```python
features = analyzer.analyze(data)

# Check topology type
if features.modality > 1:
    print("Multimodal distribution detected")
    print(f"Modes at: {features.local_maxima}")
    
if features.gaps:
    print(f"Found {len(features.gaps)} significant gaps")
    largest_gap = max(features.gaps, key=lambda g: g[1] - g[0])
    print(f"Largest gap: {largest_gap}")

if abs(features.skewness) > 1:
    direction = "right" if features.skewness > 0 else "left"
    print(f"Distribution is {direction}-skewed")
```

---

## Persistent Homology

Functions for computing topological persistence.

### `compute_persistence_diagram()`

Compute persistence diagram for 1D data.
```python
def compute_persistence_diagram(
    data: np.ndarray,
    maxdim: int = 1
) -> Dict[str, np.ndarray]
```

**Parameters:**
- `data` (np.ndarray): 1D array of data points
- `maxdim` (int): Maximum homology dimension to compute (0 or 1). Default: 1

**Returns:**
- `dict`: Dictionary with keys:
  - `'dgm0'`: H0 persistence diagram (connected components)
  - `'dgm1'`: H1 persistence diagram (holes/loops)
  
Each diagram is an (n, 2) array where each row is [birth, death].

**Example:**
```python
from channelpy.topology import compute_persistence_diagram

# Compute persistence
result = compute_persistence_diagram(data, maxdim=1)

# Analyze H0 (components)
dgm0 = result['dgm0']
print(f"Found {len(dgm0)} connected components")

# Analyze H1 (holes)
dgm1 = result['dgm1']
persistent_holes = dgm1[dgm1[:, 1] - dgm1[:, 0] > 0.1]  # Filter by persistence
print(f"Found {len(persistent_holes)} persistent holes")
```

---

### `compute_betti_numbers()`

Compute Betti numbers (topological invariants).
```python
def compute_betti_numbers(
    data: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[int, int]
```

**Parameters:**
- `data` (np.ndarray): Data points
- `threshold` (float, optional): Persistence threshold for filtering noise. If None, uses 10% of max persistence.

**Returns:**
- `dict`: Betti numbers by dimension:
  - `{0: b0, 1: b1}` where:
    - b0 = number of connected components
    - b1 = number of holes/loops

**Example:**
```python
from channelpy.topology import compute_betti_numbers

betti = compute_betti_numbers(data, threshold=0.1)

print(f"b0 (components): {betti[0]}")
print(f"b1 (holes): {betti[1]}")

# Interpret results
if betti[0] > 1:
    print("Data has multiple disconnected clusters")
if betti[1] > 0:
    print("Data has topological holes (voids)")
```

---

### `persistence_entropy()`

Compute entropy of persistence diagram.
```python
def persistence_entropy(dgm: np.ndarray) -> float
```

**Parameters:**
- `dgm` (np.ndarray): Persistence diagram (n, 2) array of [birth, death] pairs

**Returns:**
- `float`: Persistence entropy (higher = more complex topology)

**Example:**
```python
from channelpy.topology import compute_persistence_diagram, persistence_entropy

result = compute_persistence_diagram(data)
dgm1 = result['dgm1']

entropy = persistence_entropy(dgm1)
print(f"Topological complexity: {entropy:.3f}")

# Interpretation
if entropy > 2.0:
    print("Complex topology with many features")
elif entropy < 0.5:
    print("Simple topology")
```

---

## State Transition Topology

Classes for analyzing transitions between channel states.

### `Cobordism`

Represents a transition path between two states.
```python
class Cobordism:
    """
    A cobordism is a transition between two states
    
    In topology, a cobordism is a manifold whose boundary consists of
    two parts. Here, it represents a path through state space.
    
    Different paths have different properties:
    - Cost: How expensive is this transition?
    - Risk: Does it pass through unstable states (δ, ∅)?
    - Length: How many steps?
    
    Examples
    --------
    >>> from channelpy.core import EMPTY, PSI
    >>> from channelpy.topology import Cobordism
    >>> 
    >>> # Find path from EMPTY to PSI
    >>> cob = Cobordism(source=EMPTY, target=PSI)
    >>> paths = cob.enumerate_paths()
    >>> 
    >>> # Find lowest-cost path
    >>> best_path = cob.optimal_path(cost_function=step_cost)
    >>> print(f"Optimal path: {' → '.join(str(s) for s in best_path)}")
    """
```

#### Constructor
```python
def __init__(self, source: State, target: State)
```

**Parameters:**
- `source` (State): Starting state
- `target` (State): Ending state

#### Methods

##### `enumerate_paths()`

Find all possible transition paths.
```python
def enumerate_paths(
    self,
    max_length: int = 4
) -> List[List[State]]
```

**Parameters:**
- `max_length` (int): Maximum path length. Default: 4

**Returns:**
- `List[List[State]]`: List of paths, where each path is a list of states

**Example:**
```python
from channelpy.core import EMPTY, PSI, DELTA, PHI
from channelpy.topology import Cobordism

cob = Cobordism(source=EMPTY, target=PSI)
paths = cob.enumerate_paths(max_length=3)

print(f"Found {len(paths)} possible paths:")
for i, path in enumerate(paths):
    print(f"  Path {i+1}: {' → '.join(str(s) for s in path)}")

# Example output:
# Path 1: ∅ → δ → ψ
# Path 2: ∅ → φ → ψ
# Path 3: ∅ → ψ
```

##### `optimal_path()`

Find lowest-cost transition path.
```python
def optimal_path(
    self,
    cost_function: Callable[[State, State], float]
) -> List[State]
```

**Parameters:**
- `cost_function` (Callable): Function that takes (from_state, to_state) and returns cost

**Returns:**
- `List[State]`: Optimal path (lowest total cost)

**Example:**
```python
def transition_cost(from_state, to_state):
    """
    Define transition costs
    """
    # Cost of changing bits
    i_change = abs(from_state.i - to_state.i)
    q_change = abs(from_state.q - to_state.q)
    
    # Penalty for passing through EMPTY
    if to_state == EMPTY:
        return 10.0
    
    return i_change + q_change

cob = Cobordism(source=DELTA, target=PHI)
best_path = cob.optimal_path(cost_function=transition_cost)

print(f"Best path: {' → '.join(str(s) for s in best_path)}")
print(f"Total cost: {cob.path_cost(best_path, transition_cost):.2f}")
```

##### `path_risk()`

Evaluate risk of a transition path.
```python
def path_risk(
    self,
    path: List[State],
    risk_weights: Optional[Dict[State, float]] = None
) -> float
```

**Parameters:**
- `path` (List[State]): Path to evaluate
- `risk_weights` (Dict[State, float], optional): Risk weight for each state type

**Returns:**
- `float`: Total path risk

**Example:**
```python
# Define risk weights
risks = {
    EMPTY: 1.0,   # High risk (absent)
    DELTA: 0.5,   # Medium risk (puncture)
    PHI: 0.3,     # Low risk (hole)
    PSI: 0.0      # No risk (resonant)
}

path = [EMPTY, DELTA, PSI]
risk = cob.path_risk(path, risk_weights=risks)
print(f"Path risk: {risk:.2f}")
```

---

### `StateTransitionGraph`

Graph of all possible state transitions.
```python
class StateTransitionGraph:
    """
    Complete graph of state transitions
    
    Nodes: All 4 possible states
    Edges: Possible transitions (can be weighted)
    
    Useful for:
    - Visualizing state space
    - Finding reachable states
    - Analyzing transition patterns
    - Planning state sequences
    
    Examples
    --------
    >>> graph = StateTransitionGraph()
    >>> 
    >>> # Add transition rules
    >>> graph.add_transition(EMPTY, DELTA, weight=1.0)
    >>> graph.add_transition(DELTA, PSI, weight=0.5)
    >>> 
    >>> # Find path
    >>> path = graph.shortest_path(EMPTY, PSI)
    >>> print(path)
    """
```

#### Constructor
```python
def __init__(self)
```

Creates empty transition graph.

#### Methods

##### `add_transition()`

Add allowed transition.
```python
def add_transition(
    self,
    from_state: State,
    to_state: State,
    weight: float = 1.0,
    bidirectional: bool = False
)
```

**Parameters:**
- `from_state` (State): Source state
- `to_state` (State): Target state
- `weight` (float): Transition weight/cost. Default: 1.0
- `bidirectional` (bool): Add reverse transition too. Default: False

##### `shortest_path()`

Find shortest path between states.
```python
def shortest_path(
    self,
    source: State,
    target: State
) -> Optional[List[State]]
```

**Parameters:**
- `source` (State): Starting state
- `target` (State): Goal state

**Returns:**
- `List[State]` or `None`: Shortest path, or None if no path exists

**Example:**
```python
from channelpy.topology import StateTransitionGraph
from channelpy.core import EMPTY, DELTA, PHI, PSI

# Build transition graph
graph = StateTransitionGraph()

# Define allowed transitions (with costs)
graph.add_transition(EMPTY, DELTA, weight=1.0)
graph.add_transition(EMPTY, PHI, weight=2.0)
graph.add_transition(DELTA, PSI, weight=0.5)
graph.add_transition(PHI, PSI, weight=0.5)

# Find shortest path
path = graph.shortest_path(EMPTY, PSI)
print(f"Shortest path: {' → '.join(str(s) for s in path)}")
# Output: ∅ → δ → ψ (cost 1.5)
```

##### `reachable_states()`

Find all states reachable from a given state.
```python
def reachable_states(self, source: State) -> Set[State]
```

**Parameters:**
- `source` (State): Starting state

**Returns:**
- `Set[State]`: Set of reachable states

##### `is_connected()`

Check if state space is fully connected.
```python
def is_connected(self) -> bool
```

**Returns:**
- `bool`: True if any state can reach any other state

---

## Visualization Functions

### `plot_persistence_diagram()`

Plot persistence diagram.
```python
def plot_persistence_diagram(
    dgm: np.ndarray,
    title: str = "Persistence Diagram",
    ax: Optional[plt.Axes] = None
) -> plt.Figure
```

**Parameters:**
- `dgm` (np.ndarray): Persistence diagram (n, 2) array
- `title` (str): Plot title
- `ax` (plt.Axes, optional): Matplotlib axes to plot on

**Returns:**
- `plt.Figure`: Matplotlib figure object

**Example:**
```python
from channelpy.topology import (
    compute_persistence_diagram,
    plot_persistence_diagram
)

result = compute_persistence_diagram(data)
fig = plot_persistence_diagram(result['dgm1'], title="H1 Persistence")
plt.show()
```

---

### `plot_state_transition_graph()`

Visualize state transition graph.
```python
def plot_state_transition_graph(
    graph: StateTransitionGraph,
    layout: str = 'circular',
    ax: Optional[plt.Axes] = None
) -> plt.Figure
```

**Parameters:**
- `graph` (StateTransitionGraph): Graph to plot
- `layout` (str): Layout algorithm ('circular', 'spring', 'grid'). Default: 'circular'
- `ax` (plt.Axes, optional): Matplotlib axes

**Returns:**
- `plt.Figure`: Figure object

**Example:**
```python
from channelpy.topology import (
    StateTransitionGraph,
    plot_state_transition_graph
)

graph = StateTransitionGraph()
# ... add transitions ...

fig = plot_state_transition_graph(graph, layout='spring')
plt.show()
```

---

## Complete Example: Topology-Aware Analysis
```python
import numpy as np
from channelpy.topology import (
    TopologyAnalyzer,
    compute_persistence_diagram,
    compute_betti_numbers,
    Cobordism
)
from channelpy.core import EMPTY, PSI

# Generate data with complex topology
data = np.concatenate([
    np.random.normal(0, 0.5, 300),
    np.random.normal(3, 0.5, 300),
    np.random.normal(6, 0.5, 300)
])

# Analyze topology
analyzer = TopologyAnalyzer()
features = analyzer.analyze(data)

print("=== Topological Features ===")
print(f"Modality: {features.modality}")
print(f"Modes at: {features.local_maxima}")
print(f"Gaps: {features.gaps}")
print(f"Skewness: {features.skewness:.3f}")
print(f"Connected components: {features.connected_components}")

# Compute persistence
persistence = compute_persistence_diagram(data)
betti = compute_betti_numbers(data)

print("\n=== Persistent Homology ===")
print(f"Betti numbers: {betti}")
print(f"H0 features: {len(persistence['dgm0'])}")
print(f"H1 features: {len(persistence['dgm1'])}")

# Analyze state transitions
cob = Cobordism(source=EMPTY, target=PSI)
paths = cob.enumerate_paths()

print(f"\n=== State Transitions ===")
print(f"Possible paths from ∅ to ψ: {len(paths)}")
for path in paths[:3]:  # Show first 3
    print(f"  {' → '.join(str(s) for s in path)}")
```

---

## Integration with Adaptive Module

The topology module integrates seamlessly with adaptive thresholding:
```python
from channelpy.adaptive import TopologyAdaptiveThreshold
from channelpy.topology import TopologyAnalyzer

# Create topology-aware threshold
threshold = TopologyAdaptiveThreshold(window_size=1000)

# Process data stream
for value in data_stream:
    threshold.update(value)
    
    # Get topology
    topology = threshold.get_topology()
    
    # Make decisions based on topology
    if topology.modality > 1:
        print("Multimodal distribution - using mode-based thresholds")
    elif abs(topology.skewness) > 1:
        print("Skewed distribution - using asymmetric thresholds")
    
    # Encode
    state = threshold.encode(value)
```

---

## See Also

- [Adaptive API Reference](adaptive.md) - Topology-adaptive thresholds
- [Core API Reference](core.md) - Channel states and operations
- [Tutorial: Topology Features](../tutorials/06_topology_features.md)