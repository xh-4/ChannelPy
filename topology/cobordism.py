"""
State Transition Topology - The Geometry of Change

A cobordism is a transition path between two channel states. This module
provides tools for analyzing, optimizing, and visualizing state transitions.

Key Concepts:
- Different paths between states have different properties (cost, risk, stability)
- Some paths are "safer" (avoid ∅, minimize time in δ)
- Optimal paths depend on context and cost function
- Transition topology reveals system dynamics
"""

from typing import List, Tuple, Dict, Optional, Callable, Set
import numpy as np
from dataclasses import dataclass, field
from itertools import product
from collections import deque

from ..core.state import State, EMPTY, DELTA, PHI, PSI
from ..core.operations import gate, admit, overlay, weave, comp


@dataclass
class TransitionEdge:
    """
    A single state transition edge
    
    Attributes
    ----------
    from_state : State
        Source state
    to_state : State
        Target state
    operation : str
        Operation name (e.g., 'gate', 'admit')
    cost : float
        Transition cost
    risk : float
        Transition risk (0-1)
    """
    from_state: State
    to_state: State
    operation: str
    cost: float = 1.0
    risk: float = 0.0
    
    def __repr__(self):
        return f"{self.from_state} --[{self.operation}]--> {self.to_state} (cost={self.cost:.2f}, risk={self.risk:.2f})"


@dataclass
class TransitionPath:
    """
    A sequence of state transitions
    
    Attributes
    ----------
    edges : List[TransitionEdge]
        Sequence of transition edges
    total_cost : float
        Sum of edge costs
    total_risk : float
        Combined risk measure
    length : int
        Number of transitions
    """
    edges: List[TransitionEdge] = field(default_factory=list)
    
    @property
    def total_cost(self) -> float:
        """Sum of all edge costs"""
        return sum(edge.cost for edge in self.edges)
    
    @property
    def total_risk(self) -> float:
        """Maximum risk along path (worst-case)"""
        if not self.edges:
            return 0.0
        return max(edge.risk for edge in self.edges)
    
    @property
    def average_risk(self) -> float:
        """Average risk along path"""
        if not self.edges:
            return 0.0
        return np.mean([edge.risk for edge in self.edges])
    
    @property
    def length(self) -> int:
        """Number of transitions"""
        return len(self.edges)
    
    @property
    def states(self) -> List[State]:
        """All states visited (including start and end)"""
        if not self.edges:
            return []
        states = [self.edges[0].from_state]
        states.extend(edge.to_state for edge in self.edges)
        return states
    
    def passes_through(self, state: State) -> bool:
        """Check if path passes through a specific state"""
        return state in self.states
    
    def avoids(self, state: State) -> bool:
        """Check if path avoids a specific state"""
        return not self.passes_through(state)
    
    def __repr__(self):
        if not self.edges:
            return "EmptyPath()"
        
        path_str = " -> ".join(str(s) for s in self.states)
        return f"Path({path_str}) [cost={self.total_cost:.2f}, risk={self.total_risk:.2f}]"


class StateTransitionGraph:
    """
    Graph of all possible state transitions
    
    Nodes: Channel states (∅, δ, φ, ψ)
    Edges: Operations that transform states
    
    Examples
    --------
    >>> graph = StateTransitionGraph()
    >>> graph.build_default_graph()
    >>> paths = graph.find_all_paths(DELTA, PSI)
    >>> optimal = graph.find_optimal_path(DELTA, PSI, cost_function='shortest')
    """
    
    def __init__(self):
        self.edges: List[TransitionEdge] = []
        self.adjacency: Dict[State, List[TransitionEdge]] = {}
    
    def add_edge(self, edge: TransitionEdge):
        """Add a transition edge to the graph"""
        self.edges.append(edge)
        
        if edge.from_state not in self.adjacency:
            self.adjacency[edge.from_state] = []
        self.adjacency[edge.from_state].append(edge)
    
    def add_transition(
        self,
        from_state: State,
        to_state: State,
        operation: str,
        cost: float = 1.0,
        risk: float = 0.0
    ):
        """Add a transition (convenience method)"""
        edge = TransitionEdge(
            from_state=from_state,
            to_state=to_state,
            operation=operation,
            cost=cost,
            risk=risk
        )
        self.add_edge(edge)
    
    def build_default_graph(self):
        """
        Build default transition graph with standard operations
        
        Standard transitions:
        - gate: ∅→∅, δ→∅, φ→φ, ψ→ψ
        - admit: ∅→∅, δ→ψ, φ→φ, ψ→ψ
        - comp: ∅↔ψ, δ↔φ
        """
        # Gate transitions
        self.add_transition(EMPTY, EMPTY, 'gate', cost=1.0, risk=0.0)
        self.add_transition(DELTA, EMPTY, 'gate', cost=1.0, risk=0.3)  # Loses information
        self.add_transition(PHI, PHI, 'gate', cost=1.0, risk=0.0)
        self.add_transition(PSI, PSI, 'gate', cost=1.0, risk=0.0)
        
        # Admit transitions
        self.add_transition(EMPTY, EMPTY, 'admit', cost=1.0, risk=0.0)
        self.add_transition(DELTA, PSI, 'admit', cost=1.0, risk=0.1)  # Validation
        self.add_transition(PHI, PHI, 'admit', cost=1.0, risk=0.0)
        self.add_transition(PSI, PSI, 'admit', cost=1.0, risk=0.0)
        
        # Complement transitions (both directions)
        self.add_transition(EMPTY, PSI, 'comp', cost=1.0, risk=0.5)  # Total flip
        self.add_transition(PSI, EMPTY, 'comp', cost=1.0, risk=0.5)
        self.add_transition(DELTA, PHI, 'comp', cost=1.0, risk=0.3)
        self.add_transition(PHI, DELTA, 'comp', cost=1.0, risk=0.3)
        
        # Direct bit flips (more granular)
        # Flip i-bit only
        self.add_transition(EMPTY, DELTA, 'set_i', cost=0.5, risk=0.2)
        self.add_transition(DELTA, EMPTY, 'clear_i', cost=0.5, risk=0.3)
        self.add_transition(PHI, PSI, 'set_i', cost=0.5, risk=0.1)
        self.add_transition(PSI, PHI, 'clear_i', cost=0.5, risk=0.4)
        
        # Flip q-bit only
        self.add_transition(EMPTY, PHI, 'set_q', cost=0.5, risk=0.3)
        self.add_transition(PHI, EMPTY, 'clear_q', cost=0.5, risk=0.4)
        self.add_transition(DELTA, PSI, 'set_q', cost=0.5, risk=0.1)
        self.add_transition(PSI, DELTA, 'clear_q', cost=0.5, risk=0.3)
    
    def get_neighbors(self, state: State) -> List[TransitionEdge]:
        """Get all outgoing edges from a state"""
        return self.adjacency.get(state, [])
    
    def find_all_paths(
        self,
        source: State,
        target: State,
        max_length: int = 10
    ) -> List[TransitionPath]:
        """
        Find all paths from source to target
        
        Parameters
        ----------
        source : State
            Starting state
        target : State
            Goal state
        max_length : int
            Maximum path length to consider
            
        Returns
        -------
        paths : List[TransitionPath]
            All paths from source to target
        """
        if source == target:
            return [TransitionPath(edges=[])]
        
        all_paths = []
        
        # BFS with path tracking
        queue = deque([(source, [])])  # (current_state, path_so_far)
        
        while queue:
            current_state, path_edges = queue.popleft()
            
            # Check path length
            if len(path_edges) >= max_length:
                continue
            
            # Explore neighbors
            for edge in self.get_neighbors(current_state):
                new_path = path_edges + [edge]
                
                # Check for target
                if edge.to_state == target:
                    all_paths.append(TransitionPath(edges=new_path))
                else:
                    # Check for cycles (don't revisit states in this path)
                    visited_states = {e.from_state for e in new_path}
                    if edge.to_state not in visited_states:
                        queue.append((edge.to_state, new_path))
        
        return all_paths
    
    def find_optimal_path(
        self,
        source: State,
        target: State,
        cost_function: str = 'total_cost',
        constraints: Optional[Dict] = None
    ) -> Optional[TransitionPath]:
        """
        Find optimal path using specified cost function
        
        Parameters
        ----------
        source : State
            Starting state
        target : State
            Goal state
        cost_function : str
            Cost function to minimize:
            - 'total_cost': Minimize sum of edge costs
            - 'total_risk': Minimize maximum risk
            - 'length': Minimize number of steps
            - 'average_risk': Minimize average risk
        constraints : Dict, optional
            Constraints on paths:
            - 'avoid_states': List of states to avoid
            - 'max_length': Maximum path length
            - 'max_risk': Maximum acceptable risk
            
        Returns
        -------
        path : TransitionPath or None
            Optimal path, or None if no valid path exists
        """
        if constraints is None:
            constraints = {}
        
        max_length = constraints.get('max_length', 10)
        avoid_states = set(constraints.get('avoid_states', []))
        max_risk = constraints.get('max_risk', float('inf'))
        
        # Find all paths
        all_paths = self.find_all_paths(source, target, max_length)
        
        # Filter by constraints
        valid_paths = []
        for path in all_paths:
            # Check avoidance constraint
            if any(state in avoid_states for state in path.states[1:-1]):  # Exclude source/target
                continue
            
            # Check risk constraint
            if path.total_risk > max_risk:
                continue
            
            valid_paths.append(path)
        
        if not valid_paths:
            return None
        
        # Select optimal path
        if cost_function == 'total_cost':
            return min(valid_paths, key=lambda p: p.total_cost)
        elif cost_function == 'total_risk':
            return min(valid_paths, key=lambda p: p.total_risk)
        elif cost_function == 'length':
            return min(valid_paths, key=lambda p: p.length)
        elif cost_function == 'average_risk':
            return min(valid_paths, key=lambda p: p.average_risk)
        else:
            raise ValueError(f"Unknown cost function: {cost_function}")
    
    def find_safest_path(
        self,
        source: State,
        target: State,
        max_length: int = 10
    ) -> Optional[TransitionPath]:
        """
        Find path that minimizes risk
        
        Convenience method for find_optimal_path with cost_function='total_risk'
        """
        return self.find_optimal_path(
            source, 
            target, 
            cost_function='total_risk',
            constraints={'max_length': max_length}
        )
    
    def find_shortest_path(
        self,
        source: State,
        target: State
    ) -> Optional[TransitionPath]:
        """
        Find shortest path (minimum number of steps)
        
        Convenience method for find_optimal_path with cost_function='length'
        """
        return self.find_optimal_path(
            source, 
            target, 
            cost_function='length'
        )
    
    def analyze_reachability(self) -> Dict[Tuple[State, State], bool]:
        """
        Analyze which states can reach which other states
        
        Returns
        -------
        reachability : Dict[Tuple[State, State], bool]
            reachability[(s1, s2)] = True if s1 can reach s2
        """
        states = [EMPTY, DELTA, PHI, PSI]
        reachability = {}
        
        for source in states:
            for target in states:
                paths = self.find_all_paths(source, target, max_length=5)
                reachability[(source, target)] = len(paths) > 0
        
        return reachability
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get transition probability matrix
        
        Returns
        -------
        matrix : np.ndarray
            4x4 matrix where matrix[i,j] = probability of transitioning from state i to j
        """
        states = [EMPTY, DELTA, PHI, PSI]
        matrix = np.zeros((4, 4))
        
        for i, from_state in enumerate(states):
            neighbors = self.get_neighbors(from_state)
            
            if neighbors:
                # Equal probability for all neighbors (could be weighted)
                prob = 1.0 / len(neighbors)
                
                for edge in neighbors:
                    j = states.index(edge.to_state)
                    matrix[i, j] += prob
        
        return matrix
    
    def visualize(self, highlight_path: Optional[TransitionPath] = None):
        """
        Visualize the state transition graph
        
        Parameters
        ----------
        highlight_path : TransitionPath, optional
            Path to highlight in the visualization
            
        Requires
        --------
        matplotlib, networkx
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            raise ImportError("Visualization requires matplotlib and networkx")
        
        # Build networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        states = [EMPTY, DELTA, PHI, PSI]
        state_labels = {s: str(s) for s in states}
        G.add_nodes_from(states)
        
        # Add edges
        edge_labels = {}
        for edge in self.edges:
            G.add_edge(edge.from_state, edge.to_state)
            edge_labels[(edge.from_state, edge.to_state)] = f"{edge.operation}\nc={edge.cost:.1f},r={edge.risk:.1f}"
        
        # Layout
        pos = {
            EMPTY: (0, 0),
            DELTA: (1, 0),
            PHI: (0, 1),
            PSI: (1, 1)
        }
        
        # Draw
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw all edges (gray)
        nx.draw_networkx_edges(
            G, pos, 
            edge_color='gray', 
            width=1, 
            alpha=0.3,
            arrows=True,
            arrowsize=20,
            ax=ax
        )
        
        # Highlight path if provided
        if highlight_path and highlight_path.edges:
            path_edges = [
                (edge.from_state, edge.to_state) 
                for edge in highlight_path.edges
            ]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=path_edges,
                edge_color='red',
                width=3,
                arrows=True,
                arrowsize=20,
                ax=ax
            )
        
        # Draw nodes
        node_colors = ['lightgray', 'lightyellow', 'lightblue', 'lightgreen']
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            labels=state_labels,
            font_size=20,
            font_weight='bold',
            ax=ax
        )
        
        # Draw edge labels (simplified)
        # Only show operation names for clarity
        simple_edge_labels = {
            k: v.split('\n')[0] for k, v in edge_labels.items()
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=simple_edge_labels,
            font_size=8,
            ax=ax
        )
        
        ax.set_title('State Transition Graph', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig


class Cobordism:
    """
    A cobordism: a "space" connecting two channel states
    
    Represents the transition structure between states, including
    all possible paths and their properties.
    
    Examples
    --------
    >>> # Create cobordism between δ and ψ
    >>> cob = Cobordism(DELTA, PSI)
    >>> cob.build_transition_graph()
    >>> 
    >>> # Find optimal path
    >>> optimal = cob.find_optimal_transition(cost_function='total_risk')
    >>> print(f"Optimal path: {optimal}")
    >>> 
    >>> # Analyze all paths
    >>> analysis = cob.analyze_transition_space()
    >>> print(f"Number of paths: {analysis['num_paths']}")
    >>> print(f"Safest path risk: {analysis['min_risk']:.3f}")
    """
    
    def __init__(self, source: State, target: State):
        """
        Initialize cobordism
        
        Parameters
        ----------
        source : State
            Starting state
        target : State
            Ending state
        """
        self.source = source
        self.target = target
        self.graph = StateTransitionGraph()
        self.all_paths: Optional[List[TransitionPath]] = None
    
    def build_transition_graph(self):
        """Build the transition graph"""
        self.graph.build_default_graph()
    
    def compute_all_paths(self, max_length: int = 10):
        """
        Compute all possible paths
        
        Parameters
        ----------
        max_length : int
            Maximum path length to consider
        """
        self.all_paths = self.graph.find_all_paths(
            self.source, 
            self.target, 
            max_length
        )
    
    def find_optimal_transition(
        self,
        cost_function: str = 'total_cost',
        constraints: Optional[Dict] = None
    ) -> Optional[TransitionPath]:
        """
        Find optimal transition path
        
        Parameters
        ----------
        cost_function : str
            Cost function to optimize
        constraints : Dict, optional
            Constraints on valid paths
            
        Returns
        -------
        path : TransitionPath or None
            Optimal path
        """
        return self.graph.find_optimal_path(
            self.source,
            self.target,
            cost_function,
            constraints
        )
    
    def analyze_transition_space(self) -> Dict:
        """
        Comprehensive analysis of the transition space
        
        Returns
        -------
        analysis : Dict
            Statistics about all possible transitions
        """
        if self.all_paths is None:
            self.compute_all_paths()
        
        if not self.all_paths:
            return {
                'num_paths': 0,
                'reachable': False
            }
        
        costs = [p.total_cost for p in self.all_paths]
        risks = [p.total_risk for p in self.all_paths]
        lengths = [p.length for p in self.all_paths]
        
        # Find paths with specific properties
        safest = min(self.all_paths, key=lambda p: p.total_risk)
        shortest = min(self.all_paths, key=lambda p: p.length)
        cheapest = min(self.all_paths, key=lambda p: p.total_cost)
        
        # Paths that avoid dangerous states
        safe_paths = [p for p in self.all_paths if p.avoids(EMPTY)]
        
        return {
            'num_paths': len(self.all_paths),
            'reachable': True,
            'cost_stats': {
                'min': min(costs),
                'max': max(costs),
                'mean': np.mean(costs),
                'std': np.std(costs)
            },
            'risk_stats': {
                'min': min(risks),
                'max': max(risks),
                'mean': np.mean(risks),
                'std': np.std(risks)
            },
            'length_stats': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'std': np.std(lengths)
            },
            'optimal_paths': {
                'safest': safest,
                'shortest': shortest,
                'cheapest': cheapest
            },
            'num_safe_paths': len(safe_paths)  # Paths avoiding ∅
        }
    
    def visualize(self):
        """Visualize the cobordism with optimal path highlighted"""
        optimal = self.find_optimal_transition(cost_function='total_risk')
        return self.graph.visualize(highlight_path=optimal)
    
    def __repr__(self):
        return f"Cobordism({self.source} → {self.target})"


# ============================================================================
# Utility Functions
# ============================================================================

def compare_transition_strategies(
    source: State,
    target: State,
    strategies: List[str] = None
) -> Dict:
    """
    Compare different transition strategies
    
    Parameters
    ----------
    source : State
        Starting state
    target : State
        Goal state
    strategies : List[str], optional
        Strategies to compare. Default: all strategies
        
    Returns
    -------
    comparison : Dict
        Comparison of strategies
        
    Examples
    --------
    >>> results = compare_transition_strategies(DELTA, PSI)
    >>> print(results['best_for_safety'])
    >>> print(results['best_for_speed'])
    """
    if strategies is None:
        strategies = ['total_cost', 'total_risk', 'length', 'average_risk']
    
    cob = Cobordism(source, target)
    cob.build_transition_graph()
    
    results = {}
    
    for strategy in strategies:
        path = cob.find_optimal_transition(cost_function=strategy)
        if path:
            results[strategy] = {
                'path': path,
                'cost': path.total_cost,
                'risk': path.total_risk,
                'length': path.length
            }
    
    # Determine best for different objectives
    if results:
        results['best_for_safety'] = min(
            results.items(), 
            key=lambda x: x[1]['risk']
        )[0]
        results['best_for_speed'] = min(
            results.items(),
            key=lambda x: x[1]['length']
        )[0]
        results['best_for_cost'] = min(
            results.items(),
            key=lambda x: x[1]['cost']
        )[0]
    
    return results


def find_transition_avoiding_states(
    source: State,
    target: State,
    avoid: List[State]
) -> Optional[TransitionPath]:
    """
    Find transition that avoids specific states
    
    Parameters
    ----------
    source : State
        Starting state
    target : State
        Goal state
    avoid : List[State]
        States to avoid
        
    Returns
    -------
    path : TransitionPath or None
        Path avoiding specified states, or None if impossible
        
    Examples
    --------
    >>> # Find path from ∅ to ψ that avoids δ
    >>> path = find_transition_avoiding_states(EMPTY, PSI, avoid=[DELTA])
    """
    cob = Cobordism(source, target)
    cob.build_transition_graph()
    
    return cob.find_optimal_transition(
        cost_function='length',
        constraints={'avoid_states': avoid}
    )