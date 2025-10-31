"""
Channel-specific combinators for field operations

This module provides functional combinators that operate on channel fields,
enabling compositional spatial reasoning.

Key insight: Instead of just SKI combinators on functions, we create
combinators that work on:
- States (point-wise)
- Fields (spatial)
- Neighborhoods (local context)

This creates a functional algebra for spatial channel computations.
"""

from typing import Callable, Tuple, List, Optional, Any
import numpy as np
from functools import wraps

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..core.operations import gate, admit, overlay, weave, comp
from ..fields.field import ChannelField2D


# ============================================================================
# Basic Point Combinators (operate on States)
# ============================================================================

class PointCombinator:
    """
    Combinator that operates on individual states
    
    Examples
    --------
    >>> # Apply gate to every point in field
    >>> gate_combinator = PointCombinator(gate)
    >>> new_field = gate_combinator.apply_to_field(field)
    """
    
    def __init__(self, operation: Callable[[State], State]):
        """
        Parameters
        ----------
        operation : Callable[[State], State]
            Function that operates on a single state
        """
        self.operation = operation
    
    def __call__(self, state: State) -> State:
        """Apply operation to state"""
        return self.operation(state)
    
    def apply_to_field(self, field: ChannelField2D) -> ChannelField2D:
        """
        Apply operation to every point in field
        
        This is the KEY operation: lift point operation to field operation
        """
        new_i = np.zeros_like(field.i)
        new_q = np.zeros_like(field.q)
        
        for idx in np.ndindex(field.shape):
            state = field.get_state(idx)
            new_state = self.operation(state)
            new_i[idx] = new_state.i
            new_q[idx] = new_state.q
        
        return ChannelField2D(new_i, new_q)
    
    def compose(self, other: 'PointCombinator') -> 'PointCombinator':
        """
        Compose two point combinators
        
        (f ∘ g)(x) = f(g(x))
        """
        def composed(state: State) -> State:
            return self.operation(other.operation(state))
        
        return PointCombinator(composed)


# ============================================================================
# Field Combinators (operate on entire fields)
# ============================================================================

class FieldCombinator:
    """
    Combinator that operates on entire fields
    
    Examples
    --------
    >>> # Detect boundaries in field
    >>> boundary_detector = FieldCombinator(detect_boundaries)
    >>> boundary_field = boundary_detector(input_field)
    """
    
    def __init__(self, operation: Callable[[ChannelField2D], ChannelField2D]):
        """
        Parameters
        ----------
        operation : Callable[[ChannelField2D], ChannelField2D]
            Function that transforms a field
        """
        self.operation = operation
    
    def __call__(self, field: ChannelField2D) -> ChannelField2D:
        """Apply operation to field"""
        return self.operation(field)
    
    def compose(self, other: 'FieldCombinator') -> 'FieldCombinator':
        """Compose two field combinators"""
        def composed(field: ChannelField2D) -> ChannelField2D:
            return self.operation(other.operation(field))
        
        return FieldCombinator(composed)
    
    def parallel(self, other: 'FieldCombinator') -> 'FieldCombinator':
        """
        Apply two combinators in parallel and overlay results
        
        parallel(f, g)(field) = overlay(f(field), g(field))
        """
        def parallel_op(field: ChannelField2D) -> ChannelField2D:
            result1 = self.operation(field)
            result2 = other.operation(field)
            return result1.overlay(result2)
        
        return FieldCombinator(parallel_op)


# ============================================================================
# Neighborhood Combinators (local context operations)
# ============================================================================

class NeighborhoodCombinator:
    """
    Combinator that operates on local neighborhoods
    
    Key for spatial reasoning: decision at point depends on neighbors
    
    Examples
    --------
    >>> # Smooth field using neighborhood voting
    >>> smoother = NeighborhoodCombinator(majority_vote, radius=1)
    >>> smoothed_field = smoother(noisy_field)
    """
    
    def __init__(
        self, 
        operation: Callable[[List[State], State], State],
        radius: int = 1,
        include_center: bool = True
    ):
        """
        Parameters
        ----------
        operation : Callable[[List[State], State], State]
            Function(neighbors, center) → new_state
        radius : int
            Neighborhood radius
        include_center : bool
            Include center state in neighbors list
        """
        self.operation = operation
        self.radius = radius
        self.include_center = include_center
    
    def __call__(self, field: ChannelField2D) -> ChannelField2D:
        """Apply neighborhood operation to field"""
        new_i = np.zeros_like(field.i)
        new_q = np.zeros_like(field.q)
        
        for idx in np.ndindex(field.shape):
            # Get neighborhood
            neighbors = field.get_neighborhood(idx, self.radius)
            center = field.get_state(idx)
            
            if not self.include_center:
                # Remove center from neighbors
                neighbors = [n for n in neighbors if n != center]
            
            # Apply operation
            new_state = self.operation(neighbors, center)
            new_i[idx] = new_state.i
            new_q[idx] = new_state.q
        
        return ChannelField2D(new_i, new_q)


# ============================================================================
# Reduction Combinators (Field → State)
# ============================================================================

class ReductionCombinator:
    """
    Combinator that reduces field to single state
    
    Examples
    --------
    >>> # Check if field contains any PSI states
    >>> any_psi = ReductionCombinator(lambda states: PSI in states)
    >>> has_resonance = any_psi(field)
    """
    
    def __init__(self, operation: Callable[[List[State]], State]):
        """
        Parameters
        ----------
        operation : Callable[[List[State]], State]
            Function that aggregates states to single state
        """
        self.operation = operation
    
    def __call__(self, field: ChannelField2D) -> State:
        """Reduce field to single state"""
        all_states = []
        for idx in np.ndindex(field.shape):
            all_states.append(field.get_state(idx))
        
        return self.operation(all_states)


# ============================================================================
# Binary Field Combinators (Field × Field → Field)
# ============================================================================

class BinaryFieldCombinator:
    """
    Combinator for binary operations on fields
    
    Examples
    --------
    >>> # Overlay two fields
    >>> overlay_combinator = BinaryFieldCombinator(
    ...     lambda s1, s2: overlay(s1, s2)
    ... )
    >>> result = overlay_combinator(field1, field2)
    """
    
    def __init__(self, operation: Callable[[State, State], State]):
        """
        Parameters
        ----------
        operation : Callable[[State, State], State]
            Binary operation on states
        """
        self.operation = operation
    
    def __call__(
        self, 
        field1: ChannelField2D, 
        field2: ChannelField2D
    ) -> ChannelField2D:
        """Apply binary operation point-wise"""
        if field1.shape != field2.shape:
            raise ValueError("Fields must have same shape")
        
        new_i = np.zeros_like(field1.i)
        new_q = np.zeros_like(field1.q)
        
        for idx in np.ndindex(field1.shape):
            state1 = field1.get_state(idx)
            state2 = field2.get_state(idx)
            new_state = self.operation(state1, state2)
            new_i[idx] = new_state.i
            new_q[idx] = new_state.q
        
        return ChannelField2D(new_i, new_q)


# ============================================================================
# Higher-Order Combinators (combinators that create combinators)
# ============================================================================

def lift_point_to_field(point_op: Callable[[State], State]) -> FieldCombinator:
    """
    Lift point operation to field operation
    
    This is the fundamental "map" operation for fields
    
    Examples
    --------
    >>> # Lift gate to field operation
    >>> gate_field = lift_point_to_field(gate)
    >>> gated_field = gate_field(input_field)
    """
    def field_op(field: ChannelField2D) -> ChannelField2D:
        pc = PointCombinator(point_op)
        return pc.apply_to_field(field)
    
    return FieldCombinator(field_op)


def lift_binary_to_field(
    binary_op: Callable[[State, State], State]
) -> BinaryFieldCombinator:
    """
    Lift binary operation to field operation
    
    Examples
    --------
    >>> overlay_fields = lift_binary_to_field(overlay)
    >>> result = overlay_fields(field1, field2)
    """
    return BinaryFieldCombinator(binary_op)


def compose_many(*combinators: FieldCombinator) -> FieldCombinator:
    """
    Compose multiple field combinators
    
    compose_many(f, g, h) = f ∘ g ∘ h
    
    Examples
    --------
    >>> # Pipeline: detect edges → enhance → filter
    >>> pipeline = compose_many(detect_edges, enhance_features, noise_filter)
    >>> result = pipeline(input_field)
    """
    if not combinators:
        return FieldCombinator(lambda f: f)  # Identity
    
    def composed(field: ChannelField2D) -> ChannelField2D:
        result = field
        for combinator in reversed(combinators):
            result = combinator(result)
        return result
    
    return FieldCombinator(composed)


# ============================================================================
# Spatial Operations (using combinators)
# ============================================================================

def spatial_filter(
    kernel: np.ndarray,
    operation: Callable[[List[State], List[float]], State]
) -> FieldCombinator:
    """
    Create spatial filter using kernel weights
    
    Parameters
    ----------
    kernel : np.ndarray
        Filter kernel (weights for neighborhood)
    operation : Callable
        Function(neighbor_states, weights) → state
    
    Examples
    --------
    >>> # Gaussian blur on channel states
    >>> gaussian = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16
    >>> blur = spatial_filter(gaussian, weighted_overlay)
    >>> blurred = blur(field)
    """
    def filter_op(field: ChannelField2D) -> ChannelField2D:
        new_i = np.zeros_like(field.i)
        new_q = np.zeros_like(field.q)
        
        kernel_radius = kernel.shape[0] // 2
        
        for idx in np.ndindex(field.shape):
            # Get neighborhood matching kernel size
            neighbors = []
            weights = []
            
            for ki in range(kernel.shape[0]):
                for kj in range(kernel.shape[1]):
                    ni = idx[0] + ki - kernel_radius
                    nj = idx[1] + kj - kernel_radius
                    
                    if 0 <= ni < field.shape[0] and 0 <= nj < field.shape[1]:
                        neighbors.append(field.get_state((ni, nj)))
                        weights.append(kernel[ki, kj])
            
            # Apply weighted operation
            new_state = operation(neighbors, weights)
            new_i[idx] = new_state.i
            new_q[idx] = new_state.q
        
        return ChannelField2D(new_i, new_q)
    
    return FieldCombinator(filter_op)


def propagate(
    rules: Callable[[List[State], State], State],
    iterations: int = 1
) -> FieldCombinator:
    """
    Propagate states according to rules (cellular automaton style)
    
    Parameters
    ----------
    rules : Callable[[List[State], State], State]
        Function(neighbors, center) → new_state
    iterations : int
        Number of propagation steps
    
    Examples
    --------
    >>> # Conway's Game of Life with channel states
    >>> def life_rules(neighbors, center):
    ...     psi_count = sum(1 for n in neighbors if n == PSI)
    ...     if center == PSI:
    ...         return PSI if 2 <= psi_count <= 3 else EMPTY
    ...     else:
    ...         return PSI if psi_count == 3 else center
    >>> 
    >>> automaton = propagate(life_rules, iterations=10)
    >>> evolved = automaton(initial_field)
    """
    def propagate_op(field: ChannelField2D) -> ChannelField2D:
        result = field
        nc = NeighborhoodCombinator(rules, radius=1, include_center=False)
        
        for _ in range(iterations):
            result = nc(result)
        
        return result
    
    return FieldCombinator(propagate_op)


# ============================================================================
# Pre-built Combinators
# ============================================================================

# Point operations lifted to fields
gate_field = lift_point_to_field(gate)
admit_field = lift_point_to_field(admit)
comp_field = lift_point_to_field(comp)

# Binary operations on fields
overlay_fields = lift_binary_to_field(overlay)
weave_fields = lift_binary_to_field(weave)


def detect_boundaries() -> FieldCombinator:
    """
    Detect boundaries where states change
    
    Returns field where PSI marks boundaries
    
    Examples
    --------
    >>> boundaries = detect_boundaries()
    >>> boundary_field = boundaries(input_field)
    """
    def boundary_op(neighbors: List[State], center: State) -> State:
        # Boundary if any neighbor differs from center
        for neighbor in neighbors:
            if neighbor != center:
                return PSI
        return EMPTY
    
    nc = NeighborhoodCombinator(boundary_op, radius=1, include_center=False)
    return FieldCombinator(nc)


def smooth_field(radius: int = 1) -> FieldCombinator:
    """
    Smooth field using neighborhood majority vote
    
    Examples
    --------
    >>> smoother = smooth_field(radius=1)
    >>> smoothed = smoother(noisy_field)
    """
    def majority_vote(neighbors: List[State], center: State) -> State:
        # Count each state type
        counts = {EMPTY: 0, DELTA: 0, PHI: 0, PSI: 0}
        for state in neighbors + [center]:
            counts[state] = counts.get(state, 0) + 1
        
        # Return most common state
        return max(counts, key=counts.get)
    
    nc = NeighborhoodCombinator(majority_vote, radius=radius, include_center=True)
    return FieldCombinator(nc)


def dilate_psi(radius: int = 1) -> FieldCombinator:
    """
    Dilate PSI states (morphological dilation)
    
    If any neighbor is PSI, become PSI
    
    Examples
    --------
    >>> dilate = dilate_psi(radius=1)
    >>> dilated = dilate(field)
    """
    def dilation_op(neighbors: List[State], center: State) -> State:
        if center == PSI:
            return PSI
        if any(n == PSI for n in neighbors):
            return PSI
        return center
    
    nc = NeighborhoodCombinator(dilation_op, radius=radius, include_center=True)
    return FieldCombinator(nc)


def erode_psi(radius: int = 1) -> FieldCombinator:
    """
    Erode PSI states (morphological erosion)
    
    Remain PSI only if all neighbors are PSI
    
    Examples
    --------
    >>> erode = erode_psi(radius=1)
    >>> eroded = erode(field)
    """
    def erosion_op(neighbors: List[State], center: State) -> State:
        if center != PSI:
            return center
        if all(n == PSI for n in neighbors):
            return PSI
        return DELTA  # Was PSI but lost resonance
    
    nc = NeighborhoodCombinator(erosion_op, radius=radius, include_center=False)
    return FieldCombinator(nc)


def connected_components() -> Callable[[ChannelField2D], Tuple[ChannelField2D, int]]:
    """
    Label connected components of PSI states
    
    Returns labeled field and count of components
    
    Examples
    --------
    >>> label_components = connected_components()
    >>> labeled_field, num_components = label_components(field)
    """
    def component_labeling(field: ChannelField2D) -> Tuple[ChannelField2D, int]:
        # Simple flood-fill labeling
        labeled = np.zeros(field.shape, dtype=int)
        current_label = 0
        
        def flood_fill(start_idx, label):
            """Flood fill from start_idx with label"""
            stack = [start_idx]
            visited = set()
            
            while stack:
                idx = stack.pop()
                if idx in visited:
                    continue
                if not (0 <= idx[0] < field.shape[0] and 0 <= idx[1] < field.shape[1]):
                    continue
                if field.get_state(idx) != PSI:
                    continue
                
                visited.add(idx)
                labeled[idx] = label
                
                # Add neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        neighbor_idx = (idx[0] + di, idx[1] + dj)
                        stack.append(neighbor_idx)
        
        # Find all PSI states and label components
        for idx in np.ndindex(field.shape):
            if field.get_state(idx) == PSI and labeled[idx] == 0:
                current_label += 1
                flood_fill(idx, current_label)
        
        # Convert labels to field (PSI where labeled, EMPTY elsewhere)
        result_i = (labeled > 0).astype(np.int8)
        result_q = (labeled > 0).astype(np.int8)
        
        return ChannelField2D(result_i, result_q), current_label
    
    return component_labeling


# ============================================================================
# Combinator Composition Syntax (functional pipeline)
# ============================================================================

class Pipeline:
    """
    Functional pipeline for composing field operations
    
    Examples
    --------
    >>> # Image processing pipeline
    >>> result = (Pipeline(input_field)
    ...     .apply(detect_boundaries())
    ...     .apply(dilate_psi(radius=2))
    ...     .apply(smooth_field(radius=1))
    ...     .get())
    """
    
    def __init__(self, field: ChannelField2D):
        self.field = field
    
    def apply(self, combinator: FieldCombinator) -> 'Pipeline':
        """Apply combinator and return new pipeline"""
        return Pipeline(combinator(self.field))
    
    def apply_if(
        self, 
        condition: Callable[[ChannelField2D], bool],
        combinator: FieldCombinator
    ) -> 'Pipeline':
        """Conditionally apply combinator"""
        if condition(self.field):
            return self.apply(combinator)
        return self
    
    def parallel(self, *combinators: FieldCombinator) -> 'Pipeline':
        """Apply multiple combinators and overlay results"""
        results = [c(self.field) for c in combinators]
        combined = results[0]
        for result in results[1:]:
            combined = combined.overlay(result)
        return Pipeline(combined)
    
    def get(self) -> ChannelField2D:
        """Get final field"""
        return self.field


# ============================================================================
# Example: Image Processing with Channel Combinators
# ============================================================================

def edge_detection_pipeline() -> FieldCombinator:
    """
    Edge detection using channel combinators
    
    Pipeline:
    1. Detect boundaries (state changes)
    2. Thin edges
    3. Remove noise
    
    Examples
    --------
    >>> detector = edge_detection_pipeline()
    >>> edges = detector(image_field)
    """
    return compose_many(
        detect_boundaries(),
        erode_psi(radius=1),  # Thin
        dilate_psi(radius=1),  # Reconnect
        smooth_field(radius=1)  # Denoise
    )


def feature_extraction_pipeline() -> FieldCombinator:
    """
    Extract features from channel field
    
    Pipeline:
    1. Enhance resonant states
    2. Smooth regions
    3. Detect boundaries
    
    Examples
    --------
    >>> extractor = feature_extraction_pipeline()
    >>> features = extractor(input_field)
    """
    return compose_many(
        admit_field,  # Validate present elements
        smooth_field(radius=2),  # Group regions
        detect_boundaries()  # Find region boundaries
    )