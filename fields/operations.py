"""
Operations on channel fields
"""

from typing import Callable, Optional, Tuple, List, Union, Any, Dict
import numpy as np
from scipy import ndimage
from collections import deque

from ..core.state import State, EMPTY, DELTA, PHI, PSI
from ..core.operations import gate, admit, overlay, weave
from .field import (
    ChannelField, ChannelField1D, ChannelField2D, ChannelField3D,
    BoundaryCondition
)


def map_field(
    field: ChannelField,
    operation: Callable[[State], State]
) -> ChannelField:
    """
    Apply operation element-wise to field
    
    Parameters
    ----------
    field : ChannelField
        Input field
    operation : Callable[[State], State]
        Function to apply to each state
        
    Returns
    -------
    result : ChannelField
        Transformed field
        
    Examples
    --------
    >>> field = ChannelField2D(10, 10)
    >>> gated = map_field(field, gate)
    >>> admitted = map_field(field, admit)
    """
    result = field.copy()
    
    if isinstance(field, ChannelField1D):
        for x in range(field.length):
            state = field.get(x)
            result.set(x, operation(state))
    
    elif isinstance(field, ChannelField2D):
        for y in range(field.height):
            for x in range(field.width):
                state = field.get(x, y)
                result.set(x, y, operation(state))
    
    elif isinstance(field, ChannelField3D):
        for z in range(field.depth):
            for y in range(field.height):
                for x in range(field.width):
                    state = field.get(x, y, z)
                    result.set(x, y, z, operation(state))
    
    return result


def reduce_field(
    field: ChannelField,
    operation: Callable[[State, State], State],
    initial: Optional[State] = None
) -> State:
    """
    Reduce field to single state
    
    Parameters
    ----------
    field : ChannelField
        Input field
    operation : Callable[[State, State], State]
        Binary reduction operation
    initial : State, optional
        Initial value
        
    Returns
    -------
    result : State
        Reduced state
        
    Examples
    --------
    >>> field = ChannelField2D(10, 10)
    >>> # Overlay all states
    >>> combined = reduce_field(field, overlay)
    >>> # Weave all states (intersection)
    >>> intersection = reduce_field(field, weave, initial=PSI)
    """
    if initial is None:
        initial = EMPTY
    
    result = initial
    
    if isinstance(field, ChannelField1D):
        for x in range(field.length):
            result = operation(result, field.get(x))
    
    elif isinstance(field, ChannelField2D):
        for y in range(field.height):
            for x in range(field.width):
                result = operation(result, field.get(x, y))
    
    elif isinstance(field, ChannelField3D):
        for z in range(field.depth):
            for y in range(field.height):
                for x in range(field.width):
                    result = operation(result, field.get(x, y, z))
    
    return result


def convolve_field(
    field: Union[ChannelField2D, ChannelField3D],
    kernel: Union[str, np.ndarray],
    operation: Callable[[List[State]], State] = None,
    boundary: BoundaryCondition = BoundaryCondition.ZERO
) -> Union[ChannelField2D, ChannelField3D]:
    """
    Convolve field with kernel
    
    Parameters
    ----------
    field : ChannelField2D or ChannelField3D
        Input field
    kernel : str or np.ndarray
        Kernel type ('box', 'gaussian', 'sobel') or custom kernel
    operation : Callable[[List[State]], State], optional
        How to combine neighbor states. Default: majority vote
    boundary : BoundaryCondition
        How to handle boundaries
        
    Returns
    -------
    result : ChannelField
        Convolved field
        
    Examples
    --------
    >>> field = ChannelField2D(100, 100)
    >>> # Blur with box filter
    >>> blurred = convolve_field(field, 'box')
    >>> # Edge detection
    >>> edges = convolve_field(field, 'sobel')
    """
    if operation is None:
        # Default: majority vote among neighbors
        operation = _majority_vote_operation
    
    # Get kernel
    if isinstance(kernel, str):
        if kernel == 'box':
            if isinstance(field, ChannelField2D):
                kernel_array = np.ones((3, 3)) / 9
            else:
                kernel_array = np.ones((3, 3, 3)) / 27
        elif kernel == 'gaussian':
            if isinstance(field, ChannelField2D):
                kernel_array = _gaussian_kernel_2d(size=5, sigma=1.0)
            else:
                kernel_array = _gaussian_kernel_3d(size=5, sigma=1.0)
        elif kernel == 'sobel':
            if isinstance(field, ChannelField2D):
                kernel_array = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            else:
                raise ValueError("Sobel kernel only for 2D fields")
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
    else:
        kernel_array = kernel
    
    result = field.copy()
    
    if isinstance(field, ChannelField2D):
        kh, kw = kernel_array.shape
        kh_half = kh // 2
        kw_half = kw // 2
        
        for y in range(field.height):
            for x in range(field.width):
                # Collect neighbor states
                neighbors = []
                for ky in range(kh):
                    for kx in range(kw):
                        nx = x + kx - kw_half
                        ny = y + ky - kh_half
                        
                        # Handle boundary
                        neighbor_state = _get_with_boundary(
                            field, nx, ny, boundary
                        )
                        
                        if neighbor_state is not None:
                            neighbors.append(neighbor_state)
                
                # Apply operation
                if neighbors:
                    new_state = operation(neighbors)
                    result.set(x, y, new_state)
    
    elif isinstance(field, ChannelField3D):
        kd, kh, kw = kernel_array.shape
        kd_half = kd // 2
        kh_half = kh // 2
        kw_half = kw // 2
        
        for z in range(field.depth):
            for y in range(field.height):
                for x in range(field.width):
                    # Collect neighbor states
                    neighbors = []
                    for kz in range(kd):
                        for ky in range(kh):
                            for kx in range(kw):
                                nx = x + kx - kw_half
                                ny = y + ky - kh_half
                                nz = z + kz - kd_half
                                
                                # Handle boundary
                                neighbor_state = _get_with_boundary_3d(
                                    field, nx, ny, nz, boundary
                                )
                                
                                if neighbor_state is not None:
                                    neighbors.append(neighbor_state)
                    
                    # Apply operation
                    if neighbors:
                        new_state = operation(neighbors)
                        result.set(x, y, z, new_state)
    
    return result


def gradient_field(
    field: Union[ChannelField2D, ChannelField3D]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient of field (changes between states)
    
    Returns magnitude of state transitions
    
    Parameters
    ----------
    field : ChannelField2D or ChannelField3D
        Input field
        
    Returns
    -------
    gradient_i : np.ndarray
        Gradient of i-bit
    gradient_q : np.ndarray
        Gradient of q-bit
    """
    if isinstance(field, ChannelField2D):
        grad_i_y, grad_i_x = np.gradient(field._i.astype(float))
        grad_q_y, grad_q_x = np.gradient(field._q.astype(float))
        
        gradient_i = np.sqrt(grad_i_x**2 + grad_i_y**2)
        gradient_q = np.sqrt(grad_q_x**2 + grad_q_y**2)
    
    elif isinstance(field, ChannelField3D):
        grad_i_z, grad_i_y, grad_i_x = np.gradient(field._i.astype(float))
        grad_q_z, grad_q_y, grad_q_x = np.gradient(field._q.astype(float))
        
        gradient_i = np.sqrt(grad_i_x**2 + grad_i_y**2 + grad_i_z**2)
        gradient_q = np.sqrt(grad_q_x**2 + grad_q_y**2 + grad_q_z**2)
    
    else:
        raise TypeError("Gradient only for 2D or 3D fields")
    
    return gradient_i, gradient_q


def find_connected_components(
    field: Union[ChannelField2D, ChannelField3D],
    target_state: Optional[State] = None,
    connectivity: int = 1
) -> Tuple[np.ndarray, int]:
    """
    Find connected components in field
    
    Parameters
    ----------
    field : ChannelField2D or ChannelField3D
        Input field
    target_state : State, optional
        State to find components of. If None, find components for each state.
    connectivity : int
        Connectivity (1=face, 2=edge+face, 3=vertex+edge+face)
        
    Returns
    -------
    labels : np.ndarray
        Label array (0=background, 1..N=components)
    num_components : int
        Number of components found
        
    Examples
    --------
    >>> field = ChannelField2D(100, 100)
    >>> # Find all PSI regions
    >>> labels, n = find_connected_components(field, target_state=PSI)
    >>> print(f"Found {n} PSI regions")
    """
    if target_state is not None:
        # Create binary mask for target state
        if isinstance(field, ChannelField2D):
            mask = np.zeros((field.height, field.width), dtype=bool)
            for y in range(field.height):
                for x in range(field.width):
                    mask[y, x] = (field.get(x, y) == target_state)
        elif isinstance(field, ChannelField3D):
            mask = np.zeros((field.depth, field.height, field.width), dtype=bool)
            for z in range(field.depth):
                for y in range(field.height):
                    for x in range(field.width):
                        mask[z, y, x] = (field.get(x, y, z) == target_state)
        
        # Find connected components
        structure = _get_connectivity_structure(field, connectivity)
        labels, num_components = ndimage.label(mask, structure=structure)
    
    else:
        # Find components for all states
        # Encode states as integers: i*2 + q
        if isinstance(field, ChannelField2D):
            state_array = field._i * 2 + field._q
        elif isinstance(field, ChannelField3D):
            state_array = field._i * 2 + field._q
        
        structure = _get_connectivity_structure(field, connectivity)
        labels, num_components = ndimage.label(state_array, structure=structure)
    
    return labels, num_components


def detect_patterns(
    field: ChannelField2D,
    pattern: ChannelField2D,
    threshold: float = 0.8
) -> List[Tuple[int, int, float]]:
    """
    Detect occurrences of pattern in field
    
    Parameters
    ----------
    field : ChannelField2D
        Field to search
    pattern : ChannelField2D
        Pattern to find
    threshold : float
        Similarity threshold (0-1)
        
    Returns
    -------
    matches : List[Tuple[int, int, float]]
        List of (x, y, similarity) for each match
        
    Examples
    --------
    >>> field = ChannelField2D(100, 100)
    >>> # Create small pattern
    >>> pattern = ChannelField2D(5, 5)
    >>> # ... set pattern ...
    >>> matches = detect_patterns(field, pattern, threshold=0.9)
    >>> print(f"Found {len(matches)} matches")
    """
    matches = []
    
    pw, ph = pattern.width, pattern.height
    
    for y in range(field.height - ph + 1):
        for x in range(field.width - pw + 1):
            # Extract region
            region = field.get_region(x, x + pw, y, y + ph)
            
            # Compute similarity
            similarity = _compute_field_similarity(region, pattern)
            
            if similarity >= threshold:
                matches.append((x, y, similarity))
    
    return matches


def field_topology(
    field: Union[ChannelField2D, ChannelField3D],
    target_state: Optional[State] = None
) -> Dict[str, Any]:
    """
    Compute topological features of field
    
    Parameters
    ----------
    field : ChannelField2D or ChannelField3D
        Input field
    target_state : State, optional
        State to analyze (default: PSI)
        
    Returns
    -------
    topology : Dict
        Dictionary with:
        - 'betti_0': Number of connected components
        - 'betti_1': Number of holes (2D) or tunnels (3D)
        - 'betti_2': Number of voids (3D only)
        - 'euler_characteristic': χ = β₀ - β₁ + β₂
        
    Examples
    --------
    >>> field = ChannelField2D(100, 100)
    >>> topo = field_topology(field, target_state=PSI)
    >>> print(f"Components: {topo['betti_0']}, Holes: {topo['betti_1']}")
    """
    if target_state is None:
        target_state = PSI
    
    # Create binary mask
    if isinstance(field, ChannelField2D):
        mask = np.zeros((field.height, field.width), dtype=bool)
        for y in range(field.height):
            for x in range(field.width):
                mask[y, x] = (field.get(x, y) == target_state)
    elif isinstance(field, ChannelField3D):
        mask = np.zeros((field.depth, field.height, field.width), dtype=bool)
        for z in range(field.depth):
            for y in range(field.height):
                for x in range(field.width):
                    mask[z, y, x] = (field.get(x, y, z) == target_state)
    
    # Compute Betti numbers using persistent homology
    try:
        from ..topology.persistence import compute_betti_numbers
        betti = compute_betti_numbers(mask)
    except (ImportError, AttributeError):
        # Fallback: simple component counting
        labels, num_components = ndimage.label(mask)
        betti = {
            'betti_0': num_components,
            'betti_1': 0,  # Would need more sophisticated analysis
            'betti_2': 0
        }
    
    # Compute Euler characteristic
    if isinstance(field, ChannelField2D):
        euler = betti['betti_0'] - betti['betti_1']
    else:
        euler = betti['betti_0'] - betti['betti_1'] + betti['betti_2']
    
    return {
        'betti_0': betti['betti_0'],
        'betti_1': betti.get('betti_1', 0),
        'betti_2': betti.get('betti_2', 0),
        'euler_characteristic': euler
    }


def field_distance(
    field1: ChannelField,
    field2: ChannelField,
    metric: str = 'hamming'
) -> float:
    """
    Compute distance between two fields
    
    Parameters
    ----------
    field1, field2 : ChannelField
        Fields to compare (must have same shape)
    metric : str
        Distance metric:
        - 'hamming': Fraction of differing states
        - 'euclidean': Euclidean distance in bit space
        - 'topology': Topological distance
        
    Returns
    -------
    distance : float
        Distance measure
    """
    if field1.shape() != field2.shape():
        raise ValueError("Fields must have same shape")
    
    if metric == 'hamming':
        # Fraction of positions with different states
        if isinstance(field1, ChannelField1D):
            diff_i = np.sum(field1._i != field2._i)
            diff_q = np.sum(field1._q != field2._q)
            total = field1.length
        elif isinstance(field1, ChannelField2D):
            diff_i = np.sum(field1._i != field2._i)
            diff_q = np.sum(field1._q != field2._q)
            total = field1.width * field1.height
        elif isinstance(field1, ChannelField3D):
            diff_i = np.sum(field1._i != field2._i)
            diff_q = np.sum(field1._q != field2._q)
            total = field1.width * field1.height * field1.depth
        
        # Average over both bits
        distance = (diff_i + diff_q) / (2 * total)
    
    elif metric == 'euclidean':
        # Euclidean distance in flattened bit space
        diff_i = (field1._i - field2._i).astype(float)
        diff_q = (field1._q - field2._q).astype(float)
        
        distance = np.sqrt(np.sum(diff_i**2 + diff_q**2))
    
    elif metric == 'topology':
        # Compare topological features
        topo1 = field_topology(field1)
        topo2 = field_topology(field2)
        
        # Distance based on Betti numbers
        distance = (
            abs(topo1['betti_0'] - topo2['betti_0']) +
            abs(topo1['betti_1'] - topo2['betti_1']) +
            abs(topo1.get('betti_2', 0) - topo2.get('betti_2', 0))
        )
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distance


# Helper functions

def _majority_vote_operation(states: List[State]) -> State:
    """Majority vote among states"""
    if not states:
        return EMPTY
    
    # Count each state type
    counts = {EMPTY: 0, DELTA: 0, PHI: 0, PSI: 0}
    for state in states:
        counts[state] = counts.get(state, 0) + 1
    
    # Return most common
    return max(counts, key=counts.get)


def _get_with_boundary(
    field: ChannelField2D,
    x: int,
    y: int,
    boundary: BoundaryCondition
) -> Optional[State]:
    """Get state with boundary condition handling"""
    if 0 <= x < field.width and 0 <= y < field.height:
        return field.get(x, y)
    
    if boundary == BoundaryCondition.ZERO:
        return EMPTY
    elif boundary == BoundaryCondition.PERIODIC:
        x = x % field.width
        y = y % field.height
        return field.get(x, y)
    elif boundary == BoundaryCondition.REFLECT:
        x = abs(x) if x < 0 else (2 * field.width - x - 1 if x >= field.width else x)
        y = abs(y) if y < 0 else (2 * field.height - y - 1 if y >= field.height else y)
        x = min(x, field.width - 1)
        y = min(y, field.height - 1)
        return field.get(x, y)
    elif boundary == BoundaryCondition.EXTEND:
        x = max(0, min(x, field.width - 1))
        y = max(0, min(y, field.height - 1))
        return field.get(x, y)
    else:
        return None


def _get_with_boundary_3d(
    field: ChannelField3D,
    x: int,
    y: int,
    z: int,
    boundary: BoundaryCondition
) -> Optional[State]:
    """Get state with boundary condition handling (3D)"""
    if 0 <= x < field.width and 0 <= y < field.height and 0 <= z < field.depth:
        return field.get(x, y, z)
    
    if boundary == BoundaryCondition.ZERO:
        return EMPTY
    elif boundary == BoundaryCondition.PERIODIC:
        x = x % field.width
        y = y % field.height
        z = z % field.depth
        return field.get(x, y, z)
    # ... similar for other boundary conditions
    else:
        return None


def _gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """Generate 2D Gaussian kernel"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def _gaussian_kernel_3d(size: int, sigma: float) -> np.ndarray:
    """Generate 3D Gaussian kernel"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def _get_connectivity_structure(field: ChannelField, connectivity: int) -> np.ndarray:
    """Get connectivity structure for ndimage.label"""
    if isinstance(field, ChannelField2D):
        if connectivity == 1:
            return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        else:
            return np.ones((3, 3))
    elif isinstance(field, ChannelField3D):
        if connectivity == 1:
            structure = np.zeros((3, 3, 3))
            structure[1, 1, :] = 1
            structure[1, :, 1] = 1
            structure[:, 1, 1] = 1
            return structure
        else:
            return np.ones((3, 3, 3))


def _compute_field_similarity(field1: ChannelField2D, field2: ChannelField2D) -> float:
    """Compute similarity between two fields (0-1)"""
    if field1.shape() != field2.shape():
        return 0.0
    
    # Count matching states
    matches = np.sum((field1._i == field2._i) & (field1._q == field2._q))
    total = field1.width * field1.height
    
    return matches / total