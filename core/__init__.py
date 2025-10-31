"""
ChannelPy Core Module

Fundamental channel algebra operations and structures.

This module provides:
- State: Two-bit channel states (∅, δ, φ, ψ)
- StateArray: Efficient arrays of states
- Operations: gate, admit, overlay, weave, comp
- Nested: Hierarchical channel structures
- Parallel: Independent channel dimensions
- Lattice: Partial order and lattice operations
"""

from .state import (
    State,
    StateArray,
    EMPTY,
    DELTA,
    PHI,
    PSI,
)

from .operations import (
    gate,
    admit,
    overlay,
    weave,
    comp,
    neg_i,
    neg_q,
    compose,
    pipe,
)

from .nested import (
    NestedState,
    NestedStateArray,
)

from .parallel import (
    ParallelChannels,
    ParallelChannelArray,
)

from .lattice import (
    partial_order,
    meet,
    join,
    is_comparable,
    lattice_distance,
    bottom,
    top,
    LatticeStructure,
)

__all__ = [
    # Basic states
    'State',
    'StateArray',
    'EMPTY',
    'DELTA',
    'PHI',
    'PSI',
    
    # Operations
    'gate',
    'admit',
    'overlay',
    'weave',
    'comp',
    'neg_i',
    'neg_q',
    'compose',
    'pipe',
    
    # Nested structures
    'NestedState',
    'NestedStateArray',
    
    # Parallel structures
    'ParallelChannels',
    'ParallelChannelArray',
    
    # Lattice operations
    'partial_order',
    'meet',
    'join',
    'is_comparable',
    'lattice_distance',
    'bottom',
    'top',
    'LatticeStructure',
]

__version__ = '0.1.0'