"""
Functional combinators for channel field operations

This module provides a compositional algebra for spatial channel operations,
enabling elegant, efficient, and interpretable field transformations.

Key Components
--------------
Point Combinators : Operate on individual states
Field Combinators : Transform entire fields
Neighborhood Combinators : Local spatial operations
Binary Combinators : Combine two fields
Reduction Combinators : Field → State aggregation
Lazy Evaluation : Deferred execution with optimization

Examples
--------
>>> from channelpy.combinators import (
...     lazy, query, Pipeline,
...     gate_field, admit_field,
...     detect_boundaries, smooth_field, dilate_psi
... )
>>> 
>>> # Lazy evaluation
>>> result = (lazy(field)
...     .smooth(radius=2)
...     .detect_boundaries()
...     .dilate(radius=1)
...     .compute())
>>> 
>>> # Query interface
>>> psi_regions = (query(field)
...     .where(lambda s: s == PSI)
...     .dilate(radius=3)
...     .compute())
>>> 
>>> # Pipeline composition
>>> pipeline = Pipeline(field).apply(detect_boundaries()).apply(smooth_field())
>>> result = pipeline.get()
"""

# Core combinator types
from .channel_combinators import (
    # Base combinator classes
    PointCombinator,
    FieldCombinator,
    NeighborhoodCombinator,
    ReductionCombinator,
    BinaryFieldCombinator,
    
    # Higher-order combinators
    lift_point_to_field,
    lift_binary_to_field,
    compose_many,
    
    # Spatial operations
    spatial_filter,
    propagate,
    
    # Pre-built combinators
    gate_field,
    admit_field,
    comp_field,
    overlay_fields,
    weave_fields,
    
    # Common operations
    detect_boundaries,
    smooth_field,
    dilate_psi,
    erode_psi,
    connected_components,
    
    # Pipeline interface
    Pipeline,
    
    # Pre-built pipelines
    edge_detection_pipeline,
    feature_extraction_pipeline,
)

# Lazy evaluation
from .lazy import (
    # Lazy field classes
    LazyField,
    LazyFieldQuery,
    StreamingField,
    
    # Convenience functions
    lazy,
    query,
    
    # Operation representation
    Operation,
)


# Version
__version__ = '0.1.0'


# Module-level convenience functions
def compose(*combinators):
    """
    Compose multiple field combinators
    
    Alias for compose_many with more Pythonic name
    
    Examples
    --------
    >>> pipeline = compose(smooth_field(), detect_boundaries(), dilate_psi())
    >>> result = pipeline(field)
    """
    return compose_many(*combinators)


def parallel(*combinators):
    """
    Apply combinators in parallel and overlay results
    
    Examples
    --------
    >>> result = parallel(detect_boundaries(), smooth_field())(field)
    """
    if not combinators:
        return FieldCombinator(lambda f: f)
    
    def parallel_op(field):
        from ..fields.field import ChannelField2D
        results = [c(field) for c in combinators]
        combined = results[0]
        for result in results[1:]:
            combined = combined.overlay(result)
        return combined
    
    return FieldCombinator(parallel_op)


# ============================================================================
# Module Documentation
# ============================================================================

__all__ = [
    # Base combinator types
    'PointCombinator',
    'FieldCombinator',
    'NeighborhoodCombinator',
    'ReductionCombinator',
    'BinaryFieldCombinator',
    
    # Higher-order combinators
    'lift_point_to_field',
    'lift_binary_to_field',
    'compose_many',
    'compose',
    'parallel',
    
    # Spatial operations
    'spatial_filter',
    'propagate',
    
    # Pre-built combinators
    'gate_field',
    'admit_field',
    'comp_field',
    'overlay_fields',
    'weave_fields',
    
    # Common operations
    'detect_boundaries',
    'smooth_field',
    'dilate_psi',
    'erode_psi',
    'connected_components',
    
    # Pipeline interface
    'Pipeline',
    
    # Pre-built pipelines
    'edge_detection_pipeline',
    'feature_extraction_pipeline',
    
    # Lazy evaluation
    'LazyField',
    'LazyFieldQuery',
    'StreamingField',
    'lazy',
    'query',
    'Operation',
]


# Quick reference documentation
QUICK_REFERENCE = """
ChannelPy Combinators Quick Reference
=====================================

BASIC USAGE:
    from channelpy.combinators import lazy, query, Pipeline
    
    # Lazy evaluation
    result = lazy(field).smooth().dilate().compute()
    
    # Query interface
    result = query(field).where(lambda s: s == PSI).compute()
    
    # Pipeline
    result = Pipeline(field).apply(smooth_field()).get()

POINT OPERATIONS:
    gate_field      - Remove unvalidated elements
    admit_field     - Validate present elements
    comp_field      - Complement states

SPATIAL OPERATIONS:
    detect_boundaries()    - Find state transitions
    smooth_field(r)        - Neighborhood smoothing
    dilate_psi(r)          - Expand PSI regions
    erode_psi(r)           - Shrink PSI regions

COMPOSITION:
    compose(f, g, h)       - Function composition
    parallel(f, g)         - Parallel application
    compose_many(...)      - Compose many operations

LAZY EVALUATION:
    lazy(field)            - Create lazy field
        .smooth()          - Add smoothing
        .dilate()          - Add dilation
        .compute()         - Execute plan
        .explain()         - Show execution plan
    
    query(field)           - SQL-like interface
        .where(pred)       - Filter states
        .select(proj)      - Project attributes
        .compute()         - Execute query

PIPELINES:
    Pipeline(field)        - Create pipeline
        .apply(op)         - Add operation
        .parallel(ops)     - Parallel ops
        .get()             - Get result

For full documentation: https://channelpy.readthedocs.io/combinators
"""


def help():
    """Print quick reference"""
    print(QUICK_REFERENCE)