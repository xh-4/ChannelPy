"""
Lazy evaluation for channel field operations

This module provides lazy evaluation capabilities for channel fields,
enabling:
- Query optimization (combine operations before execution)
- Memory efficiency (avoid intermediate field allocations)
- Streaming processing (process huge fields in chunks)
- Composable field transformations

Key Classes
-----------
LazyField : Deferred field operations with optimization
LazyFieldQuery : SQL-like query interface for fields
StreamingField : Chunk-by-chunk processing for massive fields
"""

from typing import List, Callable, Optional, Iterator, Tuple, Any, Dict
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..fields.field import ChannelField2D
from .channel_combinators import (
    PointCombinator,
    FieldCombinator,
    NeighborhoodCombinator,
    BinaryFieldCombinator,
    ReductionCombinator
)


# ============================================================================
# Operation Representation
# ============================================================================

@dataclass
class Operation:
    """
    Represents a deferred field operation
    
    Attributes
    ----------
    combinator : Callable
        The combinator to apply
    name : str
        Human-readable operation name
    params : Dict
        Operation parameters
    """
    combinator: Callable
    name: str
    params: Dict[str, Any]
    
    def __repr__(self):
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})" if params_str else self.name


# ============================================================================
# Lazy Field
# ============================================================================

class LazyField:
    """
    Lazy channel field with deferred operations
    
    Operations are recorded but not executed until .compute() is called.
    This enables:
    - Operation fusion (combine multiple passes)
    - Memory efficiency (no intermediate allocations)
    - Query optimization (reorder operations)
    
    Examples
    --------
    >>> # Eager evaluation (3 passes, 2 intermediate fields):
    >>> result = dilate(erode(smooth(field)))
    >>> 
    >>> # Lazy evaluation (optimized to fewer passes):
    >>> lazy = LazyField(field)
    >>> lazy = lazy.smooth().erode().dilate()
    >>> result = lazy.compute()  # Execute optimized plan
    >>> 
    >>> # Inspect plan before execution:
    >>> print(lazy.explain())
    """
    
    def __init__(self, source):
        """
        Parameters
        ----------
        source : ChannelField2D or LazyField
            Source field or lazy field
        """
        if isinstance(source, ChannelField2D):
            self._source = source
            self._operations: List[Operation] = []
        elif isinstance(source, LazyField):
            self._source = source._source
            self._operations = source._operations.copy()
        else:
            raise TypeError("Source must be ChannelField2D or LazyField")
    
    def apply(
        self, 
        combinator: Callable, 
        name: str = "custom",
        **params
    ) -> 'LazyField':
        """
        Add operation to execution plan
        
        Parameters
        ----------
        combinator : Callable
            Field combinator to apply
        name : str
            Operation name (for debugging)
        **params
            Operation parameters
            
        Returns
        -------
        lazy : LazyField
            New lazy field with operation added
        """
        new_lazy = LazyField(self)
        operation = Operation(
            combinator=combinator,
            name=name,
            params=params
        )
        new_lazy._operations.append(operation)
        return new_lazy
    
    def optimize(self) -> List[Operation]:
        """
        Optimize operation sequence
        
        Optimizations:
        1. Merge consecutive point operations
        2. Eliminate redundant operations
        3. Reorder independent operations
        4. Simplify operation chains
        
        Returns
        -------
        optimized : List[Operation]
            Optimized operation sequence
        """
        if not self._operations:
            return []
        
        optimized = []
        pending_point_ops = []
        
        for op in self._operations:
            # Check if this is a point operation
            if self._is_point_operation(op):
                pending_point_ops.append(op)
            else:
                # Flush pending point operations
                if pending_point_ops:
                    merged = self._merge_point_operations(pending_point_ops)
                    optimized.append(merged)
                    pending_point_ops = []
                
                # Add the non-point operation
                optimized.append(op)
        
        # Flush remaining point operations
        if pending_point_ops:
            merged = self._merge_point_operations(pending_point_ops)
            optimized.append(merged)
        
        # Additional optimizations
        optimized = self._eliminate_redundant(optimized)
        optimized = self._reorder_for_efficiency(optimized)
        
        return optimized
    
    def _is_point_operation(self, op: Operation) -> bool:
        """Check if operation is point-wise"""
        return isinstance(op.combinator, PointCombinator) or \
               op.name in ['gate', 'admit', 'comp']
    
    def _merge_point_operations(self, ops: List[Operation]) -> Operation:
        """
        Merge consecutive point operations into single pass
        
        Instead of: field -> op1 -> intermediate -> op2 -> result
        Do: field -> (op1 ∘ op2) -> result
        """
        if len(ops) == 1:
            return ops[0]
        
        # Compose operations right-to-left
        def merged_func(field: ChannelField2D) -> ChannelField2D:
            result = field
            for op in ops:
                result = op.combinator(result)
            return result
        
        merged_name = " → ".join(op.name for op in ops)
        
        return Operation(
            combinator=merged_func,
            name=f"merged({merged_name})",
            params={}
        )
    
    def _eliminate_redundant(self, ops: List[Operation]) -> List[Operation]:
        """
        Eliminate redundant operations
        
        Examples:
        - gate → gate => gate
        - admit → admit => admit
        """
        if len(ops) <= 1:
            return ops
        
        optimized = [ops[0]]
        
        for op in ops[1:]:
            last = optimized[-1]
            
            # Check for redundancy
            if op.name == last.name and op.name in ['gate', 'admit']:
                # Duplicate operation - skip
                continue
            
            optimized.append(op)
        
        return optimized
    
    def _reorder_for_efficiency(self, ops: List[Operation]) -> List[Operation]:
        """
        Reorder operations for efficiency
        
        Heuristics:
        - Do cheap operations (point ops) before expensive ones (neighborhoods)
        - Do filtering operations early to reduce data
        """
        # For now, keep order (more sophisticated reordering needs dependency analysis)
        return ops
    
    def compute(self) -> ChannelField2D:
        """
        Execute optimized operation plan
        
        Returns
        -------
        result : ChannelField2D
            Computed result field
        """
        optimized = self.optimize()
        result = self._source
        
        for op in optimized:
            result = op.combinator(result)
        
        return result
    
    def compute_inplace(self) -> ChannelField2D:
        """
        Compute with in-place operations where possible
        
        More memory efficient but modifies source
        """
        # For now, same as compute (true in-place needs careful analysis)
        return self.compute()
    
    def compute_streaming(
        self, 
        chunk_size: Tuple[int, int] = (100, 100)
    ) -> Iterator[Tuple[Tuple[int, int], ChannelField2D]]:
        """
        Compute in chunks for huge fields
        
        Parameters
        ----------
        chunk_size : Tuple[int, int]
            Size of each chunk (height, width)
            
        Yields
        ------
        position : Tuple[int, int]
            Top-left position of chunk
        chunk : ChannelField2D
            Processed chunk
        """
        h, w = self._source.shape
        ch, cw = chunk_size
        
        optimized = self.optimize()
        
        for i in range(0, h, ch):
            for j in range(0, w, cw):
                # Extract chunk (with padding for neighborhood ops)
                padding = self._calculate_required_padding(optimized)
                
                i_start = max(0, i - padding)
                i_end = min(h, i + ch + padding)
                j_start = max(0, j - padding)
                j_end = min(w, j + cw + padding)
                
                # Get chunk with padding
                chunk_i = self._source.i[i_start:i_end, j_start:j_end]
                chunk_q = self._source.q[i_start:i_end, j_start:j_end]
                chunk_field = ChannelField2D(chunk_i, chunk_q)
                
                # Process chunk
                result_chunk = chunk_field
                for op in optimized:
                    result_chunk = op.combinator(result_chunk)
                
                # Remove padding
                if padding > 0:
                    actual_i_start = padding if i > 0 else 0
                    actual_j_start = padding if j > 0 else 0
                    actual_i_end = actual_i_start + min(ch, h - i)
                    actual_j_end = actual_j_start + min(cw, w - j)
                    
                    result_chunk = ChannelField2D(
                        result_chunk.i[actual_i_start:actual_i_end, actual_j_start:actual_j_end],
                        result_chunk.q[actual_i_start:actual_i_end, actual_j_start:actual_j_end]
                    )
                
                yield (i, j), result_chunk
    
    def _calculate_required_padding(self, ops: List[Operation]) -> int:
        """Calculate padding needed for neighborhood operations"""
        max_radius = 0
        
        for op in ops:
            if isinstance(op.combinator, NeighborhoodCombinator):
                radius = op.combinator.radius
                max_radius = max(max_radius, radius)
            elif 'radius' in op.params:
                max_radius = max(max_radius, op.params['radius'])
        
        return max_radius
    
    def explain(self) -> str:
        """
        Explain execution plan
        
        Returns
        -------
        explanation : str
            Human-readable explanation of operations
        """
        if not self._operations:
            return "No operations (identity)"
        
        lines = ["Lazy Field Execution Plan", "=" * 40]
        lines.append(f"Source shape: {self._source.shape}")
        lines.append(f"Number of operations: {len(self._operations)}")
        lines.append("")
        
        lines.append("Original operations:")
        for i, op in enumerate(self._operations, 1):
            lines.append(f"  {i}. {op}")
        
        lines.append("")
        optimized = self.optimize()
        lines.append(f"Optimized operations ({len(optimized)}):")
        for i, op in enumerate(optimized, 1):
            lines.append(f"  {i}. {op}")
        
        lines.append("")
        savings = len(self._operations) - len(optimized)
        if savings > 0:
            lines.append(f"Optimization: {savings} operations eliminated")
        else:
            lines.append("Optimization: No operations eliminated")
        
        return "\n".join(lines)
    
    # ========================================================================
    # Convenience Methods (common operations)
    # ========================================================================
    
    def gate(self) -> 'LazyField':
        """Apply gate operation"""
        from .channel_combinators import gate_field
        return self.apply(gate_field, name="gate")
    
    def admit(self) -> 'LazyField':
        """Apply admit operation"""
        from .channel_combinators import admit_field
        return self.apply(admit_field, name="admit")
    
    def comp(self) -> 'LazyField':
        """Apply complement operation"""
        from .channel_combinators import comp_field
        return self.apply(comp_field, name="comp")
    
    def smooth(self, radius: int = 1) -> 'LazyField':
        """Apply smoothing"""
        from .channel_combinators import smooth_field
        return self.apply(smooth_field(radius), name="smooth", radius=radius)
    
    def detect_boundaries(self) -> 'LazyField':
        """Detect boundaries"""
        from .channel_combinators import detect_boundaries
        return self.apply(detect_boundaries(), name="detect_boundaries")
    
    def dilate(self, radius: int = 1) -> 'LazyField':
        """Dilate PSI states"""
        from .channel_combinators import dilate_psi
        return self.apply(dilate_psi(radius), name="dilate", radius=radius)
    
    def erode(self, radius: int = 1) -> 'LazyField':
        """Erode PSI states"""
        from .channel_combinators import erode_psi
        return self.apply(erode_psi(radius), name="erode", radius=radius)
    
    def overlay(self, other: 'LazyField') -> 'LazyField':
        """Overlay with another lazy field"""
        # Compute both and overlay
        from .channel_combinators import overlay_fields
        
        def overlay_op(field: ChannelField2D) -> ChannelField2D:
            other_field = other.compute()
            return overlay_fields(field, other_field)
        
        return self.apply(overlay_op, name="overlay")
    
    def weave(self, other: 'LazyField') -> 'LazyField':
        """Weave with another lazy field"""
        from .channel_combinators import weave_fields
        
        def weave_op(field: ChannelField2D) -> ChannelField2D:
            other_field = other.compute()
            return weave_fields(field, other_field)
        
        return self.apply(weave_op, name="weave")


# ============================================================================
# Lazy Field Query (SQL-like interface)
# ============================================================================

class LazyFieldQuery:
    """
    SQL-like query interface for channel fields
    
    Provides declarative syntax for field operations:
    - WHERE: Filter states by predicate
    - SELECT: Project to specific attributes
    - GROUP BY: Group states by key
    - HAVING: Filter groups
    - ORDER BY: Sort states
    
    Examples
    --------
    >>> # Find all PSI states and dilate them
    >>> result = (LazyFieldQuery(field)
    ...     .where(lambda s: s == PSI)
    ...     .dilate(radius=2)
    ...     .compute())
    >>> 
    >>> # Count states by type
    >>> counts = (LazyFieldQuery(field)
    ...     .group_by(lambda s: s)
    ...     .count()
    ...     .compute())
    """
    
    def __init__(self, source):
        """
        Parameters
        ----------
        source : ChannelField2D or LazyField
            Source field
        """
        if isinstance(source, LazyField):
            self._lazy = source
        else:
            self._lazy = LazyField(source)
        
        self._filters = []
        self._projections = []
        self._group_key = None
    
    def where(self, predicate: Callable[[State], bool]) -> 'LazyFieldQuery':
        """
        Filter states by predicate
        
        Parameters
        ----------
        predicate : Callable[[State], bool]
            Function that returns True for states to keep
            
        Returns
        -------
        query : LazyFieldQuery
            Query with filter added
        
        Examples
        --------
        >>> query.where(lambda s: s == PSI)
        >>> query.where(lambda s: s.i == 1)
        """
        new_query = LazyFieldQuery(self._lazy)
        new_query._filters = self._filters + [predicate]
        new_query._projections = self._projections.copy()
        new_query._group_key = self._group_key
        
        # Add filter operation to lazy field
        def filter_op(field: ChannelField2D) -> ChannelField2D:
            new_i = np.zeros_like(field.i)
            new_q = np.zeros_like(field.q)
            
            for idx in np.ndindex(field.shape):
                state = field.get_state(idx)
                
                # Apply all filters
                keep = all(pred(state) for pred in new_query._filters)
                
                if keep:
                    new_i[idx] = state.i
                    new_q[idx] = state.q
            
            return ChannelField2D(new_i, new_q)
        
        new_query._lazy = self._lazy.apply(filter_op, name="where")
        return new_query
    
    def select(self, projection: Callable[[State], Any]) -> 'LazyFieldQuery':
        """
        Project states to new representation
        
        Parameters
        ----------
        projection : Callable[[State], Any]
            Function to transform states
            
        Examples
        --------
        >>> query.select(lambda s: s.i)  # Extract i bit
        >>> query.select(lambda s: State(s.i, 0))  # Zero out q bit
        """
        new_query = LazyFieldQuery(self._lazy)
        new_query._filters = self._filters.copy()
        new_query._projections = self._projections + [projection]
        new_query._group_key = self._group_key
        
        # Add projection operation
        def project_op(field: ChannelField2D) -> ChannelField2D:
            new_i = np.zeros_like(field.i)
            new_q = np.zeros_like(field.q)
            
            for idx in np.ndindex(field.shape):
                state = field.get_state(idx)
                
                # Apply all projections
                result = state
                for proj in new_query._projections:
                    result = proj(result)
                
                # Convert back to state if needed
                if isinstance(result, State):
                    new_i[idx] = result.i
                    new_q[idx] = result.q
                elif isinstance(result, int):
                    # Single bit projection
                    new_i[idx] = result
                    new_q[idx] = result
            
            return ChannelField2D(new_i, new_q)
        
        new_query._lazy = self._lazy.apply(project_op, name="select")
        return new_query
    
    def dilate(self, radius: int = 1) -> 'LazyFieldQuery':
        """Add dilation operation"""
        new_query = LazyFieldQuery(self._lazy.dilate(radius))
        new_query._filters = self._filters.copy()
        new_query._projections = self._projections.copy()
        return new_query
    
    def erode(self, radius: int = 1) -> 'LazyFieldQuery':
        """Add erosion operation"""
        new_query = LazyFieldQuery(self._lazy.erode(radius))
        new_query._filters = self._filters.copy()
        new_query._projections = self._projections.copy()
        return new_query
    
    def smooth(self, radius: int = 1) -> 'LazyFieldQuery':
        """Add smoothing operation"""
        new_query = LazyFieldQuery(self._lazy.smooth(radius))
        new_query._filters = self._filters.copy()
        new_query._projections = self._projections.copy()
        return new_query
    
    def compute(self) -> ChannelField2D:
        """Execute query"""
        return self._lazy.compute()
    
    def explain(self) -> str:
        """Explain query execution plan"""
        lines = ["Query Execution Plan", "=" * 40]
        
        if self._filters:
            lines.append(f"Filters: {len(self._filters)}")
            for i, _ in enumerate(self._filters, 1):
                lines.append(f"  {i}. WHERE <predicate>")
        
        if self._projections:
            lines.append(f"Projections: {len(self._projections)}")
            for i, _ in enumerate(self._projections, 1):
                lines.append(f"  {i}. SELECT <projection>")
        
        lines.append("")
        lines.append("Field operations:")
        lines.append(self._lazy.explain())
        
        return "\n".join(lines)


# ============================================================================
# Streaming Field (for massive fields)
# ============================================================================

class StreamingField:
    """
    Process massive fields in chunks
    
    For fields too large to fit in memory, process piece by piece
    
    Examples
    --------
    >>> # Process 10GB field in 100MB chunks
    >>> streaming = StreamingField(
    ...     source_path='huge_field.npy',
    ...     chunk_size=(1000, 1000)
    ... )
    >>> 
    >>> for chunk_pos, chunk in streaming.process(operations):
    ...     save_chunk(chunk, chunk_pos)
    """
    
    def __init__(
        self,
        source_path: str,
        chunk_size: Tuple[int, int] = (100, 100)
    ):
        """
        Parameters
        ----------
        source_path : str
            Path to field data (memory-mapped)
        chunk_size : Tuple[int, int]
            Size of processing chunks
        """
        self.source_path = source_path
        self.chunk_size = chunk_size
        
        # Memory-map the source
        self._mmap = None  # Would use np.memmap in practice
    
    def process(
        self, 
        operations: List[Callable]
    ) -> Iterator[Tuple[Tuple[int, int], ChannelField2D]]:
        """
        Process field in chunks
        
        Parameters
        ----------
        operations : List[Callable]
            Operations to apply to each chunk
            
        Yields
        ------
        position : Tuple[int, int]
            Chunk position
        chunk : ChannelField2D
            Processed chunk
        """
        # Implementation would handle memory-mapped processing
        raise NotImplementedError("Streaming implementation requires memory mapping")


# ============================================================================
# Utility Functions
# ============================================================================

def lazy(field: ChannelField2D) -> LazyField:
    """
    Convenience function to create lazy field
    
    Examples
    --------
    >>> result = lazy(field).smooth().dilate().compute()
    """
    return LazyField(field)


def query(field: ChannelField2D) -> LazyFieldQuery:
    """
    Convenience function to create field query
    
    Examples
    --------
    >>> result = query(field).where(lambda s: s == PSI).dilate().compute()
    """
    return LazyFieldQuery(field)