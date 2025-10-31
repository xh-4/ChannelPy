"""
Lazy evaluation for large channel fields

Enables working with fields too large to fit in memory by:
- Computing states on-demand
- Caching frequently accessed regions
- Streaming processing
"""

from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
from functools import lru_cache

from ..core.state import State, EMPTY
from .field import ChannelField2D, ChannelField3D, BoundaryCondition


class LazyChannelField:
    """
    Base class for lazy-evaluated fields
    
    States are computed on-demand rather than stored
    """
    
    def __init__(
        self,
        generator: Callable[..., State],
        cache_size: int = 10000
    ):
        """
        Parameters
        ----------
        generator : Callable
            Function to generate state at coordinates
        cache_size : int
            Number of states to cache
        """
        self.generator = generator
        self.cache_size = cache_size
        self._cache = {}
    
    def _get_cached(self, *coords) -> State:
        """Get state with caching"""
        if coords in self._cache:
            return self._cache[coords]
        
        state = self.generator(*coords)
        
        # Manage cache size
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[coords] = state
        return state
    
    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self._cache),
            'max_size': self.cache_size,
            'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_access_count', 1), 1)
        }


class LazyChannelField2D(LazyChannelField):
    """
    Lazy 2D channel field
    
    Examples
    --------
    >>> # Create infinite field with generation function
    >>> def mandelbrot_state(x, y):
    ...     # Complex mandelbrot set encoding
    ...     c = complex((x - 500) / 200, (y - 500) / 200)
    ...     z = 0
    ...     for i in range(100):
    ...         z = z*z + c
    ...         if abs(z) > 2:
    ...             return State(i=1, q=int(i > 50))
    ...     return State(i=0, q=0)
    >>> 
    >>> field = LazyChannelField2D(width=1000, height=1000, generator=mandelbrot_state)
    >>> state = field.get(500, 500)
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        generator: Callable[[int, int], State],
        cache_size: int = 10000
    ):
        """
        Parameters
        ----------
        width, height : int
            Field dimensions
        generator : Callable[[int, int], State]
            Function (x, y) -> State
        cache_size : int
            Number of states to cache
        """
        super().__init__(generator, cache_size)
        self.width = width
        self.height = height
    
    def get(self, x: int, y: int) -> State:
        """Get state at (x, y)"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Index ({x}, {y}) out of range")
        
        return self._get_cached(x, y)
    
    def shape(self) -> Tuple[int, int]:
        """Get field shape"""
        return (self.width, self.height)
    
    def materialize_region(
        self,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int
    ) -> ChannelField2D:
        """
        Materialize (compute and store) a region
        
        Parameters
        ----------
        x_start, x_end : int
            X range [x_start, x_end)
        y_start, y_end : int
            Y range [y_start, y_end)
            
        Returns
        -------
        region : ChannelField2D
            Materialized region
        """
        region_width = x_end - x_start
        region_height = y_end - y_start
        
        region = ChannelField2D(region_width, region_height)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                state = self.get(x, y)
                region.set(x - x_start, y - y_start, state)
        
        return region
    
    def materialize_all(self) -> ChannelField2D:
        """
        Materialize entire field
        
        Warning: May require significant memory
        """
        return self.materialize_region(0, self.width, 0, self.height)
    
    def stream_rows(self, batch_size: int = 1) -> Iterator[ChannelField2D]:
        """
        Stream field row by row
        
        Parameters
        ----------
        batch_size : int
            Number of rows per batch
            
        Yields
        ------
        batch : ChannelField2D
            Batch of rows
        """
        for y_start in range(0, self.height, batch_size):
            y_end = min(y_start + batch_size, self.height)
            yield self.materialize_region(0, self.width, y_start, y_end)
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory if materialized"""
        # 2 bytes per pixel (i and q)
        bytes_total = self.width * self.height * 2
        return bytes_total / (1024 * 1024)


class LazyChannelField3D(LazyChannelField):
    """
    Lazy 3D channel field
    
    Examples
    --------
    >>> # Create volumetric field
    >>> def sphere_state(x, y, z):
    ...     r_sq = (x - 50)**2 + (y - 50)**2 + (z - 50)**2
    ...     return State(i=int(r_sq < 2500), q=int(r_sq < 1600))
    >>> 
    >>> field = LazyChannelField3D(100, 100, 100, sphere_state)
    >>> state = field.get(50, 50, 50)
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        generator: Callable[[int, int, int], State],
        cache_size: int = 10000
    ):
        """
        Parameters
        ----------
        width, height, depth : int
            Field dimensions
        generator : Callable[[int, int, int], State]
            Function (x, y, z) -> State
        cache_size : int
            Number of states to cache
        """
        super().__init__(generator, cache_size)
        self.width = width
        self.height = height
        self.depth = depth
    
    def get(self, x: int, y: int, z: int) -> State:
        """Get state at (x, y, z)"""
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth):
            raise IndexError(f"Index ({x}, {y}, {z}) out of range")
        
        return self._get_cached(x, y, z)
    
    def shape(self) -> Tuple[int, int, int]:
        """Get field shape"""
        return (self.width, self.height, self.depth)
    
    def materialize_slice(self, axis: str, index: int) -> ChannelField2D:
        """
        Materialize a 2D slice
        
        Parameters
        ----------
        axis : str
            'x', 'y', or 'z'
        index : int
            Slice position
            
        Returns
        -------
        slice_field : ChannelField2D
            Materialized slice
        """
        if axis == 'x':
            if not 0 <= index < self.width:
                raise IndexError(f"X index {index} out of range")
            slice_field = ChannelField2D(self.depth, self.height)
            for z in range(self.depth):
                for y in range(self.height):
                    state = self.get(index, y, z)
                    slice_field.set(z, y, state)
        
        elif axis == 'y':
            if not 0 <= index < self.height:
                raise IndexError(f"Y index {index} out of range")
            slice_field = ChannelField2D(self.width, self.depth)
            for z in range(self.depth):
                for x in range(self.width):
                    state = self.get(x, index, z)
                    slice_field.set(x, z, state)
        
        elif axis == 'z':
            if not 0 <= index < self.depth:
                raise IndexError(f"Z index {index} out of range")
            slice_field = ChannelField2D(self.width, self.height)
            for y in range(self.height):
                for x in range(self.width):
                    state = self.get(x, y, index)
                    slice_field.set(x, y, state)
        
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        return slice_field
    
    def stream_slices(self, axis: str = 'z', batch_size: int = 1) -> Iterator[ChannelField2D]:
        """
        Stream field slice by slice
        
        Parameters
        ----------
        axis : str
            Axis to slice along ('x', 'y', 'z')
        batch_size : int
            Number of slices per batch
            
        Yields
        ------
        slice : ChannelField2D
            Batch of slices
        """
        if axis == 'z':
            num_slices = self.depth
        elif axis == 'y':
            num_slices = self.height
        elif axis == 'x':
            num_slices = self.width
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        for i in range(num_slices):
            yield self.materialize_slice(axis, i)
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory if materialized"""
        # 2 bytes per voxel (i and q)
        bytes_total = self.width * self.height * self.depth * 2
        return bytes_total / (1024 * 1024)


# Utility functions

def materialize_region(
    lazy_field: Union[LazyChannelField2D, LazyChannelField3D],
    *region_spec
) -> Union[ChannelField2D, ChannelField3D]:
    """
    Materialize a region of a lazy field
    
    Parameters
    ----------
    lazy_field : LazyChannelField
        Lazy field to materialize from
    *region_spec : int
        Region specification (x_start, x_end, y_start, y_end[, z_start, z_end])
        
    Returns
    -------
    field : ChannelField
        Materialized field
    """
    if isinstance(lazy_field, LazyChannelField2D):
        if len(region_spec) != 4:
            raise ValueError("2D region requires 4 coordinates")
        return lazy_field.materialize_region(*region_spec)
    elif isinstance(lazy_field, LazyChannelField3D):
        raise NotImplementedError("3D region materialization not yet implemented")
    else:
        raise TypeError("Unknown lazy field type")


def stream_field(
    lazy_field: Union[LazyChannelField2D, LazyChannelField3D],
    axis: str = 'row',
    batch_size: int = 1
) -> Iterator:
    """
    Stream a lazy field for processing
    
    Parameters
    ----------
    lazy_field : LazyChannelField
        Lazy field to stream
    axis : str
        Stream direction ('row' for 2D, 'x'/'y'/'z' for 3D)
    batch_size : int
        Batch size
        
    Yields
    ------
    batch : ChannelField
        Batch of field data
    """
    if isinstance(lazy_field, LazyChannelField2D):
        yield from lazy_field.stream_rows(batch_size)
    elif isinstance(lazy_field, LazyChannelField3D):
        yield from lazy_field.stream_slices(axis, batch_size)
    else:
        raise TypeError("Unknown lazy field type")


# Factory functions for common patterns

def create_procedural_field_2d(
    width: int,
    height: int,
    pattern: str,
    **kwargs
) -> LazyChannelField2D:
    """
    Create procedural field with common patterns
    
    Parameters
    ----------
    width, height : int
        Dimensions
    pattern : str
        Pattern type:
        - 'checkerboard': Alternating states
        - 'gradient': Gradient pattern
        - 'noise': Random noise
        - 'sine': Sine wave pattern
        - 'mandelbrot': Mandelbrot set
    **kwargs
        Pattern-specific parameters
        
    Returns
    -------
    field : LazyChannelField2D
        Lazy field with pattern
    """
    if pattern == 'checkerboard':
        size = kwargs.get('size', 10)
        def generator(x, y):
            return State(i=((x // size) + (y // size)) % 2, q=1)
    
    elif pattern == 'gradient':
        def generator(x, y):
            i_val = int(x > width / 2)
            q_val = int(y > height / 2)
            return State(i=i_val, q=q_val)
    
    elif pattern == 'noise':
        import random
        def generator(x, y):
            random.seed(x * 10000 + y)
            return State(i=random.randint(0, 1), q=random.randint(0, 1))
    
    elif pattern == 'sine':
        frequency = kwargs.get('frequency', 0.1)
        def generator(x, y):
            val = np.sin(x * frequency) * np.sin(y * frequency)
            return State(i=int(val > 0), q=int(val > 0.5))
    
    elif pattern == 'mandelbrot':
        max_iter = kwargs.get('max_iter', 100)
        scale = kwargs.get('scale', 200)
        def generator(x, y):
            c = complex((x - width/2) / scale, (y - height/2) / scale)
            z = 0
            for i in range(max_iter):
                z = z*z + c
                if abs(z) > 2:
                    return State(i=1, q=int(i > max_iter/2))
            return State(i=0, q=0)
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return LazyChannelField2D(width, height, generator)