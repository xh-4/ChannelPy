"""
Core channel field classes for spatial analysis
"""

from typing import Callable, Optional, Tuple, List, Union, Iterator
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..core.operations import gate, admit, overlay, weave


class BoundaryCondition(Enum):
    """Boundary condition types for field edges"""
    ZERO = "zero"          # Pad with EMPTY states
    PERIODIC = "periodic"  # Wrap around (torus topology)
    REFLECT = "reflect"    # Mirror at boundaries
    EXTEND = "extend"      # Extend edge values
    CONSTANT = "constant"  # Use constant value


class ChannelField(ABC):
    """
    Abstract base class for channel fields
    
    A field is a spatial domain where each point has a channel state
    """
    
    @abstractmethod
    def get(self, *coords) -> State:
        """Get state at coordinates"""
        pass
    
    @abstractmethod
    def set(self, *coords, state: State):
        """Set state at coordinates"""
        pass
    
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Get field dimensions"""
        pass
    
    @abstractmethod
    def copy(self) -> 'ChannelField':
        """Create deep copy"""
        pass


class ChannelField1D(ChannelField):
    """
    1D channel field (line)
    
    Examples
    --------
    >>> field = ChannelField1D(length=100)
    >>> field.set(50, PSI)
    >>> state = field.get(50)
    >>> print(state)
    ψ
    """
    
    def __init__(self, length: int, default_state: State = EMPTY):
        """
        Parameters
        ----------
        length : int
            Number of points
        default_state : State
            Default state for all points
        """
        self.length = length
        self.default_state = default_state
        
        # Store as i and q arrays for efficiency
        self._i = np.full(length, default_state.i, dtype=np.int8)
        self._q = np.full(length, default_state.q, dtype=np.int8)
    
    def get(self, x: int) -> State:
        """Get state at position x"""
        if not 0 <= x < self.length:
            raise IndexError(f"Index {x} out of range [0, {self.length})")
        return State(int(self._i[x]), int(self._q[x]))
    
    def set(self, x: int, state: State):
        """Set state at position x"""
        if not 0 <= x < self.length:
            raise IndexError(f"Index {x} out of range [0, {self.length})")
        self._i[x] = state.i
        self._q[x] = state.q
    
    def shape(self) -> Tuple[int]:
        """Get field shape"""
        return (self.length,)
    
    def to_state_array(self) -> StateArray:
        """Convert to StateArray"""
        return StateArray(self._i.copy(), self._q.copy())
    
    @classmethod
    def from_state_array(cls, states: StateArray) -> 'ChannelField1D':
        """Create field from StateArray"""
        field = cls(len(states))
        field._i = states.i.copy()
        field._q = states.q.copy()
        return field
    
    @classmethod
    def from_function(
        cls, 
        length: int, 
        func: Callable[[int], State]
    ) -> 'ChannelField1D':
        """
        Create field from function
        
        Parameters
        ----------
        length : int
            Field length
        func : Callable[[int], State]
            Function x -> State
        """
        field = cls(length)
        for x in range(length):
            field.set(x, func(x))
        return field
    
    def copy(self) -> 'ChannelField1D':
        """Create deep copy"""
        new_field = ChannelField1D(self.length)
        new_field._i = self._i.copy()
        new_field._q = self._q.copy()
        return new_field
    
    def __len__(self) -> int:
        return self.length
    
    def __iter__(self) -> Iterator[State]:
        """Iterate over states"""
        for x in range(self.length):
            yield self.get(x)
    
    def count_states(self) -> dict:
        """Count occurrences of each state"""
        states = self.to_state_array()
        return states.count_by_state()


class ChannelField2D(ChannelField):
    """
    2D channel field (image/plane)
    
    Examples
    --------
    >>> # Create 100x100 field
    >>> field = ChannelField2D(width=100, height=100)
    >>> 
    >>> # Set a region
    >>> for x in range(40, 60):
    ...     for y in range(40, 60):
    ...         field.set(x, y, PSI)
    >>> 
    >>> # Get state
    >>> state = field.get(50, 50)
    >>> 
    >>> # Create from function
    >>> field = ChannelField2D.from_function(
    ...     width=100, height=100,
    ...     func=lambda x, y: State(i=int(x+y > 100), q=int(x*y > 2500))
    ... )
    """
    
    def __init__(
        self, 
        width: int, 
        height: int, 
        default_state: State = EMPTY
    ):
        """
        Parameters
        ----------
        width : int
            Width (x dimension)
        height : int
            Height (y dimension)
        default_state : State
            Default state for all points
        """
        self.width = width
        self.height = height
        self.default_state = default_state
        
        # Store as 2D arrays
        self._i = np.full((height, width), default_state.i, dtype=np.int8)
        self._q = np.full((height, width), default_state.q, dtype=np.int8)
    
    def get(self, x: int, y: int) -> State:
        """Get state at (x, y)"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Index ({x}, {y}) out of range")
        return State(int(self._i[y, x]), int(self._q[y, x]))
    
    def set(self, x: int, y: int, state: State):
        """Set state at (x, y)"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"Index ({x}, {y}) out of range")
        self._i[y, x] = state.i
        self._q[y, x] = state.q
    
    def shape(self) -> Tuple[int, int]:
        """Get field shape (width, height)"""
        return (self.width, self.height)
    
    def get_region(
        self, 
        x_start: int, 
        x_end: int, 
        y_start: int, 
        y_end: int
    ) -> 'ChannelField2D':
        """
        Extract rectangular region
        
        Parameters
        ----------
        x_start, x_end : int
            X range [x_start, x_end)
        y_start, y_end : int
            Y range [y_start, y_end)
        """
        region_width = x_end - x_start
        region_height = y_end - y_start
        
        region = ChannelField2D(region_width, region_height)
        region._i = self._i[y_start:y_end, x_start:x_end].copy()
        region._q = self._q[y_start:y_end, x_start:x_end].copy()
        
        return region
    
    def set_region(
        self, 
        x_start: int, 
        y_start: int, 
        region: 'ChannelField2D'
    ):
        """
        Set rectangular region
        
        Parameters
        ----------
        x_start, y_start : int
            Top-left corner of region
        region : ChannelField2D
            Region to insert
        """
        x_end = x_start + region.width
        y_end = y_start + region.height
        
        if x_end > self.width or y_end > self.height:
            raise ValueError("Region extends beyond field boundaries")
        
        self._i[y_start:y_end, x_start:x_end] = region._i
        self._q[y_start:y_end, x_start:x_end] = region._q
    
    @classmethod
    def from_function(
        cls,
        width: int,
        height: int,
        func: Callable[[int, int], State]
    ) -> 'ChannelField2D':
        """
        Create field from function
        
        Parameters
        ----------
        width, height : int
            Field dimensions
        func : Callable[[int, int], State]
            Function (x, y) -> State
        """
        field = cls(width, height)
        for y in range(height):
            for x in range(width):
                field.set(x, y, func(x, y))
        return field
    
    @classmethod
    def from_image(cls, image: np.ndarray, threshold: float = 0.5) -> 'ChannelField2D':
        """
        Create field from grayscale image
        
        Parameters
        ----------
        image : np.ndarray
            2D array with values in [0, 1]
        threshold : float
            Threshold for state encoding
        """
        height, width = image.shape
        field = cls(width, height)
        
        # Simple threshold encoding
        field._i = (image > threshold).astype(np.int8)
        field._q = (image > threshold * 1.5).astype(np.int8)
        
        return field
    
    def to_image(self, mode: str = 'i') -> np.ndarray:
        """
        Convert field to image
        
        Parameters
        ----------
        mode : str
            'i': Use i-bit
            'q': Use q-bit
            'both': Use i*2 + q (0-3 range)
            'psi': Only PSI states as 1
        """
        if mode == 'i':
            return self._i.astype(float)
        elif mode == 'q':
            return self._q.astype(float)
        elif mode == 'both':
            return (self._i * 2 + self._q).astype(float) / 3.0
        elif mode == 'psi':
            return (self._i & self._q).astype(float)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def copy(self) -> 'ChannelField2D':
        """Create deep copy"""
        new_field = ChannelField2D(self.width, self.height)
        new_field._i = self._i.copy()
        new_field._q = self._q.copy()
        return new_field
    
    def count_states(self) -> dict:
        """Count occurrences of each state"""
        flat_i = self._i.flatten()
        flat_q = self._q.flatten()
        states = StateArray(flat_i, flat_q)
        return states.count_by_state()
    
    def __iter__(self) -> Iterator[Tuple[int, int, State]]:
        """Iterate over (x, y, state) tuples"""
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y, self.get(x, y))


class ChannelField3D(ChannelField):
    """
    3D channel field (volume)
    
    Examples
    --------
    >>> # Create 50x50x50 volume
    >>> field = ChannelField3D(width=50, height=50, depth=50)
    >>> 
    >>> # Set a voxel
    >>> field.set(25, 25, 25, PSI)
    >>> 
    >>> # Create from function
    >>> field = ChannelField3D.from_function(
    ...     width=50, height=50, depth=50,
    ...     func=lambda x, y, z: State(
    ...         i=int(x**2 + y**2 + z**2 < 625),
    ...         q=int(x**2 + y**2 + z**2 < 400)
    ...     )
    ... )
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        default_state: State = EMPTY
    ):
        """
        Parameters
        ----------
        width : int
            Width (x dimension)
        height : int
            Height (y dimension)
        depth : int
            Depth (z dimension)
        default_state : State
            Default state for all points
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.default_state = default_state
        
        # Store as 3D arrays (z, y, x order for consistency)
        self._i = np.full((depth, height, width), default_state.i, dtype=np.int8)
        self._q = np.full((depth, height, width), default_state.q, dtype=np.int8)
    
    def get(self, x: int, y: int, z: int) -> State:
        """Get state at (x, y, z)"""
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth):
            raise IndexError(f"Index ({x}, {y}, {z}) out of range")
        return State(int(self._i[z, y, x]), int(self._q[z, y, x]))
    
    def set(self, x: int, y: int, z: int, state: State):
        """Set state at (x, y, z)"""
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth):
            raise IndexError(f"Index ({x}, {y}, {z}) out of range")
        self._i[z, y, x] = state.i
        self._q[z, y, x] = state.q
    
    def shape(self) -> Tuple[int, int, int]:
        """Get field shape (width, height, depth)"""
        return (self.width, self.height, self.depth)
    
    def get_slice(self, axis: str, index: int) -> ChannelField2D:
        """
        Get 2D slice along axis
        
        Parameters
        ----------
        axis : str
            'x', 'y', or 'z'
        index : int
            Slice index
        """
        if axis == 'x':
            if not 0 <= index < self.width:
                raise IndexError(f"X index {index} out of range")
            slice_field = ChannelField2D(self.depth, self.height)
            slice_field._i = self._i[:, :, index].copy()
            slice_field._q = self._q[:, :, index].copy()
        elif axis == 'y':
            if not 0 <= index < self.height:
                raise IndexError(f"Y index {index} out of range")
            slice_field = ChannelField2D(self.width, self.depth)
            slice_field._i = self._i[:, index, :].copy()
            slice_field._q = self._q[:, index, :].copy()
        elif axis == 'z':
            if not 0 <= index < self.depth:
                raise IndexError(f"Z index {index} out of range")
            slice_field = ChannelField2D(self.width, self.height)
            slice_field._i = self._i[index, :, :].copy()
            slice_field._q = self._q[index, :, :].copy()
        else:
            raise ValueError(f"Unknown axis: {axis}")
        
        return slice_field
    
    @classmethod
    def from_function(
        cls,
        width: int,
        height: int,
        depth: int,
        func: Callable[[int, int, int], State]
    ) -> 'ChannelField3D':
        """
        Create field from function
        
        Parameters
        ----------
        width, height, depth : int
            Field dimensions
        func : Callable[[int, int, int], State]
            Function (x, y, z) -> State
        """
        field = cls(width, height, depth)
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    field.set(x, y, z, func(x, y, z))
        return field
    
    def copy(self) -> 'ChannelField3D':
        """Create deep copy"""
        new_field = ChannelField3D(self.width, self.height, self.depth)
        new_field._i = self._i.copy()
        new_field._q = self._q.copy()
        return new_field
    
    def count_states(self) -> dict:
        """Count occurrences of each state"""
        flat_i = self._i.flatten()
        flat_q = self._q.flatten()
        states = StateArray(flat_i, flat_q)
        return states.count_by_state()
    
    def memory_size_mb(self) -> float:
        """Estimate memory size in MB"""
        # 2 bytes per voxel (i and q, both int8)
        total_voxels = self.width * self.height * self.depth
        bytes_total = total_voxels * 2
        return bytes_total / (1024 * 1024)


# Utility functions

def field_from_array(array: np.ndarray, encoding: str = 'threshold') -> Union[ChannelField1D, ChannelField2D, ChannelField3D]:
    """
    Create field from numpy array
    
    Parameters
    ----------
    array : np.ndarray
        1D, 2D, or 3D array
    encoding : str
        Encoding method:
        - 'threshold': Use simple thresholding
        - 'percentile': Use percentile-based encoding
        - 'adaptive': Use adaptive encoding
    
    Returns
    -------
    field : ChannelField
        Appropriate field type
    """
    if array.ndim == 1:
        length = array.shape[0]
        field = ChannelField1D(length)
        
        if encoding == 'threshold':
            threshold_i = np.median(array)
            threshold_q = np.percentile(array, 75)
        elif encoding == 'percentile':
            threshold_i = np.percentile(array, 50)
            threshold_q = np.percentile(array, 75)
        else:
            threshold_i = np.mean(array)
            threshold_q = np.mean(array) + 0.5 * np.std(array)
        
        field._i = (array > threshold_i).astype(np.int8)
        field._q = (array > threshold_q).astype(np.int8)
        
        return field
    
    elif array.ndim == 2:
        height, width = array.shape
        field = ChannelField2D(width, height)
        
        if encoding == 'threshold':
            threshold_i = np.median(array)
            threshold_q = np.percentile(array, 75)
        elif encoding == 'percentile':
            threshold_i = np.percentile(array, 50)
            threshold_q = np.percentile(array, 75)
        else:
            threshold_i = np.mean(array)
            threshold_q = np.mean(array) + 0.5 * np.std(array)
        
        field._i = (array > threshold_i).astype(np.int8)
        field._q = (array > threshold_q).astype(np.int8)
        
        return field
    
    elif array.ndim == 3:
        depth, height, width = array.shape
        field = ChannelField3D(width, height, depth)
        
        if encoding == 'threshold':
            threshold_i = np.median(array)
            threshold_q = np.percentile(array, 75)
        elif encoding == 'percentile':
            threshold_i = np.percentile(array, 50)
            threshold_q = np.percentile(array, 75)
        else:
            threshold_i = np.mean(array)
            threshold_q = np.mean(array) + 0.5 * np.std(array)
        
        field._i = (array > threshold_i).astype(np.int8)
        field._q = (array > threshold_q).astype(np.int8)
        
        return field
    
    else:
        raise ValueError(f"Array must be 1D, 2D, or 3D, got {array.ndim}D")