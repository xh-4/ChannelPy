"""
Nested channel states (hierarchical structure)

Nested states represent hierarchical channel structures where
each level is itself a channel state. This enables multi-scale
or multi-resolution analysis.
"""

from typing import Dict, List, Optional, Union, Tuple
from .state import State, StateArray, EMPTY, DELTA, PHI, PSI


class NestedState:
    """
    Nested channel state with multiple levels
    
    Each level is a State, forming a tree structure.
    Levels are numbered starting from 0.
    
    Examples
    --------
    >>> # Two-level nested state
    >>> state = NestedState(
    ...     level0=PSI,
    ...     level1=PHI
    ... )
    >>> print(state)
    ψ.φ
    >>> state.depth
    1
    >>> state.get_level(0)
    State(i=1, q=1)
    
    >>> # Three-level nested state
    >>> state = NestedState(
    ...     level0=PSI,
    ...     level1=DELTA,
    ...     level2=PHI
    ... )
    >>> state.path_string()
    'ψ.δ.φ'
    >>> state.all_psi()
    False
    """
    
    def __init__(self, **levels):
        """
        Initialize nested state
        
        Parameters
        ----------
        **levels : State
            Keyword arguments level0, level1, level2, etc.
            Must be contiguous starting from level0
        
        Raises
        ------
        ValueError
            If levels are not contiguous or don't start at 0
        TypeError
            If level values are not State objects
        """
        self._levels: Dict[int, State] = {}
        level_nums = []
        
        for key, value in levels.items():
            # Validate key format
            if not key.startswith('level'):
                raise ValueError(f"Keys must be 'levelN', got '{key}'")
            
            # Extract level number
            try:
                level_num = int(key[5:])
            except ValueError:
                raise ValueError(f"Invalid level key: '{key}'")
            
            # Validate value type
            if not isinstance(value, State):
                raise TypeError(f"Level values must be State, got {type(value)}")
            
            level_nums.append(level_num)
            self._levels[level_num] = value
        
        # Validate contiguous levels starting from 0
        if level_nums:
            level_nums.sort()
            if level_nums[0] != 0:
                raise ValueError("Levels must start at 0")
            for i in range(len(level_nums) - 1):
                if level_nums[i+1] != level_nums[i] + 1:
                    raise ValueError(f"Levels must be contiguous, missing level {level_nums[i]+1}")
        
        self._depth = max(level_nums) if level_nums else -1
    
    @property
    def depth(self) -> int:
        """Maximum level index"""
        return self._depth
    
    @property
    def num_levels(self) -> int:
        """Number of levels"""
        return self._depth + 1
    
    @property
    def total_bits(self) -> int:
        """Total number of bits across all levels"""
        return 2 * self.num_levels
    
    @property
    def total_states(self) -> int:
        """Total number of possible nested states"""
        return 4 ** self.num_levels
    
    def get_level(self, level: int) -> State:
        """
        Get state at specific level
        
        Parameters
        ----------
        level : int
            Level index
        
        Returns
        -------
        state : State
            State at that level
        
        Raises
        ------
        IndexError
            If level doesn't exist
        """
        if level not in self._levels:
            raise IndexError(f"Level {level} does not exist")
        return self._levels[level]
    
    def set_level(self, level: int, state: State):
        """
        Set state at specific level
        
        Can extend depth by 1 if setting level = depth + 1
        
        Parameters
        ----------
        level : int
            Level index
        state : State
            New state for that level
        
        Raises
        ------
        IndexError
            If trying to skip levels
        """
        if level < 0:
            raise IndexError(f"Level must be non-negative, got {level}")
        if level > self._depth + 1:
            raise IndexError(
                f"Cannot skip levels. Current depth is {self._depth}, "
                f"can only add level {self._depth + 1}, got {level}"
            )
        
        self._levels[level] = state
        if level > self._depth:
            self._depth = level
    
    def all_levels(self) -> List[State]:
        """
        Return list of all level states in order
        
        Returns
        -------
        levels : List[State]
            States from level 0 to depth
        """
        return [self._levels[i] for i in range(self.num_levels)]
    
    def all_psi(self) -> bool:
        """Check if all levels are ψ (fully resonant)"""
        return all(s == PSI for s in self.all_levels())
    
    def all_empty(self) -> bool:
        """Check if all levels are ∅"""
        return all(s == EMPTY for s in self.all_levels())
    
    def any_empty(self) -> bool:
        """Check if any level is ∅"""
        return any(s == EMPTY for s in self.all_levels())
    
    def any_psi(self) -> bool:
        """Check if any level is ψ"""
        return any(s == PSI for s in self.all_levels())
    
    def count_psi(self) -> int:
        """Count number of ψ levels"""
        return sum(1 for s in self.all_levels() if s == PSI)
    
    def count_empty(self) -> int:
        """Count number of ∅ levels"""
        return sum(1 for s in self.all_levels() if s == EMPTY)
    
    def path_string(self, separator: str = '.') -> str:
        """
        Return path as string
        
        Parameters
        ----------
        separator : str
            Separator between levels (default: '.')
        
        Returns
        -------
        path : str
            String representation of path
        
        Examples
        --------
        >>> state = NestedState(level0=PSI, level1=PHI, level2=DELTA)
        >>> state.path_string()
        'ψ.φ.δ'
        >>> state.path_string(separator='/')
        'ψ/φ/δ'
        """
        return separator.join(str(s) for s in self.all_levels())
    
    def path_matches(self, pattern: str, separator: str = '.') -> bool:
        """
        Check if path matches pattern
        
        Pattern can include wildcards (*)
        
        Parameters
        ----------
        pattern : str
            Pattern string (use * for wildcard)
        separator : str
            Level separator
        
        Returns
        -------
        matches : bool
            True if path matches pattern
        
        Examples
        --------
        >>> state = NestedState(level0=PSI, level1=PHI)
        >>> state.path_matches("ψ.*")
        True
        >>> state.path_matches("*.φ")
        True
        >>> state.path_matches("δ.*")
        False
        """
        pattern_parts = pattern.split(separator)
        path_parts = self.path_string(separator).split(separator)
        
        if len(pattern_parts) != len(path_parts):
            return False
        
        for pattern_part, path_part in zip(pattern_parts, path_parts):
            if pattern_part != '*' and pattern_part != path_part:
                return False
        
        return True
    
    def to_tuple(self) -> Tuple[State, ...]:
        """Convert to tuple of states"""
        return tuple(self.all_levels())
    
    def to_dict(self) -> Dict[int, State]:
        """Convert to dictionary"""
        return self._levels.copy()
    
    def __eq__(self, other) -> bool:
        """
        Equality comparison
        
        Can compare with:
        - Another NestedState
        - A string path (e.g., "ψ.φ.δ")
        - A tuple of states
        """
        if isinstance(other, str):
            return self.path_string() == other
        elif isinstance(other, NestedState):
            return self._levels == other._levels
        elif isinstance(other, tuple):
            return self.to_tuple() == other
        return False
    
    def __hash__(self) -> int:
        """Hash for use in sets/dicts"""
        return hash(self.to_tuple())
    
    def __str__(self) -> str:
        """String representation"""
        return self.path_string()
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        levels_str = ', '.join(
            f"level{i}={repr(s)}" 
            for i, s in self._levels.items()
        )
        return f"NestedState({levels_str})"
    
    def __len__(self) -> int:
        """Number of levels"""
        return self.num_levels
    
    @classmethod
    def from_path(cls, path: str, separator: str = '.') -> 'NestedState':
        """
        Create nested state from path string
        
        Parameters
        ----------
        path : str
            Path string (e.g., "ψ.φ.δ")
        separator : str
            Level separator
        
        Returns
        -------
        state : NestedState
            Nested state corresponding to path
        
        Examples
        --------
        >>> state = NestedState.from_path("ψ.φ.δ")
        >>> state.depth
        2
        >>> state.get_level(0)
        State(i=1, q=1)
        """
        parts = path.split(separator)
        levels = {}
        for i, part in enumerate(parts):
            levels[f'level{i}'] = State.from_name(part)
        return cls(**levels)
    
    @classmethod
    def from_tuple(cls, states: Tuple[State, ...]) -> 'NestedState':
        """Create nested state from tuple of states"""
        levels = {f'level{i}': state for i, state in enumerate(states)}
        return cls(**levels)
    
    @classmethod
    def from_list(cls, states: List[State]) -> 'NestedState':
        """Create nested state from list of states"""
        levels = {f'level{i}': state for i, state in enumerate(states)}
        return cls(**levels)


class NestedStateArray:
    """
    Array of nested states (all same depth)
    
    Examples
    --------
    >>> # Create array of nested states
    >>> states = [
    ...     NestedState(level0=PSI, level1=PHI),
    ...     NestedState(level0=DELTA, level1=EMPTY),
    ...     NestedState(level0=PSI, level1=PSI)
    ... ]
    >>> array = NestedStateArray.from_states(states)
    >>> len(array)
    3
    >>> array.depth
    1
    """
    
    def __init__(self, level_arrays: Dict[int, StateArray]):
        """
        Initialize nested state array
        
        Parameters
        ----------
        level_arrays : Dict[int, StateArray]
            Dictionary mapping level index to StateArray
            All arrays must have same shape
        """
        if not level_arrays:
            raise ValueError("Must provide at least one level")
        
        # Validate all arrays have same shape
        shapes = [arr.shape for arr in level_arrays.values()]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("All level arrays must have same shape")
        
        self._level_arrays = level_arrays
        self._depth = max(level_arrays.keys())
        self._shape = shapes[0]
    
    @classmethod
    def from_states(cls, states: List[NestedState]) -> 'NestedStateArray':
        """
        Create from list of NestedState objects
        
        All states must have same depth
        """
        if not states:
            raise ValueError("Must provide at least one state")
        
        # Check all same depth
        depth = states[0].depth
        if not all(s.depth == depth for s in states):
            raise ValueError("All states must have same depth")
        
        # Extract level arrays
        level_arrays = {}
        for level in range(depth + 1):
            level_states = [s.get_level(level) for s in states]
            level_arrays[level] = StateArray.from_states(level_states)
        
        return cls(level_arrays)
    
    @property
    def depth(self) -> int:
        """Maximum level index"""
        return self._depth
    
    @property
    def num_levels(self) -> int:
        """Number of levels"""
        return self._depth + 1
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of array"""
        return self._shape
    
    def __len__(self) -> int:
        """Length of array"""
        return self._shape[0] if self._shape else 0
    
    def __getitem__(self, idx) -> Union[NestedState, 'NestedStateArray']:
        """Get nested state(s) by index"""
        if isinstance(idx, (int, np.integer)):
            # Single element
            levels = {}
            for level, arr in self._level_arrays.items():
                levels[f'level{level}'] = arr[idx]
            return NestedState(**levels)
        else:
            # Slice or array
            level_arrays = {}
            for level, arr in self._level_arrays.items():
                level_arrays[level] = arr[idx]
            return NestedStateArray(level_arrays)
    
    def get_level_array(self, level: int) -> StateArray:
        """Get StateArray for specific level"""
        if level not in self._level_arrays:
            raise IndexError(f"Level {level} does not exist")
        return self._level_arrays[level]
    
    def to_states(self) -> List[NestedState]:
        """Convert to list of NestedState objects"""
        return [self[i] for i in range(len(self))]
    
    def all_paths(self) -> List[str]:
        """Get list of all path strings"""
        return [state.path_string() for state in self.to_states()]