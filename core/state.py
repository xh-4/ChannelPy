"""
Core channel state representation

A State is a two-bit structure:
- i (presence bit): Is the element present?
- q (membership bit): Is the element a member of the set?

Four possible states:
- ∅ (EMPTY):  i=0, q=0 - Absent
- δ (DELTA):  i=1, q=0 - Present but not member (puncture)
- φ (PHI):    i=0, q=1 - Not present but expected (hole)
- ψ (PSI):    i=1, q=1 - Present and member (resonant)
"""

from typing import Union, Tuple, List, Optional
import numpy as np
from dataclasses import dataclass


class State:
    """
    Channel algebra state with two bits: i (presence) and q (membership)
    
    The State class is immutable and hashable, making it suitable for
    use in sets and as dictionary keys.
    
    Parameters
    ----------
    i : int
        Presence bit (0 or 1)
    q : int
        Membership bit (0 or 1)
    
    Examples
    --------
    >>> state = State(i=1, q=1)  # Create ψ state
    >>> print(state)
    ψ
    >>> state == PSI
    True
    >>> state.i
    1
    >>> state.q
    1
    
    >>> # Create from name
    >>> phi_state = State.from_name('phi')
    >>> print(phi_state)
    φ
    
    >>> # Convert to integer
    >>> state.to_int()
    3
    >>> State.from_int(3) == PSI
    True
    """
    
    __slots__ = ('_i', '_q', '_hash')
    
    def __init__(self, i: int, q: int):
        """
        Initialize channel state
        
        Raises
        ------
        ValueError
            If i or q not in {0, 1}
        """
        if i not in (0, 1):
            raise ValueError(f"i must be 0 or 1, got {i}")
        if q not in (0, 1):
            raise ValueError(f"q must be 0 or 1, got {q}")
        
        object.__setattr__(self, '_i', i)
        object.__setattr__(self, '_q', q)
        object.__setattr__(self, '_hash', hash((i, q)))
    
    @property
    def i(self) -> int:
        """Presence bit (0 or 1)"""
        return self._i
    
    @property
    def q(self) -> int:
        """Membership bit (0 or 1)"""
        return self._q
    
    def __setattr__(self, name, value):
        """Prevent modification (immutable)"""
        raise AttributeError("State objects are immutable")
    
    def __delattr__(self, name):
        """Prevent deletion"""
        raise AttributeError("State objects are immutable")
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, State):
            return False
        return self._i == other._i and self._q == other._q
    
    def __hash__(self) -> int:
        """Hash for use in sets and dicts"""
        return self._hash
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return f"State(i={self._i}, q={self._q})"
    
    def __str__(self) -> str:
        """Unicode symbol representation"""
        symbols = {
            (0, 0): '∅',
            (1, 0): 'δ',
            (0, 1): 'φ',
            (1, 1): 'ψ'
        }
        return symbols[(self._i, self._q)]
    
    def __lt__(self, other) -> bool:
        """
        Less than comparison (for sorting)
        Order: ∅ < δ < φ < ψ
        """
        if not isinstance(other, State):
            return NotImplemented
        return self.to_int() < other.to_int()
    
    def __le__(self, other) -> bool:
        """Less than or equal"""
        if not isinstance(other, State):
            return NotImplemented
        return self.to_int() <= other.to_int()
    
    def __gt__(self, other) -> bool:
        """Greater than"""
        if not isinstance(other, State):
            return NotImplemented
        return self.to_int() > other.to_int()
    
    def __ge__(self, other) -> bool:
        """Greater than or equal"""
        if not isinstance(other, State):
            return NotImplemented
        return self.to_int() >= other.to_int()
    
    def to_bits(self) -> Tuple[int, int]:
        """
        Return as (i, q) tuple
        
        Returns
        -------
        bits : Tuple[int, int]
            (presence_bit, membership_bit)
        """
        return (self._i, self._q)
    
    def to_int(self) -> int:
        """
        Convert to integer 0-3
        
        Mapping:
        - ∅ (0, 0) → 0
        - φ (0, 1) → 1
        - δ (1, 0) → 2
        - ψ (1, 1) → 3
        
        Returns
        -------
        value : int
            Integer representation in range [0, 3]
        """
        return self._i * 2 + self._q
    
    @classmethod
    def from_int(cls, value: int) -> 'State':
        """
        Create state from integer 0-3
        
        Parameters
        ----------
        value : int
            Integer in range [0, 3]
        
        Returns
        -------
        state : State
            Corresponding state
        
        Raises
        ------
        ValueError
            If value not in [0, 3]
        """
        if not 0 <= value <= 3:
            raise ValueError(f"Value must be in [0, 3], got {value}")
        i = value // 2
        q = value % 2
        return cls(i, q)
    
    @classmethod
    def from_name(cls, name: str) -> 'State':
        """
        Create state from name or symbol
        
        Parameters
        ----------
        name : str
            State name: 'empty', 'delta', 'phi', 'psi' or symbols '∅', 'δ', 'φ', 'ψ'
        
        Returns
        -------
        state : State
            Corresponding state
        
        Raises
        ------
        ValueError
            If name not recognized
        
        Examples
        --------
        >>> State.from_name('psi')
        State(i=1, q=1)
        >>> State.from_name('ψ')
        State(i=1, q=1)
        """
        name_map = {
            'empty': (0, 0), '∅': (0, 0), 'null': (0, 0),
            'delta': (1, 0), 'δ': (1, 0), 'puncture': (1, 0),
            'phi': (0, 1), 'φ': (0, 1), 'hole': (0, 1),
            'psi': (1, 1), 'ψ': (1, 1), 'resonant': (1, 1),
        }
        
        name_lower = name.lower().strip()
        if name_lower not in name_map:
            raise ValueError(
                f"Unknown state name: '{name}'. "
                f"Valid names: {list(name_map.keys())}"
            )
        
        i, q = name_map[name_lower]
        return cls(i, q)
    
    def to_complex(self) -> complex:
        """
        Convert to complex number: i + iq
        
        Useful for quantum/phase space interpretations
        
        Returns
        -------
        c : complex
            Complex representation
        """
        return self._i + 1j * self._q
    
    def to_vector(self) -> np.ndarray:
        """
        Convert to 2D vector [i, q]
        
        Returns
        -------
        vec : np.ndarray
            2D vector representation
        """
        return np.array([self._i, self._q], dtype=np.int8)
    
    def is_present(self) -> bool:
        """Check if element is present (i=1)"""
        return self._i == 1
    
    def is_member(self) -> bool:
        """Check if element is member (q=1)"""
        return self._q == 1
    
    def is_empty(self) -> bool:
        """Check if state is ∅"""
        return self._i == 0 and self._q == 0
    
    def is_delta(self) -> bool:
        """Check if state is δ (puncture)"""
        return self._i == 1 and self._q == 0
    
    def is_phi(self) -> bool:
        """Check if state is φ (hole)"""
        return self._i == 0 and self._q == 1
    
    def is_psi(self) -> bool:
        """Check if state is ψ (resonant)"""
        return self._i == 1 and self._q == 1


# Pre-defined state constants
EMPTY = State(0, 0)  # ∅
DELTA = State(1, 0)  # δ
PHI = State(0, 1)    # φ
PSI = State(1, 1)    # ψ


class StateArray:
    """
    Efficient array of channel states using numpy
    
    StateArray stores states as two separate bit arrays (i and q),
    enabling vectorized operations and efficient memory usage.
    
    Parameters
    ----------
    i : array-like
        Array of presence bits
    q : array-like
        Array of membership bits
    
    Examples
    --------
    >>> # Create from bits
    >>> states = StateArray.from_bits(i=[1,0,1,1], q=[1,1,0,1])
    >>> len(states)
    4
    >>> states[0]
    State(i=1, q=1)
    
    >>> # Create from State objects
    >>> state_list = [PSI, EMPTY, DELTA, PHI]
    >>> states = StateArray.from_states(state_list)
    >>> states.to_strings()
    array(['ψ', '∅', 'δ', 'φ'], dtype='<U1')
    
    >>> # Count states
    >>> states.count_by_state()
    {State(i=0, q=0): 1, State(i=1, q=0): 1, State(i=0, q=1): 1, State(i=1, q=1): 1}
    """
    
    def __init__(self, i: np.ndarray, q: np.ndarray):
        """
        Initialize state array
        
        Raises
        ------
        ValueError
            If i and q have different shapes
        """
        i = np.asarray(i, dtype=np.int8)
        q = np.asarray(q, dtype=np.int8)
        
        if i.shape != q.shape:
            raise ValueError(
                f"i and q must have same shape, got {i.shape} and {q.shape}"
            )
        
        # Validate bit values
        if not np.all((i == 0) | (i == 1)):
            raise ValueError("i array must contain only 0 and 1")
        if not np.all((q == 0) | (q == 1)):
            raise ValueError("q array must contain only 0 and 1")
        
        self._i = i
        self._q = q
        self._shape = i.shape
    
    @classmethod
    def from_bits(cls, i, q) -> 'StateArray':
        """
        Create from bit arrays
        
        Parameters
        ----------
        i : array-like
            Presence bits
        q : array-like
            Membership bits
        """
        return cls(i, q)
    
    @classmethod
    def from_states(cls, states: List[State]) -> 'StateArray':
        """
        Create from list of State objects
        
        Parameters
        ----------
        states : List[State]
            List of states
        """
        i = np.array([s.i for s in states], dtype=np.int8)
        q = np.array([s.q for s in states], dtype=np.int8)
        return cls(i, q)
    
    @classmethod
    def from_ints(cls, values: np.ndarray) -> 'StateArray':
        """
        Create from integer array (0-3)
        
        Parameters
        ----------
        values : array-like
            Integer values in range [0, 3]
        """
        values = np.asarray(values, dtype=np.int8)
        i = values // 2
        q = values % 2
        return cls(i, q)
    
    @classmethod
    def empty(cls, shape: Tuple[int, ...]) -> 'StateArray':
        """
        Create array of EMPTY states
        
        Parameters
        ----------
        shape : tuple
            Shape of array
        """
        i = np.zeros(shape, dtype=np.int8)
        q = np.zeros(shape, dtype=np.int8)
        return cls(i, q)
    
    @classmethod
    def full(cls, shape: Tuple[int, ...], state: State) -> 'StateArray':
        """
        Create array filled with specific state
        
        Parameters
        ----------
        shape : tuple
            Shape of array
        state : State
            State to fill with
        """
        i = np.full(shape, state.i, dtype=np.int8)
        q = np.full(shape, state.q, dtype=np.int8)
        return cls(i, q)
    
    @property
    def i(self) -> np.ndarray:
        """Presence bit array"""
        return self._i
    
    @property
    def q(self) -> np.ndarray:
        """Membership bit array"""
        return self._q
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of state array"""
        return self._shape
    
    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return len(self._shape)
    
    @property
    def size(self) -> int:
        """Total number of states"""
        return self._i.size
    
    def __len__(self) -> int:
        """Length of array (first dimension)"""
        return self._shape[0] if self._shape else 0
    
    def __getitem__(self, idx) -> Union[State, 'StateArray']:
        """
        Get state(s) by index
        
        Returns single State for integer index,
        StateArray for slice or array index
        """
        if isinstance(idx, (int, np.integer)):
            # Single element
            return State(int(self._i.flat[idx]), int(self._q.flat[idx]))
        else:
            # Slice or array
            return StateArray(self._i[idx], self._q[idx])
    
    def __setitem__(self, idx, value: Union[State, 'StateArray']):
        """Set state(s) by index"""
        if isinstance(value, State):
            self._i[idx] = value.i
            self._q[idx] = value.q
        elif isinstance(value, StateArray):
            self._i[idx] = value.i
            self._q[idx] = value.q
        else:
            raise TypeError(f"Value must be State or StateArray, got {type(value)}")
    
    def __iter__(self):
        """Iterate over states"""
        for i in range(len(self)):
            yield self[i]
    
    def __repr__(self) -> str:
        """Developer-friendly representation"""
        return f"StateArray(shape={self.shape}, i={self._i}, q={self._q})"
    
    def __str__(self) -> str:
        """String representation"""
        if self.size <= 10:
            states_str = ', '.join(str(s) for s in self)
            return f"[{states_str}]"
        else:
            first = ', '.join(str(self[i]) for i in range(3))
            last = ', '.join(str(self[i]) for i in range(-3, 0))
            return f"[{first}, ..., {last}]"
    
    def to_ints(self) -> np.ndarray:
        """
        Convert to integer array (0-3)
        
        Returns
        -------
        ints : np.ndarray
            Integer array with same shape
        """
        return self._i * 2 + self._q
    
    def to_strings(self) -> np.ndarray:
        """
        Convert to string array of symbols
        
        Returns
        -------
        strings : np.ndarray
            String array of symbols
        """
        symbols = np.array(['∅', 'φ', 'δ', 'ψ'])
        return symbols[self.to_ints()]
    
    def to_states(self) -> List[State]:
        """
        Convert to list of State objects
        
        Returns
        -------
        states : List[State]
            List of states
        """
        return [State(int(i), int(q)) for i, q in zip(self._i.flat, self._q.flat)]
    
    def count_by_state(self) -> dict:
        """
        Count occurrences of each state
        
        Returns
        -------
        counts : dict
            Dictionary mapping State to count
        """
        ints = self.to_ints()
        return {
            EMPTY: int(np.sum(ints == 0)),
            PHI: int(np.sum(ints == 1)),
            DELTA: int(np.sum(ints == 2)),
            PSI: int(np.sum(ints == 3))
        }
    
    def count_present(self) -> int:
        """Count states where i=1"""
        return int(np.sum(self._i == 1))
    
    def count_member(self) -> int:
        """Count states where q=1"""
        return int(np.sum(self._q == 1))
    
    def fraction_psi(self) -> float:
        """Fraction of states that are ψ"""
        return float(np.sum((self._i == 1) & (self._q == 1)) / self.size)
    
    def fraction_empty(self) -> float:
        """Fraction of states that are ∅"""
        return float(np.sum((self._i == 0) & (self._q == 0)) / self.size)
    
    def reshape(self, shape: Tuple[int, ...]) -> 'StateArray':
        """
        Reshape state array
        
        Parameters
        ----------
        shape : tuple
            New shape
        
        Returns
        -------
        reshaped : StateArray
            Reshaped array
        """
        return StateArray(self._i.reshape(shape), self._q.reshape(shape))
    
    def flatten(self) -> 'StateArray':
        """Flatten to 1D array"""
        return StateArray(self._i.flatten(), self._q.flatten())
    
    def copy(self) -> 'StateArray':
        """Create a copy"""
        return StateArray(self._i.copy(), self._q.copy())