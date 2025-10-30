# Core API Reference

Complete reference for `channelpy.core` module.

## Table of Contents

1. [State](#state)
2. [StateArray](#statearray)
3. [Operations](#operations)
4. [NestedState](#nestedstate)
5. [ParallelChannels](#parallelchannels)
6. [Lattice](#lattice)

---

## State
```python
from channelpy.core import State
```

The fundamental building block of channel algebra.

### Class: `State`
```python
State(i: int, q: int)
```

A channel state with two bits:
- `i` (presence): Is the element present?
- `q` (membership): Does it satisfy membership criteria?

**Parameters:**
- `i` (int): Presence bit (0 or 1)
- `q` (int): Membership bit (0 or 1)

**Raises:**
- `ValueError`: If `i` or `q` not in {0, 1}

**Attributes:**
- `i` (int): Presence bit
- `q` (int): Membership bit

#### Example
```python
# Create states
empty = State(0, 0)  # ∅ - absent
delta = State(1, 0)  # δ - present but not member
phi = State(0, 1)    # φ - expected but not present
psi = State(1, 1)    # ψ - present and member

# String representation
print(empty)  # Output: ∅
print(psi)    # Output: ψ
```

### Constants

Pre-defined state constants for convenience:
```python
from channelpy.core import EMPTY, DELTA, PHI, PSI

EMPTY = State(0, 0)  # ∅
DELTA = State(1, 0)  # δ
PHI = State(0, 1)    # φ
PSI = State(1, 1)    # ψ
```

### Methods

#### `to_bits()`
```python
state.to_bits() -> Tuple[int, int]
```

Return state as (i, q) tuple.

**Returns:** Tuple of (i_bit, q_bit)

#### `to_int()`
```python
state.to_int() -> int
```

Convert state to integer 0-3.

**Returns:** Integer representation (0=∅, 1=φ, 2=δ, 3=ψ)

#### `to_complex()`
```python
state.to_complex() -> complex
```

Convert to complex number: `i + iq` (useful for phase space interpretations).

**Returns:** Complex representation

#### `from_int(value)` (classmethod)
```python
State.from_int(value: int) -> State
```

Create state from integer 0-3.

**Parameters:**
- `value` (int): Integer 0-3

**Returns:** State object

**Example:**
```python
state = State.from_int(3)  # Creates ψ
```

#### `from_name(name)` (classmethod)
```python
State.from_name(name: str) -> State
```

Create state from name.

**Parameters:**
- `name` (str): State name ('empty', '∅', 'delta', 'δ', 'phi', 'φ', 'psi', 'ψ')

**Returns:** State object

**Example:**
```python
state = State.from_name('psi')  # Creates ψ
state = State.from_name('ψ')    # Also creates ψ
```

---

## StateArray
```python
from channelpy.core import StateArray
```

Efficient array of channel states using NumPy.

### Class: `StateArray`
```python
StateArray(i: np.ndarray, q: np.ndarray)
```

Array of channel states for efficient batch processing.

**Parameters:**
- `i` (np.ndarray): Array of presence bits
- `q` (np.ndarray): Array of membership bits

**Attributes:**
- `i` (np.ndarray): Presence bit array
- `q` (np.ndarray): Membership bit array

#### Example
```python
import numpy as np
from channelpy.core import StateArray

# Create state array
states = StateArray(
    i=np.array([1, 0, 1, 1]),
    q=np.array([1, 1, 0, 1])
)

# Access individual states
print(states[0])  # ψ
print(states[1])  # φ
print(states[2])  # δ
```

### Methods

#### `from_bits(i, q)` (classmethod)
```python
StateArray.from_bits(i, q) -> StateArray
```

Create from bit arrays.

**Parameters:**
- `i`: Array-like of presence bits
- `q`: Array-like of membership bits

**Returns:** StateArray object

#### `from_states(states)` (classmethod)
```python
StateArray.from_states(states: List[State]) -> StateArray
```

Create from list of State objects.

**Parameters:**
- `states` (List[State]): List of State objects

**Returns:** StateArray object

**Example:**
```python
states_list = [PSI, PHI, DELTA, EMPTY]
array = StateArray.from_states(states_list)
```

#### `to_ints()`
```python
array.to_ints() -> np.ndarray
```

Convert to integer array 0-3.

**Returns:** NumPy array of integers

#### `to_strings()`
```python
array.to_strings() -> np.ndarray
```

Convert to string array.

**Returns:** NumPy array of strings ('∅', 'φ', 'δ', 'ψ')

#### `count_by_state()`
```python
array.count_by_state() -> Dict[State, int]
```

Count occurrences of each state.

**Returns:** Dictionary mapping State → count

**Example:**
```python
counts = states.count_by_state()
print(f"PSI count: {counts[PSI]}")
print(f"EMPTY count: {counts[EMPTY]}")
```

---

## Operations
```python
from channelpy.core import gate, admit, overlay, weave, comp
```

Core channel algebra operations.

### Function: `gate()`
```python
gate(state: Union[State, StateArray]) -> Union[State, StateArray]
```

Gate operation: Remove elements not validated by membership.

**Rule:** If q=0, set i=0

**Transformations:**
- ∅ → ∅
- δ → ∅ (puncture removed)
- φ → φ (hole preserved)
- ψ → ψ (resonant preserved)

**Example:**
```python
gate(DELTA)  # Returns EMPTY
gate(PSI)    # Returns PSI
```

### Function: `admit()`
```python
admit(state: Union[State, StateArray]) -> Union[State, StateArray]
```

Admit operation: Grant membership to present elements.

**Rule:** If i=1, set q=1

**Transformations:**
- ∅ → ∅
- δ → ψ (puncture validated)
- φ → φ (hole remains)
- ψ → ψ (already resonant)

**Example:**
```python
admit(DELTA)  # Returns PSI
admit(EMPTY)  # Returns EMPTY
```

### Function: `overlay()`
```python
overlay(state1: Union[State, StateArray], 
        state2: Union[State, StateArray]) -> Union[State, StateArray]
```

Overlay operation: Bitwise OR (union).

Takes maximum information from both states.

**Example:**
```python
overlay(DELTA, PHI)  # Returns PSI (i=1|0, q=0|1)
overlay(EMPTY, PSI)  # Returns PSI
```

### Function: `weave()`
```python
weave(state1: Union[State, StateArray], 
      state2: Union[State, StateArray]) -> Union[State, StateArray]
```

Weave operation: Bitwise AND (intersection).

Keeps only common information.

**Example:**
```python
weave(PSI, DELTA)  # Returns DELTA (i=1&1, q=1&0)
weave(PHI, DELTA)  # Returns EMPTY
```

### Function: `comp()`
```python
comp(state: Union[State, StateArray]) -> Union[State, StateArray]
```

Complement operation: Flip both bits.

**Transformations:**
- ∅ ↔ ψ
- δ ↔ φ

**Example:**
```python
comp(EMPTY)  # Returns PSI
comp(DELTA)  # Returns PHI
```

### Function: `neg_i()`
```python
neg_i(state: Union[State, StateArray]) -> Union[State, StateArray]
```

Negate i-bit only.

### Function: `neg_q()`
```python
neg_q(state: Union[State, StateArray]) -> Union[State, StateArray]
```

Negate q-bit only.

### Function: `compose()`
```python
compose(*operations) -> Callable
```

Compose operations right-to-left (mathematical composition).

**Example:**
```python
admit_then_gate = compose(gate, admit)
result = admit_then_gate(DELTA)  # Equivalent to gate(admit(DELTA))
```

### Function: `pipe()`
```python
pipe(*operations) -> Callable
```

Compose operations left-to-right (pipeline style).

**Example:**
```python
process = pipe(admit, gate)
result = process(DELTA)  # Equivalent to gate(admit(DELTA))
```

---

## NestedState
```python
from channelpy.core import NestedState
```

Hierarchical channel states for multi-level structures.

### Class: `NestedState`
```python
NestedState(level0=state0, level1=state1, ...)
```

Nested channel state with multiple levels forming a tree structure.

**Parameters:**
- `**levels`: Keyword arguments level0, level1, level2, etc. (State objects)

**Attributes:**
- `depth` (int): Maximum level index
- `num_levels` (int): Number of levels
- `total_bits` (int): Total number of bits (2 * num_levels)

#### Example
```python
# Two-level nested state
state = NestedState(
    level0=PSI,   # ψ
    level1=PHI    # φ
)

print(state)  # Output: ψ.φ
print(state.depth)  # Output: 1
print(state.num_levels)  # Output: 2
```

### Methods

#### `get_level(level)`
```python
nested.get_level(level: int) -> State
```

Get state at specific level.

#### `set_level(level, state)`
```python
nested.set_level(level: int, state: State)
```

Set state at specific level.

#### `all_levels()`
```python
nested.all_levels() -> List[State]
```

Return list of all level states.

#### `all_psi()`
```python
nested.all_psi() -> bool
```

Check if all levels are ψ.

#### `any_empty()`
```python
nested.any_empty() -> bool
```

Check if any level is ∅.

#### `count_psi()`
```python
nested.count_psi() -> int
```

Count number of ψ levels.

#### `path_string()`
```python
nested.path_string() -> str
```

Return path as string (e.g., "ψ.φ.δ").

#### `path_matches(pattern)`
```python
nested.path_matches(pattern: str) -> bool
```

Check if path matches pattern (supports wildcards).

**Example:**
```python
state = NestedState(level0=PSI, level1=PHI)
state.path_matches("ψ.*")  # True
state.path_matches("*.φ")  # True
```

#### `from_path(path)` (classmethod)
```python
NestedState.from_path(path: str) -> NestedState
```

Create nested state from path string.

**Example:**
```python
state = NestedState.from_path("ψ.φ.δ")
```

---

## ParallelChannels
```python
from channelpy.core import ParallelChannels
```

Multiple independent channel states.

### Class: `ParallelChannels`
```python
ParallelChannels(**channels)
```

Multiple independent channel states for parallel dimensions.

**Parameters:**
- `**channels`: Named channel states (State objects)

#### Example
```python
channels = ParallelChannels(
    technical=PSI,
    business=DELTA,
    team=PHI
)

print(channels['technical'])  # ψ
print(channels.all_names())   # ['technical', 'business', 'team']
```

### Methods

#### `all_names()`
```python
channels.all_names() -> List[str]
```

List of channel names.

#### `all_states()`
```python
channels.all_states() -> List[State]
```

List of all states.

#### `to_dict()`
```python
channels.to_dict() -> Dict[str, State]
```

Convert to dictionary.

#### `count_psi()`
```python
channels.count_psi() -> int
```

Count channels in ψ state.

#### `all_psi()`
```python
channels.all_psi() -> bool
```

Check if all channels are ψ.

#### `any_empty()`
```python
channels.any_empty() -> bool
```

Check if any channel is ∅.

---

## Lattice
```python
from channelpy.core import partial_order, meet, join
```

Lattice operations on channel states.

### Function: `partial_order()`
```python
partial_order(state1: State, state2: State) -> bool
```

Check if state1 ≤ state2 in the lattice order.

**Returns:** True if state1 ≤ state2

**Example:**
```python
partial_order(EMPTY, PSI)  # True (∅ ≤ anything)
partial_order(DELTA, PSI)  # True (δ ≤ ψ)
partial_order(PSI, EMPTY)  # False
```

### Function: `meet()`
```python
meet(state1: State, state2: State) -> State
```

Greatest lower bound (infimum) in the lattice.

**Returns:** State that is the meet of state1 and state2

**Example:**
```python
meet(PSI, DELTA)  # Returns DELTA
meet(DELTA, PHI)  # Returns EMPTY
```

### Function: `join()`
```python
join(state1: State, state2: State) -> State
```

Least upper bound (supremum) in the lattice.

**Returns:** State that is the join of state1 and state2

**Example:**
```python
join(DELTA, PHI)  # Returns PSI
join(EMPTY, DELTA)  # Returns DELTA
```

### Function: `is_comparable()`
```python
is_comparable(state1: State, state2: State) -> bool
```

Check if two states are comparable in the lattice.

**Returns:** True if states can be compared

### Function: `lattice_distance()`
```python
lattice_distance(state1: State, state2: State) -> int
```

Compute distance in the lattice (Hamming distance of bits).

**Returns:** Distance (0-2)

---

## Complete Example
```python
from channelpy.core import (
    State, StateArray, EMPTY, DELTA, PHI, PSI,
    gate, admit, overlay, weave, compose,
    NestedState, ParallelChannels,
    partial_order, meet, join
)
import numpy as np

# Create individual states
s1 = State(1, 0)  # δ
s2 = PSI          # ψ

# Operations
s3 = admit(s1)    # δ → ψ
s4 = gate(s1)     # δ → ∅
s5 = overlay(DELTA, PHI)  # δ ∨ φ = ψ

# State arrays
states = StateArray.from_bits(
    i=[1, 0, 1, 1],
    q=[1, 1, 0, 1]
)
counts = states.count_by_state()

# Nested states
nested = NestedState(level0=PSI, level1=PHI, level2=DELTA)
print(nested.path_string())  # "ψ.φ.δ"

# Parallel channels
parallel = ParallelChannels(
    feature1=PSI,
    feature2=DELTA,
    feature3=PHI
)

# Lattice operations
is_less = partial_order(DELTA, PSI)  # True
m = meet(DELTA, PHI)  # ∅
j = join(DELTA, PHI)  # ψ
```

---

## See Also

- [Pipeline API](pipeline.md)
- [Adaptive API](adaptive.md)
- [Tutorial: Basic States](../tutorials/01_basic_states.md)