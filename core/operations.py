"""
Core channel algebra operations

Operations on channel states:
- gate: Remove unvalidated elements (if q=0, set i=0)
- admit: Validate present elements (if i=1, set q=1)
- overlay: Bitwise OR (union)
- weave: Bitwise AND (intersection)
- comp: Complement both bits
- neg_i: Flip i-bit
- neg_q: Flip q-bit

Functional composition:
- compose: Right-to-left composition
- pipe: Left-to-right composition
"""

from typing import Union, Callable, TypeVar, List
import numpy as np
from .state import State, StateArray, EMPTY, DELTA, PHI, PSI


T = TypeVar('T', State, StateArray)


# ============================================================================
# Basic Operations
# ============================================================================

def gate(state: T) -> T:
    """
    Gate operation: Remove elements not validated by membership
    
    Rule: If q=0, set i=0
    
    Truth table:
    - ∅ → ∅  (nothing to remove)
    - δ → ∅  (puncture removed)
    - φ → φ  (hole preserved)
    - ψ → ψ  (resonant preserved)
    
    Interpretation: Only members pass through the gate.
    
    Parameters
    ----------
    state : State or StateArray
        Input state(s)
    
    Returns
    -------
    result : State or StateArray
        State(s) after gating
    
    Examples
    --------
    >>> gate(DELTA)
    State(i=0, q=0)  # ∅
    >>> gate(PSI)
    State(i=1, q=1)  # ψ
    
    >>> states = StateArray.from_states([PSI, DELTA, PHI, EMPTY])
    >>> gate(states).to_strings()
    array(['ψ', '∅', 'φ', '∅'])
    """
    if isinstance(state, State):
        return State(state.i & state.q, state.q)
    elif isinstance(state, StateArray):
        return StateArray(state.i & state.q, state.q)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


def admit(state: T) -> T:
    """
    Admit operation: Grant membership to present elements
    
    Rule: If i=1, set q=1
    
    Truth table:
    - ∅ → ∅  (nothing to admit)
    - δ → ψ  (puncture validated)
    - φ → φ  (hole remains)
    - ψ → ψ  (already resonant)
    
    Interpretation: Present elements are accepted as members.
    
    Parameters
    ----------
    state : State or StateArray
        Input state(s)
    
    Returns
    -------
    result : State or StateArray
        State(s) after admission
    
    Examples
    --------
    >>> admit(DELTA)
    State(i=1, q=1)  # ψ
    >>> admit(PHI)
    State(i=0, q=1)  # φ
    
    >>> states = StateArray.from_states([PSI, DELTA, PHI, EMPTY])
    >>> admit(states).to_strings()
    array(['ψ', 'ψ', 'φ', '∅'])
    """
    if isinstance(state, State):
        return State(state.i, state.q | state.i)
    elif isinstance(state, StateArray):
        return StateArray(state.i, state.q | state.i)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


def overlay(state1: T, state2: T) -> T:
    """
    Overlay operation: Bitwise OR (union of information)
    
    Takes maximum information from both states.
    
    Truth table (examples):
    - ∅ ∪ ψ → ψ
    - δ ∪ φ → ψ
    - δ ∪ δ → δ
    
    Parameters
    ----------
    state1, state2 : State or StateArray
        Input states (must be same type)
    
    Returns
    -------
    result : State or StateArray
        Overlaid state(s)
    
    Examples
    --------
    >>> overlay(DELTA, PHI)
    State(i=1, q=1)  # ψ
    >>> overlay(EMPTY, PSI)
    State(i=1, q=1)  # ψ
    
    >>> s1 = StateArray.from_states([EMPTY, DELTA])
    >>> s2 = StateArray.from_states([PHI, PHI])
    >>> overlay(s1, s2).to_strings()
    array(['φ', 'ψ'])
    """
    if isinstance(state1, State) and isinstance(state2, State):
        return State(state1.i | state2.i, state1.q | state2.q)
    elif isinstance(state1, StateArray) and isinstance(state2, StateArray):
        if state1.shape != state2.shape:
            raise ValueError(
                f"StateArrays must have same shape, got {state1.shape} and {state2.shape}"
            )
        return StateArray(state1.i | state2.i, state1.q | state2.q)
    else:
        raise TypeError("Both arguments must be same type (State or StateArray)")


def weave(state1: T, state2: T) -> T:
    """
    Weave operation: Bitwise AND (intersection of information)
    
    Keeps only common information.
    
    Truth table (examples):
    - ψ ∩ δ → δ
    - ψ ∩ φ → φ
    - φ ∩ δ → ∅
    
    Parameters
    ----------
    state1, state2 : State or StateArray
        Input states (must be same type)
    
    Returns
    -------
    result : State or StateArray
        Woven state(s)
    
    Examples
    --------
    >>> weave(PSI, DELTA)
    State(i=1, q=0)  # δ
    >>> weave(PHI, DELTA)
    State(i=0, q=0)  # ∅
    
    >>> s1 = StateArray.from_states([PSI, PSI])
    >>> s2 = StateArray.from_states([DELTA, PHI])
    >>> weave(s1, s2).to_strings()
    array(['δ', 'φ'])
    """
    if isinstance(state1, State) and isinstance(state2, State):
        return State(state1.i & state2.i, state1.q & state2.q)
    elif isinstance(state1, StateArray) and isinstance(state2, StateArray):
        if state1.shape != state2.shape:
            raise ValueError(
                f"StateArrays must have same shape, got {state1.shape} and {state2.shape}"
            )
        return StateArray(state1.i & state2.i, state1.q & state2.q)
    else:
        raise TypeError("Both arguments must be same type (State or StateArray)")


def comp(state: T) -> T:
    """
    Complement operation: Flip both bits
    
    Truth table:
    - ∅ ↔ ψ
    - δ ↔ φ
    
    This creates a duality between states.
    
    Parameters
    ----------
    state : State or StateArray
        Input state(s)
    
    Returns
    -------
    result : State or StateArray
        Complemented state(s)
    
    Examples
    --------
    >>> comp(EMPTY)
    State(i=1, q=1)  # ψ
    >>> comp(DELTA)
    State(i=0, q=1)  # φ
    >>> comp(comp(PSI)) == PSI
    True
    """
    if isinstance(state, State):
        return State(1 - state.i, 1 - state.q)
    elif isinstance(state, StateArray):
        return StateArray(1 - state.i, 1 - state.q)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


def neg_i(state: T) -> T:
    """
    Flip i-bit only
    
    Truth table:
    - ∅ ↔ δ
    - φ ↔ ψ
    
    Parameters
    ----------
    state : State or StateArray
        Input state(s)
    
    Returns
    -------
    result : State or StateArray
        State(s) with i-bit flipped
    
    Examples
    --------
    >>> neg_i(EMPTY)
    State(i=1, q=0)  # δ
    >>> neg_i(PHI)
    State(i=1, q=1)  # ψ
    """
    if isinstance(state, State):
        return State(1 - state.i, state.q)
    elif isinstance(state, StateArray):
        return StateArray(1 - state.i, state.q)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


def neg_q(state: T) -> T:
    """
    Flip q-bit only
    
    Truth table:
    - ∅ ↔ φ
    - δ ↔ ψ
    
    Parameters
    ----------
    state : State or StateArray
        Input state(s)
    
    Returns
    -------
    result : State or StateArray
        State(s) with q-bit flipped
    
    Examples
    --------
    >>> neg_q(EMPTY)
    State(i=0, q=1)  # φ
    >>> neg_q(DELTA)
    State(i=1, q=1)  # ψ
    """
    if isinstance(state, State):
        return State(state.i, 1 - state.q)
    elif isinstance(state, StateArray):
        return StateArray(state.i, 1 - state.q)
    else:
        raise TypeError(f"Expected State or StateArray, got {type(state)}")


# ============================================================================
# Functional Composition
# ============================================================================

def compose(*operations: Callable) -> Callable:
    """
    Compose operations right-to-left (mathematical style)
    
    compose(f, g, h)(x) = f(g(h(x)))
    
    Parameters
    ----------
    *operations : Callable
        Operations to compose
    
    Returns
    -------
    composed : Callable
        Composed operation
    
    Examples
    --------
    >>> # admit then gate
    >>> admit_then_gate = compose(gate, admit)
    >>> admit_then_gate(DELTA)
    State(i=1, q=1)  # ψ
    
    >>> # Complex composition
    >>> process = compose(gate, admit, neg_i)
    >>> process(EMPTY)
    State(i=1, q=1)  # ψ
    """
    if not operations:
        return lambda x: x
    
    def composed(state):
        result = state
        for op in reversed(operations):
            result = op(result)
        return result
    
    # Set helpful name
    op_names = [getattr(op, '__name__', 'op') for op in reversed(operations)]
    composed.__name__ = ' → '.join(op_names)
    
    return composed


def pipe(*operations: Callable) -> Callable:
    """
    Compose operations left-to-right (pipeline style)
    
    pipe(f, g, h)(x) = h(g(f(x)))
    
    Parameters
    ----------
    *operations : Callable
        Operations to pipe
    
    Returns
    -------
    piped : Callable
        Piped operation
    
    Examples
    --------
    >>> # admit then gate (pipeline style)
    >>> process = pipe(admit, gate)
    >>> process(DELTA)
    State(i=1, q=1)  # ψ
    
    >>> # Multi-stage pipeline
    >>> transform = pipe(neg_i, admit, gate)
    >>> transform(EMPTY)
    State(i=1, q=1)  # ψ
    """
    if not operations:
        return lambda x: x
    
    def piped(state):
        result = state
        for op in operations:
            result = op(result)
        return result
    
    # Set helpful name
    op_names = [getattr(op, '__name__', 'op') for op in operations]
    piped.__name__ = ' → '.join(op_names)
    
    return piped


# ============================================================================
# Batch Operations
# ============================================================================

def apply_operation(
    operation: Callable[[State], State],
    states: StateArray
) -> StateArray:
    """
    Apply operation to each state in array
    
    Parameters
    ----------
    operation : Callable
        Operation to apply
    states : StateArray
        Input states
    
    Returns
    -------
    result : StateArray
        Transformed states
    
    Examples
    --------
    >>> states = StateArray.from_states([DELTA, PHI, EMPTY])
    >>> apply_operation(admit, states).to_strings()
    array(['ψ', 'φ', '∅'])
    """
    # For built-in operations, use vectorized version
    if operation in [gate, admit, comp, neg_i, neg_q]:
        return operation(states)
    
    # For custom operations, apply element-wise
    result_states = [operation(s) for s in states]
    return StateArray.from_states(result_states)


def apply_binary_operation(
    operation: Callable[[State, State], State],
    states1: StateArray,
    states2: StateArray
) -> StateArray:
    """
    Apply binary operation element-wise
    
    Parameters
    ----------
    operation : Callable
        Binary operation
    states1, states2 : StateArray
        Input state arrays
    
    Returns
    -------
    result : StateArray
        Result of operation
    
    Examples
    --------
    >>> s1 = StateArray.from_states([DELTA, PHI])
    >>> s2 = StateArray.from_states([PHI, DELTA])
    >>> apply_binary_operation(overlay, s1, s2).to_strings()
    array(['ψ', 'ψ'])
    """
    if states1.shape != states2.shape:
        raise ValueError(
            f"StateArrays must have same shape, got {states1.shape} and {states2.shape}"
        )
    
    # For built-in operations, use vectorized version
    if operation in [overlay, weave]:
        return operation(states1, states2)
    
    # For custom operations, apply element-wise
    result_states = [operation(s1, s2) for s1, s2 in zip(states1, states2)]
    return StateArray.from_states(result_states)


# ============================================================================
# Utility Functions
# ============================================================================

def validate_operation(
    operation: Callable[[State], State]
) -> bool:
    """
    Test if operation is valid on all states
    
    Parameters
    ----------
    operation : Callable
        Operation to validate
    
    Returns
    -------
    valid : bool
        True if operation is valid
    """
    try:
        for state in [EMPTY, DELTA, PHI, PSI]:
            result = operation(state)
            if not isinstance(result, State):
                return False
        return True
    except Exception:
        return False


def operation_table(operation: Callable[[State], State]) -> dict:
    """
    Generate truth table for unary operation
    
    Parameters
    ----------
    operation : Callable
        Unary operation
    
    Returns
    -------
    table : dict
        Mapping from input state to output state
    
    Examples
    --------
    >>> table = operation_table(gate)
    >>> table[DELTA]
    State(i=0, q=0)  # ∅
    """
    return {
        EMPTY: operation(EMPTY),
        DELTA: operation(DELTA),
        PHI: operation(PHI),
        PSI: operation(PSI)
    }


def binary_operation_table(
    operation: Callable[[State, State], State]
) -> dict:
    """
    Generate truth table for binary operation
    
    Parameters
    ----------
    operation : Callable
        Binary operation
    
    Returns
    -------
    table : dict
        Mapping from (state1, state2) to result
    
    Examples
    --------
    >>> table = binary_operation_table(overlay)
    >>> table[(DELTA, PHI)]
    State(i=1, q=1)  # ψ
    """
    states = [EMPTY, DELTA, PHI, PSI]
    return {
        (s1, s2): operation(s1, s2)
        for s1 in states
        for s2 in states
    }