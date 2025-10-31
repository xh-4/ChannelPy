"""
Parallel channel systems for independent dimensions

Parallel channels represent multiple independent features or dimensions,
each with its own channel state. This enables:
- Multi-dimensional feature encoding
- Feature combination and aggregation
- Parallel state transformations
- Pattern matching across dimensions
"""

from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np
from collections import defaultdict

from .state import State, StateArray, EMPTY, DELTA, PHI, PSI
from .operations import gate, admit, overlay, weave, comp
from .lattice import meet, join


class ParallelChannels:
    """
    Multiple independent channel states
    
    Each channel represents a different feature or dimension,
    allowing multi-dimensional encoding and analysis.
    
    Examples
    --------
    >>> channels = ParallelChannels(
    ...     technical=State(1, 1),
    ...     business=State(1, 0),
    ...     team=State(0, 1)
    ... )
    >>> channels['technical']
    ψ
    >>> channels.count_psi()
    1
    >>> channels.all_names()
    ['technical', 'business', 'team']
    """
    
    def __init__(self, **channels):
        """
        Initialize parallel channels
        
        Parameters
        ----------
        **channels : State
            Named channel states
        """
        self._channels: Dict[str, State] = {}
        
        for name, state in channels.items():
            if not isinstance(state, State):
                raise TypeError(f"Channel values must be State, got {type(state)}")
            self._channels[name] = state
    
    def __getitem__(self, name: str) -> State:
        """Get channel state by name"""
        return self._channels[name]
    
    def __setitem__(self, name: str, state: State):
        """Set channel state by name"""
        if not isinstance(state, State):
            raise TypeError(f"Value must be State, got {type(state)}")
        self._channels[name] = state
    
    def __len__(self) -> int:
        """Number of channels"""
        return len(self._channels)
    
    def __iter__(self):
        """Iterate over channel names"""
        return iter(self._channels)
    
    def __contains__(self, name: str) -> bool:
        """Check if channel exists"""
        return name in self._channels
    
    def all_names(self) -> List[str]:
        """List of channel names"""
        return list(self._channels.keys())
    
    def all_states(self) -> List[State]:
        """List of all states"""
        return list(self._channels.values())
    
    def items(self):
        """Iterate over (name, state) pairs"""
        return self._channels.items()
    
    def to_dict(self) -> Dict[str, State]:
        """Convert to dictionary"""
        return self._channels.copy()
    
    # ========================================================================
    # State Counting and Analysis
    # ========================================================================
    
    def count_state(self, state: State) -> int:
        """Count channels in given state"""
        return sum(1 for s in self._channels.values() if s == state)
    
    def count_psi(self) -> int:
        """Count channels in ψ state (resonant)"""
        return self.count_state(PSI)
    
    def count_delta(self) -> int:
        """Count channels in δ state (puncture)"""
        return self.count_state(DELTA)
    
    def count_phi(self) -> int:
        """Count channels in φ state (hole)"""
        return self.count_state(PHI)
    
    def count_empty(self) -> int:
        """Count channels in ∅ state (absent)"""
        return self.count_state(EMPTY)
    
    def get_state_distribution(self) -> Dict[State, int]:
        """Get count of each state type"""
        return {
            EMPTY: self.count_empty(),
            DELTA: self.count_delta(),
            PHI: self.count_phi(),
            PSI: self.count_psi()
        }
    
    def all_psi(self) -> bool:
        """Check if all channels are ψ"""
        return all(s == PSI for s in self._channels.values())
    
    def any_psi(self) -> bool:
        """Check if any channel is ψ"""
        return any(s == PSI for s in self._channels.values())
    
    def any_empty(self) -> bool:
        """Check if any channel is ∅"""
        return any(s == EMPTY for s in self._channels.values())
    
    def all_empty(self) -> bool:
        """Check if all channels are ∅"""
        return all(s == EMPTY for s in self._channels.values())
    
    def get_channels_by_state(self, state: State) -> List[str]:
        """Get names of channels in given state"""
        return [name for name, s in self._channels.items() if s == state]
    
    # ========================================================================
    # Operations on All Channels
    # ========================================================================
    
    def apply_to_all(self, operation: Callable[[State], State]) -> 'ParallelChannels':
        """
        Apply operation to all channels
        
        Parameters
        ----------
        operation : Callable[[State], State]
            Operation to apply (e.g., gate, admit, comp)
            
        Returns
        -------
        result : ParallelChannels
            New ParallelChannels with operation applied
            
        Examples
        --------
        >>> channels = ParallelChannels(a=DELTA, b=PHI, c=PSI)
        >>> gated = channels.apply_to_all(gate)
        >>> gated['a']  # δ → ∅
        ∅
        """
        return ParallelChannels(**{
            name: operation(state)
            for name, state in self._channels.items()
        })
    
    def gate_all(self) -> 'ParallelChannels':
        """Apply gate operation to all channels"""
        return self.apply_to_all(gate)
    
    def admit_all(self) -> 'ParallelChannels':
        """Apply admit operation to all channels"""
        return self.apply_to_all(admit)
    
    def comp_all(self) -> 'ParallelChannels':
        """Apply complement to all channels"""
        return self.apply_to_all(comp)
    
    def filter_channels(self, predicate: Callable[[str, State], bool]) -> 'ParallelChannels':
        """
        Filter channels by predicate
        
        Parameters
        ----------
        predicate : Callable[[str, State], bool]
            Function that takes (name, state) and returns bool
            
        Returns
        -------
        filtered : ParallelChannels
            New ParallelChannels with only channels where predicate is True
            
        Examples
        --------
        >>> channels = ParallelChannels(a=PSI, b=DELTA, c=PSI)
        >>> psi_only = channels.filter_channels(lambda n, s: s == PSI)
        >>> psi_only.all_names()
        ['a', 'c']
        """
        return ParallelChannels(**{
            name: state
            for name, state in self._channels.items()
            if predicate(name, state)
        })
    
    def select_channels(self, names: List[str]) -> 'ParallelChannels':
        """
        Select subset of channels by name
        
        Parameters
        ----------
        names : List[str]
            Channel names to select
            
        Returns
        -------
        subset : ParallelChannels
            New ParallelChannels with selected channels
        """
        return ParallelChannels(**{
            name: self._channels[name]
            for name in names
            if name in self._channels
        })
    
    # ========================================================================
    # Aggregation Operations
    # ========================================================================
    
    def aggregate_meet(self) -> State:
        """
        Aggregate all channels using meet (∧)
        
        Returns the greatest lower bound across all channels
        
        Examples
        --------
        >>> channels = ParallelChannels(a=PSI, b=DELTA, c=PHI)
        >>> channels.aggregate_meet()
        ∅
        """
        if not self._channels:
            return PSI  # Identity for meet
        
        result = list(self._channels.values())[0]
        for state in list(self._channels.values())[1:]:
            result = meet(result, state)
        
        return result
    
    def aggregate_join(self) -> State:
        """
        Aggregate all channels using join (∨)
        
        Returns the least upper bound across all channels
        
        Examples
        --------
        >>> channels = ParallelChannels(a=EMPTY, b=DELTA, c=PHI)
        >>> channels.aggregate_join()
        ψ
        """
        if not self._channels:
            return EMPTY  # Identity for join
        
        result = list(self._channels.values())[0]
        for state in list(self._channels.values())[1:]:
            result = join(result, state)
        
        return result
    
    def aggregate_overlay(self) -> State:
        """
        Aggregate using overlay (bitwise OR)
        
        Takes maximum information from all channels
        """
        if not self._channels:
            return EMPTY
        
        result = list(self._channels.values())[0]
        for state in list(self._channels.values())[1:]:
            result = overlay(result, state)
        
        return result
    
    def aggregate_weave(self) -> State:
        """
        Aggregate using weave (bitwise AND)
        
        Keeps only common information across all channels
        """
        if not self._channels:
            return PSI
        
        result = list(self._channels.values())[0]
        for state in list(self._channels.values())[1:]:
            result = weave(result, state)
        
        return result
    
    def weighted_vote(self, weights: Optional[Dict[str, float]] = None) -> State:
        """
        Weighted voting to aggregate channels
        
        Each state votes with a weight, result is majority
        
        Parameters
        ----------
        weights : Dict[str, float], optional
            Weight for each channel (default: equal weights)
            
        Returns
        -------
        result : State
            Aggregated state by weighted voting
        """
        if weights is None:
            weights = {name: 1.0 for name in self._channels}
        
        # Vote for each bit independently
        i_vote = sum(
            weights.get(name, 1.0) * state.i
            for name, state in self._channels.items()
        )
        q_vote = sum(
            weights.get(name, 1.0) * state.q
            for name, state in self._channels.items()
        )
        
        total_weight = sum(weights.get(name, 1.0) for name in self._channels)
        
        # Majority vote
        i = 1 if i_vote > total_weight / 2 else 0
        q = 1 if q_vote > total_weight / 2 else 0
        
        return State(i, q)
    
    def unanimous_psi(self) -> bool:
        """
        Check if unanimous ψ across all channels
        
        Useful for "all conditions met" logic
        """
        return self.all_psi()
    
    def majority_psi(self) -> bool:
        """Check if majority of channels are ψ"""
        return self.count_psi() > len(self._channels) / 2
    
    # ========================================================================
    # Pattern Matching
    # ========================================================================
    
    def matches_pattern(self, pattern: Dict[str, Union[State, List[State]]]) -> bool:
        """
        Check if channels match pattern
        
        Parameters
        ----------
        pattern : Dict[str, State or List[State]]
            Pattern to match. Value can be:
            - Single State: exact match required
            - List[State]: channel must be in list
            - '*': any state (wildcard)
            
        Returns
        -------
        matches : bool
            True if all specified channels match pattern
            
        Examples
        --------
        >>> channels = ParallelChannels(a=PSI, b=DELTA, c=PHI)
        >>> channels.matches_pattern({'a': PSI, 'b': [DELTA, PHI]})
        True
        >>> channels.matches_pattern({'a': DELTA})
        False
        """
        for name, expected in pattern.items():
            if name not in self._channels:
                return False
            
            actual = self._channels[name]
            
            if expected == '*':
                continue
            elif isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        
        return True
    
    def find_matching_channels(self, state: State) -> List[str]:
        """Find all channel names with given state"""
        return self.get_channels_by_state(state)
    
    # ========================================================================
    # Combination with Other ParallelChannels
    # ========================================================================
    
    def combine_with(
        self, 
        other: 'ParallelChannels',
        operation: Callable[[State, State], State],
        handle_missing: str = 'skip'
    ) -> 'ParallelChannels':
        """
        Combine with another ParallelChannels using operation
        
        Parameters
        ----------
        other : ParallelChannels
            Other parallel channels
        operation : Callable
            Binary operation (e.g., meet, join, overlay)
        handle_missing : str
            How to handle channels in one but not the other:
            - 'skip': Only include common channels
            - 'keep_left': Keep channels from self
            - 'keep_right': Keep channels from other
            - 'keep_both': Keep all channels (use identity for missing)
            
        Returns
        -------
        combined : ParallelChannels
            Result of combining channels
        """
        result = {}
        
        if handle_missing == 'skip':
            # Only common channels
            common = set(self._channels.keys()) & set(other._channels.keys())
            for name in common:
                result[name] = operation(self._channels[name], other._channels[name])
        
        elif handle_missing == 'keep_left':
            for name in self._channels:
                if name in other._channels:
                    result[name] = operation(self._channels[name], other._channels[name])
                else:
                    result[name] = self._channels[name]
        
        elif handle_missing == 'keep_right':
            for name in other._channels:
                if name in self._channels:
                    result[name] = operation(self._channels[name], other._channels[name])
                else:
                    result[name] = other._channels[name]
        
        elif handle_missing == 'keep_both':
            all_names = set(self._channels.keys()) | set(other._channels.keys())
            for name in all_names:
                if name in self._channels and name in other._channels:
                    result[name] = operation(self._channels[name], other._channels[name])
                elif name in self._channels:
                    result[name] = self._channels[name]
                else:
                    result[name] = other._channels[name]
        
        return ParallelChannels(**result)
    
    def merge(self, other: 'ParallelChannels') -> 'ParallelChannels':
        """
        Merge with another ParallelChannels (overlay operation)
        
        For common channels, combines using overlay
        Keeps all channels from both
        """
        return self.combine_with(other, overlay, handle_missing='keep_both')
    
    # ========================================================================
    # String Representation
    # ========================================================================
    
    def __str__(self) -> str:
        parts = [f"{name}:{state}" for name, state in self._channels.items()]
        return f"({', '.join(parts)})"
    
    def __repr__(self) -> str:
        items = ', '.join(
            f"{name}={repr(state)}" 
            for name, state in self._channels.items()
        )
        return f"ParallelChannels({items})"
    
    def to_string_compact(self) -> str:
        """Compact string representation"""
        return ''.join(str(s) for s in self._channels.values())
    
    def summary(self) -> str:
        """
        Human-readable summary
        
        Returns
        -------
        summary : str
            Multi-line summary of channels
        """
        lines = [
            f"ParallelChannels with {len(self._channels)} channels:",
            ""
        ]
        
        dist = self.get_state_distribution()
        lines.append("State distribution:")
        for state, count in dist.items():
            pct = 100 * count / len(self._channels) if self._channels else 0
            lines.append(f"  {state}: {count} ({pct:.1f}%)")
        
        lines.append("")
        lines.append("Channels:")
        for name, state in sorted(self._channels.items()):
            lines.append(f"  {name:15s}: {state}")
        
        return "\n".join(lines)
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def plot_distribution(self):
        """
        Plot state distribution
        
        Requires matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Requires matplotlib for plotting")
        
        dist = self.get_state_distribution()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        states = [EMPTY, DELTA, PHI, PSI]
        labels = ['∅', 'δ', 'φ', 'ψ']
        values = [dist[s] for s in states]
        colors = ['lightgray', 'lightyellow', 'lightblue', 'lightgreen']
        
        ax.bar(labels, values, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Channel State Distribution', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentages
        total = sum(values)
        for i, (label, value) in enumerate(zip(labels, values)):
            if value > 0:
                pct = 100 * value / total
                ax.text(i, value, f'{pct:.1f}%', 
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_channels(self):
        """
        Plot all channels as bars
        
        Requires matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Requires matplotlib for plotting")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        names = self.all_names()
        state_ints = [self._channels[name].to_int() for name in names]
        
        colors = ['lightgray', 'lightblue', 'lightyellow', 'lightgreen']
        bar_colors = [colors[i] for i in state_ints]
        
        ax.bar(range(len(names)), [1]*len(names), color=bar_colors, 
              edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticks([])
        ax.set_title('Channel States', fontsize=14)
        
        # Add state symbols on bars
        for i, (name, state_int) in enumerate(zip(names, state_ints)):
            state = self._channels[name]
            ax.text(i, 0.5, str(state), ha='center', va='center', 
                   fontsize=16, fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='black', label='∅ (Empty)'),
            Patch(facecolor='lightblue', edgecolor='black', label='φ (Hole)'),
            Patch(facecolor='lightyellow', edgecolor='black', label='δ (Puncture)'),
            Patch(facecolor='lightgreen', edgecolor='black', label='ψ (Resonant)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig