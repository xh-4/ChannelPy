"""
Lattice structure for channel states

Channel states form a lattice under the partial order:
    s1 ≤ s2  iff  s1.i ≤ s2.i  AND  s1.q ≤ s2.q

The lattice has:
- Bottom element: ∅ (0,0)
- Top element: ψ (1,1)
- Meet (∧): greatest lower bound
- Join (∨): least upper bound

This structure provides:
- Rigorous mathematical foundation
- Lattice-based operations
- Distance measures
- Path finding between states
"""

from typing import List, Set, Tuple, Optional, Dict
import numpy as np
from itertools import product

from .state import State, StateArray, EMPTY, DELTA, PHI, PSI


class StateLattice:
    """
    The lattice of channel states
    
    Partial order: s1 ≤ s2 iff s1.i ≤ s2.i AND s1.q ≤ s2.q
    
    Hasse diagram:
    
           ψ (1,1)
          / \
    δ (1,0)  φ (0,1)
          \ /
           ∅ (0,0)
    
    Examples
    --------
    >>> lattice = StateLattice()
    >>> lattice.partial_order(EMPTY, PSI)
    True
    >>> lattice.meet(DELTA, PHI)
    ∅
    >>> lattice.join(DELTA, PHI)
    ψ
    """
    
    def __init__(self):
        # All states in the lattice
        self.elements = [EMPTY, DELTA, PHI, PSI]
        
        # Precompute partial order relations
        self._partial_order_matrix = self._compute_partial_order_matrix()
        
        # Precompute meet and join tables
        self._meet_table = self._compute_meet_table()
        self._join_table = self._compute_join_table()
    
    def _compute_partial_order_matrix(self) -> np.ndarray:
        """
        Compute matrix of partial order relations
        
        Returns
        -------
        matrix : np.ndarray
            matrix[i,j] = 1 if elements[i] ≤ elements[j]
        """
        n = len(self.elements)
        matrix = np.zeros((n, n), dtype=bool)
        
        for i, s1 in enumerate(self.elements):
            for j, s2 in enumerate(self.elements):
                matrix[i, j] = self.partial_order(s1, s2)
        
        return matrix
    
    def _compute_meet_table(self) -> Dict[Tuple[State, State], State]:
        """Precompute meet (∧) for all pairs"""
        table = {}
        for s1 in self.elements:
            for s2 in self.elements:
                table[(s1, s2)] = self.meet(s1, s2)
        return table
    
    def _compute_join_table(self) -> Dict[Tuple[State, State], State]:
        """Precompute join (∨) for all pairs"""
        table = {}
        for s1 in self.elements:
            for s2 in self.elements:
                table[(s1, s2)] = self.join(s1, s2)
        return table
    
    def partial_order(self, s1: State, s2: State) -> bool:
        """
        Check if s1 ≤ s2 in the lattice order
        
        s1 ≤ s2  iff  s1.i ≤ s2.i  AND  s1.q ≤ s2.q
        
        Parameters
        ----------
        s1, s2 : State
            States to compare
            
        Returns
        -------
        ordered : bool
            True if s1 ≤ s2
            
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.partial_order(EMPTY, PSI)  # ∅ ≤ ψ
        True
        >>> lattice.partial_order(DELTA, PHI)  # δ and φ incomparable
        False
        """
        return s1.i <= s2.i and s1.q <= s2.q
    
    def is_comparable(self, s1: State, s2: State) -> bool:
        """
        Check if two states are comparable
        
        s1 and s2 are comparable if s1 ≤ s2 or s2 ≤ s1
        
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.is_comparable(EMPTY, PSI)  # ∅ and ψ comparable
        True
        >>> lattice.is_comparable(DELTA, PHI)  # δ and φ incomparable
        False
        """
        return self.partial_order(s1, s2) or self.partial_order(s2, s1)
    
    def meet(self, s1: State, s2: State) -> State:
        """
        Compute meet (greatest lower bound, ∧)
        
        s1 ∧ s2 = (min(s1.i, s2.i), min(s1.q, s2.q))
        
        The meet is the "most information both agree on"
        
        Parameters
        ----------
        s1, s2 : State
            States to meet
            
        Returns
        -------
        meet : State
            s1 ∧ s2
            
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.meet(DELTA, PHI)  # δ ∧ φ = ∅
        ∅
        >>> lattice.meet(PSI, DELTA)  # ψ ∧ δ = δ
        δ
        """
        return State(
            i=min(s1.i, s2.i),
            q=min(s1.q, s2.q)
        )
    
    def join(self, s1: State, s2: State) -> State:
        """
        Compute join (least upper bound, ∨)
        
        s1 ∨ s2 = (max(s1.i, s2.i), max(s1.q, s2.q))
        
        The join is the "combination of information from both"
        
        Parameters
        ----------
        s1, s2 : State
            States to join
            
        Returns
        -------
        join : State
            s1 ∨ s2
            
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.join(DELTA, PHI)  # δ ∨ φ = ψ
        ψ
        >>> lattice.join(EMPTY, DELTA)  # ∅ ∨ δ = δ
        δ
        """
        return State(
            i=max(s1.i, s2.i),
            q=max(s1.q, s2.q)
        )
    
    def meet_array(self, states: List[State]) -> State:
        """
        Compute meet of multiple states
        
        ⋀ states = meet of all states
        
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.meet_array([PSI, DELTA, PHI])
        ∅
        """
        if not states:
            return PSI  # Top element is identity for meet
        
        result = states[0]
        for state in states[1:]:
            result = self.meet(result, state)
        
        return result
    
    def join_array(self, states: List[State]) -> State:
        """
        Compute join of multiple states
        
        ⋁ states = join of all states
        
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.join_array([EMPTY, DELTA, PHI])
        ψ
        """
        if not states:
            return EMPTY  # Bottom element is identity for join
        
        result = states[0]
        for state in states[1:]:
            result = self.join(result, state)
        
        return result
    
    def lattice_distance(self, s1: State, s2: State) -> int:
        """
        Compute lattice distance between states
        
        Distance = minimum number of covering steps between states
        
        Covering relation: s1 ⋖ s2 if s1 < s2 and no s3 with s1 < s3 < s2
        
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.lattice_distance(EMPTY, PSI)  # ∅ → ψ
        2
        >>> lattice.lattice_distance(DELTA, PHI)  # δ → φ (via ∅ or ψ)
        2
        """
        # BFS to find shortest path
        from collections import deque
        
        if s1 == s2:
            return 0
        
        queue = deque([(s1, 0)])
        visited = {s1}
        
        while queue:
            current, dist = queue.popleft()
            
            # Try all possible steps (covers and covered-by)
            for next_state in self.elements:
                if next_state in visited:
                    continue
                
                # Check if there's a covering relation
                if self._covers(current, next_state) or self._covers(next_state, current):
                    if next_state == s2:
                        return dist + 1
                    
                    visited.add(next_state)
                    queue.append((next_state, dist + 1))
        
        # No path found (shouldn't happen in connected lattice)
        return float('inf')
    
    def _covers(self, s1: State, s2: State) -> bool:
        """
        Check if s1 covers s2 (s1 ⋖ s2)
        
        s1 ⋖ s2 if:
        - s1 < s2 (s1 ≤ s2 and s1 ≠ s2)
        - No s3 with s1 < s3 < s2
        """
        if not (self.partial_order(s1, s2) and s1 != s2):
            return False
        
        # Check if immediate successor (Hamming distance = 1)
        bit_diff = abs(s1.i - s2.i) + abs(s1.q - s2.q)
        return bit_diff == 1
    
    def covers(self, s: State) -> List[State]:
        """
        Get all states that cover s (immediate successors)
        
        Returns states s' where s ⋖ s'
        
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.covers(EMPTY)  # ∅ is covered by δ and φ
        [δ, φ]
        """
        return [s2 for s2 in self.elements if self._covers(s, s2)]
    
    def covered_by(self, s: State) -> List[State]:
        """
        Get all states covered by s (immediate predecessors)
        
        Returns states s' where s' ⋖ s
        
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.covered_by(PSI)  # ψ covers δ and φ
        [δ, φ]
        """
        return [s1 for s1 in self.elements if self._covers(s1, s)]
    
    def get_chains(self, s1: State, s2: State) -> List[List[State]]:
        """
        Get all maximal chains from s1 to s2
        
        A chain is a totally ordered subset
        A maximal chain has no gaps
        
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.get_chains(EMPTY, PSI)
        [[∅, δ, ψ], [∅, φ, ψ]]
        """
        if not self.partial_order(s1, s2):
            return []
        
        if s1 == s2:
            return [[s1]]
        
        chains = []
        
        # Find all paths through covering relations
        def find_chains_recursive(current, target, path):
            if current == target:
                chains.append(path[:])
                return
            
            for next_state in self.covers(current):
                if self.partial_order(next_state, target):
                    path.append(next_state)
                    find_chains_recursive(next_state, target, path)
                    path.pop()
        
        find_chains_recursive(s1, s2, [s1])
        return chains
    
    def is_lattice(self) -> bool:
        """
        Verify that the structure is indeed a lattice
        
        A lattice requires:
        1. Every pair has a meet
        2. Every pair has a join
        3. Meet and join are associative, commutative, idempotent
        4. Absorption laws hold
        
        Returns
        -------
        is_lattice : bool
            True if structure satisfies lattice properties
        """
        # Check that meet and join exist for all pairs
        for s1 in self.elements:
            for s2 in self.elements:
                # Meet exists
                m = self.meet(s1, s2)
                if m not in self.elements:
                    return False
                
                # Join exists
                j = self.join(s1, s2)
                if j not in self.elements:
                    return False
                
                # Absorption laws
                # s1 ∧ (s1 ∨ s2) = s1
                if self.meet(s1, self.join(s1, s2)) != s1:
                    return False
                
                # s1 ∨ (s1 ∧ s2) = s1
                if self.join(s1, self.meet(s1, s2)) != s1:
                    return False
        
        return True
    
    def get_atoms(self) -> List[State]:
        """
        Get atoms (minimal non-bottom elements)
        
        An atom covers the bottom element
        
        Returns
        -------
        atoms : List[State]
            [δ, φ] in the channel lattice
        """
        return self.covers(EMPTY)
    
    def get_coatoms(self) -> List[State]:
        """
        Get coatoms (maximal non-top elements)
        
        A coatom is covered by the top element
        
        Returns
        -------
        coatoms : List[State]
            [δ, φ] in the channel lattice
        """
        return self.covered_by(PSI)
    
    def is_complemented(self) -> bool:
        """
        Check if lattice is complemented
        
        A lattice is complemented if every element has a complement:
        For each s, exists s' such that s ∧ s' = ⊥ and s ∨ s' = ⊤
        
        Returns
        -------
        complemented : bool
            True if every element has a complement
        """
        for s in self.elements:
            has_complement = False
            for s_prime in self.elements:
                if (self.meet(s, s_prime) == EMPTY and 
                    self.join(s, s_prime) == PSI):
                    has_complement = True
                    break
            
            if not has_complement:
                return False
        
        return True
    
    def get_complement(self, s: State) -> Optional[State]:
        """
        Get the complement of a state (if it exists)
        
        Complement s' satisfies: s ∧ s' = ⊥ and s ∨ s' = ⊤
        
        Examples
        --------
        >>> lattice = StateLattice()
        >>> lattice.get_complement(EMPTY)
        ψ
        >>> lattice.get_complement(DELTA)
        φ
        """
        for s_prime in self.elements:
            if (self.meet(s, s_prime) == EMPTY and 
                self.join(s, s_prime) == PSI):
                return s_prime
        
        return None
    
    def plot_hasse_diagram(self, show_labels: bool = True):
        """
        Plot Hasse diagram of the lattice
        
        Requires matplotlib and networkx
        
        Parameters
        ----------
        show_labels : bool
            Whether to show state labels
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            raise ImportError("Requires matplotlib and networkx for plotting")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for state in self.elements:
            G.add_node(str(state))
        
        # Add edges (covering relations)
        for s1 in self.elements:
            for s2 in self.covers(s1):
                G.add_edge(str(s1), str(s2))
        
        # Position nodes (manual for nice layout)
        pos = {
            '∅': (0.5, 0),
            'δ': (0, 1),
            'φ': (1, 1),
            'ψ': (0.5, 2)
        }
        
        # Draw
        fig, ax = plt.subplots(figsize=(8, 6))
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=3000, ax=ax)
        nx.draw_networkx_edges(G, pos, arrows=True, 
                              arrowsize=20, ax=ax,
                              edge_color='gray', width=2)
        
        if show_labels:
            nx.draw_networkx_labels(G, pos, font_size=24, 
                                   font_family='monospace', ax=ax)
        
        ax.set_title('Hasse Diagram of Channel State Lattice', fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def get_properties(self) -> Dict[str, bool]:
        """
        Get all lattice properties
        
        Returns
        -------
        properties : Dict[str, bool]
            Dictionary of lattice properties
        """
        return {
            'is_lattice': self.is_lattice(),
            'is_complemented': self.is_complemented(),
            'is_bounded': True,  # Has ⊥ and ⊤
            'is_distributive': self._is_distributive(),
            'is_modular': self._is_modular(),
            'num_elements': len(self.elements),
            'num_atoms': len(self.get_atoms()),
            'num_coatoms': len(self.get_coatoms())
        }
    
    def _is_distributive(self) -> bool:
        """
        Check if lattice is distributive
        
        s1 ∧ (s2 ∨ s3) = (s1 ∧ s2) ∨ (s1 ∧ s3) for all s1, s2, s3
        """
        for s1 in self.elements:
            for s2 in self.elements:
                for s3 in self.elements:
                    lhs = self.meet(s1, self.join(s2, s3))
                    rhs = self.join(self.meet(s1, s2), self.meet(s1, s3))
                    
                    if lhs != rhs:
                        return False
        
        return True
    
    def _is_modular(self) -> bool:
        """
        Check if lattice is modular
        
        If s1 ≤ s3, then s1 ∨ (s2 ∧ s3) = (s1 ∨ s2) ∧ s3
        """
        for s1 in self.elements:
            for s2 in self.elements:
                for s3 in self.elements:
                    if self.partial_order(s1, s3):
                        lhs = self.join(s1, self.meet(s2, s3))
                        rhs = self.meet(self.join(s1, s2), s3)
                        
                        if lhs != rhs:
                            return False
        
        return True


# Global instance
LATTICE = StateLattice()


# Convenience functions

def partial_order(s1: State, s2: State) -> bool:
    """Check if s1 ≤ s2"""
    return LATTICE.partial_order(s1, s2)


def meet(s1: State, s2: State) -> State:
    """Compute s1 ∧ s2"""
    return LATTICE.meet(s1, s2)


def join(s1: State, s2: State) -> State:
    """Compute s1 ∨ s2"""
    return LATTICE.join(s1, s2)


def lattice_distance(s1: State, s2: State) -> int:
    """Compute lattice distance between states"""
    return LATTICE.lattice_distance(s1, s2)


def complement(s: State) -> Optional[State]:
    """Get complement of state"""
    return LATTICE.get_complement(s)