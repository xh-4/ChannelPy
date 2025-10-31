"""
Interpreters: Channel States → Decisions

Stage 3 of the pipeline: convert states to actionable decisions
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..core.operations import gate, admit


class Interpreter(ABC):
    """
    Base class for interpreters
    
    All interpreters convert states to decisions
    """
    
    @abstractmethod
    def interpret(self, state: State) -> Any:
        """Interpret single state"""
        pass
    
    def __call__(self, states):
        """Make interpreter callable"""
        if isinstance(states, State):
            return self.interpret(states)
        elif isinstance(states, StateArray):
            return self.interpret_array(states)
        elif isinstance(states, list):
            return [self.interpret(s) for s in states]
        else:
            raise TypeError(f"Cannot interpret type: {type(states)}")
    
    def interpret_array(self, states: StateArray):
        """Interpret array of states"""
        return np.array([self.interpret(states[i]) for i in range(len(states))])
    
    def fit(self, states, y=None):
        """Optional: learn from states and labels"""
        return self


class RuleBasedInterpreter(Interpreter):
    """
    Interpret using explicit rules
    
    Maps state patterns to decisions
    
    Examples
    --------
    >>> rules = {
    ...     PSI: 'strong_buy',
    ...     DELTA: 'weak_buy',
    ...     PHI: 'hold',
    ...     EMPTY: 'sell'
    ... }
    >>> interpreter = RuleBasedInterpreter(rules)
    >>> decision = interpreter(PSI)  # Returns 'strong_buy'
    """
    
    def __init__(self, rules: Optional[Dict[State, Any]] = None):
        """
        Parameters
        ----------
        rules : Dict[State, Any], optional
            Mapping from states to decisions
        """
        if rules is None:
            # Default trading rules
            rules = {
                PSI: 'buy',
                DELTA: 'hold',
                PHI: 'hold',
                EMPTY: 'sell'
            }
        
        self.rules = rules
    
    def interpret(self, state: State) -> Any:
        """Apply rules"""
        if state in self.rules:
            return self.rules[state]
        else:
            # Default: return state as string
            return str(state)
    
    def add_rule(self, state: State, decision: Any):
        """Add or update rule"""
        self.rules[state] = decision
    
    def remove_rule(self, state: State):
        """Remove rule"""
        if state in self.rules:
            del self.rules[state]


class LookupTableInterpreter(Interpreter):
    """
    Learn decision mapping from training data
    
    Builds lookup table: state → most common label
    
    Examples
    --------
    >>> interpreter = LookupTableInterpreter()
    >>> interpreter.fit(train_states, train_labels)
    >>> predictions = interpreter(test_states)
    """
    
    def __init__(self):
        self.lookup_table = {}
        self.default_decision = None
        self.is_fitted = False
    
    def fit(self, states, y):
        """Build lookup table from data"""
        if isinstance(states, StateArray):
            states = [states[i] for i in range(len(states))]
        
        y = np.asarray(y)
        
        # Count label occurrences for each state
        state_labels = {}
        for state, label in zip(states, y):
            if state not in state_labels:
                state_labels[state] = []
            state_labels[state].append(label)
        
        # Most common label for each state
        for state, labels in state_labels.items():
            unique, counts = np.unique(labels, return_counts=True)
            most_common = unique[np.argmax(counts)]
            self.lookup_table[state] = most_common
        
        # Default decision (most common overall)
        unique, counts = np.unique(y, return_counts=True)
        self.default_decision = unique[np.argmax(counts)]
        
        self.is_fitted = True
        return self
    
    def interpret(self, state: State) -> Any:
        """Look up decision"""
        if not self.is_fitted:
            raise RuntimeError("Interpreter not fitted. Call fit() first.")
        
        return self.lookup_table.get(state, self.default_decision)


class FSMInterpreter(Interpreter):
    """
    Finite State Machine interpreter
    
    Maintains internal state and makes decisions based on state transitions
    
    Examples
    --------
    >>> interpreter = FSMInterpreter()
    >>> interpreter.add_transition(EMPTY, PSI, action='buy', next_state='invested')
    >>> interpreter.add_transition(PSI, EMPTY, action='sell', next_state='cash')
    >>> 
    >>> for state in state_sequence:
    ...     action = interpreter(state)
    """
    
    def __init__(self, initial_fsm_state: str = 'start'):
        """
        Parameters
        ----------
        initial_fsm_state : str
            Initial FSM state (not channel state)
        """
        self.initial_fsm_state = initial_fsm_state
        self.current_fsm_state = initial_fsm_state
        self.transitions = {}
        self.default_action = 'hold'
    
    def add_transition(
        self, 
        from_channel_state: State,
        to_channel_state: State,
        action: Any,
        next_fsm_state: Optional[str] = None
    ):
        """
        Add state transition rule
        
        Parameters
        ----------
        from_channel_state : State
            Previous channel state
        to_channel_state : State
            Current channel state
        action : Any
            Action to take
        next_fsm_state : str, optional
            Next FSM state
        """
        key = (self.current_fsm_state, from_channel_state, to_channel_state)
        self.transitions[key] = {
            'action': action,
            'next_state': next_fsm_state or self.current_fsm_state
        }
    
    def interpret(self, state: State) -> Any:
        """Interpret with FSM logic"""
        # Look for matching transition
        for key, transition in self.transitions.items():
            fsm_state, from_state, to_state = key
            
            if fsm_state == self.current_fsm_state and to_state == state:
                # Found matching transition
                action = transition['action']
                self.current_fsm_state = transition['next_state']
                return action
        
        # No matching transition
        return self.default_action
    
    def reset(self):
        """Reset FSM to initial state"""
        self.current_fsm_state = self.initial_fsm_state


class PatternMatcher(Interpreter):
    """
    Match state sequences to patterns
    
    Looks for specific patterns in state history
    
    Examples
    --------
    >>> matcher = PatternMatcher()
    >>> matcher.add_pattern([EMPTY, DELTA, PSI], decision='uptrend', confidence=0.9)
    >>> matcher.add_pattern([PSI, DELTA, EMPTY], decision='downtrend', confidence=0.9)
    >>> 
    >>> for state in state_sequence:
    ...     decision = matcher(state)
    """
    
    def __init__(self, window_size: int = 3):
        """
        Parameters
        ----------
        window_size : int
            Size of pattern window
        """
        self.window_size = window_size
        self.patterns = []
        self.state_history = []
        self.default_decision = 'hold'
    
    def add_pattern(
        self, 
        pattern: List[State], 
        decision: Any,
        confidence: float = 1.0
    ):
        """
        Add pattern to match
        
        Parameters
        ----------
        pattern : List[State]
            Sequence of states to match
        decision : Any
            Decision if pattern matches
        confidence : float
            Confidence score (0-1)
        """
        self.patterns.append({
            'pattern': pattern,
            'decision': decision,
            'confidence': confidence
        })
    
    def interpret(self, state: State) -> Any:
        """Match patterns"""
        # Add to history
        self.state_history.append(state)
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
        
        # Check patterns
        best_match = None
        best_confidence = 0
        
        for pattern_info in self.patterns:
            pattern = pattern_info['pattern']
            
            if len(self.state_history) < len(pattern):
                continue
            
            # Check if pattern matches recent history
            recent = self.state_history[-len(pattern):]
            if recent == pattern:
                if pattern_info['confidence'] > best_confidence:
                    best_match = pattern_info['decision']
                    best_confidence = pattern_info['confidence']
        
        return best_match if best_match else self.default_decision
    
    def reset(self):
        """Clear history"""
        self.state_history = []


class ScoreBasedInterpreter(Interpreter):
    """
    Interpret based on accumulated score
    
    Each state contributes to a score, decision based on score threshold
    
    Examples
    --------
    >>> interpreter = ScoreBasedInterpreter()
    >>> interpreter.set_scores({PSI: 1.0, DELTA: 0.5, PHI: -0.5, EMPTY: -1.0})
    >>> interpreter.set_thresholds(buy=0.8, sell=-0.8)
    >>> 
    >>> for state in state_sequence:
    ...     decision = interpreter(state)
    """
    
    def __init__(self):
        self.state_scores = {
            PSI: 1.0,
            DELTA: 0.3,
            PHI: -0.3,
            EMPTY: -1.0
        }
        self.thresholds = {
            'buy': 0.7,
            'sell': -0.7
        }
        self.current_score = 0.0
        self.decay_rate = 0.9  # Score decays over time
    
    def set_scores(self, scores: Dict[State, float]):
        """Set score for each state"""
        self.state_scores = scores
    
    def set_thresholds(self, **thresholds):
        """Set decision thresholds"""
        self.thresholds = thresholds
    
    def interpret(self, state: State) -> str:
        """Accumulate score and make decision"""
        # Decay previous score
        self.current_score *= self.decay_rate
        
        # Add new score
        self.current_score += self.state_scores.get(state, 0)
        
        # Clip score to reasonable range
        self.current_score = np.clip(self.current_score, -10, 10)
        
        # Make decision based on thresholds
        if self.current_score >= self.thresholds.get('buy', 0.7):
            return 'buy'
        elif self.current_score <= self.thresholds.get('sell', -0.7):
            return 'sell'
        else:
            return 'hold'
    
    def reset(self):
        """Reset score"""
        self.current_score = 0.0


class EnsembleInterpreter(Interpreter):
    """
    Ensemble of multiple interpreters
    
    Combines decisions from multiple interpreters using voting or averaging
    
    Examples
    --------
    >>> rule_interp = RuleBasedInterpreter()
    >>> score_interp = ScoreBasedInterpreter()
    >>> pattern_interp = PatternMatcher()
    >>> 
    >>> ensemble = EnsembleInterpreter([rule_interp, score_interp, pattern_interp])
    >>> ensemble.set_aggregation('majority_vote')
    >>> 
    >>> decision = ensemble(state)
    """
    
    def __init__(self, interpreters: List[Interpreter]):
        """
        Parameters
        ----------
        interpreters : List[Interpreter]
            List of interpreters to ensemble
        """
        self.interpreters = interpreters
        self.aggregation = 'majority_vote'
        self.weights = [1.0] * len(interpreters)
    
    def set_aggregation(self, method: str):
        """
        Set aggregation method
        
        Parameters
        ----------
        method : str
            'majority_vote', 'weighted_vote', 'unanimous'
        """
        self.aggregation = method
    
    def set_weights(self, weights: List[float]):
        """Set weights for weighted voting"""
        if len(weights) != len(self.interpreters):
            raise ValueError("Number of weights must match number of interpreters")
        self.weights = weights
    
    def interpret(self, state: State) -> Any:
        """Ensemble interpretation"""
        # Get decisions from all interpreters
        decisions = [interp(state) for interp in self.interpreters]
        
        if self.aggregation == 'majority_vote':
            # Most common decision
            unique, counts = np.unique(decisions, return_counts=True)
            return unique[np.argmax(counts)]
        
        elif self.aggregation == 'weighted_vote':
            # Weighted voting
            decision_scores = {}
            for decision, weight in zip(decisions, self.weights):
                decision_scores[decision] = decision_scores.get(decision, 0) + weight
            return max(decision_scores, key=decision_scores.get)
        
        elif self.aggregation == 'unanimous':
            # All must agree
            if len(set(decisions)) == 1:
                return decisions[0]
            else:
                return 'hold'  # Default if no agreement
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")