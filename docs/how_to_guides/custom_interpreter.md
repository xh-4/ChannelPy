# How to Create Custom Interpreters

This guide shows you how to create custom interpreters to convert channel states into domain-specific decisions.

## Table of Contents
- [Understanding Interpreters](#understanding-interpreters)
- [Simple Function-Based Interpreters](#simple-function-based-interpreters)
- [Class-Based Interpreters](#class-based-interpreters)
- [Stateful Interpreters](#stateful-interpreters)
- [Multi-State Interpreters](#multi-state-interpreters)
- [Advanced Patterns](#advanced-patterns)

---

## Understanding Interpreters

Interpreters are the final stage in a channel pipeline. They convert channel states into actionable decisions.
```
Features → Encoder → States → Interpreter → Decisions
```

**Key Principles:**
- **Interpretability**: Every decision should be explainable
- **Consistency**: Same state should produce same decision (unless stateful)
- **Domain Alignment**: Output should match domain needs

---

## Simple Function-Based Interpreters

The simplest interpreter is a pure function.

### Example 1: Binary Classification
```python
from channelpy.core import State, PSI, EMPTY, DELTA, PHI

def binary_classifier(state: State) -> str:
    """
    Simple binary classification
    
    Rules:
    - ψ (PSI): Positive class
    - All others: Negative class
    """
    if state == PSI:
        return "POSITIVE"
    else:
        return "NEGATIVE"

# Usage
state = State(1, 1)  # PSI
decision = binary_classifier(state)
print(decision)  # "POSITIVE"
```

### Example 2: Three-Way Decision
```python
def three_way_decision(state: State) -> dict:
    """
    Three-way decision with confidence
    
    Returns:
    - action: 'accept', 'reject', or 'review'
    - confidence: float between 0 and 1
    """
    if state == PSI:
        return {'action': 'accept', 'confidence': 1.0}
    elif state == EMPTY:
        return {'action': 'reject', 'confidence': 1.0}
    else:  # DELTA or PHI
        return {'action': 'review', 'confidence': 0.5}

# Usage
state = State(1, 0)  # DELTA
decision = three_way_decision(state)
print(f"Action: {decision['action']}, Confidence: {decision['confidence']}")
```

### Example 3: Scoring-Based Decision
```python
def scored_decision(state: State) -> dict:
    """
    Convert state to numerical score
    """
    # Define state scores
    state_scores = {
        (0, 0): 0.0,  # EMPTY
        (0, 1): 0.3,  # PHI
        (1, 0): 0.6,  # DELTA
        (1, 1): 1.0   # PSI
    }
    
    score = state_scores[(state.i, state.q)]
    
    return {
        'score': score,
        'decision': 'pass' if score >= 0.5 else 'fail',
        'state': str(state)
    }
```

---

## Class-Based Interpreters

For more complex logic, use classes.

### Template: Base Interpreter
```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseInterpreter(ABC):
    """Base class for interpreters"""
    
    def __init__(self):
        self.is_fitted = False
    
    @abstractmethod
    def interpret(self, state: State) -> Any:
        """Convert state to decision"""
        pass
    
    def fit(self, states, labels=None):
        """Optional: Learn from data"""
        self.is_fitted = True
        return self
    
    def explain(self, state: State) -> str:
        """Explain why this decision was made"""
        return f"State {state} interpreted"
```

### Example 1: Rule-Based Interpreter
```python
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class Rule:
    """A single interpretation rule"""
    condition: Callable[[State], bool]
    action: Any
    priority: int
    description: str

class RuleBasedInterpreter(BaseInterpreter):
    """
    Interpreter based on prioritized rules
    
    Examples
    --------
    >>> interpreter = RuleBasedInterpreter()
    >>> interpreter.add_rule(
    ...     condition=lambda s: s == PSI,
    ...     action={'decision': 'BUY', 'confidence': 1.0},
    ...     priority=1,
    ...     description="Strong buy signal"
    ... )
    >>> interpreter.add_rule(
    ...     condition=lambda s: s == DELTA,
    ...     action={'decision': 'HOLD', 'confidence': 0.5},
    ...     priority=2,
    ...     description="Uncertain signal"
    ... )
    """
    
    def __init__(self):
        super().__init__()
        self.rules: List[Rule] = []
    
    def add_rule(
        self, 
        condition: Callable[[State], bool],
        action: Any,
        priority: int = 0,
        description: str = ""
    ):
        """Add interpretation rule"""
        rule = Rule(
            condition=condition,
            action=action,
            priority=priority,
            description=description
        )
        self.rules.append(rule)
        # Sort by priority (lower number = higher priority)
        self.rules.sort(key=lambda r: r.priority)
    
    def interpret(self, state: State) -> Any:
        """Apply first matching rule"""
        for rule in self.rules:
            if rule.condition(state):
                return rule.action
        
        # Default if no rule matches
        return {'decision': 'UNKNOWN', 'confidence': 0.0}
    
    def explain(self, state: State) -> str:
        """Explain which rule was applied"""
        for rule in self.rules:
            if rule.condition(state):
                return f"Rule applied: {rule.description}"
        return "No matching rule found"
```

### Example 2: Lookup Table Interpreter
```python
class LookupTableInterpreter(BaseInterpreter):
    """
    Simple lookup table: state → action
    
    Examples
    --------
    >>> interpreter = LookupTableInterpreter()
    >>> interpreter.set_mapping(PSI, "approve")
    >>> interpreter.set_mapping(EMPTY, "reject")
    >>> interpreter.set_mapping(DELTA, "review")
    >>> interpreter.set_mapping(PHI, "review")
    >>> 
    >>> decision = interpreter.interpret(State(1, 1))
    >>> print(decision)  # "approve"
    """
    
    def __init__(self, default_action=None):
        super().__init__()
        self.lookup = {}
        self.default_action = default_action
    
    def set_mapping(self, state: State, action: Any):
        """Map state to action"""
        key = (state.i, state.q)
        self.lookup[key] = action
    
    def interpret(self, state: State) -> Any:
        """Lookup action for state"""
        key = (state.i, state.q)
        return self.lookup.get(key, self.default_action)
    
    def explain(self, state: State) -> str:
        """Explain lookup"""
        key = (state.i, state.q)
        if key in self.lookup:
            return f"State {state} maps to: {self.lookup[key]}"
        else:
            return f"State {state} not in lookup table, using default: {self.default_action}"
```

---

## Stateful Interpreters

Interpreters that maintain state across calls.

### Example: FSM Interpreter
```python
from enum import Enum

class FSMState(Enum):
    """Finite state machine states"""
    INIT = "init"
    WAITING = "waiting"
    ACTIVE = "active"
    COOLDOWN = "cooldown"

class FSMInterpreter(BaseInterpreter):
    """
    Finite state machine interpreter
    
    Decision depends on both channel state AND FSM state
    
    Examples
    --------
    >>> interpreter = FSMInterpreter()
    >>> 
    >>> # Process sequence of states
    >>> for channel_state in state_sequence:
    ...     decision = interpreter.interpret(channel_state)
    ...     print(f"FSM State: {interpreter.current_state}, Decision: {decision}")
    """
    
    def __init__(self):
        super().__init__()
        self.current_state = FSMState.INIT
        self.transition_count = 0
        self.history = []
    
    def interpret(self, state: State) -> dict:
        """
        Interpret based on channel state and FSM state
        """
        decision = {
            'action': None,
            'fsm_state': self.current_state.value,
            'transition_count': self.transition_count
        }
        
        # State-dependent interpretation
        if self.current_state == FSMState.INIT:
            if state == PSI:
                decision['action'] = 'activate'
                self._transition_to(FSMState.ACTIVE)
            else:
                decision['action'] = 'wait'
                self._transition_to(FSMState.WAITING)
        
        elif self.current_state == FSMState.WAITING:
            if state == PSI:
                decision['action'] = 'activate'
                self._transition_to(FSMState.ACTIVE)
            else:
                decision['action'] = 'continue_waiting'
        
        elif self.current_state == FSMState.ACTIVE:
            if state == EMPTY:
                decision['action'] = 'deactivate'
                self._transition_to(FSMState.COOLDOWN)
            else:
                decision['action'] = 'continue_active'
        
        elif self.current_state == FSMState.COOLDOWN:
            # Always transition back to waiting after cooldown
            decision['action'] = 'cooldown_complete'
            self._transition_to(FSMState.WAITING)
        
        self.history.append({
            'channel_state': state,
            'fsm_state': self.current_state,
            'decision': decision
        })
        
        return decision
    
    def _transition_to(self, new_state: FSMState):
        """Perform state transition"""
        self.current_state = new_state
        self.transition_count += 1
    
    def reset(self):
        """Reset FSM to initial state"""
        self.current_state = FSMState.INIT
        self.transition_count = 0
        self.history = []
    
    def explain(self, state: State) -> str:
        """Explain current FSM state and transition logic"""
        return f"FSM in {self.current_state.value} state. " \
               f"Channel state {state} will trigger appropriate transition."
```

### Example: Accumulator Interpreter
```python
class AccumulatorInterpreter(BaseInterpreter):
    """
    Accumulate evidence over time before deciding
    
    Examples
    --------
    >>> interpreter = AccumulatorInterpreter(threshold=3)
    >>> 
    >>> # Need 3 PSI states before triggering
    >>> for state in [PSI, PSI, DELTA, PSI, PSI, PSI]:
    ...     decision = interpreter.interpret(state)
    ...     if decision['triggered']:
    ...         print("Action triggered!")
    """
    
    def __init__(self, threshold: int = 3, decay: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.accumulator = 0.0
    
    def interpret(self, state: State) -> dict:
        """
        Accumulate evidence, trigger when threshold reached
        """
        # Add evidence based on state
        if state == PSI:
            self.accumulator += 1.0
        elif state == DELTA:
            self.accumulator += 0.5
        elif state == PHI:
            self.accumulator += 0.3
        # EMPTY adds nothing
        
        # Check threshold
        triggered = self.accumulator >= self.threshold
        
        if triggered:
            # Reset after trigger
            self.accumulator = 0.0
        else:
            # Decay over time
            self.accumulator *= self.decay
        
        return {
            'triggered': triggered,
            'accumulator': self.accumulator,
            'threshold': self.threshold,
            'confidence': min(self.accumulator / self.threshold, 1.0)
        }
    
    def reset(self):
        """Reset accumulator"""
        self.accumulator = 0.0
```

---

## Multi-State Interpreters

Interpreters that handle multiple channel states.

### Example: Parallel Channel Interpreter
```python
from channelpy.core import ParallelChannels

class ParallelInterpreter(BaseInterpreter):
    """
    Interpret multiple parallel channels
    
    Examples
    --------
    >>> interpreter = ParallelInterpreter()
    >>> 
    >>> # Define channel weights
    >>> interpreter.set_weights({
    ...     'technical': 2.0,
    ...     'fundamental': 1.5,
    ...     'sentiment': 1.0
    ... })
    >>> 
    >>> channels = ParallelChannels(
    ...     technical=PSI,
    ...     fundamental=DELTA,
    ...     sentiment=PSI
    ... )
    >>> decision = interpreter.interpret(channels)
    """
    
    def __init__(self):
        super().__init__()
        self.weights = {}
        self.state_values = {
            (0, 0): 0.0,  # EMPTY
            (0, 1): 0.3,  # PHI
            (1, 0): 0.6,  # DELTA
            (1, 1): 1.0   # PSI
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """Set channel weights"""
        self.weights = weights
    
    def interpret(self, channels: ParallelChannels) -> dict:
        """
        Aggregate across channels with weights
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        channel_details = {}
        
        for name, state in channels.to_dict().items():
            # Get weight (default 1.0)
            weight = self.weights.get(name, 1.0)
            
            # Get state value
            value = self.state_values[(state.i, state.q)]
            
            weighted_sum += weight * value
            total_weight += weight
            
            channel_details[name] = {
                'state': str(state),
                'value': value,
                'weight': weight,
                'contribution': weight * value
            }
        
        # Compute aggregate score
        aggregate_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            'aggregate_score': aggregate_score,
            'decision': 'approve' if aggregate_score >= 0.7 else 'reject',
            'channel_details': channel_details,
            'total_weight': total_weight
        }
    
    def explain(self, channels: ParallelChannels) -> str:
        """Explain aggregation"""
        decision = self.interpret(channels)
        
        lines = [f"Aggregate score: {decision['aggregate_score']:.3f}"]
        lines.append("Channel breakdown:")
        
        for name, details in decision['channel_details'].items():
            lines.append(
                f"  {name}: {details['state']} "
                f"(value={details['value']:.2f}, weight={details['weight']:.1f}, "
                f"contribution={details['contribution']:.2f})"
            )
        
        return "\n".join(lines)
```

---

## Advanced Patterns

### Pattern 1: Probabilistic Interpreter
```python
import numpy as np

class ProbabilisticInterpreter(BaseInterpreter):
    """
    Return probabilities rather than hard decisions
    
    Learn probabilities from training data
    """
    
    def __init__(self):
        super().__init__()
        self.class_probs = {}  # (i, q) → {class: prob}
    
    def fit(self, states: List[State], labels: List[str]):
        """Learn P(class | state) from data"""
        from collections import defaultdict
        
        # Count occurrences
        counts = defaultdict(lambda: defaultdict(int))
        
        for state, label in zip(states, labels):
            key = (state.i, state.q)
            counts[key][label] += 1
        
        # Convert to probabilities
        for key, label_counts in counts.items():
            total = sum(label_counts.values())
            self.class_probs[key] = {
                label: count / total 
                for label, count in label_counts.items()
            }
        
        self.is_fitted = True
        return self
    
    def interpret(self, state: State) -> dict:
        """Return probability distribution over classes"""
        key = (state.i, state.q)
        
        if key not in self.class_probs:
            return {'error': 'State not seen during training'}
        
        probs = self.class_probs[key]
        most_likely = max(probs.items(), key=lambda x: x[1])
        
        return {
            'probabilities': probs,
            'most_likely_class': most_likely[0],
            'confidence': most_likely[1]
        }
```

### Pattern 2: Ensemble Interpreter
```python
class EnsembleInterpreter(BaseInterpreter):
    """
    Combine multiple interpreters via voting or averaging
    """
    
    def __init__(self, interpreters: List[BaseInterpreter], method='voting'):
        super().__init__()
        self.interpreters = interpreters
        self.method = method  # 'voting' or 'averaging'
    
    def interpret(self, state: State) -> dict:
        """Combine interpreter outputs"""
        outputs = [interp.interpret(state) for interp in self.interpreters]
        
        if self.method == 'voting':
            return self._voting(outputs)
        elif self.method == 'averaging':
            return self._averaging(outputs)
    
    def _voting(self, outputs: List[dict]) -> dict:
        """Majority vote on decision"""
        from collections import Counter
        
        # Extract decisions
        decisions = [out.get('decision', out.get('action')) for out in outputs]
        
        # Count votes
        votes = Counter(decisions)
        winner, count = votes.most_common(1)[0]
        
        return {
            'decision': winner,
            'votes': dict(votes),
            'confidence': count / len(decisions),
            'num_interpreters': len(decisions)
        }
    
    def _averaging(self, outputs: List[dict]) -> dict:
        """Average numerical outputs"""
        # Assume outputs have 'score' key
        scores = [out.get('score', 0.0) for out in outputs]
        avg_score = np.mean(scores)
        
        return {
            'score': avg_score,
            'individual_scores': scores,
            'decision': 'pass' if avg_score >= 0.5 else 'fail'
        }
```

---

## Integration with Pipeline
```python
from channelpy.pipeline import ChannelPipeline

# Create custom interpreter
interpreter = RuleBasedInterpreter()
interpreter.add_rule(
    condition=lambda s: s == PSI,
    action={'buy': True, 'amount': 1.0},
    priority=1,
    description="Strong buy signal"
)
interpreter.add_rule(
    condition=lambda s: s == EMPTY,
    action={'buy': False, 'amount': 0.0},
    priority=2,
    description="No signal"
)

# Add to pipeline
pipeline = ChannelPipeline()
pipeline.add_preprocessor(my_preprocessor)
pipeline.add_encoder(my_encoder)
pipeline.add_interpreter(interpreter.interpret)  # Pass the method

# Use pipeline
decisions, states = pipeline.transform(data)
```

---

## Best Practices

### 1. **Keep It Simple**
Start with simple function-based interpreters. Only use classes when you need state or complexity.

### 2. **Make It Explainable**
Always implement an `explain()` method. Users need to understand why decisions were made.

### 3. **Validate Inputs**
Check that inputs match expected format:
```python
def interpret(self, state: State) -> dict:
    if not isinstance(state, State):
        raise TypeError(f"Expected State, got {type(state)}")
    # ... rest of logic
```

### 4. **Return Structured Output**
Use dictionaries with consistent keys:
```python
return {
    'decision': 'approve',
    'confidence': 0.95,
    'reasoning': 'Strong positive signal',
    'metadata': {...}
}
```

### 5. **Test Edge Cases**
Test with all four states:
```python
def test_interpreter():
    interpreter = MyInterpreter()
    
    for state in [EMPTY, DELTA, PHI, PSI]:
        decision = interpreter.interpret(state)
        assert 'decision' in decision
        print(f"{state}: {decision}")
```

### 6. **Log Decisions**
For production, log all decisions:
```python
class LoggingInterpreter(BaseInterpreter):
    def __init__(self, base_interpreter):
        self.base = base_interpreter
        self.log = []
    
    def interpret(self, state: State) -> dict:
        decision = self.base.interpret(state)
        
        self.log.append({
            'timestamp': time.time(),
            'state': str(state),
            'decision': decision
        })
        
        return decision
```

---

## Summary

You've learned:
- ✅ Simple function-based interpreters
- ✅ Class-based interpreters with rules
- ✅ Stateful interpreters (FSM, accumulator)
- ✅ Multi-state interpreters for parallel channels
- ✅ Advanced patterns (probabilistic, ensemble)
- ✅ Best practices for production use

**Next Steps:**
- See [Handle Missing Data](handle_missing_data.md) for preprocessing
- See [Debug Pipeline](debug_pipeline.md) for troubleshooting
- Check [API Reference](../api_reference/pipeline.md) for full details