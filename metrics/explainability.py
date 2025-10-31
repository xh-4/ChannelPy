"""
Explainability metrics

Measures how interpretable and understandable the system is.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
from collections import Counter

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI


@dataclass
class ExplainabilityMetrics:
    """
    Container for explainability metrics
    
    Attributes
    ----------
    clarity : float
        Decision clarity [0, 1]
    complexity : float
        Rule complexity (lower is better) [0, ∞]
    completeness : float
        Explanation completeness [0, 1]
    consistency : float
        Interpretation consistency [0, 1]
    """
    clarity: float = 0.0
    complexity: float = 0.0
    completeness: float = 0.0
    consistency: float = 0.0
    
    def overall_score(self) -> float:
        """
        Overall explainability score
        
        Returns
        -------
        score : float
            Overall score [0, 1]
        """
        # Normalize complexity (assume max reasonable complexity = 10)
        normalized_complexity = max(0, 1 - self.complexity / 10)
        
        score = (
            0.3 * self.clarity +
            0.3 * normalized_complexity +
            0.2 * self.completeness +
            0.2 * self.consistency
        )
        
        return score


def decision_clarity(
    interpreter,
    states: Optional[Union[StateArray, List[State]]] = None,
    num_samples: int = 100
) -> float:
    """
    Measure clarity of decision rules
    
    Clear rules have:
    - Unambiguous mappings (state → decision)
    - Few edge cases
    - Consistent outputs
    
    Parameters
    ----------
    interpreter : Interpreter
        Interpreter to evaluate
    states : StateArray or List[State], optional
        States to test on (if None, tests all possible states)
    num_samples : int
        Number of samples for monte carlo (if testing all states)
        
    Returns
    -------
    clarity : float
        Clarity score [0, 1], where 1 = perfectly clear
        
    Examples
    --------
    >>> from channelpy.pipeline.interpreters import RuleBasedInterpreter
    >>> 
    >>> interpreter = RuleBasedInterpreter()
    >>> interpreter.add_rule(PSI, "BUY")
    >>> interpreter.add_rule(EMPTY, "SELL")
    >>> 
    >>> clarity = decision_clarity(interpreter)
    >>> print(f"Clarity: {clarity:.3f}")
    """
    if states is None:
        # Test all possible states
        test_states = [EMPTY, PHI, DELTA, PSI]
    else:
        if isinstance(states, list):
            test_states = states
        else:
            # Sample from StateArray
            indices = np.random.choice(len(states), min(num_samples, len(states)), replace=False)
            test_states = [states[i] for i in indices]
    
    # Get decisions for each state
    decisions = []
    for state in test_states:
        try:
            decision = interpreter(state)
            decisions.append(decision)
        except:
            decisions.append(None)  # Ambiguous
    
    # Measure clarity
    
    # 1. No ambiguous decisions
    has_ambiguous = None in decisions
    ambiguous_penalty = 0.5 if has_ambiguous else 0.0
    
    # 2. Deterministic (same state → same decision)
    state_to_decisions = {}
    for state, decision in zip(test_states, decisions):
        state_key = str(state)
        if state_key not in state_to_decisions:
            state_to_decisions[state_key] = []
        state_to_decisions[state_key].append(decision)
    
    determinism = np.mean([
        len(set(decs)) == 1 
        for decs in state_to_decisions.values() 
        if None not in decs
    ])
    
    # 3. Decision coverage (all states have decisions)
    coverage = np.mean([d is not None for d in decisions])
    
    # Combine
    clarity = (0.3 * (1 - ambiguous_penalty) + 0.4 * determinism + 0.3 * coverage)
    
    return float(clarity)


def rule_complexity(
    interpreter,
    method: str = 'count'
) -> float:
    """
    Measure complexity of interpretation rules
    
    Simpler rules are more interpretable
    
    Parameters
    ----------
    interpreter : Interpreter
        Interpreter to evaluate
    method : str
        Complexity measure:
        - 'count': Number of rules
        - 'depth': Maximum decision tree depth
        - 'conditions': Number of conditions
        
    Returns
    -------
    complexity : float
        Complexity score (lower is simpler)
        
    Examples
    --------
    >>> interpreter = RuleBasedInterpreter()
    >>> interpreter.add_rule(PSI, "BUY")
    >>> interpreter.add_rule(DELTA, "HOLD")
    >>> interpreter.add_rule(EMPTY, "SELL")
    >>> 
    >>> complexity = rule_complexity(interpreter)
    >>> print(f"Complexity: {complexity:.1f}")  # 3.0 (3 rules)
    """
    if method == 'count':
        # Count number of rules
        if hasattr(interpreter, 'rules'):
            return float(len(interpreter.rules))
        elif hasattr(interpreter, 'lookup_table'):
            return float(len(interpreter.lookup_table))
        else:
            # Can't determine - assume moderate complexity
            return 5.0
    
    elif method == 'depth':
        # Estimate decision tree depth
        # For now, use heuristic based on rule count
        if hasattr(interpreter, 'rules'):
            num_rules = len(interpreter.rules)
            # Approximate depth: log2(rules)
            return float(np.log2(max(num_rules, 1)))
        else:
            return 3.0  # Default moderate depth
    
    elif method == 'conditions':
        # Count total number of conditions
        if hasattr(interpreter, 'rules'):
            # Each rule has 1 condition (the state)
            return float(len(interpreter.rules))
        else:
            return 5.0
    
    else:
        raise ValueError(f"Unknown method: {method}")


def explanation_completeness(
    interpreter,
    test_states: Optional[List[State]] = None
) -> float:
    """
    Measure completeness of explanations
    
    Complete explanations cover:
    - All possible inputs
    - Reasoning for decision
    - Alternative possibilities
    
    Parameters
    ----------
    interpreter : Interpreter
        Interpreter to evaluate
    test_states : List[State], optional
        States to test (default: all 4 states)
        
    Returns
    -------
    completeness : float
        Completeness score [0, 1]
        
    Examples
    --------
    >>> completeness = explanation_completeness(interpreter)
    >>> print(f"Completeness: {completeness:.3f}")
    """
    if test_states is None:
        test_states = [EMPTY, PHI, DELTA, PSI]
    
    # Check coverage
    decisions = []
    for state in test_states:
        try:
            decision = interpreter(state)
            decisions.append(decision is not None)
        except:
            decisions.append(False)
    
    coverage = np.mean(decisions)
    
    # Check if interpreter can explain (has explain method)
    has_explain = hasattr(interpreter, 'explain')
    explain_capability = 1.0 if has_explain else 0.5
    
    # If has explain, check quality
    if has_explain:
        explanations_quality = []
        for state in test_states:
            try:
                explanation = interpreter.explain(state)
                # Quality based on length (longer = more detailed)
                quality = min(len(str(explanation)) / 100, 1.0)
                explanations_quality.append(quality)
            except:
                explanations_quality.append(0.0)
        
        avg_explanation_quality = np.mean(explanations_quality)
    else:
        avg_explanation_quality = 0.0
    
    # Combine
    completeness = (
        0.5 * coverage +
        0.2 * explain_capability +
        0.3 * avg_explanation_quality
    )
    
    return float(completeness)


def interpretation_consistency(
    interpreter,
    states: Union[StateArray, List[State]]
) -> float:
    """
    Measure consistency of interpretations
    
    Similar states should yield similar decisions
    
    Parameters
    ----------
    interpreter : Interpreter
        Interpreter to evaluate
    states : StateArray or List[State]
        States to test
        
    Returns
    -------
    consistency : float
        Consistency score [0, 1]
        
    Examples
    --------
    >>> states = StateArray.from_bits(i=[1,1,1,0], q=[1,1,0,0])
    >>> consistency = interpretation_consistency(interpreter, states)
    """
    if isinstance(states, StateArray):
        state_list = [states[i] for i in range(len(states))]
    else:
        state_list = states
    
    # Get decisions
    decisions = []
    for state in state_list:
        try:
            decision = interpreter(state)
            decisions.append(decision)
        except:
            decisions.append(None)
    
    # Measure consistency: states differing by 1 bit should have similar decisions
    
    consistencies = []
    for i, state1 in enumerate(state_list):
        for j, state2 in enumerate(state_list[i+1:], i+1):
            # Hamming distance
            hamming = (state1.i != state2.i) + (state1.q != state2.q)
            
            if hamming == 1:  # Differ by 1 bit
                # Check if decisions are "similar"
                dec1 = decisions[i]
                dec2 = decisions[j]
                
                if dec1 is None or dec2 is None:
                    similar = 0.0
                elif dec1 == dec2:
                    similar = 1.0
                else:
                    # Try to measure similarity numerically
                    try:
                        # If numeric decisions
                        diff = abs(float(dec1) - float(dec2))
                        similar = 1.0 / (1.0 + diff)
                    except:
                        # Different decisions
                        similar = 0.0
                
                consistencies.append(similar)
    
    if not consistencies:
        return 0.5  # Not enough data
    
    return float(np.mean(consistencies))


def explainability_score(
    interpreter,
    states: Optional[Union[StateArray, List[State]]] = None
) -> ExplainabilityMetrics:
    """
    Comprehensive explainability evaluation
    
    Parameters
    ----------
    interpreter : Interpreter
        Interpreter to evaluate
    states : StateArray or List[State], optional
        States for testing
        
    Returns
    -------
    metrics : ExplainabilityMetrics
        Complete explainability metrics
        
    Examples
    --------
    >>> metrics = explainability_score(interpreter, test_states)
    >>> print(metrics)
    >>> print(f"Overall: {metrics.overall_score():.3f}")
    """
    metrics = ExplainabilityMetrics()
    
    # Clarity
    metrics.clarity = decision_clarity(interpreter, states)
    
    # Complexity
    metrics.complexity = rule_complexity(interpreter)
    
    # Completeness
    test_states = None
    if states is not None:
        if isinstance(states, StateArray):
            test_states = [states[i] for i in range(min(4, len(states)))]
        else:
            test_states = states[:4]
    
    metrics.completeness = explanation_completeness(interpreter, test_states)
    
    # Consistency
    if states is not None:
        metrics.consistency = interpretation_consistency(interpreter, states)
    
    return metrics


def generate_explanation(
    state: State,
    decision: Any,
    context: Optional[Dict] = None,
    verbosity: str = 'medium'
) -> str:
    """
    Generate human-readable explanation for a decision
    
    Parameters
    ----------
    state : State
        Input state
    decision : Any
        Decision made
    context : Dict, optional
        Additional context (thresholds, features, etc.)
    verbosity : str
        'brief', 'medium', or 'detailed'
        
    Returns
    -------
    explanation : str
        Human-readable explanation
        
    Examples
    --------
    >>> state = PSI
    >>> decision = "BUY"
    >>> context = {
    ...     'feature_value': 0.85,
    ...     'threshold_i': 0.5,
    ...     'threshold_q': 0.75
    ... }
    >>> 
    >>> explanation = generate_explanation(state, decision, context, verbosity='detailed')
    >>> print(explanation)
    """
    lines = []
    
    # State description
    state_desc = {
        EMPTY: "absent (below all thresholds)",
        PHI: "expected but not present (above q-threshold only)",
        DELTA: "present but not validated (above i-threshold only)",
        PSI: "resonant (above both thresholds)"
    }
    
    lines.append(f"State: {state} - {state_desc.get(state, 'unknown')}")
    lines.append(f"Decision: {decision}")
    
    if verbosity in ['medium', 'detailed']:
        lines.append("")
        lines.append("Reasoning:")
        
        # Explain based on state
        if state == PSI:
            lines.append("  • Signal is strong (i=1): Feature exceeds presence threshold")
            lines.append("  • Signal is validated (q=1): Feature exceeds quality threshold")
            lines.append("  • Both conditions met → Confident positive signal")
        
        elif state == DELTA:
            lines.append("  • Signal is present (i=1): Feature exceeds presence threshold")
            lines.append("  • Signal not validated (q=0): Feature below quality threshold")
            lines.append("  • Puncture state → Tentative/uncertain signal")
        
        elif state == PHI:
            lines.append("  • Signal not present (i=0): Feature below presence threshold")
            lines.append("  • Signal expected (q=1): Feature exceeds quality threshold")
            lines.append("  • Hole state → Unexpected absence or opportunity")
        
        elif state == EMPTY:
            lines.append("  • Signal absent (i=0): Feature below presence threshold")
            lines.append("  • Signal not expected (q=0): Feature below quality threshold")
            lines.append("  • Empty state → Clear negative signal")
    
    if verbosity == 'detailed' and context:
        lines.append("")
        lines.append("Context:")
        
        if 'feature_value' in context:
            lines.append(f"  • Feature value: {context['feature_value']:.3f}")
        
        if 'threshold_i' in context:
            lines.append(f"  • Presence threshold (i): {context['threshold_i']:.3f}")
        
        if 'threshold_q' in context:
            lines.append(f"  • Quality threshold (q): {context['threshold_q']:.3f}")
        
        if 'confidence' in context:
            lines.append(f"  • Confidence: {context['confidence']:.1%}")
        
        # Additional context
        for key, value in context.items():
            if key not in ['feature_value', 'threshold_i', 'threshold_q', 'confidence']:
                lines.append(f"  • {key}: {value}")
    
    return "\n".join(lines)


def compare_explainability(
    interpreter1,
    interpreter2,
    test_states: Optional[List[State]] = None,
    names: Tuple[str, str] = ("Interpreter 1", "Interpreter 2")
) -> Dict[str, Any]:
    """
    Compare explainability of two interpreters
    
    Parameters
    ----------
    interpreter1, interpreter2 : Interpreter
        Interpreters to compare
    test_states : List[State], optional
        States for testing
    names : Tuple[str, str]
        Names for interpreters
        
    Returns
    -------
    comparison : Dict
        Comparison results
        
    Examples
    --------
    >>> simple_interp = RuleBasedInterpreter()
    >>> complex_interp = FSMInterpreter()
    >>> 
    >>> comparison = compare_explainability(
    ...     simple_interp, complex_interp,
    ...     names=("Simple", "Complex")
    ... )
    >>> print(comparison['summary'])
    """
    # Evaluate both
    metrics1 = explainability_score(interpreter1, test_states)
    metrics2 = explainability_score(interpreter2, test_states)
    
    comparison = {
        'names': names,
        'metrics': {
            names[0]: metrics1,
            names[1]: metrics2
        },
        'differences': {
            'clarity': metrics2.clarity - metrics1.clarity,
            'complexity': metrics1.complexity - metrics2.complexity,  # Lower is better
            'completeness': metrics2.completeness - metrics1.completeness,
            'consistency': metrics2.consistency - metrics1.consistency
        },
        'winner': None
    }
    
    # Determine winner
    score1 = metrics1.overall_score()
    score2 = metrics2.overall_score()
    
    if score2 > score1:
        comparison['winner'] = names[1]
    elif score1 > score2:
        comparison['winner'] = names[0]
    else:
        comparison['winner'] = "Tie"
    
    # Summary
    comparison['summary'] = f"""
Explainability Comparison
=========================

{names[0]}:
  Overall Score: {score1:.3f}
  Clarity: {metrics1.clarity:.3f}
  Complexity: {metrics1.complexity:.1f} (lower is better)
  Completeness: {metrics1.completeness:.3f}
  Consistency: {metrics1.consistency:.3f}

{names[1]}:
  Overall Score: {score2:.3f}
  Clarity: {metrics2.clarity:.3f}
  Complexity: {metrics2.complexity:.1f} (lower is better)
  Completeness: {metrics2.completeness:.3f}
  Consistency: {metrics2.consistency:.3f}

Most Explainable: {comparison['winner']}
    """.strip()
    
    return comparison