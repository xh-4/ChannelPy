"""
Encoding quality metrics

Measures how well features are encoded to channel states.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI


@dataclass
class QualityMetrics:
    """
    Container for quality metrics
    
    Attributes
    ----------
    accuracy : float
        Classification accuracy (if labels available)
    discrimination_power : float
        Ability to separate classes [0, 1]
    information_content : float
        Bits of information in encoding [0, 2]
    balance : float
        State distribution balance [0, 1]
    consistency : float
        Encoding consistency [0, 1]
    """
    accuracy: float = 0.0
    discrimination_power: float = 0.0
    information_content: float = 0.0
    balance: float = 0.0
    consistency: float = 0.0
    
    def overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted overall quality score
        
        Parameters
        ----------
        weights : Dict, optional
            Weights for each metric (default: equal weights)
            
        Returns
        -------
        score : float
            Overall quality score [0, 1]
        """
        if weights is None:
            weights = {
                'discrimination_power': 0.3,
                'information_content': 0.25,
                'balance': 0.25,
                'consistency': 0.2
            }
        
        score = (
            weights.get('discrimination_power', 0) * self.discrimination_power +
            weights.get('information_content', 0) * self.information_content / 2.0 +  # Normalize to [0,1]
            weights.get('balance', 0) * self.balance +
            weights.get('consistency', 0) * self.consistency
        )
        
        return score


def encoding_accuracy(
    states: Union[StateArray, List[State]],
    labels: np.ndarray,
    method: str = 'state_label'
) -> float:
    """
    Measure encoding accuracy
    
    Computes how well states correspond to true labels
    
    Parameters
    ----------
    states : StateArray or List[State]
        Encoded states
    labels : np.ndarray
        True labels (binary or multi-class)
    method : str
        Accuracy computation method:
        - 'state_label': Align states to labels
        - 'psi_positive': Treat ψ as positive class
        - 'i_bit': Use i-bit as prediction
        
    Returns
    -------
    accuracy : float
        Accuracy score [0, 1]
        
    Examples
    --------
    >>> states = StateArray.from_bits(i=[1,1,0,1], q=[1,0,1,1])
    >>> labels = np.array([1, 0, 0, 1])
    >>> accuracy = encoding_accuracy(states, labels, method='i_bit')
    >>> print(f"Accuracy: {accuracy:.3f}")
    """
    if isinstance(states, list):
        states = StateArray.from_states(states)
    
    labels = np.asarray(labels)
    
    if len(states) != len(labels):
        raise ValueError("States and labels must have same length")
    
    if method == 'state_label':
        # For each unique state, find most common label
        state_ints = states.to_ints()
        predictions = np.zeros(len(states))
        
        for state_val in np.unique(state_ints):
            mask = state_ints == state_val
            most_common_label = np.bincount(labels[mask].astype(int)).argmax()
            predictions[mask] = most_common_label
        
        accuracy = np.mean(predictions == labels)
    
    elif method == 'psi_positive':
        # ψ = positive class, others = negative
        predictions = (states.to_ints() == 3).astype(int)
        accuracy = np.mean(predictions == labels)
    
    elif method == 'i_bit':
        # i-bit as binary prediction
        predictions = states.i
        accuracy = np.mean(predictions == labels)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(accuracy)


def state_distribution_balance(
    states: Union[StateArray, List[State]]
) -> float:
    """
    Measure balance of state distribution
    
    A balanced distribution uses all states roughly equally.
    Unbalanced distributions (e.g., all ψ) lose discriminative power.
    
    Parameters
    ----------
    states : StateArray or List[State]
        Encoded states
        
    Returns
    -------
    balance : float
        Balance score [0, 1], where 1 = perfectly balanced
        
    Examples
    --------
    >>> # Perfectly balanced (equal states)
    >>> states = StateArray.from_bits(
    ...     i=[0,0,1,1], 
    ...     q=[0,1,0,1]
    ... )
    >>> balance = state_distribution_balance(states)
    >>> print(f"Balance: {balance:.3f}")  # Close to 1.0
    """
    if isinstance(states, list):
        states = StateArray.from_states(states)
    
    counts = states.count_by_state()
    total = len(states)
    
    # Compute entropy of distribution
    proportions = np.array([
        counts[EMPTY] / total,
        counts[PHI] / total,
        counts[DELTA] / total,
        counts[PSI] / total
    ])
    
    # Remove zeros for log
    proportions = proportions[proportions > 0]
    
    # Entropy
    entropy = -np.sum(proportions * np.log2(proportions))
    
    # Normalize by max entropy (log2(4) = 2)
    balance = entropy / 2.0
    
    return float(balance)


def discrimination_power(
    states: Union[StateArray, List[State]],
    labels: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None
) -> float:
    """
    Measure discriminative power of encoding
    
    How well do states separate different classes or feature values?
    
    Parameters
    ----------
    states : StateArray or List[State]
        Encoded states
    labels : np.ndarray, optional
        Class labels (for supervised discrimination)
    features : np.ndarray, optional
        Original feature values (for feature separation)
        
    Returns
    -------
    power : float
        Discrimination power [0, 1]
        
    Examples
    --------
    >>> states = StateArray.from_bits(i=[1,1,0,0], q=[1,0,1,0])
    >>> labels = np.array([1, 1, 0, 0])
    >>> power = discrimination_power(states, labels=labels)
    >>> print(f"Discrimination: {power:.3f}")
    """
    if isinstance(states, list):
        states = StateArray.from_states(states)
    
    state_ints = states.to_ints()
    
    if labels is not None:
        # Supervised: How well do states separate labels?
        labels = np.asarray(labels)
        
        # Compute mutual information between states and labels
        power = _mutual_information(state_ints, labels)
        
    elif features is not None:
        # Unsupervised: How well do states separate feature space?
        features = np.asarray(features)
        
        # Compute variance ratio (between-state vs within-state)
        power = _variance_ratio(state_ints, features)
        
    else:
        # No reference: Use state entropy as proxy
        power = state_distribution_balance(states)
    
    return float(np.clip(power, 0, 1))


def information_content(
    states: Union[StateArray, List[State]]
) -> float:
    """
    Measure information content in bits
    
    How many bits of information are actually used?
    
    Parameters
    ----------
    states : StateArray or List[State]
        Encoded states
        
    Returns
    -------
    bits : float
        Information content in bits [0, 2]
        
    Examples
    --------
    >>> # All same state = 0 bits
    >>> states = StateArray.from_bits(i=[1,1,1], q=[1,1,1])
    >>> info = information_content(states)
    >>> print(f"Information: {info:.3f} bits")  # ~0
    >>> 
    >>> # Perfectly random = 2 bits
    >>> states = StateArray.from_bits(
    ...     i=[0,0,1,1] * 25,
    ...     q=[0,1,0,1] * 25
    ... )
    >>> info = information_content(states)
    >>> print(f"Information: {info:.3f} bits")  # ~2.0
    """
    if isinstance(states, list):
        states = StateArray.from_states(states)
    
    state_ints = states.to_ints()
    
    # Compute entropy
    unique, counts = np.unique(state_ints, return_counts=True)
    proportions = counts / len(state_ints)
    
    entropy = -np.sum(proportions * np.log2(proportions))
    
    return float(entropy)


def encoding_consistency(
    states: Union[StateArray, List[State]],
    features: np.ndarray,
    window_size: int = 10
) -> float:
    """
    Measure encoding consistency
    
    Similar features should encode to similar states
    
    Parameters
    ----------
    states : StateArray or List[State]
        Encoded states
    features : np.ndarray
        Original feature values
    window_size : int
        Window for local consistency check
        
    Returns
    -------
    consistency : float
        Consistency score [0, 1]
        
    Examples
    --------
    >>> features = np.array([0.1, 0.11, 0.12, 0.5, 0.51, 0.52])
    >>> states = encoder(features)
    >>> consistency = encoding_consistency(states, features)
    """
    if isinstance(states, list):
        states = StateArray.from_states(states)
    
    features = np.asarray(features)
    state_ints = states.to_ints()
    
    if len(states) < window_size:
        window_size = len(states)
    
    # Sort by feature value
    sorted_indices = np.argsort(features)
    sorted_states = state_ints[sorted_indices]
    
    # Measure consistency in sliding windows
    consistencies = []
    for i in range(len(sorted_states) - window_size + 1):
        window_states = sorted_states[i:i + window_size]
        
        # Most common state in window
        most_common = np.bincount(window_states).argmax()
        
        # Proportion matching most common
        consistency = np.mean(window_states == most_common)
        consistencies.append(consistency)
    
    return float(np.mean(consistencies))


def encoding_quality_report(
    states: Union[StateArray, List[State]],
    labels: Optional[np.ndarray] = None,
    features: Optional[np.ndarray] = None
) -> QualityMetrics:
    """
    Comprehensive encoding quality report
    
    Parameters
    ----------
    states : StateArray or List[State]
        Encoded states
    labels : np.ndarray, optional
        True labels
    features : np.ndarray, optional
        Original features
        
    Returns
    -------
    metrics : QualityMetrics
        Complete quality metrics
        
    Examples
    --------
    >>> report = encoding_quality_report(states, labels=labels, features=features)
    >>> print(report)
    >>> print(f"Overall score: {report.overall_score():.3f}")
    """
    metrics = QualityMetrics()
    
    # Accuracy (if labels available)
    if labels is not None:
        metrics.accuracy = encoding_accuracy(states, labels)
    
    # Discrimination power
    metrics.discrimination_power = discrimination_power(
        states, 
        labels=labels, 
        features=features
    )
    
    # Information content
    metrics.information_content = information_content(states)
    
    # Balance
    metrics.balance = state_distribution_balance(states)
    
    # Consistency (if features available)
    if features is not None:
        metrics.consistency = encoding_consistency(states, features)
    
    return metrics


def compare_encoders(
    encoder1,
    encoder2,
    test_data: np.ndarray,
    test_labels: Optional[np.ndarray] = None,
    names: Tuple[str, str] = ("Encoder 1", "Encoder 2")
) -> Dict[str, Any]:
    """
    Compare two encoders
    
    Parameters
    ----------
    encoder1, encoder2 : Encoder
        Encoders to compare
    test_data : np.ndarray
        Test features
    test_labels : np.ndarray, optional
        Test labels
    names : Tuple[str, str]
        Names for encoders
        
    Returns
    -------
    comparison : Dict
        Comparison results
        
    Examples
    --------
    >>> from channelpy.pipeline.encoders import ThresholdEncoder, LearnedThresholdEncoder
    >>> 
    >>> encoder1 = ThresholdEncoder(threshold_i=0.5, threshold_q=0.75)
    >>> encoder2 = LearnedThresholdEncoder()
    >>> encoder2.fit(train_data, train_labels)
    >>> 
    >>> comparison = compare_encoders(
    ...     encoder1, encoder2, 
    ...     test_data, test_labels,
    ...     names=("Fixed", "Learned")
    ... )
    >>> print(comparison['summary'])
    """
    # Encode with both
    states1 = encoder1(test_data)
    states2 = encoder2(test_data)
    
    # Compute metrics
    metrics1 = encoding_quality_report(states1, labels=test_labels, features=test_data)
    metrics2 = encoding_quality_report(states2, labels=test_labels, features=test_data)
    
    # Compare
    comparison = {
        'names': names,
        'metrics': {
            names[0]: metrics1,
            names[1]: metrics2
        },
        'differences': {
            'accuracy': metrics2.accuracy - metrics1.accuracy,
            'discrimination_power': metrics2.discrimination_power - metrics1.discrimination_power,
            'information_content': metrics2.information_content - metrics1.information_content,
            'balance': metrics2.balance - metrics1.balance,
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
Encoder Comparison
==================

{names[0]}:
  Overall Score: {score1:.3f}
  Discrimination: {metrics1.discrimination_power:.3f}
  Information: {metrics1.information_content:.3f} bits
  Balance: {metrics1.balance:.3f}
  Consistency: {metrics1.consistency:.3f}

{names[1]}:
  Overall Score: {score2:.3f}
  Discrimination: {metrics2.discrimination_power:.3f}
  Information: {metrics2.information_content:.3f} bits
  Balance: {metrics2.balance:.3f}
  Consistency: {metrics2.consistency:.3f}

Winner: {comparison['winner']}
    """.strip()
    
    return comparison


# ============================================================================
# Helper Functions
# ============================================================================

def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mutual information between x and y
    
    Normalized to [0, 1]
    """
    # Compute joint and marginal probabilities
    xy = np.column_stack([x, y])
    
    # Unique joint outcomes
    unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
    p_xy = counts_xy / len(x)
    
    # Marginal probabilities
    unique_x, counts_x = np.unique(x, return_counts=True)
    p_x = counts_x / len(x)
    
    unique_y, counts_y = np.unique(y, return_counts=True)
    p_y = counts_y / len(y)
    
    # Mutual information
    mi = 0.0
    for i, (x_val, y_val) in enumerate(unique_xy):
        p_joint = p_xy[i]
        p_x_marg = p_x[unique_x == x_val][0]
        p_y_marg = p_y[unique_y == y_val][0]
        
        if p_joint > 0:
            mi += p_joint * np.log2(p_joint / (p_x_marg * p_y_marg))
    
    # Normalize by min entropy
    h_x = -np.sum(p_x * np.log2(p_x))
    h_y = -np.sum(p_y * np.log2(p_y))
    min_h = min(h_x, h_y)
    
    if min_h > 0:
        mi_normalized = mi / min_h
    else:
        mi_normalized = 0.0
    
    return mi_normalized


def _variance_ratio(groups: np.ndarray, values: np.ndarray) -> float:
    """
    Compute variance ratio (between-group / within-group)
    
    Normalized to [0, 1]
    """
    # Between-group variance
    overall_mean = np.mean(values)
    group_means = []
    group_sizes = []
    
    for group in np.unique(groups):
        group_values = values[groups == group]
        group_means.append(np.mean(group_values))
        group_sizes.append(len(group_values))
    
    group_means = np.array(group_means)
    group_sizes = np.array(group_sizes)
    
    between_var = np.sum(group_sizes * (group_means - overall_mean) ** 2) / len(values)
    
    # Within-group variance
    within_var = 0.0
    for group in np.unique(groups):
        group_values = values[groups == group]
        group_mean = np.mean(group_values)
        within_var += np.sum((group_values - group_mean) ** 2)
    
    within_var /= len(values)
    
    # Ratio
    if within_var > 0:
        ratio = between_var / (between_var + within_var)
    else:
        ratio = 1.0  # Perfect separation
    
    return ratio