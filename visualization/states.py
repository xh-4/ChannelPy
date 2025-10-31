"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import numpy as np
from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI


def plot_states(states: StateArray, title: str = "Channel States"):
    """
    Plot state sequence
    
    Examples
    --------
    >>> states = StateArray.from_bits(i=[1,0,1,1], q=[1,1,0,1])
    >>> plot_states(states)
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Convert to integers for plotting
    state_ints = states.to_ints()
    
    # Plot as steps
    ax.step(range(len(states)), state_ints, where='post', linewidth=2)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['∅', 'φ', 'δ', 'ψ'])
    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Color regions
    colors = ['lightgray', 'lightblue', 'lightyellow', 'lightgreen']
    for i in range(4):
        mask = state_ints == i
        if np.any(mask):
            ax.axhspan(i-0.5, i+0.5, alpha=0.3, color=colors[i])
    
    return fig, ax


def plot_state_distribution(states: StateArray, title: str = "State Distribution"):
    """
    Plot distribution of states
    
    Examples
    --------
    >>> states = StateArray.from_bits(i=np.random.randint(0,2,100),
    ...                                q=np.random.randint(0,2,100))
    >>> plot_state_distribution(states)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    counts = states.count_by_state()
    
    labels = ['∅', 'δ', 'φ', 'ψ']
    values = [counts[s] for s in [EMPTY, DELTA, PHI, PSI]]
    colors = ['lightgray', 'lightyellow', 'lightblue', 'lightgreen']
    
    ax.bar(labels, values, color=colors, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentages
    total = sum(values)
    for i, (label, value) in enumerate(zip(labels, values)):
        pct = 100 * value / total
        ax.text(i, value, f'{pct:.1f}%', ha='center', va='bottom')
    
    return fig, ax


def plot_threshold_adaptation(values, thresholds_i, thresholds_q,
                              title="Adaptive Thresholds"):
    """
    Plot values with adaptive thresholds
    
    Examples
    --------
    >>> values = np.random.randn(1000)
    >>> threshold = StreamingAdaptiveThreshold()
    >>> thresholds_i, thresholds_q = [], []
    >>> for v in values:
    ...     threshold.update(v)
    ...     thresholds_i.append(threshold.threshold_i)
    ...     thresholds_q.append(threshold.threshold_q)
    >>> plot_threshold_adaptation(values, thresholds_i, thresholds_q)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(values, alpha=0.5, label='Values', linewidth=0.5)
    ax.plot(thresholds_i, label='Threshold i', linewidth=2, color='orange')
    ax.plot(thresholds_q, label='Threshold q', linewidth=2, color='red')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax




