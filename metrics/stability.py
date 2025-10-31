"""
Threshold stability metrics

Measures how stable thresholds are over time and across distribution changes.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class StabilityMetrics:
    """
    Container for stability metrics
    
    Attributes
    ----------
    variance : float
        Threshold variance over time
    drift_rate : float
        Rate of threshold drift
    regime_consistency : float
        Consistency across regimes [0, 1]
    adaptation_quality : float
        Quality of adaptation [0, 1]
    """
    variance: float = 0.0
    drift_rate: float = 0.0
    regime_consistency: float = 0.0
    adaptation_quality: float = 0.0
    
    def overall_score(self) -> float:
        """
        Overall stability score
        
        Returns
        -------
        score : float
            Overall score [0, 1], where 1 = very stable
        """
        # Normalize variance (assume max reasonable std = 1.0)
        normalized_variance = max(0, 1 - min(self.variance, 1.0))
        
        # Normalize drift rate (assume max reasonable rate = 0.01)
        normalized_drift = max(0, 1 - min(self.drift_rate / 0.01, 1.0))
        
        score = (
            0.3 * normalized_variance +
            0.3 * normalized_drift +
            0.2 * self.regime_consistency +
            0.2 * self.adaptation_quality
        )
        
        return score


def threshold_stability(
    threshold_history: List[Dict[str, float]],
    window_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Measure threshold stability over time
    
    Parameters
    ----------
    threshold_history : List[Dict]
        History of thresholds with keys 'threshold_i', 'threshold_q'
    window_size : int, optional
        Window for computing rolling statistics
        
    Returns
    -------
    stability : Dict[str, float]
        Stability metrics
        
    Examples
    --------
    >>> history = [
    ...     {'threshold_i': 0.50, 'threshold_q': 0.75},
    ...     {'threshold_i': 0.51, 'threshold_q': 0.76},
    ...     {'threshold_i': 0.49, 'threshold_q': 0.74},
    ... ]
    >>> stability = threshold_stability(history)
    >>> print(f"Variance: {stability['variance_i']:.4f}")
    >>> print(f"Drift rate: {stability['drift_rate_i']:.4f}")
    """
    if not threshold_history:
        return {
            'variance_i': 0.0,
            'variance_q': 0.0,
            'drift_rate_i': 0.0,
            'drift_rate_q': 0.0,
            'range_i': 0.0,
            'range_q': 0.0
        }
    
    # Extract threshold sequences
    thresholds_i = np.array([h.get('threshold_i', 0) for h in threshold_history])
    thresholds_q = np.array([h.get('threshold_q', 0) for h in threshold_history])
    
    # Variance
    variance_i = np.var(thresholds_i)
    variance_q = np.var(thresholds_q)
    
    # Drift rate (average absolute change per step)
    if len(thresholds_i) > 1:
        changes_i = np.abs(np.diff(thresholds_i))
        changes_q = np.abs(np.diff(thresholds_q))
        drift_rate_i = np.mean(changes_i)
        drift_rate_q = np.mean(changes_q)
    else:
        drift_rate_i = 0.0
        drift_rate_q = 0.0
    
    # Range
    range_i = np.max(thresholds_i) - np.min(thresholds_i)
    range_q = np.max(thresholds_q) - np.min(thresholds_q)
    
    # Rolling stability (if window specified)
    if window_size and len(thresholds_i) >= window_size:
        rolling_vars_i = []
        rolling_vars_q = []
        
        for i in range(len(thresholds_i) - window_size + 1):
            window_i = thresholds_i[i:i + window_size]
            window_q = thresholds_q[i:i + window_size]
            rolling_vars_i.append(np.var(window_i))
            rolling_vars_q.append(np.var(window_q))
        
        rolling_variance_i = np.mean(rolling_vars_i)
        rolling_variance_q = np.mean(rolling_vars_q)
    else:
        rolling_variance_i = variance_i
        rolling_variance_q = variance_q
    
    return {
        'variance_i': float(variance_i),
        'variance_q': float(variance_q),
        'drift_rate_i': float(drift_rate_i),
        'drift_rate_q': float(drift_rate_q),
        'range_i': float(range_i),
        'range_q': float(range_q),
        'rolling_variance_i': float(rolling_variance_i),
        'rolling_variance_q': float(rolling_variance_q)
    }


def drift_detection(
    threshold_history: List[Dict[str, float]],
    window_size: int = 50,
    significance: float = 0.05
) -> Dict[str, Any]:
    """
    Detect significant threshold drift
    
    Uses statistical tests to identify drift points
    
    Parameters
    ----------
    threshold_history : List[Dict]
        History of thresholds
    window_size : int
        Window for comparison
    significance : float
        Significance level for tests
        
    Returns
    -------
    drift_info : Dict
        Drift detection results
        
    Examples
    --------
    >>> drift = drift_detection(threshold_history, window_size=50)
    >>> if drift['drift_detected']:
    ...     print(f"Drift detected at update {drift['drift_points']}")
    """
    thresholds_i = np.array([h.get('threshold_i', 0) for h in threshold_history])
    thresholds_q = np.array([h.get('threshold_q', 0) for h in threshold_history])
    
    drift_points_i = []
    drift_points_q = []
    
    # Sliding window comparison
    if len(thresholds_i) >= 2 * window_size:
        for i in range(window_size, len(thresholds_i) - window_size + 1, window_size // 2):
            # Compare window before and after
            before_i = thresholds_i[i - window_size:i]
            after_i = thresholds_i[i:i + window_size]
            
            before_q = thresholds_q[i - window_size:i]
            after_q = thresholds_q[i:i + window_size]
            
            # Statistical test (Mann-Whitney U)
            try:
                _, p_value_i = stats.mannwhitneyu(before_i, after_i, alternative='two-sided')
                _, p_value_q = stats.mannwhitneyu(before_q, after_q, alternative='two-sided')
                
                if p_value_i < significance:
                    drift_points_i.append(i)
                
                if p_value_q < significance:
                    drift_points_q.append(i)
            except:
                pass  # Not enough data for test
    
    return {
        'drift_detected': len(drift_points_i) > 0 or len(drift_points_q) > 0,
        'drift_points_i': drift_points_i,
        'drift_points_q': drift_points_q,
        'num_drifts': len(drift_points_i) + len(drift_points_q),
        'drift_rate': (len(drift_points_i) + len(drift_points_q)) / max(len(threshold_history), 1)
    }


def regime_consistency(
    threshold_history: List[Dict[str, float]],
    regime_labels: List[str]
) -> Dict[str, float]:
    """
    Measure consistency of thresholds within regimes
    
    Thresholds should be stable within a regime and adapt between regimes
    
    Parameters
    ----------
    threshold_history : List[Dict]
        History of thresholds
    regime_labels : List[str]
        Regime label for each threshold
        
    Returns
    -------
    consistency : Dict[str, float]
        Consistency metrics
        
    Examples
    --------
    >>> regimes = ['stable', 'stable', 'volatile', 'volatile', 'stable']
    >>> consistency = regime_consistency(threshold_history, regimes)
    >>> print(f"Within-regime variance: {consistency['within_regime_variance']:.4f}")
    >>> print(f"Between-regime variance: {consistency['between_regime_variance']:.4f}")
    """
    if len(threshold_history) != len(regime_labels):
        raise ValueError("History and labels must have same length")
    
    thresholds_i = np.array([h.get('threshold_i', 0) for h in threshold_history])
    thresholds_q = np.array([h.get('threshold_q', 0) for h in threshold_history])
    regime_labels = np.array(regime_labels)
    
    # Within-regime variance
    within_vars_i = []
    within_vars_q = []
    
    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        regime_thresholds_i = thresholds_i[mask]
        regime_thresholds_q = thresholds_q[mask]
        
        if len(regime_thresholds_i) > 1:
            within_vars_i.append(np.var(regime_thresholds_i))
            within_vars_q.append(np.var(regime_thresholds_q))
    
    within_variance_i = np.mean(within_vars_i) if within_vars_i else 0.0
    within_variance_q = np.mean(within_vars_q) if within_vars_q else 0.0
    
    # Between-regime variance
    regime_means_i = []
    regime_means_q = []
    
    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        regime_means_i.append(np.mean(thresholds_i[mask]))
        regime_means_q.append(np.mean(thresholds_q[mask]))
    
    between_variance_i = np.var(regime_means_i) if len(regime_means_i) > 1 else 0.0
    between_variance_q = np.var(regime_means_q) if len(regime_means_q) > 1 else 0.0
    
    # Consistency score (low within, high between is good)
    total_var_i = within_variance_i + between_variance_i
    total_var_q = within_variance_q + between_variance_q
    
    if total_var_i > 0:
        consistency_i = between_variance_i / total_var_i
    else:
        consistency_i = 1.0
    
    if total_var_q > 0:
        consistency_q = between_variance_q / total_var_q
    else:
        consistency_q = 1.0
    
    return {
        'within_regime_variance_i': float(within_variance_i),
        'within_regime_variance_q': float(within_variance_q),
        'between_regime_variance_i': float(between_variance_i),
        'between_regime_variance_q': float(between_variance_q),
        'consistency_score_i': float(consistency_i),
        'consistency_score_q': float(consistency_q),
        'overall_consistency': float((consistency_i + consistency_q) / 2)
    }


def adaptation_quality(
    threshold_history: List[Dict[str, float]],
    data_history: List[float],
    optimal_percentile: float = 0.5
) -> Dict[str, float]:
    """
    Measure quality of threshold adaptation
    
    Good adaptation keeps thresholds near optimal percentile of data
    
    Parameters
    ----------
    threshold_history : List[Dict]
        History of thresholds
    data_history : List[float]
        History of data values
    optimal_percentile : float
        Target percentile for threshold_i (default: 0.5 = median)
        
    Returns
    -------
    quality : Dict[str, float]
        Adaptation quality metrics
        
    Examples
    --------
    >>> quality = adaptation_quality(threshold_history, data_values)
    >>> print(f"Tracking error: {quality['tracking_error']:.4f}")
    >>> print(f"Adaptation lag: {quality['lag']:.1f}")
    """
    if len(threshold_history) != len(data_history):
        raise ValueError("History lengths must match")
    
    thresholds_i = np.array([h.get('threshold_i', 0) for h in threshold_history])
    data = np.array(data_history)
    
    # Compute optimal thresholds at each point (hindsight)
    window_size = 100
    optimal_thresholds = []
    
    for i in range(len(data)):
        # Use past window
        start = max(0, i - window_size)
        window_data = data[start:i+1]
        
        if len(window_data) > 0:
            optimal = np.percentile(window_data, optimal_percentile * 100)
        else:
            optimal = 0.0
        
        optimal_thresholds.append(optimal)
    
    optimal_thresholds = np.array(optimal_thresholds)
    
    # Tracking error (how close to optimal)
    tracking_errors = np.abs(thresholds_i - optimal_thresholds)
    mean_tracking_error = np.mean(tracking_errors)
    
    # Adaptation lag (cross-correlation)
    if len(thresholds_i) > 10:
        # Compute cross-correlation
        correlation = np.correlate(
            thresholds_i - np.mean(thresholds_i),
            optimal_thresholds - np.mean(optimal_thresholds),
            mode='full'
        )
        lags = np.arange(-len(thresholds_i) + 1, len(thresholds_i))
        
        # Find lag at maximum correlation
        max_corr_idx = np.argmax(correlation)
        lag = lags[max_corr_idx]
    else:
        lag = 0.0
    
    # Responsiveness (how quickly thresholds respond to changes)
    if len(data) > 1:
        data_changes = np.abs(np.diff(data))
        threshold_changes = np.abs(np.diff(thresholds_i))
        
        # Correlation between data changes and threshold changes
        if np.std(data_changes) > 0 and np.std(threshold_changes) > 0:
            responsiveness = np.corrcoef(data_changes, threshold_changes)[0, 1]
        else:
            responsiveness = 0.0
    else:
        responsiveness = 0.0
    
    # Stability vs responsiveness tradeoff
    threshold_variance = np.var(thresholds_i)
    
    return {
        'tracking_error': float(mean_tracking_error),
        'lag': float(lag),
        'responsiveness': float(max(0, responsiveness)),  # Clip negative correlations
        'stability': float(1.0 / (1.0 + threshold_variance)),
        'tradeoff_score': float(max(0, responsiveness) * (1.0 / (1.0 + threshold_variance)))
    }


def stability_report(
    threshold_history: List[Dict[str, float]],
    data_history: Optional[List[float]] = None,
    regime_labels: Optional[List[str]] = None
) -> StabilityMetrics:
    """
    Comprehensive stability report
    
    Parameters
    ----------
    threshold_history : List[Dict]
        History of thresholds
    data_history : List[float], optional
        History of data values
    regime_labels : List[str], optional
        Regime labels
        
    Returns
    -------
    metrics : StabilityMetrics
        Complete stability metrics
        
    Examples
    --------
    >>> report = stability_report(
    ...     threshold_history,
    ...     data_history=data_values,
    ...     regime_labels=regimes
    ... )
    >>> print(report)
    >>> print(f"Overall stability: {report.overall_score():.3f}")
    """
    metrics = StabilityMetrics()
    
    # Basic stability
    stability = threshold_stability(threshold_history)
    metrics.variance = (stability['variance_i'] + stability['variance_q']) / 2
    metrics.drift_rate = (stability['drift_rate_i'] + stability['drift_rate_q']) / 2
    
    # Regime consistency
    if regime_labels is not None:
        consistency = regime_consistency(threshold_history, regime_labels)
        metrics.regime_consistency = consistency['overall_consistency']
    
    # Adaptation quality
    if data_history is not None:
        quality = adaptation_quality(threshold_history, data_history)
        metrics.adaptation_quality = quality['tradeoff_score']
    
    return metrics


def visualize_stability(
    threshold_history: List[Dict[str, float]],
    data_history: Optional[List[float]] = None,
    regime_labels: Optional[List[str]] = None
):
    """
    Visualize threshold stability over time
    
    Requires matplotlib
    
    Parameters
    ----------
    threshold_history : List[Dict]
        History of thresholds
    data_history : List[float], optional
        History of data values
    regime_labels : List[str], optional
        Regime labels
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with stability plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Requires matplotlib for visualization")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    thresholds_i = [h.get('threshold_i', 0) for h in threshold_history]
    thresholds_q = [h.get('threshold_q', 0) for h in threshold_history]
    time = np.arange(len(thresholds_i))
    
    # Plot 1: Thresholds over time
    ax1 = axes[0]
    ax1.plot(time, thresholds_i, label='Threshold i', linewidth=2, color='orange')
    ax1.plot(time, thresholds_q, label='Threshold q', linewidth=2, color='red')
    
    if data_history is not None:
        ax1.plot(time, data_history, label='Data', alpha=0.3, color='gray')
    
    ax1.set_ylabel('Value')
    ax1.set_title('Threshold Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Drift rate
    ax2 = axes[1]
    if len(thresholds_i) > 1:
        drift_i = np.abs(np.diff(thresholds_i))
        drift_q = np.abs(np.diff(thresholds_q))
        
        ax2.plot(time[1:], drift_i, label='Drift i', alpha=0.7, color='orange')
        ax2.plot(time[1:], drift_q, label='Drift q', alpha=0.7, color='red')
        
        # Rolling average
        window = 10
        if len(drift_i) >= window:
            rolling_drift = np.convolve(drift_i, np.ones(window)/window, mode='valid')
            ax2.plot(time[window:], rolling_drift, label=f'{window}-period avg', 
                    linewidth=2, color='blue')
    
    ax2.set_ylabel('Drift Rate')
    ax2.set_title('Threshold Drift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime indicators
    ax3 = axes[2]
    if regime_labels is not None:
        regime_colors = {
            'stable': 'green',
            'volatile': 'red',
            'transitioning': 'yellow',
            'trending': 'blue',
            'mean_reverting': 'purple'
        }
        
        for i, regime in enumerate(regime_labels):
            color = regime_colors.get(regime, 'gray')
            ax3.axvspan(i, i+1, color=color, alpha=0.3)
        
        ax3.set_ylabel('Regime')
        ax3.set_yticks([])
        ax3.set_title('Regime Evolution')
        
        # Legend
        handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.3) 
                  for color in regime_colors.values()]
        labels = list(regime_colors.keys())
        ax3.legend(handles, labels, loc='upper right', ncol=len(labels))
    
    ax3.set_xlabel('Time')
    
    plt.tight_layout()
    return fig