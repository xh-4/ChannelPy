"""
Multi-scale threshold tracking for regime detection

Maintains thresholds at multiple timescales to detect distribution shifts
and adapt appropriately to different volatility regimes.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..core.state import State, StateArray
from .streaming import StreamingAdaptiveThreshold
from .topology_adaptive import TopologyAdaptiveThreshold


class RegimeType(Enum):
    """Detected regime types"""
    STABLE = "stable"
    TRANSITIONING = "transitioning"
    VOLATILE = "volatile"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    UNKNOWN = "unknown"


@dataclass
class RegimeChange:
    """
    Information about a detected regime change
    
    Attributes
    ----------
    timestamp : int
        Update count when change detected
    from_regime : RegimeType
        Previous regime
    to_regime : RegimeType
        New regime
    confidence : float
        Confidence in detection (0-1)
    divergence_measure : float
        Magnitude of divergence between scales
    """
    timestamp: int
    from_regime: RegimeType
    to_regime: RegimeType
    confidence: float
    divergence_measure: float


class MultiScaleAdaptiveThreshold:
    """
    Track thresholds at multiple timescales
    
    Maintains three scales:
    - Fast (100 samples): Quick reaction to changes
    - Medium (1000 samples): Balanced responsiveness
    - Slow (10000 samples): Long-term baseline
    
    Comparing scales enables:
    - Regime change detection
    - Appropriate threshold selection
    - Volatility estimation
    
    Examples
    --------
    >>> tracker = MultiScaleAdaptiveThreshold()
    >>> 
    >>> for value in data_stream:
    ...     tracker.update(value)
    ...     
    ...     # Check for regime change
    ...     if tracker.regime_changed():
    ...         change = tracker.get_last_regime_change()
    ...         print(f"Regime change: {change.from_regime} → {change.to_regime}")
    ...     
    ...     # Get appropriate threshold for current regime
    ...     state = tracker.encode_adaptive(value)
    """
    
    def __init__(
        self,
        use_topology: bool = True,
        fast_window: int = 100,
        medium_window: int = 1000,
        slow_window: int = 10000
    ):
        """
        Parameters
        ----------
        use_topology : bool
            Use topology-aware thresholds (recommended)
        fast_window : int
            Window size for fast scale
        medium_window : int
            Window size for medium scale
        slow_window : int
            Window size for slow scale
        """
        self.use_topology = use_topology
        
        # Create threshold trackers for each scale
        if use_topology:
            self.fast = TopologyAdaptiveThreshold(
                window_size=fast_window,
                adaptation_rate=0.1  # Fast adaptation
            )
            self.medium = TopologyAdaptiveThreshold(
                window_size=medium_window,
                adaptation_rate=0.01  # Balanced
            )
            self.slow = TopologyAdaptiveThreshold(
                window_size=slow_window,
                adaptation_rate=0.001  # Slow, stable
            )
        else:
            self.fast = StreamingAdaptiveThreshold(
                window_size=fast_window,
                adaptation_rate=0.1
            )
            self.medium = StreamingAdaptiveThreshold(
                window_size=medium_window,
                adaptation_rate=0.01
            )
            self.slow = StreamingAdaptiveThreshold(
                window_size=slow_window,
                adaptation_rate=0.001
            )
        
        # State tracking
        self.update_count = 0
        self.current_regime = RegimeType.UNKNOWN
        self.previous_regime = RegimeType.UNKNOWN
        self.regime_history: List[RegimeChange] = []
        
        # Divergence tracking
        self.divergence_history = []
    
    def update(self, value: float):
        """
        Update all scales with new value
        
        Parameters
        ----------
        value : float
            New data value
        """
        self.fast.update(value)
        self.medium.update(value)
        self.slow.update(value)
        
        self.update_count += 1
        
        # Check for regime change periodically
        if self.update_count % 50 == 0:
            self._detect_regime()
    
    def _detect_regime(self):
        """Detect current regime from scale divergence"""
        # Get thresholds from each scale
        fast_stats = self.fast.get_thresholds() if hasattr(self.fast, 'get_thresholds') else self.fast.get_stats()
        medium_stats = self.medium.get_thresholds() if hasattr(self.medium, 'get_thresholds') else self.medium.get_stats()
        slow_stats = self.slow.get_thresholds() if hasattr(self.slow, 'get_thresholds') else self.slow.get_stats()
        
        # Compute divergences
        fast_medium_div = self._compute_divergence(fast_stats, medium_stats)
        fast_slow_div = self._compute_divergence(fast_stats, slow_stats)
        medium_slow_div = self._compute_divergence(medium_stats, slow_stats)
        
        # Record divergence
        self.divergence_history.append({
            'update_count': self.update_count,
            'fast_medium': fast_medium_div,
            'fast_slow': fast_slow_div,
            'medium_slow': medium_slow_div
        })
        
        # Classify regime
        self.previous_regime = self.current_regime
        self.current_regime = self._classify_regime(
            fast_medium_div, 
            fast_slow_div, 
            medium_slow_div
        )
        
        # Record regime change
        if self.current_regime != self.previous_regime:
            change = RegimeChange(
                timestamp=self.update_count,
                from_regime=self.previous_regime,
                to_regime=self.current_regime,
                confidence=self._compute_confidence(fast_medium_div, fast_slow_div),
                divergence_measure=fast_slow_div
            )
            self.regime_history.append(change)
    
    def _compute_divergence(
        self, 
        stats1: Dict, 
        stats2: Dict
    ) -> float:
        """
        Compute divergence between two threshold sets
        
        Returns normalized divergence measure
        """
        # Get thresholds
        if 'threshold_i' in stats1:
            t1_i = stats1['threshold_i']
            t1_q = stats1['threshold_q']
        else:
            t1_i = stats1['mean']
            t1_q = stats1['mean'] + 0.5 * stats1['std']
        
        if 'threshold_i' in stats2:
            t2_i = stats2['threshold_i']
            t2_q = stats2['threshold_q']
        else:
            t2_i = stats2['mean']
            t2_q = stats2['mean'] + 0.5 * stats2['std']
        
        # Normalize by scale
        if 'std' in stats2:
            scale = stats2['std']
        elif 'threshold_q' in stats2 and 'threshold_i' in stats2:
            scale = stats2['threshold_q'] - stats2['threshold_i']
        else:
            scale = 1.0
        
        if scale == 0:
            return 0.0
        
        # Compute normalized differences
        i_div = abs(t1_i - t2_i) / scale
        q_div = abs(t1_q - t2_q) / scale
        
        # Combined divergence
        divergence = np.sqrt(i_div**2 + q_div**2)
        
        return divergence
    
    def _classify_regime(
        self, 
        fast_medium_div: float,
        fast_slow_div: float,
        medium_slow_div: float
    ) -> RegimeType:
        """
        Classify regime based on scale divergences
        
        Returns
        -------
        regime : RegimeType
            Detected regime
        """
        # Thresholds for regime classification
        LOW_DIV = 0.5
        HIGH_DIV = 2.0
        
        # All scales aligned → stable
        if fast_slow_div < LOW_DIV and medium_slow_div < LOW_DIV:
            return RegimeType.STABLE
        
        # Fast diverges from slow → transitioning or volatile
        if fast_slow_div > HIGH_DIV:
            # Check if medium also diverges
            if medium_slow_div > HIGH_DIV:
                return RegimeType.VOLATILE  # Sustained change
            else:
                return RegimeType.TRANSITIONING  # Recent change
        
        # Medium diverges from slow → trending
        if medium_slow_div > LOW_DIV and fast_medium_div < LOW_DIV:
            return RegimeType.TRENDING
        
        # Fast and medium differ but both near slow → mean reverting
        if fast_medium_div > LOW_DIV and fast_slow_div < LOW_DIV:
            return RegimeType.MEAN_REVERTING
        
        return RegimeType.UNKNOWN
    
    def _compute_confidence(
        self, 
        fast_medium_div: float, 
        fast_slow_div: float
    ) -> float:
        """
        Compute confidence in regime detection
        
        Higher divergence → higher confidence
        """
        # Combine divergences
        avg_div = (fast_medium_div + fast_slow_div) / 2
        
        # Map to [0, 1]
        confidence = min(avg_div / 3.0, 1.0)
        
        return confidence
    
    def regime_changed(self) -> bool:
        """
        Check if regime changed on last update
        
        Returns
        -------
        changed : bool
            True if regime changed
        """
        return (
            len(self.regime_history) > 0 and 
            self.regime_history[-1].timestamp == self.update_count
        )
    
    def get_last_regime_change(self) -> Optional[RegimeChange]:
        """Get most recent regime change"""
        if self.regime_history:
            return self.regime_history[-1]
        return None
    
    def get_current_regime(self) -> RegimeType:
        """Get current regime"""
        return self.current_regime
    
    def encode_adaptive(self, value: float) -> State:
        """
        Encode using appropriate scale for current regime
        
        Parameters
        ----------
        value : float
            Value to encode
            
        Returns
        -------
        state : State
            Encoded state using regime-appropriate threshold
        """
        # Choose scale based on regime
        if self.current_regime == RegimeType.STABLE:
            # Use slow scale (stable baseline)
            return self.slow.encode(value)
        
        elif self.current_regime == RegimeType.VOLATILE:
            # Use fast scale (quick adaptation)
            return self.fast.encode(value)
        
        elif self.current_regime == RegimeType.TRANSITIONING:
            # Use medium scale (balanced)
            return self.medium.encode(value)
        
        elif self.current_regime == RegimeType.TRENDING:
            # Use medium scale
            return self.medium.encode(value)
        
        elif self.current_regime == RegimeType.MEAN_REVERTING:
            # Use slow scale (resist noise)
            return self.slow.encode(value)
        
        else:
            # Unknown → use medium scale
            return self.medium.encode(value)
    
    def get_all_thresholds(self) -> Dict[str, Dict]:
        """
        Get thresholds from all scales
        
        Returns
        -------
        thresholds : Dict
            Dictionary with 'fast', 'medium', 'slow' keys
        """
        return {
            'fast': self.fast.get_thresholds() if hasattr(self.fast, 'get_thresholds') else self.fast.get_stats(),
            'medium': self.medium.get_thresholds() if hasattr(self.medium, 'get_thresholds') else self.medium.get_stats(),
            'slow': self.slow.get_thresholds() if hasattr(self.slow, 'get_thresholds') else self.slow.get_stats()
        }
    
    def get_regime_info(self) -> Dict:
        """
        Get comprehensive regime information
        
        Returns
        -------
        info : Dict
            Current regime, divergences, history
        """
        recent_divs = self.divergence_history[-10:] if self.divergence_history else []
        
        return {
            'current_regime': self.current_regime.value,
            'update_count': self.update_count,
            'recent_divergences': recent_divs,
            'num_regime_changes': len(self.regime_history),
            'last_change': self.get_last_regime_change()
        }
    
    def plot_multiscale(self):
        """
        Plot thresholds across scales with regime annotations
        
        Requires matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Requires matplotlib for plotting")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Get threshold history (simplified - would need to track this)
        # For now, plot current thresholds
        thresholds = self.get_all_thresholds()
        
        # Plot 1: Thresholds
        ax1 = axes[0]
        ax1.axhline(thresholds['fast']['threshold_i'], 
                   label='Fast i', linestyle='--', color='lightblue', linewidth=2)
        ax1.axhline(thresholds['fast']['threshold_q'], 
                   label='Fast q', linestyle='--', color='blue', linewidth=2)
        ax1.axhline(thresholds['medium']['threshold_i'], 
                   label='Medium i', linestyle='--', color='lightgreen', linewidth=2)
        ax1.axhline(thresholds['medium']['threshold_q'], 
                   label='Medium q', linestyle='--', color='green', linewidth=2)
        ax1.axhline(thresholds['slow']['threshold_i'], 
                   label='Slow i', linestyle='--', color='lightcoral', linewidth=2)
        ax1.axhline(thresholds['slow']['threshold_q'], 
                   label='Slow q', linestyle='--', color='red', linewidth=2)
        ax1.set_ylabel('Threshold Value')
        ax1.set_title('Multi-Scale Thresholds')
        ax1.legend(loc='upper right', ncol=3)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Divergences
        if self.divergence_history:
            ax2 = axes[1]
            updates = [d['update_count'] for d in self.divergence_history]
            fast_medium = [d['fast_medium'] for d in self.divergence_history]
            fast_slow = [d['fast_slow'] for d in self.divergence_history]
            medium_slow = [d['medium_slow'] for d in self.divergence_history]
            
            ax2.plot(updates, fast_medium, label='Fast-Medium', linewidth=2)
            ax2.plot(updates, fast_slow, label='Fast-Slow', linewidth=2)
            ax2.plot(updates, medium_slow, label='Medium-Slow', linewidth=2)
            ax2.axhline(0.5, color='orange', linestyle=':', alpha=0.5, label='Low threshold')
            ax2.axhline(2.0, color='red', linestyle=':', alpha=0.5, label='High threshold')
            ax2.set_ylabel('Divergence')
            ax2.set_title('Scale Divergences')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regime history
        if self.regime_history:
            ax3 = axes[2]
            
            # Create regime timeline
            regime_colors = {
                RegimeType.STABLE: 'green',
                RegimeType.TRANSITIONING: 'yellow',
                RegimeType.VOLATILE: 'red',
                RegimeType.TRENDING: 'blue',
                RegimeType.MEAN_REVERTING: 'purple',
                RegimeType.UNKNOWN: 'gray'
            }
            
            for i, change in enumerate(self.regime_history):
                start = change.timestamp
                end = self.regime_history[i+1].timestamp if i+1 < len(self.regime_history) else self.update_count
                
                ax3.axvspan(start, end, 
                           color=regime_colors.get(change.to_regime, 'gray'),
                           alpha=0.3,
                           label=change.to_regime.value if i == 0 else "")
            
            ax3.set_ylabel('Regime')
            ax3.set_xlabel('Update Count')
            ax3.set_title('Regime Evolution')
            ax3.set_ylim(-0.5, 0.5)
            ax3.set_yticks([])
            
            # Add legend with unique regimes
            handles, labels = ax3.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax3.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        return fig




