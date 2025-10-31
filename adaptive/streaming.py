"""
Streaming adaptive thresholds
"""
import numpy as np
from ..core.state import State, StateArray


class StreamingAdaptiveThreshold:
    """
    Online threshold adaptation for streaming data
    
    Examples
    --------
    >>> threshold = StreamingAdaptiveThreshold(window_size=1000)
    >>> for value in stream:
    ...     threshold.update(value)
    ...     state = threshold.encode(value)
    ...     process(state)
    """
    
    def __init__(self, window_size: int = 1000, adaptation_rate: float = 0.01):
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        # Streaming statistics (Welford's algorithm)
        self.running_mean = 0.0
        self.running_m2 = 0.0
        self.n_samples = 0
        
        # Sliding window
        self.window = []
        
        # Current thresholds
        self.threshold_i = 0.0
        self.threshold_q = 0.0
    
    def update(self, value: float):
        """Update with new value"""
        # Update running statistics
        self.n_samples += 1
        delta = value - self.running_mean
        self.running_mean += delta / self.n_samples
        delta2 = value - self.running_mean
        self.running_m2 += delta * delta2
        
        # Update window
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        
        # Update thresholds
        self._update_thresholds()
    
    def _update_thresholds(self):
        """Update thresholds based on statistics"""
        if self.n_samples < 2:
            return
        
        current_std = np.sqrt(self.running_m2 / (self.n_samples - 1))
        
        # Target thresholds
        target_i = self.running_mean
        target_q = self.running_mean + 0.5 * current_std
        
        # Smooth update (EMA)
        self.threshold_i = (
            (1 - self.adaptation_rate) * self.threshold_i +
            self.adaptation_rate * target_i
        )
        self.threshold_q = (
            (1 - self.adaptation_rate) * self.threshold_q +
            self.adaptation_rate * target_q
        )
    
    def encode(self, value: float) -> State:
        """Encode value with current thresholds"""
        return State(
            i=int(value > self.threshold_i),
            q=int(value > self.threshold_q)
        )
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        return {
            'mean': self.running_mean,
            'std': np.sqrt(self.running_m2 / max(self.n_samples - 1, 1)),
            'n_samples': self.n_samples,
            'threshold_i': self.threshold_i,
            'threshold_q': self.threshold_q
        }




