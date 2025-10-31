"""
Simple Adaptive System Demo

This demonstrates the exact usage pattern requested:
- Feature scoring integration
- Topology-aware adaptive thresholds
- Multi-scale regime detection
- Intelligent decision making
"""

import numpy as np
from channelpy.adaptive import (
    FeatureScorer, 
    TopologyAdaptiveThreshold,
    MultiScaleAdaptiveThreshold,
    create_trading_scorer
)

# Create scorer
scorer = create_trading_scorer()

# Create topology-aware tracker
topology_threshold = TopologyAdaptiveThreshold(
    window_size=1000,
    feature_scorer=scorer
)

# Or use multi-scale for regime detection
multiscale = MultiScaleAdaptiveThreshold(use_topology=True)

# Generate sample data stream
np.random.seed(42)
data_stream = np.random.normal(0, 1, 2000)

# Simple decision interpreter
def interpret(state):
    """Simple decision interpreter"""
    if state.to_int() == 3:  # PSI
        return "STRONG_BUY"
    elif state.to_int() == 2:  # PHI
        return "BUY"
    elif state.to_int() == 1:  # DELTA
        return "HOLD"
    else:  # EMPTY
        return "SELL"

# Process stream
print("Processing data stream with adaptive system...")
print("=" * 50)

for i, value in enumerate(data_stream):
    multiscale.update(value)
    
    # Check regime
    if multiscale.regime_changed():
        change = multiscale.get_last_regime_change()
        print(f"Regime change at sample {i}: {change.from_regime.value} → {change.to_regime.value}")
    
    # Encode with regime-appropriate threshold
    state = multiscale.encode_adaptive(value)
    
    # Make decision based on state
    decision = interpret(state)
    
    # Print progress every 500 samples
    if (i + 1) % 500 == 0:
        regime_info = multiscale.get_regime_info()
        print(f"Sample {i+1}: regime={regime_info['current_regime']}, "
              f"state={state}, decision={decision}")

print("\n" + "=" * 50)
print("Processing complete!")
print(f"Final regime: {multiscale.get_current_regime().value}")
print(f"Total regime changes: {len(multiscale.regime_history)}")




