# Tutorial 3: Adaptive Thresholds

Learn to build thresholds that adapt to changing data distributions.

## Table of Contents

1. [Why Adaptive Thresholds?](#why-adaptive-thresholds)
2. [Streaming Adaptation](#streaming-adaptation)
3. [Topology-Aware Adaptation](#topology-aware-adaptation)
4. [Multi-Scale Tracking](#multi-scale-tracking)
5. [Feature Scoring](#feature-scoring)
6. [Practical Examples](#practical-examples)

## Why Adaptive Thresholds?

### The Problem with Fixed Thresholds
```python
import numpy as np
import matplotlib.pyplot as plt
from channelpy.pipeline.encoders import ThresholdEncoder

# Regime 1: Low volatility
regime1 = np.random.randn(300) * 0.5

# Regime 2: High volatility  
regime2 = np.random.randn(300) * 3.0 + 2.0

# Combined data
data = np.concatenate([regime1, regime2])

# Fixed threshold encoder
fixed_encoder = ThresholdEncoder(threshold_i=0.0, threshold_q=0.5)
states = fixed_encoder(data)

# Problem: In regime 2, almost everything is HIGH!
regime2_states = states[300:]
high_count = sum(1 for s in regime2_states if s.i and s.q)
print(f"In regime 2, {high_count/300*100:.1f}% are HIGH")
print("Fixed threshold loses discriminative power!")
```

### The Solution: Adaptive Thresholds
```python
from channelpy.adaptive import StreamingAdaptiveThreshold

# Adaptive threshold
adaptive = StreamingAdaptiveThreshold(window_size=100)

states_adaptive = []
for value in data:
    adaptive.update(value)
    state = adaptive.encode(value)
    states_adaptive.append(state)

# Now the threshold adapts!
print(f"Regime 1 threshold: ~{adaptive.threshold_i:.2f}")
# After regime 2, threshold will be higher
```

## Streaming Adaptation

### Basic Streaming Threshold
```python
from channelpy.adaptive import StreamingAdaptiveThreshold
import numpy as np

# Create streaming threshold
threshold = StreamingAdaptiveThreshold(
    window_size=1000,      # Look at last 1000 samples
    adaptation_rate=0.01   # Smooth adaptation
)

# Generate non-stationary stream
stream = []
for i in range(2000):
    if i < 500:
        stream.append(np.random.randn())
    elif i < 1000:
        stream.append(np.random.randn() * 2 + 3)
    elif i < 1500:
        stream.append(np.random.randn() * 0.5)
    else:
        stream.append(np.random.randn() * 1.5 - 2)

# Process stream
threshold_i_history = []
threshold_q_history = []
states = []

for value in stream:
    # Update (learns from new data)
    threshold.update(value)
    
    # Encode with current thresholds
    state = threshold.encode(value)
    states.append(state)
    
    # Record thresholds
    stats = threshold.get_stats()
    threshold_i_history.append(stats['threshold_i'])
    threshold_q_history.append(stats['threshold_q'])

# Visualize adaptation
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(stream, alpha=0.5, label='Data')
plt.plot(threshold_i_history, label='Threshold i', linewidth=2)
plt.plot(threshold_q_history, label='Threshold q', linewidth=2)
plt.legend()
plt.title('Adaptive Thresholds')
plt.ylabel('Value')

plt.subplot(2, 1, 2)
state_ints = [s.to_int() for s in states]
plt.step(range(len(state_ints)), state_ints, where='post')
plt.yticks([0, 1, 2, 3], ['∅', 'φ', 'δ', 'ψ'])
plt.title('State Evolution')
plt.ylabel('State')
plt.xlabel('Time')
plt.tight_layout()
plt.show()
```

### Understanding Adaptation Parameters
```python
# Fast adaptation (responds quickly, less stable)
fast = StreamingAdaptiveThreshold(
    window_size=50,        # Short memory
    adaptation_rate=0.1    # Quick adaptation
)

# Slow adaptation (stable, slower response)
slow = StreamingAdaptiveThreshold(
    window_size=2000,      # Long memory
    adaptation_rate=0.001  # Slow adaptation
)

# Balanced (recommended for most cases)
balanced = StreamingAdaptiveThreshold(
    window_size=1000,
    adaptation_rate=0.01
)
```

## Topology-Aware Adaptation 🌟

**This is ChannelPy's unique innovation!**

### The Concept

Instead of blindly using mean/std, analyze the **shape** of your data:
- Multimodal? → Threshold between modes
- Skewed? → Asymmetric thresholds
- Clustered? → Gap-based thresholds
- Heavy-tailed? → Robust percentiles
```python
from channelpy.adaptive import TopologyAdaptiveThreshold
import numpy as np

# Create topology-aware threshold
topo_threshold = TopologyAdaptiveThreshold(
    window_size=1000,
    adaptation_rate=0.01,
    topology_update_interval=100  # Recompute topology every 100 samples
)
```

### Example 1: Bimodal Distribution
```python
# Generate bimodal data (two clusters)
cluster1 = np.random.normal(-2, 0.5, 500)
cluster2 = np.random.normal(2, 0.5, 500)
bimodal_data = np.concatenate([cluster1, cluster2])
np.random.shuffle(bimodal_data)

# Process with topology-aware threshold
for value in bimodal_data:
    topo_threshold.update(value)

# Check detected topology
topology = topo_threshold.get_topology()
print(f"Detected modality: {topology.modality}")
print(f"Mode locations: {topology.local_maxima}")

# Get thresholds
thresholds = topo_threshold.get_thresholds()
print(f"\nAdaptation strategy: {thresholds['adaptation_strategy']}")
print(f"Threshold i: {thresholds['threshold_i']:.3f}")
print(f"Threshold q: {thresholds['threshold_q']:.3f}")

# The threshold is placed BETWEEN the two modes!
print(f"\nMode 1 center: ~-2.0")
print(f"Mode 2 center: ~2.0")
print(f"Threshold between them: {thresholds['threshold_i']:.3f}")
```

### Example 2: Skewed Distribution
```python
# Generate right-skewed data
skewed_data = np.random.exponential(scale=2.0, size=1000)

# Create new topology threshold
topo_skewed = TopologyAdaptiveThreshold(window_size=1000)

for value in skewed_data:
    topo_skewed.update(value)

topology = topo_skewed.get_topology()
print(f"Skewness: {topology.skewness:.3f}")

thresholds = topo_skewed.get_thresholds()
print(f"Strategy: {thresholds['adaptation_strategy']}")
print(f"Threshold i: {thresholds['threshold_i']:.3f}")
print(f"Threshold q: {thresholds['threshold_q']:.3f}")

# For right-skewed data, thresholds are adjusted asymmetrically
```

### Example 3: Heavy-Tailed Distribution
```python
# Generate heavy-tailed data (Student's t with df=2)
from scipy.stats import t
heavy_tailed = t.rvs(df=2, size=1000)

# Topology-aware threshold
topo_heavy = TopologyAdaptiveThreshold(window_size=1000)

for value in heavy_tailed:
    topo_heavy.update(value)

topology = topo_heavy.get_topology()
print(f"Kurtosis: {topology.kurtosis:.3f}")  # High kurtosis

thresholds = topo_heavy.get_thresholds()
print(f"Strategy: {thresholds['adaptation_strategy']}")

# For heavy-tailed data, uses robust percentiles instead of mean/std
```

### Visualizing Topology Features
```python
# Generate complex distribution
complex_data = np.concatenate([
    np.random.normal(-3, 0.3, 300),
    np.random.normal(0, 0.5, 400),
    np.random.normal(3, 0.4, 300)
])
np.random.shuffle(complex_data)

# Process with topology threshold
topo = TopologyAdaptiveThreshold(window_size=1000)
for value in complex_data:
    topo.update(value)

# Plot topology and thresholds
fig = topo.plot_topology_and_thresholds()
plt.show()
```

## Multi-Scale Tracking

Track thresholds at multiple timescales for regime detection!

### Basic Multi-Scale Setup
```python
from channelpy.adaptive import MultiScaleAdaptiveThreshold

# Create multi-scale tracker
multiscale = MultiScaleAdaptiveThreshold(
    use_topology=True,     # Use topology-aware thresholds
    fast_window=100,       # Fast scale
    medium_window=1000,    # Medium scale
    slow_window=10000      # Slow scale
)
```

### Regime Detection
```python
import numpy as np

# Generate data with regime changes
data = []

# Regime 1: Stable (0-300)
data.extend(np.random.randn(300) * 0.5)

# Regime 2: Volatile (300-600)
data.extend(np.random.randn(300) * 3.0)

# Regime 3: Trending (600-900)
data.extend(np.linspace(0, 5, 300) + np.random.randn(300) * 0.3)

# Process with multi-scale
regime_changes = []

for i, value in enumerate(data):
    multiscale.update(value)
    
    # Check for regime change
    if multiscale.regime_changed():
        change = multiscale.get_last_regime_change()
        regime_changes.append({
            'time': i,
            'from': change.from_regime.value,
            'to': change.to_regime.value,
            'confidence': change.confidence
        })
        print(f"Time {i}: {change.from_regime.value} → {change.to_regime.value} "
              f"(confidence: {change.confidence:.2f})")

print(f"\nTotal regime changes detected: {len(regime_changes)}")
```

### Adaptive Encoding Based on Regime
```python
# Encode using regime-appropriate scale
states = []

for value in data:
    multiscale.update(value)
    
    # Automatically selects appropriate scale based on regime
    state = multiscale.encode_adaptive(value)
    states.append(state)
    
    # Get regime info
    regime = multiscale.get_current_regime()

# Different regimes use different scales:
# - STABLE → slow scale (long-term baseline)
# - VOLATILE → fast scale (quick adaptation)
# - TRANSITIONING → medium scale (balanced)
```

### Visualizing Multi-Scale
```python
# Plot all scales and regime changes
fig = multiscale.plot_multiscale()
plt.show()
```

## Feature Scoring

Add context-aware scoring to threshold decisions!

### Basic Feature Scoring
```python
from channelpy.adaptive import FeatureScorer, create_trading_scorer

# Create scorer with multiple dimensions
scorer = FeatureScorer()

# Add scoring dimensions
def relevance_scorer(value, context):
    """Score based on predictive power"""
    historical = context.get('historical_values', [])
    if len(historical) < 10:
        return 0.5
    # Simplified: higher values are more relevant
    return (value - min(historical)) / (max(historical) - min(historical))

def confidence_scorer(value, context):
    """Score based on data quality"""
    sample_size = context.get('sample_size', 0)
    if sample_size < 30:
        return sample_size / 30  # Penalty for small sample
    return 1.0

scorer.add_dimension('relevance', relevance_scorer, weight=2.0)
scorer.add_dimension('confidence', confidence_scorer, weight=1.0)
```

### Using Scorer with Context
```python
# Build context
context = {
    'historical_values': list(np.random.randn(100)),
    'sample_size': 100,
    'age_seconds': 10
}

# Score a value
value = 1.5
aggregate_score, dimension_scores = scorer.score_and_aggregate(value, context)

print(f"Value: {value}")
print(f"Overall score: {aggregate_score:.3f}")
print("\nDimension scores:")
for dim, score in dimension_scores.items():
    print(f"  {dim}: {score:.3f}")

# Get explanation
explanation = scorer.explain_score(value, context)
print(f"\n{explanation}")
```

### Pre-Configured Scorers
```python
from channelpy.adaptive import (
    create_trading_scorer,
    create_medical_scorer,
    create_signal_scorer
)

# Trading scorer (emphasizes timeliness and strength)
trading_scorer = create_trading_scorer()

# Medical scorer (emphasizes confidence and consistency)
medical_scorer = create_medical_scorer()

# Signal scorer (emphasizes stability and noise level)
signal_scorer = create_signal_scorer()

# Use with context
trading_context = {
    'historical_values': recent_prices,
    'historical_outcomes': recent_returns,
    'sample_size': len(recent_prices),
    'age_seconds': 0  # Real-time
}

score = trading_scorer.score_and_aggregate(signal_value, trading_context)
```

## Practical Examples

### Example 1: Stock Price Monitoring
```python
from channelpy.adaptive import TopologyAdaptiveThreshold
import numpy as np

# Simulate stock prices with volatility regimes
prices = []

# Low volatility period
prices.extend(100 + np.cumsum(np.random.randn(200) * 0.5))

# High volatility period
prices.extend(prices[-1] + np.cumsum(np.random.randn(200) * 2.0))

# Back to low volatility
prices.extend(prices[-1] + np.cumsum(np.random.randn(200) * 0.5))

# Adaptive threshold for price changes
price_changes = np.diff(prices)
threshold = TopologyAdaptiveThreshold(window_size=100)

signals = []
for change in price_changes:
    threshold.update(change)
    state = threshold.encode(change)
    
    # Generate trading signal
    if state.i and state.q:
        signals.append('STRONG_MOVE')
    elif state.i:
        signals.append('MOVE')
    else:
        signals.append('NORMAL')

print(f"Strong moves detected: {signals.count('STRONG_MOVE')}")
print(f"Regular moves: {signals.count('MOVE')}")
print(f"Normal: {signals.count('NORMAL')}")
```

### Example 2: Sensor Anomaly Detection
```python
from channelpy.adaptive import MultiScaleAdaptiveThreshold

# Simulate sensor data with anomalies
sensor_data = []

# Normal operation
sensor_data.extend(np.random.randn(400) * 0.3 + 20.0)

# Drift (gradual change)
sensor_data.extend(20.0 + np.linspace(0, 5, 200) + np.random.randn(200) * 0.3)

# Anomaly (sudden spike)
sensor_data.extend([25.0] * 10 + [35.0] * 5 + [25.0] * 10)

# Back to normal
sensor_data.extend(np.random.randn(400) * 0.3 + 25.0)

# Multi-scale detector
detector = MultiScaleAdaptiveThreshold(use_topology=True)

anomalies = []
for i, value in enumerate(sensor_data):
    detector.update(value)
    
    # Check for regime change (potential anomaly)
    if detector.regime_changed():
        change = detector.get_last_regime_change()
        if change.to_regime.value == 'volatile':
            anomalies.append(i)
            print(f"Anomaly detected at time {i}: {change.from_regime.value} → {change.to_regime.value}")

print(f"\nTotal anomalies detected: {len(anomalies)}")
```

### Example 3: Adaptive Quality Control
```python
from channelpy.adaptive import TopologyAdaptiveThreshold, FeatureScorer

# Manufacturing measurements
measurements = []

# Good production run
measurements.extend(np.random.normal(10.0, 0.1, 500))

# Slight drift
measurements.extend(np.random.normal(10.2, 0.1, 200))

# Bad batch (high variance)
measurements.extend(np.random.normal(10.0, 0.5, 100))

# Back to good
measurements.extend(np.random.normal(10.0, 0.1, 200))

# Adaptive quality checker
quality_checker = TopologyAdaptiveThreshold(window_size=200)

# Feature scorer for quality
scorer = FeatureScorer()

def measurement_quality(value, context):
    """Score based on proximity to target"""
    target = context.get('target', 10.0)
    tolerance = context.get('tolerance', 0.5)
    distance = abs(value - target)
    return max(0, 1 - distance / tolerance)

scorer.add_dimension('quality', measurement_quality, weight=1.0)

# Process measurements
quality_states = []
for i, measurement in enumerate(measurements):
    quality_checker.update(measurement)
    state = quality_checker.encode(measurement)
    
    # Score quality
    context = {'target': 10.0, 'tolerance': 0.5}
    quality_score, _ = scorer.score_and_aggregate(measurement, context)
    
    quality_states.append({
        'value': measurement,
        'state': str(state),
        'quality_score': quality_score
    })
    
    # Alert on low quality
    if quality_score < 0.5:
        print(f"Quality alert at measurement {i}: value={measurement:.3f}, score={quality_score:.3f}")

# Summary
good_count = sum(1 for q in quality_states if q['quality_score'] > 0.8)
print(f"\nHigh quality measurements: {good_count}/{len(measurements)} ({good_count/len(measurements)*100:.1f}%)")
```

## Best Practices

### 1. Choose Appropriate Window Size
```python
# Small window (50-200): Responsive to quick changes
fast_threshold = StreamingAdaptiveThreshold(window_size=100)

# Medium window (500-2000): Balanced
balanced_threshold = StreamingAdaptiveThreshold(window_size=1000)

# Large window (5000+): Stable baseline
stable_threshold = StreamingAdaptiveThreshold(window_size=10000)
```

### 2. Use Topology-Aware for Non-Normal Data
```python
# If your data is:
# - Multimodal (multiple peaks)
# - Heavily skewed
# - Clustered
# - Heavy-tailed
# → Use TopologyAdaptiveThreshold!

threshold = TopologyAdaptiveThreshold(window_size=1000)
```

### 3. Use Multi-Scale for Regime Detection
```python
# If you need to detect:
# - Volatility changes
# - Trend changes
# - Structural breaks
# → Use MultiScaleAdaptiveThreshold!

multiscale = MultiScaleAdaptiveThreshold(use_topology=True)
```

### 4. Combine with Feature Scoring
```python
# For sophisticated decision-making:
# 1. Use adaptive threshold for encoding
# 2. Use feature scorer for decision confidence

threshold = TopologyAdaptiveThreshold(window_size=1000)
scorer = create_trading_scorer()

for value in stream:
    threshold.update(value)
    state = threshold.encode(value)
    
    context = build_context(value)
    confidence, _ = scorer.score_and_aggregate(value, context)
    
    if state == PSI and confidence > 0.8:
        execute_high_confidence_action()
```

## Key Takeaways

1. **Fixed thresholds fail** when distributions change
2. **Streaming adaptation** maintains discriminative power
3. **Topology-aware adaptation** understands data shape
4. **Multi-scale tracking** detects regime changes
5. **Feature scoring** adds context-aware confidence
6. **Window size** controls responsiveness vs stability
7. **Combine techniques** for robust systems

## What's Next?

See adaptive thresholds in action with complete applications:

➡️ **[Tutorial 4: Trading Bot](04_trading_bot.md)**
➡️ **[Tutorial 5: Medical Diagnosis](05_medical_diagnosis.md)**

## Exercises

1. Create a stream with 3 regime changes and detect them
2. Build a topology-aware threshold for heavily skewed data
3. Compare fixed vs adaptive vs topology-aware on the same data
4. Build a multi-scale detector with custom regime classification

Solutions in the [GitHub repository](https://github.com/channelalgebra/channelpy/tree/main/examples).