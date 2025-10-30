# Adaptive API Reference

The adaptive module provides sophisticated threshold adaptation mechanisms that respond to data distribution characteristics and topology.

## Overview
```python
from channelpy.adaptive import (
    StreamingAdaptiveThreshold,
    TopologyAdaptiveThreshold,
    MultiScaleAdaptiveThreshold,
    FeatureScorer
)

# Simple streaming adaptation
threshold = StreamingAdaptiveThreshold()
for value in stream:
    threshold.update(value)
    state = threshold.encode(value)

# Topology-aware adaptation (recommended)
topology_threshold = TopologyAdaptiveThreshold()
for value in stream:
    topology_threshold.update(value)
    state = topology_threshold.encode(value)

# Multi-scale with regime detection
multiscale = MultiScaleAdaptiveThreshold(use_topology=True)
for value in stream:
    multiscale.update(value)
    if multiscale.regime_changed():
        print(f"Regime: {multiscale.get_current_regime()}")
    state = multiscale.encode_adaptive(value)
```

---

## Module Structure
```
channelpy.adaptive/
├── streaming.py          # Online threshold adaptation
├── topology_adaptive.py  # Topology-aware thresholds (KEY INNOVATION)
├── multiscale.py        # Multi-scale tracking and regime detection
├── thresholds.py        # Threshold learning utilities
└── scoring.py           # Multi-dimensional feature scoring
```

---

## Streaming Adaptation

### `StreamingAdaptiveThreshold`

Online threshold adaptation for streaming data using Welford's algorithm.
```python
from channelpy.adaptive import StreamingAdaptiveThreshold

threshold = StreamingAdaptiveThreshold(
    window_size=1000,
    adaptation_rate=0.01
)
```

#### Parameters

- `window_size` : int, default=1000
  - Size of sliding window for statistics
- `adaptation_rate` : float, default=0.01
  - Rate of adaptation (0 = frozen, 1 = instant)
  - Higher values → faster adaptation to changes
  - Lower values → more stable thresholds

#### Attributes

- `running_mean` : float
  - Current running mean (Welford's algorithm)
- `running_m2` : float
  - Running M2 statistic for variance
- `n_samples` : int
  - Total samples processed
- `threshold_i` : float
  - Current i-threshold (presence)
- `threshold_q` : float
  - Current q-threshold (membership)
- `window` : list
  - Sliding window of recent values

#### Methods

##### `update(value)`

Update thresholds with new value.

**Parameters:**
- `value` : float
  - New data value

**Example:**
```python
for value in data_stream:
    threshold.update(value)
```

##### `encode(value)`

Encode value using current thresholds.

**Parameters:**
- `value` : float
  - Value to encode

**Returns:**
- `state` : State
  - Encoded channel state

**Example:**
```python
state = threshold.encode(0.75)
```

##### `get_stats()`

Get current statistics and thresholds.

**Returns:**
- `stats` : dict
  - Dictionary with keys:
    - `'mean'`: Current mean
    - `'std'`: Current standard deviation
    - `'n_samples'`: Sample count
    - `'threshold_i'`: i-threshold
    - `'threshold_q'`: q-threshold

**Example:**
```python
stats = threshold.get_stats()
print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
print(f"Thresholds: i={stats['threshold_i']:.3f}, q={stats['threshold_q']:.3f}")
```

---

## Topology-Aware Adaptation ⭐

### `TopologyAdaptiveThreshold`

**This is ChannelPy's key innovation**: Thresholds that adapt to distributional topology, not just mean/variance.
```python
from channelpy.adaptive import TopologyAdaptiveThreshold

threshold = TopologyAdaptiveThreshold(
    window_size=1000,
    adaptation_rate=0.01,
    topology_update_interval=100
)
```

#### Parameters

- `window_size` : int, default=1000
  - Size of sliding window for topology analysis
- `adaptation_rate` : float, default=0.01
  - Rate of threshold adaptation
- `topology_update_interval` : int, default=100
  - How often to recompute topology (computational cost vs responsiveness)
- `feature_scorer` : FeatureScorer, optional
  - Multi-dimensional scorer for enhanced adaptation

#### Key Concepts

The adapter analyzes these **topological features**:

1. **Modality**: Number of peaks/modes in distribution
2. **Skewness**: Distribution asymmetry
3. **Kurtosis**: Tail heaviness
4. **Gaps**: Significant breaks in data
5. **Density variance**: Clustering vs uniformity
6. **Connected components**: Number of separated clusters

Based on topology, it applies different **threshold strategies**:

- **Multimodal** → Threshold between modes (valley)
- **Heavy-tailed** → Robust percentiles (IQR-based)
- **Skewed** → Asymmetric thresholds
- **Clustered** → Gap-based thresholds
- **Normal-ish** → Standard statistical thresholds

#### Methods

##### `update(value)`

Update with new value and adapt thresholds.

**Parameters:**
- `value` : float
  - New data value

**Example:**
```python
for value in data_stream:
    threshold.update(value)
```

##### `encode(value)`

Encode using topology-aware thresholds.

**Parameters:**
- `value` : float
  - Value to encode

**Returns:**
- `state` : State
  - Encoded channel state

##### `topology_changed(sensitivity=0.5)`

Check if topology has changed significantly.

**Parameters:**
- `sensitivity` : float, default=0.5
  - Sensitivity to change (0=insensitive, 1=very sensitive)

**Returns:**
- `changed` : bool
  - True if significant topology change detected

**Example:**
```python
threshold.update(value)
if threshold.topology_changed():
    print("Distribution topology has shifted!")
    print(f"New strategy: {threshold._get_current_strategy()}")
```

##### `get_topology()`

Get current topological features.

**Returns:**
- `topology` : TopologyFeatures
  - Dataclass with fields:
    - `modality`: int
    - `skewness`: float
    - `kurtosis`: float
    - `gaps`: list of (start, end) tuples
    - `local_maxima`: list of mode locations
    - `density_variance`: float
    - `connected_components`: int

**Example:**
```python
topo = threshold.get_topology()
print(f"Modality: {topo.modality}")
print(f"Skewness: {topo.skewness:.3f}")
print(f"Modes at: {topo.local_maxima}")
```

##### `get_thresholds()`

Get current thresholds with explanation.

**Returns:**
- `info` : dict
  - Complete threshold information including:
    - `'threshold_i'`, `'threshold_q'`
    - `'topology'`: Current topology features
    - `'adaptation_strategy'`: Active strategy name
    - `'n_samples'`: Sample count

**Example:**
```python
info = threshold.get_thresholds()
print(f"Strategy: {info['adaptation_strategy']}")
print(f"Thresholds: i={info['threshold_i']:.3f}, q={info['threshold_q']:.3f}")
```

##### `plot_topology_and_thresholds()`

Visualize distribution, topology, and thresholds.

**Returns:**
- `fig` : matplotlib.figure.Figure
  - Figure with two subplots:
    1. Histogram + KDE with thresholds and modes
    2. Topology evolution over time

**Requires:** matplotlib, scipy

**Example:**
```python
fig = threshold.plot_topology_and_thresholds()
fig.savefig('topology_analysis.png')
```

---

### `TopologyAnalyzer`

Compute topological features of distributions.
```python
from channelpy.adaptive.topology_adaptive import TopologyAnalyzer

analyzer = TopologyAnalyzer(bandwidth=0.1)
data = np.random.randn(1000)
features = analyzer.analyze(data)

print(f"Modality: {features.modality}")
print(f"Skewness: {features.skewness:.3f}")
print(f"Gaps: {features.gaps}")
```

#### Methods

- `analyze(data)` : Compute all topological features
- `_detect_modality(data)` : Detect number and location of modes
- `_find_gaps(data)` : Find significant gaps
- `_compute_density_variance(data)` : Measure clustering
- `_count_components(data)` : Count separated clusters

---

## Multi-Scale Adaptation

### `MultiScaleAdaptiveThreshold`

Track thresholds at multiple timescales for regime detection.
```python
from channelpy.adaptive import MultiScaleAdaptiveThreshold

multiscale = MultiScaleAdaptiveThreshold(
    use_topology=True,
    fast_window=100,
    medium_window=1000,
    slow_window=10000
)
```

#### Parameters

- `use_topology` : bool, default=True
  - Use topology-aware thresholds (recommended)
- `fast_window` : int, default=100
  - Window size for fast scale (quick reactions)
- `medium_window` : int, default=1000
  - Window size for medium scale (balanced)
- `slow_window` : int, default=10000
  - Window size for slow scale (stable baseline)

#### Attributes

- `fast` : TopologyAdaptiveThreshold or StreamingAdaptiveThreshold
  - Fast-scale tracker
- `medium` : TopologyAdaptiveThreshold or StreamingAdaptiveThreshold
  - Medium-scale tracker
- `slow` : TopologyAdaptiveThreshold or StreamingAdaptiveThreshold
  - Slow-scale tracker
- `current_regime` : RegimeType
  - Currently detected regime
- `regime_history` : list of RegimeChange
  - History of regime transitions

#### Regime Types
```python
from channelpy.adaptive.multiscale import RegimeType

# Possible regimes:
RegimeType.STABLE           # All scales aligned
RegimeType.TRANSITIONING    # Fast diverging from slow
RegimeType.VOLATILE         # Sustained high divergence
RegimeType.TRENDING         # Medium diverging gradually
RegimeType.MEAN_REVERTING   # Fast/medium differ but near slow
RegimeType.UNKNOWN          # Unclassified
```

#### Methods

##### `update(value)`

Update all scales with new value.

**Parameters:**
- `value` : float
  - New data value

##### `encode_adaptive(value)`

Encode using regime-appropriate scale.

**Parameters:**
- `value` : float
  - Value to encode

**Returns:**
- `state` : State
  - Encoded state using optimal scale for current regime

**Logic:**
- `STABLE` → Use slow scale (stable baseline)
- `VOLATILE` → Use fast scale (quick adaptation)
- `TRANSITIONING` → Use medium scale (balanced)
- `TRENDING` → Use medium scale
- `MEAN_REVERTING` → Use slow scale (resist noise)

**Example:**
```python
for value in data_stream:
    multiscale.update(value)
    state = multiscale.encode_adaptive(value)
    # Automatically uses appropriate scale!
```

##### `regime_changed()`

Check if regime changed on last update.

**Returns:**
- `changed` : bool
  - True if regime just changed

**Example:**
```python
multiscale.update(value)
if multiscale.regime_changed():
    change = multiscale.get_last_regime_change()
    print(f"Regime: {change.from_regime.value} → {change.to_regime.value}")
```

##### `get_current_regime()`

Get current regime type.

**Returns:**
- `regime` : RegimeType
  - Current regime

##### `get_last_regime_change()`

Get information about most recent regime change.

**Returns:**
- `change` : RegimeChange or None
  - Dataclass with fields:
    - `timestamp`: Update count when changed
    - `from_regime`: Previous regime
    - `to_regime`: New regime
    - `confidence`: Detection confidence (0-1)
    - `divergence_measure`: Magnitude of change

**Example:**
```python
change = multiscale.get_last_regime_change()
if change:
    print(f"Changed at update {change.timestamp}")
    print(f"Confidence: {change.confidence:.2%}")
    print(f"Divergence: {change.divergence_measure:.3f}σ")
```

##### `get_all_thresholds()`

Get thresholds from all scales.

**Returns:**
- `thresholds` : dict
  - Dictionary with 'fast', 'medium', 'slow' keys

**Example:**
```python
thresholds = multiscale.get_all_thresholds()
print(f"Fast: {thresholds['fast']['threshold_i']:.3f}")
print(f"Medium: {thresholds['medium']['threshold_i']:.3f}")
print(f"Slow: {thresholds['slow']['threshold_i']:.3f}")
```

##### `get_regime_info()`

Get comprehensive regime information.

**Returns:**
- `info` : dict
  - Current regime, divergences, change history

##### `plot_multiscale()`

Visualize multi-scale tracking and regimes.

**Returns:**
- `fig` : matplotlib.figure.Figure
  - Three subplots:
    1. Thresholds at all scales
    2. Scale divergences over time
    3. Regime timeline

**Example:**
```python
fig = multiscale.plot_multiscale()
fig.savefig('multiscale_analysis.png')
```

---

## Feature Scoring

### `FeatureScorer`

Score features across multiple dimensions for context-aware decisions.
```python
from channelpy.adaptive import FeatureScorer

scorer = FeatureScorer()

# Add scoring dimensions
scorer.add_dimension('relevance', relevance_scorer, weight=2.0)
scorer.add_dimension('confidence', confidence_scorer, weight=1.5)
scorer.add_dimension('freshness', freshness_scorer, weight=1.0)

# Score a feature
context = {
    'historical_values': recent_values,
    'historical_outcomes': outcomes,
    'sample_size': 100
}
aggregate_score, dim_scores = scorer.score_and_aggregate(value, context)
```

#### Methods

##### `add_dimension(name, scorer, weight=1.0, description="")`

Add a scoring dimension.

**Parameters:**
- `name` : str
  - Dimension name
- `scorer` : callable
  - Function with signature `scorer(value, context) -> float`
- `weight` : float, default=1.0
  - Weight for aggregation
- `description` : str, optional
  - Human-readable description

##### `score_feature(value, context)`

Score feature across all dimensions.

**Parameters:**
- `value` : float
  - Feature value to score
- `context` : dict
  - Context with additional information

**Returns:**
- `scores` : dict
  - Score for each dimension

##### `aggregate_scores(dimension_scores, method='weighted_average')`

Aggregate dimension scores.

**Parameters:**
- `dimension_scores` : dict
  - Scores from each dimension
- `method` : {'weighted_average', 'min', 'max', 'product'}
  - Aggregation method

**Returns:**
- `score` : float
  - Aggregated score in [0, 1]

##### `explain_score(value, context)`

Generate human-readable explanation.

**Parameters:**
- `value` : float
  - Feature value
- `context` : dict
  - Context dictionary

**Returns:**
- `explanation` : str
  - Multi-line explanation of score

**Example:**
```python
explanation = scorer.explain_score(0.75, context)
print(explanation)
# Output:
# Feature value: 0.750
# Overall score: 0.823
#
# Dimension breakdown:
#   relevance      : 0.900 (weight=2.0, contribution=1.800)
#   confidence     : 0.850 (weight=1.5, contribution=1.275)
#   freshness      : 0.600 (weight=1.0, contribution=0.600)
```

---

### Pre-configured Scorers

#### `create_trading_scorer()`

Create scorer configured for trading signals.
```python
from channelpy.adaptive import create_trading_scorer

scorer = create_trading_scorer()
# Includes: strength, confidence, timeliness, stability
```

#### `create_medical_scorer()`

Create scorer configured for medical diagnosis.
```python
from channelpy.adaptive import create_medical_scorer

scorer = create_medical_scorer()
# Includes: diagnostic_value, measurement_quality, consistency
```

#### `create_signal_scorer()`

Create scorer configured for signal processing.
```python
from channelpy.adaptive import create_signal_scorer

scorer = create_signal_scorer()
# Includes: signal_strength, noise_level, temporal_stability, local_density
```

---

### Standard Scoring Functions

#### `relevance_scorer(value, context)`

Score based on correlation with outcomes.

**Context keys:**
- `'historical_values'`: array
- `'historical_outcomes'`: array
- `'similarity_threshold'`: float

#### `confidence_scorer(value, context)`

Score based on data quality.

**Context keys:**
- `'sample_size'`: int
- `'missing_rate'`: float
- `'noise_level'`: float

#### `freshness_scorer(value, context)`

Score based on data recency.

**Context keys:**
- `'age_seconds'`: float
- `'half_life_seconds'`: float

#### `stability_scorer(value, context)`

Score based on feature stability.

**Context keys:**
- `'recent_values'`: array

#### `density_scorer(value, context)`

Score based on local density.

**Context keys:**
- `'all_values'`: array
- `'bandwidth'`: float

---

## Threshold Learning

### `ThresholdLearner`

Learn optimal thresholds from data.
```python
from channelpy.adaptive import ThresholdLearner

learner = ThresholdLearner(method='supervised')
learner.fit(X_train, y_train)
threshold_i, threshold_q = learner.get_thresholds()
```

#### Parameters

- `method` : {'statistical', 'supervised', 'optimized'}
  - Learning method

#### Methods

- `fit(X, y=None)` : Learn thresholds
- `get_thresholds()` : Get learned thresholds
- `score(X, y)` : Evaluate threshold quality

---

## Usage Patterns

### Pattern 1: Simple Streaming

For basic online adaptation:
```python
threshold = StreamingAdaptiveThreshold()

for value in data_stream:
    threshold.update(value)
    state = threshold.encode(value)
    process(state)
```

### Pattern 2: Topology-Aware (Recommended)

For robust adaptation to distribution changes:
```python
threshold = TopologyAdaptiveThreshold()

for value in data_stream:
    threshold.update(value)
    
    # Check for topology changes
    if threshold.topology_changed():
        topo = threshold.get_topology()
        print(f"Distribution changed: {topo.modality} modes")
    
    state = threshold.encode(value)
    process(state)
```

### Pattern 3: Multi-Scale with Regime Detection

For regime-aware trading or monitoring:
```python
multiscale = MultiScaleAdaptiveThreshold(use_topology=True)

for value in data_stream:
    multiscale.update(value)
    
    # Detect regime changes
    if multiscale.regime_changed():
        regime = multiscale.get_current_regime()
        print(f"New regime: {regime.value}")
        adjust_strategy(regime)
    
    # Use regime-appropriate threshold
    state = multiscale.encode_adaptive(value)
    decision = interpret(state, multiscale.current_regime)
```

### Pattern 4: With Feature Scoring

For context-aware adaptation:
```python
scorer = create_trading_scorer()
threshold = TopologyAdaptiveThreshold(feature_scorer=scorer)

for value in data_stream:
    # Build context
    context = {
        'historical_values': recent_values,
        'historical_outcomes': recent_outcomes,
        'sample_size': len(recent_values),
        'age_seconds': 0
    }
    
    # Score feature
    score, dim_scores = scorer.score_and_aggregate(value, context)
    
    # Update threshold
    threshold.update(value)
    
    # Encode and decide
    state = threshold.encode(value)
    decision = interpret(state, score)
```

---

## Best Practices

### 1. Choose Appropriate Window Size
```python
# High-frequency trading: small window
threshold = TopologyAdaptiveThreshold(window_size=100)

# Long-term investing: large window
threshold = TopologyAdaptiveThreshold(window_size=10000)
```

### 2. Monitor Topology Changes
```python
if threshold.topology_changed(sensitivity=0.3):
    # Distribution has shifted
    # Consider:
    # - Resetting models
    # - Alerting operators
    # - Adjusting risk parameters
    pass
```

### 3. Use Multi-Scale for Critical Applications
```python
# For production systems, use multi-scale
multiscale = MultiScaleAdaptiveThreshold(use_topology=True)

# Monitor regime changes
if multiscale.current_regime == RegimeType.VOLATILE:
    # Reduce exposure
    position_size *= 0.5
```

### 4. Combine with Feature Scoring
```python
# For nuanced decisions, score features
scorer = create_domain_scorer()  # Your domain
threshold = TopologyAdaptiveThreshold(feature_scorer=scorer)

# Scoring enhances threshold adaptation
```

---

## Performance Considerations

### Computational Cost

- `StreamingAdaptiveThreshold`: O(1) per update
- `TopologyAdaptiveThreshold`: O(N log N) per topology update
  - Control cost via `topology_update_interval`
  - Default: every 100 samples
- `MultiScaleAdaptiveThreshold`: 3× topology cost

### Memory Usage

- `StreamingAdaptiveThreshold`: O(window_size)
- `TopologyAdaptiveThreshold`: O(window_size)
- `MultiScaleAdaptiveThreshold`: O(fast + medium + slow window sizes)

### Recommendations
```python
# For real-time systems with <1ms latency requirements:
threshold = StreamingAdaptiveThreshold(window_size=1000)

# For most applications (recommended):
threshold = TopologyAdaptiveThreshold(
    window_size=1000,
    topology_update_interval=100  # Balance cost vs responsiveness
)

# For research/offline analysis:
threshold = TopologyAdaptiveThreshold(
    window_size=10000,
    topology_update_interval=10  # Frequent updates
)
```

---

## Examples

See:
- [Tutorial: Adaptive Thresholds](../tutorials/03_adaptive_thresholds.md)
- [Example: Trading Bot](../tutorials/04_trading_bot.md)
- [How-to: Handle Distribution Shifts](../how_to_guides/handle_distribution_shifts.md)

---

## See Also

- [Core API](core.md) - Channel states and operations
- [Pipeline API](pipeline.md) - Building pipelines
- [Topology API](topology.md) - Topological analysis
- [Applications API](applications.md) - Domain-specific systems