# Pipeline API Reference

The pipeline module provides the infrastructure for building end-to-end data processing pipelines using channel algebra.

## Overview

The pipeline follows a three-stage architecture:

1. **Preprocessing**: Transform raw data into features
2. **Encoding**: Convert features into channel states
3. **Interpretation**: Transform states into decisions/actions
```python
from channelpy.pipeline import ChannelPipeline

pipeline = ChannelPipeline()
pipeline.add_preprocessor(normalize)
pipeline.add_encoder(threshold_encoder)
pipeline.add_interpreter(rule_interpreter)

pipeline.fit(train_data, train_labels)
decisions, states = pipeline.transform(test_data)
```

---

## Module Structure
```
channelpy.pipeline/
├── base.py              # Base pipeline classes
├── preprocessors.py     # Data preprocessing components
├── encoders.py          # Feature → State encoders
├── interpreters.py      # State → Decision interpreters
└── full_pipeline.py     # Complete pipeline utilities
```

---

## Base Classes

### `BasePipeline`

Abstract base class for all pipelines.
```python
class BasePipeline(ABC):
    """
    Base class for channel pipelines
    
    Three-stage architecture:
    1. Preprocess: Raw data → Features
    2. Encode: Features → States
    3. Interpret: States → Decisions
    """
```

#### Methods

##### `fit(X, y=None)`

Fit the pipeline on training data.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features)
  - Input data
- `y` : array-like, shape (n_samples,), optional
  - Target labels for supervised learning

**Returns:**
- `self` : fitted pipeline

**Example:**
```python
pipeline.fit(train_features, train_labels)
```

##### `transform(X)`

Transform data through the pipeline.

**Parameters:**
- `X` : array-like, shape (n_samples, n_features)
  - Input data

**Returns:**
- `decisions` : array-like
  - Pipeline outputs (decisions/predictions)
- `states` : StateArray or list of StateArray
  - Intermediate channel states (for debugging)

**Example:**
```python
decisions, states = pipeline.transform(test_features)
```

##### `fit_transform(X, y=None)`

Fit and transform in one step.

**Parameters:**
- `X` : array-like
  - Input data
- `y` : array-like, optional
  - Target labels

**Returns:**
- `decisions` : array-like
  - Pipeline outputs
- `states` : StateArray or list
  - Intermediate states

**Example:**
```python
decisions, states = pipeline.fit_transform(data, labels)
```

---

### `ChannelPipeline`

Concrete implementation of `BasePipeline`.
```python
from channelpy.pipeline import ChannelPipeline

pipeline = ChannelPipeline()
```

#### Methods

##### `add_preprocessor(preprocessor)`

Add a preprocessing step to the pipeline.

**Parameters:**
- `preprocessor` : callable or object with fit/transform
  - Preprocessing component

**Example:**
```python
from channelpy.pipeline.preprocessors import StandardScaler

scaler = StandardScaler()
pipeline.add_preprocessor(scaler)
```

##### `add_encoder(encoder)`

Add an encoding step to the pipeline.

**Parameters:**
- `encoder` : callable or Encoder object
  - Encoding component that converts features to states

**Example:**
```python
from channelpy.pipeline.encoders import ThresholdEncoder

encoder = ThresholdEncoder(threshold_i=0.5, threshold_q=0.75)
pipeline.add_encoder(encoder)
```

##### `add_interpreter(interpreter)`

Add an interpretation step to the pipeline.

**Parameters:**
- `interpreter` : callable or Interpreter object
  - Interpretation component that converts states to decisions

**Example:**
```python
from channelpy.pipeline.interpreters import RuleBasedInterpreter

interpreter = RuleBasedInterpreter()
interpreter.add_rule(PSI, "buy")
interpreter.add_rule(EMPTY, "sell")
pipeline.add_interpreter(interpreter)
```

---

## Preprocessors

### `StandardScaler`

Standardize features by removing mean and scaling to unit variance.
```python
from channelpy.pipeline.preprocessors import StandardScaler

scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
```

#### Parameters

- `with_mean` : bool, default=True
  - If True, center the data before scaling
- `with_std` : bool, default=True
  - If True, scale the data to unit variance

#### Attributes

- `mean_` : ndarray
  - Mean of training data
- `std_` : ndarray
  - Standard deviation of training data

#### Methods

- `fit(X, y=None)` : Compute mean and std
- `transform(X)` : Apply standardization
- `inverse_transform(X)` : Reverse the scaling

---

### `RobustScaler`

Scale features using robust statistics (median and IQR).

More robust to outliers than StandardScaler.
```python
from channelpy.pipeline.preprocessors import RobustScaler

scaler = RobustScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
```

#### Parameters

- `quantile_range` : tuple (q_min, q_max), default=(25.0, 75.0)
  - Quantile range used to calculate scale

#### Attributes

- `center_` : ndarray
  - Median of training data
- `scale_` : ndarray
  - IQR of training data

---

### `MissingDataHandler`

Handle missing data using various strategies.
```python
from channelpy.pipeline.preprocessors import MissingDataHandler

handler = MissingDataHandler(strategy='mean')
handler.fit(data_with_nans)
clean_data = handler.transform(data_with_nans)
```

#### Parameters

- `strategy` : {'mean', 'median', 'most_frequent', 'constant', 'forward_fill', 'backward_fill'}
  - Imputation strategy
- `fill_value` : scalar, optional
  - Value to use for 'constant' strategy

#### Methods

- `fit(X, y=None)` : Learn imputation parameters
- `transform(X)` : Impute missing values

---

### `FeatureExtractor`

Extract features from time series or structured data.
```python
from channelpy.pipeline.preprocessors import FeatureExtractor

extractor = FeatureExtractor(features=['mean', 'std', 'trend'])
features = extractor.transform(time_series)
```

#### Parameters

- `features` : list of str
  - Features to extract: 'mean', 'std', 'min', 'max', 'trend', 'seasonality'
- `window_size` : int, optional
  - Window size for rolling features

#### Methods

- `transform(X)` : Extract features
- `get_feature_names()` : Get names of extracted features

---

### `OutlierDetector`

Detect and optionally remove outliers.
```python
from channelpy.pipeline.preprocessors import OutlierDetector

detector = OutlierDetector(method='zscore', threshold=3.0)
detector.fit(data)
clean_data, outlier_mask = detector.transform(data, return_mask=True)
```

#### Parameters

- `method` : {'zscore', 'iqr', 'isolation_forest'}
  - Outlier detection method
- `threshold` : float
  - Threshold for outlier detection
- `action` : {'remove', 'clip', 'flag'}
  - What to do with outliers

#### Methods

- `fit(X, y=None)` : Learn outlier parameters
- `transform(X, return_mask=False)` : Detect/handle outliers

---

## Encoders

### `ThresholdEncoder`

Simple threshold-based encoder.
```python
from channelpy.pipeline.encoders import ThresholdEncoder

encoder = ThresholdEncoder(threshold_i=0.5, threshold_q=0.75)
state = encoder.encode(0.8)  # Returns ψ
states = encoder.encode_array([0.3, 0.6, 0.9])
```

#### Parameters

- `threshold_i` : float, default=0.5
  - Threshold for i-bit (presence)
- `threshold_q` : float, default=0.75
  - Threshold for q-bit (membership)

#### Methods

- `encode(value)` : Encode single value → State
- `encode_array(values)` : Encode array → StateArray
- `__call__(values)` : Callable interface

---

### `LearnedThresholdEncoder`

Learn optimal thresholds from data.
```python
from channelpy.pipeline.encoders import LearnedThresholdEncoder

encoder = LearnedThresholdEncoder(method='statistical')
encoder.fit(train_features, train_labels)
states = encoder(test_features)
```

#### Parameters

- `method` : {'statistical', 'supervised'}
  - Learning method
    - `'statistical'`: Use percentiles (unsupervised)
    - `'supervised'`: Optimize for classification (requires labels)

#### Attributes

- `threshold_i` : float
  - Learned i-threshold
- `threshold_q` : float
  - Learned q-threshold

#### Methods

- `fit(X, y=None)` : Learn thresholds
- `__call__(X)` : Encode features → StateArray

---

### `DualFeatureEncoder`

Encode from two separate features (one for i, one for q).
```python
from channelpy.pipeline.encoders import DualFeatureEncoder

encoder = DualFeatureEncoder()
encoder.fit(train_features_i, train_features_q, train_labels)
states = encoder(test_features_i, test_features_q)
```

#### Methods

- `fit(X_i, X_q, y=None)` : Fit both encoders
- `__call__(X_i, X_q)` : Encode from two features → StateArray

---

### `AdaptiveEncoder`

Encoder that adapts thresholds online.
```python
from channelpy.pipeline.encoders import AdaptiveEncoder
from channelpy.adaptive import StreamingAdaptiveThreshold

threshold = StreamingAdaptiveThreshold()
encoder = AdaptiveEncoder(threshold)

for value in stream:
    state = encoder.encode_and_update(value)
```

#### Parameters

- `adaptive_threshold` : StreamingAdaptiveThreshold or TopologyAdaptiveThreshold
  - Adaptive threshold tracker

#### Methods

- `encode_and_update(value)` : Update threshold and encode
- `encode(value)` : Encode without updating
- `get_thresholds()` : Get current threshold values

---

## Interpreters

### `RuleBasedInterpreter`

Interpret states using explicit rules.
```python
from channelpy.pipeline.interpreters import RuleBasedInterpreter
from channelpy.core import PSI, DELTA, PHI, EMPTY

interpreter = RuleBasedInterpreter()
interpreter.add_rule(PSI, "buy", confidence=1.0)
interpreter.add_rule(DELTA, "hold", confidence=0.5)
interpreter.add_rule(PHI, "hold", confidence=0.5)
interpreter.add_rule(EMPTY, "sell", confidence=1.0)

decision = interpreter.interpret(PSI)
# Returns: {'action': 'buy', 'confidence': 1.0}
```

#### Methods

##### `add_rule(state, decision, confidence=1.0)`

Add interpretation rule.

**Parameters:**
- `state` : State
  - Channel state to match
- `decision` : any
  - Decision/action for this state
- `confidence` : float
  - Confidence level (0-1)

##### `interpret(state)`

Interpret a state.

**Parameters:**
- `state` : State
  - Channel state to interpret

**Returns:**
- `decision` : dict
  - Dictionary with 'action' and 'confidence' keys

##### `interpret_array(states)`

Interpret array of states.

**Parameters:**
- `states` : StateArray
  - Array of channel states

**Returns:**
- `decisions` : list of dict
  - List of decision dictionaries

---

### `LookupTableInterpreter`

Interpret using a lookup table with pattern matching.
```python
from channelpy.pipeline.interpreters import LookupTableInterpreter

interpreter = LookupTableInterpreter()

# Add patterns (can use wildcards)
interpreter.add_pattern("ψ.*", "strong_signal")
interpreter.add_pattern("*.φ", "potential_hole")
interpreter.add_pattern("∅", "no_signal")

decision = interpreter.interpret_nested(nested_state)
```

#### Methods

- `add_pattern(pattern, decision)` : Add pattern rule
- `interpret_nested(nested_state)` : Interpret nested state
- `match_pattern(state, pattern)` : Check if state matches pattern

---

### `FSMInterpreter`

Finite state machine interpreter with memory.
```python
from channelpy.pipeline.interpreters import FSMInterpreter

interpreter = FSMInterpreter(initial_state='neutral')

# Add transitions
interpreter.add_transition(
    from_state='neutral',
    on_input=PSI,
    to_state='bullish',
    action='buy'
)

# Process sequence
for state in state_sequence:
    decision = interpreter.step(state)
    print(f"State: {interpreter.current_state}, Action: {decision}")
```

#### Parameters

- `initial_state` : str
  - Initial FSM state

#### Methods

- `add_transition(from_state, on_input, to_state, action)` : Define transition
- `step(input_state)` : Process one input and transition
- `reset()` : Return to initial state

---

### `PatternMatcher`

Match complex patterns in state sequences.
```python
from channelpy.pipeline.interpreters import PatternMatcher

matcher = PatternMatcher()

# Define pattern
pattern = [PSI, PSI, DELTA]  # Two resonant followed by puncture
matcher.add_pattern('breakout', pattern, action='strong_buy')

# Match in sequence
matches = matcher.find_matches(state_sequence)
for match in matches:
    print(f"Pattern '{match['name']}' at index {match['index']}")
```

#### Methods

- `add_pattern(name, pattern, action)` : Register pattern
- `find_matches(state_sequence)` : Find pattern occurrences
- `interpret_sequence(state_sequence)` : Interpret entire sequence

---

## Complete Pipeline Utilities

### `PipelineBuilder`

Fluent interface for building pipelines.
```python
from channelpy.pipeline import PipelineBuilder

pipeline = (PipelineBuilder()
    .add_scaler('standard')
    .add_missing_handler('mean')
    .add_encoder('learned', method='supervised')
    .add_interpreter('rules')
    .build())

pipeline.fit(train_data, train_labels)
```

#### Methods

- `add_scaler(type)` : Add scaler ('standard' or 'robust')
- `add_missing_handler(strategy)` : Add missing data handler
- `add_outlier_detector(method)` : Add outlier detector
- `add_encoder(type, **kwargs)` : Add encoder
- `add_interpreter(type, **kwargs)` : Add interpreter
- `build()` : Create ChannelPipeline

---

### `save_pipeline(pipeline, filepath)`

Save pipeline to disk.
```python
from channelpy.pipeline import save_pipeline, load_pipeline

save_pipeline(pipeline, 'models/my_pipeline.pkl')
loaded = load_pipeline('models/my_pipeline.pkl')
```

**Parameters:**
- `pipeline` : ChannelPipeline
  - Pipeline to save
- `filepath` : str
  - Path to save file

---

### `load_pipeline(filepath)`

Load pipeline from disk.

**Parameters:**
- `filepath` : str
  - Path to saved pipeline

**Returns:**
- `pipeline` : ChannelPipeline
  - Loaded pipeline

---

## Best Practices

### 1. Always Fit Before Transform
```python
# ✓ Correct
pipeline.fit(train_data)
results = pipeline.transform(test_data)

# ✗ Wrong
results = pipeline.transform(test_data)  # RuntimeError!
```

### 2. Handle Missing Data Early
```python
pipeline = ChannelPipeline()
pipeline.add_preprocessor(MissingDataHandler(strategy='mean'))  # First!
pipeline.add_preprocessor(StandardScaler())
pipeline.add_encoder(ThresholdEncoder())
```

### 3. Use Adaptive Encoders for Streaming
```python
from channelpy.adaptive import StreamingAdaptiveThreshold

threshold = StreamingAdaptiveThreshold()
encoder = AdaptiveEncoder(threshold)
pipeline.add_encoder(encoder)
```

### 4. Validate Pipeline Outputs
```python
decisions, states = pipeline.transform(test_data)

# Check state distribution
from channelpy.visualization import plot_state_distribution
plot_state_distribution(states[0])  # First encoder's states

# Validate decisions
assert len(decisions) == len(test_data)
```

---

## Examples

### Simple Classification Pipeline
```python
from channelpy.pipeline import ChannelPipeline
from channelpy.pipeline.preprocessors import StandardScaler
from channelpy.pipeline.encoders import LearnedThresholdEncoder
from channelpy.pipeline.interpreters import RuleBasedInterpreter
from channelpy.core import PSI, EMPTY

# Build pipeline
pipeline = ChannelPipeline()

scaler = StandardScaler()
pipeline.add_preprocessor(scaler)

encoder = LearnedThresholdEncoder(method='supervised')
pipeline.add_encoder(encoder)

interpreter = RuleBasedInterpreter()
interpreter.add_rule(PSI, 'positive', confidence=1.0)
interpreter.add_rule(EMPTY, 'negative', confidence=1.0)
pipeline.add_interpreter(interpreter)

# Train and predict
pipeline.fit(X_train, y_train)
predictions, states = pipeline.transform(X_test)
```

### Time Series Pipeline
```python
from channelpy.pipeline import ChannelPipeline
from channelpy.pipeline.preprocessors import FeatureExtractor
from channelpy.pipeline.encoders import DualFeatureEncoder
from channelpy.pipeline.interpreters import FSMInterpreter

pipeline = ChannelPipeline()

# Extract features
extractor = FeatureExtractor(features=['mean', 'std', 'trend'])
pipeline.add_preprocessor(extractor)

# Encode mean and std separately
encoder = DualFeatureEncoder()
pipeline.add_encoder(encoder)

# Use FSM for temporal patterns
fsm = FSMInterpreter(initial_state='watching')
fsm.add_transition('watching', PSI, 'bullish', 'buy')
fsm.add_transition('bullish', EMPTY, 'watching', 'sell')
pipeline.add_interpreter(fsm)

# Process time series
for window in time_series_windows:
    decision, state = pipeline.transform(window)
```

---

## See Also

- [Core API](core.md) - Channel states and operations
- [Adaptive API](adaptive.md) - Adaptive thresholding
- [Tutorial: Building Pipelines](../tutorials/02_building_pipeline.md)
- [How-to: Custom Encoders](../how_to_guides/custom_encoder.md)