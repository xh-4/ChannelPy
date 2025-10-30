# Tutorial 2: Building Pipelines

Learn to build complete data processing pipelines with ChannelPy.

## Table of Contents

1. [Pipeline Architecture](#pipeline-architecture)
2. [Your First Pipeline](#your-first-pipeline)
3. [Preprocessors](#preprocessors)
4. [Encoders](#encoders)
5. [Interpreters](#interpreters)
6. [Complete Examples](#complete-examples)

## Pipeline Architecture

A ChannelPy pipeline has three stages:
```
Raw Data → [Preprocessors] → Features → [Encoders] → States → [Interpreters] → Decisions
```

### Stage 1: Preprocessors
Clean and prepare raw data
- Handle missing values
- Remove outliers
- Normalize/scale
- Extract features

### Stage 2: Encoders
Convert features to channel states
- Threshold-based encoding
- Learned thresholds
- Multi-feature encoding

### Stage 3: Interpreters
Convert states to decisions
- Rule-based logic
- Pattern matching
- State machines

## Your First Pipeline
```python
from channelpy.pipeline import ChannelPipeline
from channelpy.pipeline.encoders import ThresholdEncoder
import numpy as np

# Generate sample data
np.random.seed(42)
data = np.random.randn(100)

# Create pipeline
pipeline = ChannelPipeline()

# Add encoder
encoder = ThresholdEncoder(threshold_i=0.0, threshold_q=0.5)
pipeline.add_encoder(encoder)

# Add simple interpreter
def interpret_high_low(states):
    """Convert states to HIGH/MEDIUM/LOW/NONE"""
    decisions = []
    for state in states[0]:
        if state.i and state.q:
            decisions.append('HIGH')
        elif state.i:
            decisions.append('MEDIUM')
        elif state.q:
            decisions.append('LOW')
        else:
            decisions.append('NONE')
    return decisions

pipeline.add_interpreter(interpret_high_low)

# Fit pipeline
pipeline.fit(data)

# Transform new data
test_data = np.array([0.8, -0.3, 1.2, -0.8])
decisions, states = pipeline.transform(test_data)

print("Values:   ", test_data)
print("States:   ", [str(s) for s in states[0]])
print("Decisions:", decisions)
```

## Preprocessors

### Built-in Preprocessors
```python
from channelpy.pipeline.preprocessors import (
    StandardScaler,
    RobustScaler,
    MissingDataHandler,
    OutlierDetector,
    FeatureExtractor
)

# 1. Standard Scaler
scaler = StandardScaler()
data = np.random.randn(100) * 5 + 10
scaled = scaler.fit_transform(data)
print(f"Original: mean={data.mean():.2f}, std={data.std():.2f}")
print(f"Scaled: mean={scaled.mean():.2f}, std={scaled.std():.2f}")

# 2. Robust Scaler (resistant to outliers)
robust = RobustScaler()
data_with_outliers = np.concatenate([np.random.randn(95), [100, 150, -200, 250, -180]])
scaled = robust.fit_transform(data_with_outliers)
print(f"Original range: [{data_with_outliers.min():.1f}, {data_with_outliers.max():.1f}]")
print(f"Scaled range: [{scaled.min():.1f}, {scaled.max():.1f}]")

# 3. Missing Data Handler
handler = MissingDataHandler(strategy='mean')
data_with_missing = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
filled = handler.fit_transform(data_with_missing)
print(f"Original: {data_with_missing}")
print(f"Filled: {filled}")

# 4. Outlier Detector
detector = OutlierDetector(method='iqr', threshold=1.5)
data = np.concatenate([np.random.randn(95), [10, -10, 15, -15, 20]])
is_outlier = detector.fit_transform(data)
print(f"Number of outliers: {is_outlier.sum()}")

# 5. Feature Extractor (time series)
extractor = FeatureExtractor(features=['mean', 'std', 'min', 'max'])
time_series = np.random.randn(100, 10)  # 100 windows of 10 samples
features = extractor.fit_transform(time_series)
print(f"Extracted features shape: {features.shape}")
```

### Custom Preprocessor
```python
from channelpy.pipeline.base import BasePipeline

class LogTransform:
    """Custom preprocessor: log transform"""
    
    def __init__(self, offset=1.0):
        self.offset = offset
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log(X + self.offset)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Use custom preprocessor
pipeline = ChannelPipeline()
pipeline.add_preprocessor(LogTransform(offset=1.0))

data = np.array([1, 10, 100, 1000])
pipeline.fit(data)
transformed, _ = pipeline.transform(data)
print(f"Original: {data}")
print(f"Log transformed: {transformed}")
```

## Encoders

### 1. Threshold Encoder
```python
from channelpy.pipeline.encoders import ThresholdEncoder

# Fixed thresholds
encoder = ThresholdEncoder(threshold_i=0.5, threshold_q=0.75)

values = np.array([0.3, 0.6, 0.8, 0.2])
states = encoder(values)

print("Values:", values)
print("States:", states.to_strings())
```

### 2. Learned Threshold Encoder
```python
from channelpy.pipeline.encoders import LearnedThresholdEncoder

# Statistical method (uses percentiles)
encoder_stat = LearnedThresholdEncoder(method='statistical')
train_data = np.random.randn(1000)
encoder_stat.fit(train_data)

print(f"Learned threshold_i: {encoder_stat.threshold_i:.3f}")
print(f"Learned threshold_q: {encoder_stat.threshold_q:.3f}")

# Test encoding
test_values = np.array([0.5, -0.5, 1.0, -1.0])
states = encoder_stat(test_values)
print("Test states:", states.to_strings())

# Supervised method (optimizes for labels)
encoder_sup = LearnedThresholdEncoder(method='supervised')
X_train = np.random.randn(1000)
y_train = (X_train > 0).astype(int)  # Binary labels
encoder_sup.fit(X_train, y_train)

print(f"Supervised threshold_i: {encoder_sup.threshold_i:.3f}")
print(f"Supervised threshold_q: {encoder_sup.threshold_q:.3f}")
```

### 3. Dual Feature Encoder
```python
from channelpy.pipeline.encoders import DualFeatureEncoder

# Encode from two separate features
encoder = DualFeatureEncoder()

# Feature 1 determines i-bit, Feature 2 determines q-bit
feature1 = np.random.randn(100)  # e.g., signal strength
feature2 = np.random.randn(100)  # e.g., confidence score

encoder.fit(feature1, feature2)

# Encode new samples
f1_test = np.array([0.5, -0.3, 0.8])
f2_test = np.array([0.2, 0.7, -0.1])
states = encoder(f1_test, f2_test)

print("Feature 1:", f1_test)
print("Feature 2:", f2_test)
print("States:", states.to_strings())
```

### Custom Encoder
```python
class PercentileEncoder:
    """Encode based on percentile ranks"""
    
    def __init__(self, percentile_i=50, percentile_q=75):
        self.percentile_i = percentile_i
        self.percentile_q = percentile_q
        self.threshold_i = None
        self.threshold_q = None
    
    def fit(self, X, y=None):
        self.threshold_i = np.percentile(X, self.percentile_i)
        self.threshold_q = np.percentile(X, self.percentile_q)
        return self
    
    def __call__(self, X):
        from channelpy import StateArray
        X = np.asarray(X)
        return StateArray(
            i=(X > self.threshold_i).astype(np.int8),
            q=(X > self.threshold_q).astype(np.int8)
        )

# Use custom encoder
encoder = PercentileEncoder(percentile_i=33, percentile_q=66)
data = np.random.randn(1000)
encoder.fit(data)

test = np.array([-1, 0, 1])
states = encoder(test)
print("States:", states.to_strings())
```

## Interpreters

### 1. Rule-Based Interpreter
```python
from channelpy.pipeline.interpreters import RuleBasedInterpreter
from channelpy import PSI, DELTA, PHI, EMPTY

# Define rules
rules = {
    PSI: {'action': 'BUY', 'confidence': 1.0},
    DELTA: {'action': 'HOLD', 'confidence': 0.5},
    PHI: {'action': 'HOLD', 'confidence': 0.3},
    EMPTY: {'action': 'SELL', 'confidence': 0.0}
}

interpreter = RuleBasedInterpreter(rules)

# Apply rules
states = [PSI, DELTA, PHI, EMPTY, PSI]
decisions = interpreter(states)

print("States:   ", [str(s) for s in states])
print("Actions:  ", [d['action'] for d in decisions])
print("Confidence:", [d['confidence'] for d in decisions])
```

### 2. Lookup Table Interpreter
```python
from channelpy.pipeline.interpreters import LookupTableInterpreter

# Create lookup table
lookup = {
    (1, 1): 'STRONG_BUY',   # ψ
    (1, 0): 'WEAK_BUY',     # δ
    (0, 1): 'WEAK_SELL',    # φ
    (0, 0): 'STRONG_SELL'   # ∅
}

interpreter = LookupTableInterpreter(lookup)

states = [PSI, DELTA, PHI, EMPTY]
decisions = interpreter(states)

print("States:   ", [str(s) for s in states])
print("Decisions:", decisions)
```

### 3. FSM Interpreter
```python
from channelpy.pipeline.interpreters import FSMInterpreter

# Define state machine
transitions = {
    'WAIT': {
        PSI: ('ACTIVE', 'signal_detected'),
        DELTA: ('WATCH', 'weak_signal'),
        PHI: ('WAIT', 'no_change'),
        EMPTY: ('WAIT', 'no_change')
    },
    'WATCH': {
        PSI: ('ACTIVE', 'signal_strengthened'),
        DELTA: ('WATCH', 'still_weak'),
        PHI: ('WAIT', 'signal_lost'),
        EMPTY: ('WAIT', 'signal_lost')
    },
    'ACTIVE': {
        PSI: ('ACTIVE', 'signal_continues'),
        DELTA: ('WATCH', 'signal_weakening'),
        PHI: ('WAIT', 'signal_lost'),
        EMPTY: ('WAIT', 'signal_lost')
    }
}

interpreter = FSMInterpreter(
    transitions=transitions,
    initial_state='WAIT'
)

# Process sequence
states = [DELTA, PSI, PSI, DELTA, PHI]
outputs = []

for state in states:
    output = interpreter(state)
    outputs.append(output)

print("States:  ", [str(s) for s in states])
print("FSM State:", [o['current_state'] for o in outputs])
print("Actions:  ", [o['action'] for o in outputs])
```

### Custom Interpreter
```python
class ThresholdCountInterpreter:
    """Interpret based on count of high states in window"""
    
    def __init__(self, window_size=5, threshold=3):
        self.window_size = window_size
        self.threshold = threshold
        self.window = []
    
    def __call__(self, states):
        decisions = []
        
        for state in states:
            # Add to window
            self.window.append(state)
            if len(self.window) > self.window_size:
                self.window.pop(0)
            
            # Count PSI states in window
            psi_count = sum(1 for s in self.window if s == PSI)
            
            # Decide based on count
            if psi_count >= self.threshold:
                decisions.append('HIGH')
            elif psi_count >= self.threshold // 2:
                decisions.append('MEDIUM')
            else:
                decisions.append('LOW')
        
        return decisions

# Use custom interpreter
interpreter = ThresholdCountInterpreter(window_size=5, threshold=3)
states = [PSI, PSI, DELTA, PSI, PSI, PHI, PSI, DELTA]
decisions = interpreter(states)

print("States:   ", [str(s) for s in states])
print("Decisions:", decisions)
```

## Complete Examples

### Example 1: Time Series Anomaly Detection
```python
from channelpy.pipeline import ChannelPipeline
from channelpy.pipeline.preprocessors import StandardScaler, FeatureExtractor
from channelpy.pipeline.encoders import LearnedThresholdEncoder
from channelpy.pipeline.interpreters import RuleBasedInterpreter
import numpy as np

# Generate time series with anomalies
np.random.seed(42)
normal_data = np.random.randn(900)
anomalies = np.random.randn(100) * 5  # Larger variance
data = np.concatenate([normal_data, anomalies])
np.random.shuffle(data)

# Build pipeline
pipeline = ChannelPipeline()

# Stage 1: Preprocess
pipeline.add_preprocessor(StandardScaler())

# Stage 2: Encode
encoder = LearnedThresholdEncoder(method='statistical')
pipeline.add_encoder(encoder)

# Stage 3: Interpret
rules = {
    PSI: 'ANOMALY',    # High value
    DELTA: 'SUSPECT',  # Medium value
    PHI: 'NORMAL',     # Low value
    EMPTY: 'NORMAL'    # Very low value
}
interpreter = RuleBasedInterpreter(rules)
pipeline.add_interpreter(interpreter)

# Fit and transform
pipeline.fit(data)
decisions, states = pipeline.transform(data)

# Analyze results
print(f"Total samples: {len(data)}")
print(f"Anomalies detected: {decisions.count('ANOMALY')}")
print(f"Suspect: {decisions.count('SUSPECT')}")
print(f"Normal: {decisions.count('NORMAL')}")
```

### Example 2: Multi-Feature Classification
```python
from channelpy.pipeline import ChannelPipeline
from channelpy.pipeline.preprocessors import StandardScaler
from channelpy.pipeline.encoders import DualFeatureEncoder
import numpy as np

# Generate two-feature data
n_samples = 1000
feature1 = np.random.randn(n_samples)  # Feature A
feature2 = np.random.randn(n_samples)  # Feature B

# Build pipeline
pipeline = ChannelPipeline()

# Preprocess both features
scaler = StandardScaler()
pipeline.add_preprocessor(lambda X: scaler.fit_transform(X))

# Encode from both features
encoder = DualFeatureEncoder()

# Fit encoder on training data
encoder.fit(feature1[:800], feature2[:800])

# Manually encode (since we need both features)
test_states = encoder(feature1[800:], feature2[800:])

# Interpret
def interpret_quadrants(states):
    """Interpret based on feature quadrants"""
    decisions = []
    for state in states:
        if state == PSI:
            decisions.append('Quadrant I: High-High')
        elif state == DELTA:
            decisions.append('Quadrant IV: High-Low')
        elif state == PHI:
            decisions.append('Quadrant II: Low-High')
        else:
            decisions.append('Quadrant III: Low-Low')
    return decisions

decisions = interpret_quadrants(test_states)

print("Sample decisions:", decisions[:10])
```

### Example 3: Streaming Pipeline
```python
from channelpy.pipeline import ChannelPipeline
from channelpy.adaptive import StreamingAdaptiveThreshold
import numpy as np

# Streaming data source
def generate_stream(n_samples):
    """Generate streaming data with regime changes"""
    for i in range(n_samples):
        if i < 300:
            yield np.random.randn() * 1.0  # Regime 1
        elif i < 600:
            yield np.random.randn() * 3.0 + 2.0  # Regime 2
        else:
            yield np.random.randn() * 0.5 - 1.0  # Regime 3

# Create streaming encoder
streaming_encoder = StreamingAdaptiveThreshold(
    window_size=100,
    adaptation_rate=0.01
)

# Process stream
decisions = []
states = []

for value in generate_stream(900):
    # Update encoder
    streaming_encoder.update(value)
    
    # Encode
    state = streaming_encoder.encode(value)
    states.append(state)
    
    # Interpret
    if state == PSI:
        decision = 'HIGH'
    elif state == DELTA:
        decision = 'MEDIUM'
    else:
        decision = 'LOW'
    decisions.append(decision)
    
    # Monitor adaptation
    if len(states) % 100 == 0:
        stats = streaming_encoder.get_stats()
        print(f"Sample {len(states)}: "
              f"threshold_i={stats['threshold_i']:.3f}, "
              f"threshold_q={stats['threshold_q']:.3f}")

print(f"\nTotal processed: {len(states)}")
print(f"High decisions: {decisions.count('HIGH')}")
```

## Pipeline Best Practices

### 1. Always Fit Before Transform
```python
pipeline = ChannelPipeline()
# ... add components ...

# CORRECT
pipeline.fit(train_data)
decisions, states = pipeline.transform(test_data)

# INCORRECT - will raise error
# decisions, states = pipeline.transform(test_data)  # Not fitted!
```

### 2. Use fit_transform for Training Data
```python
# On training data, use fit_transform
decisions_train, states_train = pipeline.fit_transform(train_data)

# On test data, just use transform
decisions_test, states_test = pipeline.transform(test_data)
```

### 3. Chain Preprocessors for Complex Transformations
```python
pipeline = ChannelPipeline()

# Chain multiple preprocessors
pipeline.add_preprocessor(MissingDataHandler())
pipeline.add_preprocessor(OutlierDetector())
pipeline.add_preprocessor(StandardScaler())

# They execute in order
```

### 4. Save and Load Pipelines
```python
from channelpy.utils.serialization import save_pipeline, load_pipeline

# Save fitted pipeline
save_pipeline(pipeline, 'my_pipeline.pkl')

# Load later
loaded_pipeline = load_pipeline('my_pipeline.pkl')
decisions, states = loaded_pipeline.transform(new_data)
```

## Key Takeaways

1. **Three-stage architecture**: Preprocess → Encode → Interpret
2. **Preprocessors** clean data and extract features
3. **Encoders** convert features to states
4. **Interpreters** convert states to decisions
5. **Pipelines are composable** and reusable
6. **Always fit before transform**
7. **Custom components** are easy to add

## What's Next?

Learn about adaptive thresholds that respond to your data:

➡️ **[Tutorial 3: Adaptive Thresholds](03_adaptive_thresholds.md)**

## Exercises

1. Build a pipeline with 3 preprocessors, 1 encoder, 1 interpreter
2. Create a custom preprocessor that removes outliers beyond 3σ
3. Build a streaming pipeline that adapts to regime changes
4. Implement a voting interpreter that combines multiple rules

Solutions in the [GitHub repository](https://github.com/channelalgebra/channelpy/tree/main/examples).