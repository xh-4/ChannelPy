# How-To Guide: Create a Custom Encoder

Creating custom encoders lets you adapt ChannelPy to your specific domain and data characteristics.

## When to Create a Custom Encoder

- **Domain-specific logic**: Your domain has specific rules (e.g., time-of-day patterns, seasonal effects)
- **Non-linear relationships**: Standard thresholds don't capture your data's structure
- **Multiple features**: Need to encode combinations of features
- **Context-aware**: Encoding depends on external context

## Basic Custom Encoder
```python
from channelpy.pipeline.encoders import LearnedThresholdEncoder
from channelpy.core import State, StateArray
import numpy as np

class CustomEncoder:
    """
    Template for custom encoder
    """
    
    def __init__(self, **params):
        """
        Initialize encoder with parameters
        """
        self.params = params
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """
        Learn encoding parameters from data
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like, optional
            Training labels (for supervised encoding)
        """
        # Learn thresholds, patterns, etc.
        self._learn_parameters(X, y)
        self.is_fitted = True
        return self
    
    def encode(self, value):
        """
        Encode single value
        
        Parameters
        ----------
        value : float
            Value to encode
            
        Returns
        -------
        state : State
            Encoded channel state
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted")
        
        # Your encoding logic here
        i = self._compute_i_bit(value)
        q = self._compute_q_bit(value)
        
        return State(i=i, q=q)
    
    def encode_array(self, X):
        """
        Encode array of values
        
        Parameters
        ----------
        X : array-like
            Values to encode
            
        Returns
        -------
        states : StateArray
            Encoded channel states
        """
        X = np.asarray(X)
        i_bits = np.array([self._compute_i_bit(x) for x in X])
        q_bits = np.array([self._compute_q_bit(x) for x in X])
        return StateArray(i=i_bits, q=q_bits)
    
    def __call__(self, X):
        """Make encoder callable"""
        if isinstance(X, (int, float)):
            return self.encode(X)
        else:
            return self.encode_array(X)
    
    def _learn_parameters(self, X, y):
        """Implement your learning logic"""
        pass
    
    def _compute_i_bit(self, value):
        """Implement i-bit logic"""
        pass
    
    def _compute_q_bit(self, value):
        """Implement q-bit logic"""
        pass
```

## Example 1: Time-of-Day Encoder

Encode values differently based on time of day:
```python
class TimeOfDayEncoder:
    """
    Encode values with time-of-day awareness
    
    Example: High transaction volume means different things
    at 2pm (normal) vs 2am (suspicious)
    """
    
    def __init__(self):
        # Threshold for each hour
        self.hourly_thresholds_i = {}
        self.hourly_thresholds_q = {}
        self.is_fitted = False
    
    def fit(self, X, timestamps, y=None):
        """
        Learn hour-specific thresholds
        
        Parameters
        ----------
        X : array-like
            Values
        timestamps : array-like
            Timestamps or hour values (0-23)
        """
        X = np.asarray(X)
        hours = np.asarray([t.hour if hasattr(t, 'hour') else t for t in timestamps])
        
        # Learn threshold for each hour
        for hour in range(24):
            hour_mask = hours == hour
            if hour_mask.sum() > 10:  # Need enough samples
                hour_data = X[hour_mask]
                self.hourly_thresholds_i[hour] = np.percentile(hour_data, 50)
                self.hourly_thresholds_q[hour] = np.percentile(hour_data, 75)
            else:
                # Fallback to global thresholds
                self.hourly_thresholds_i[hour] = np.median(X)
                self.hourly_thresholds_q[hour] = np.percentile(X, 75)
        
        self.is_fitted = True
        return self
    
    def encode(self, value, timestamp):
        """
        Encode value with time context
        
        Parameters
        ----------
        value : float
            Value to encode
        timestamp : datetime or int
            Hour (0-23) or datetime object
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted")
        
        hour = timestamp.hour if hasattr(timestamp, 'hour') else timestamp
        
        threshold_i = self.hourly_thresholds_i[hour]
        threshold_q = self.hourly_thresholds_q[hour]
        
        return State(
            i=int(value > threshold_i),
            q=int(value > threshold_q)
        )
    
    def __call__(self, value, timestamp):
        return self.encode(value, timestamp)

# Usage
encoder = TimeOfDayEncoder()
encoder.fit(
    X=transaction_amounts,
    timestamps=transaction_times
)

# Encode new transaction
state = encoder(amount=150.00, timestamp=datetime(2024, 1, 1, 14, 30))
print(f"$150 at 2:30pm: {state}")

state = encoder(amount=150.00, timestamp=datetime(2024, 1, 1, 2, 30))
print(f"$150 at 2:30am: {state}")
```

## Example 2: Multi-Feature Encoder

Encode based on combination of features:
```python
class MultiFeatureEncoder:
    """
    Encode multiple features into a single state
    
    Useful when features should be considered together
    """
    
    def __init__(self, aggregation='weighted_sum'):
        """
        Parameters
        ----------
        aggregation : str
            How to combine features: 'weighted_sum', 'product', 'max'
        """
        self.aggregation = aggregation
        self.feature_weights = None
        self.feature_encoders = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """
        Learn to encode multiple features
        
        Parameters
        ----------
        X : pd.DataFrame or dict
            Multiple features (each column is a feature)
        """
        # Create encoder for each feature
        self.feature_encoders = {}
        
        if isinstance(X, pd.DataFrame):
            features = X.columns
        else:
            features = X.keys()
        
        for feature in features:
            encoder = LearnedThresholdEncoder()
            encoder.fit(X[feature], y)
            self.feature_encoders[feature] = encoder
        
        # Learn feature weights (if supervised)
        if y is not None:
            self.feature_weights = self._learn_weights(X, y)
        else:
            self.feature_weights = {f: 1.0 for f in features}
        
        self.is_fitted = True
        return self
    
    def encode(self, feature_values):
        """
        Encode multiple features
        
        Parameters
        ----------
        feature_values : dict
            {feature_name: value}
            
        Returns
        -------
        state : State
            Combined channel state
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted")
        
        # Encode each feature
        feature_states = {}
        for feature, value in feature_values.items():
            if feature in self.feature_encoders:
                feature_states[feature] = self.feature_encoders[feature](value)
        
        # Combine
        combined_state = self._combine_states(feature_states)
        return combined_state
    
    def _combine_states(self, feature_states):
        """Combine multiple states into one"""
        
        if self.aggregation == 'weighted_sum':
            # Weighted sum of bit values
            i_score = sum(
                state.i * self.feature_weights[feature]
                for feature, state in feature_states.items()
            )
            q_score = sum(
                state.q * self.feature_weights[feature]
                for feature, state in feature_states.items()
            )
            
            total_weight = sum(self.feature_weights.values())
            i_score /= total_weight
            q_score /= total_weight
            
            # Threshold at 0.5
            return State(i=int(i_score > 0.5), q=int(q_score > 0.5))
        
        elif self.aggregation == 'product':
            # All features must agree (AND logic)
            i_bit = int(all(state.i for state in feature_states.values()))
            q_bit = int(all(state.q for state in feature_states.values()))
            return State(i=i_bit, q=q_bit)
        
        elif self.aggregation == 'max':
            # Any feature triggers (OR logic)
            i_bit = int(any(state.i for state in feature_states.values()))
            q_bit = int(any(state.q for state in feature_states.values()))
            return State(i=i_bit, q=q_bit)
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
    
    def _learn_weights(self, X, y):
        """Learn feature weights from correlation with target"""
        weights = {}
        for feature in X.columns if hasattr(X, 'columns') else X.keys():
            correlation = np.abs(np.corrcoef(X[feature], y)[0, 1])
            weights[feature] = correlation
        return weights
    
    def __call__(self, feature_values):
        return self.encode(feature_values)

# Usage
encoder = MultiFeatureEncoder(aggregation='weighted_sum')
encoder.fit(
    X=train_df[['feature1', 'feature2', 'feature3']],
    y=train_labels
)

# Encode multiple features at once
state = encoder({
    'feature1': 0.7,
    'feature2': 0.3,
    'feature3': 0.9
})
print(f"Combined state: {state}")
```

## Example 3: Context-Aware Encoder

Encoding depends on external context:
```python
class ContextAwareEncoder:
    """
    Encode values differently based on context
    
    Example: "High latency" means different things for
    local vs international requests
    """
    
    def __init__(self):
        self.context_encoders = {}
        self.default_encoder = None
        self.is_fitted = False
    
    def fit(self, X, contexts, y=None):
        """
        Learn context-specific encoders
        
        Parameters
        ----------
        X : array-like
            Values
        contexts : array-like
            Context labels (e.g., 'local', 'international')
        """
        X = np.asarray(X)
        contexts = np.asarray(contexts)
        
        # Learn encoder for each context
        unique_contexts = np.unique(contexts)
        
        for context in unique_contexts:
            context_mask = contexts == context
            context_data = X[context_mask]
            
            if len(context_data) > 10:
                encoder = LearnedThresholdEncoder()
                encoder.fit(context_data, y[context_mask] if y is not None else None)
                self.context_encoders[context] = encoder
        
        # Default encoder for unknown contexts
        self.default_encoder = LearnedThresholdEncoder()
        self.default_encoder.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def encode(self, value, context):
        """
        Encode value with context
        
        Parameters
        ----------
        value : float
            Value to encode
        context : str
            Context identifier
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted")
        
        # Use context-specific encoder if available
        if context in self.context_encoders:
            return self.context_encoders[context](value)
        else:
            return self.default_encoder(value)
    
    def __call__(self, value, context):
        return self.encode(value, context)

# Usage
encoder = ContextAwareEncoder()
encoder.fit(
    X=latencies,
    contexts=request_types  # ['local', 'international', 'cdn']
)

# Same latency, different contexts
state_local = encoder(value=50, context='local')
state_intl = encoder(value=50, context='international')

print(f"50ms latency (local): {state_local}")
print(f"50ms latency (international): {state_intl}")
```

## Example 4: Pattern-Based Encoder

Encode based on patterns in recent history:
```python
class PatternEncoder:
    """
    Encode based on recent pattern, not just current value
    
    Example: Sudden spike vs gradual increase
    """
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.window = []
        self.base_encoder = LearnedThresholdEncoder()
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Learn base encoding"""
        self.base_encoder.fit(X, y)
        self.is_fitted = True
        return self
    
    def update(self, value):
        """Add value to window"""
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    
    def encode(self, value):
        """
        Encode based on value and recent pattern
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder not fitted")
        
        # Add to window
        self.update(value)
        
        if len(self.window) < 3:
            # Not enough history, use base encoder
            return self.base_encoder(value)
        
        # Detect pattern
        pattern = self._detect_pattern()
        
        # Adjust encoding based on pattern
        base_state = self.base_encoder(value)
        adjusted_state = self._adjust_for_pattern(base_state, pattern)
        
        return adjusted_state
    
    def _detect_pattern(self):
        """Detect pattern in recent window"""
        if len(self.window) < 3:
            return 'stable'
        
        recent = np.array(self.window[-3:])
        
        # Trend detection
        diffs = np.diff(recent)
        if all(diffs > 0):
            if diffs[-1] > 2 * diffs[0]:
                return 'accelerating'
            else:
                return 'increasing'
        elif all(diffs < 0):
            if abs(diffs[-1]) > 2 * abs(diffs[0]):
                return 'crashing'
            else:
                return 'decreasing'
        else:
            # Check volatility
            std = np.std(recent)
            mean = np.mean(recent)
            cv = std / mean if mean != 0 else 0
            
            if cv > 0.2:
                return 'volatile'
            else:
                return 'stable'
    
    def _adjust_for_pattern(self, base_state, pattern):
        """Adjust state based on pattern"""
        
        if pattern == 'accelerating':
            # Escalate: δ → ψ, φ → δ
            if base_state == DELTA:
                return PSI
            elif base_state == PHI:
                return DELTA
            else:
                return base_state
        
        elif pattern == 'crashing':
            # De-escalate: ψ → δ, δ → ∅
            if base_state == PSI:
                return DELTA
            elif base_state == DELTA:
                return EMPTY
            else:
                return base_state
        
        elif pattern == 'volatile':
            # Be conservative (require more evidence)
            if base_state == PSI:
                return DELTA  # Downgrade to borderline
            else:
                return base_state
        
        else:
            # Stable pattern: trust base state
            return base_state
    
    def __call__(self, value):
        return self.encode(value)

# Usage
encoder = PatternEncoder(window_size=10)
encoder.fit(historical_data)

# Process stream
for value in data_stream:
    state = encoder(value)
    print(f"Value: {value:.2f}, State: {state}, Pattern: {encoder._detect_pattern()}")
```

## Testing Your Custom Encoder

Always test your custom encoder thoroughly:
```python
def test_custom_encoder():
    """Test suite for custom encoder"""
    
    # Test 1: Fit and encode
    encoder = CustomEncoder()
    encoder.fit(train_data)
    state = encoder(test_value)
    assert isinstance(state, State)
    
    # Test 2: Encode array
    states = encoder(test_array)
    assert isinstance(states, StateArray)
    assert len(states) == len(test_array)
    
    # Test 3: Reproducibility
    state1 = encoder(value)
    state2 = encoder(value)
    assert state1 == state2
    
    # Test 4: Edge cases
    state_min = encoder(data.min())
    state_max = encoder(data.max())
    # Verify states make sense
    
    # Test 5: Missing/invalid data
    try:
        encoder(np.nan)
        assert False, "Should raise error on NaN"
    except ValueError:
        pass
    
    print("✅ All tests passed")

test_custom_encoder()
```

## Best Practices

1. **Always validate inputs**
```python
def encode(self, value):
    if not np.isfinite(value):
        raise ValueError(f"Invalid value: {value}")
    # ... encoding logic
```

2. **Provide clear documentation**
```python
def __init__(self, threshold_method='percentile'):
    """
    Parameters
    ----------
    threshold_method : str
        Method for threshold selection:
        - 'percentile': Use data percentiles
        - 'std': Use mean ± k*std
        - 'iqr': Use interquartile range
    """
```

3. **Handle edge cases gracefully**
```python
if len(data) < min_samples:
    warnings.warn("Insufficient data, using default thresholds")
    return self._default_encode(value)
```

4. **Make encoders serializable**
```python
def get_params(self):
    """Get encoder parameters for saving"""
    return {
        'threshold_i': self.threshold_i,
        'threshold_q': self.threshold_q,
        'is_fitted': self.is_fitted
    }

def set_params(self, params):
    """Load encoder parameters"""
    self.__dict__.update(params)
```

## Next Steps

- **How-To Guide**: Create custom interpreter for your domain
- **API Reference**: Full encoder API documentation
- **Examples**: More encoder examples in `examples/custom_encoders.py`

---

**💡 Key Takeaway**: Custom encoders let you inject domain knowledge into the encoding process while maintaining the benefits of channel algebra.