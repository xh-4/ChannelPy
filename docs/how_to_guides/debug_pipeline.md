# How to Debug a ChannelPy Pipeline

A comprehensive guide to debugging and troubleshooting ChannelPy pipelines.

## Table of Contents

1. [Common Issues and Solutions](#common-issues-and-solutions)
2. [Debugging Tools](#debugging-tools)
3. [Step-by-Step Debugging](#step-by-step-debugging)
4. [Performance Debugging](#performance-debugging)
5. [State Inspection](#state-inspection)
6. [Threshold Debugging](#threshold-debugging)
7. [Logging and Monitoring](#logging-and-monitoring)

---

## Common Issues and Solutions

### Issue 1: All States Are Empty (∅)

**Symptoms:**
```python
states = pipeline.transform(data)
print(states.count_by_state())
# Output: {EMPTY: 1000, DELTA: 0, PHI: 0, PSI: 0}
```

**Causes:**
- Thresholds are too high
- Data not normalized/scaled properly
- Encoder not fitted

**Solutions:**
```python
# Check if encoder is fitted
if not hasattr(encoder, 'threshold_i'):
    print("Encoder not fitted!")
    encoder.fit(training_data)

# Check threshold values
print(f"Threshold i: {encoder.threshold_i}")
print(f"Threshold q: {encoder.threshold_q}")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")

# If thresholds are outside data range, adjust
if encoder.threshold_i > data.max():
    print("Thresholds too high! Re-fit or manually adjust.")
    encoder.threshold_i = np.percentile(data, 50)
    encoder.threshold_q = np.percentile(data, 75)
```

### Issue 2: All States Are PSI (ψ)

**Symptoms:**
```python
# Everything is PSI
states.count_by_state()
# Output: {EMPTY: 0, DELTA: 0, PHI: 0, PSI: 1000}
```

**Causes:**
- Thresholds are too low
- Data shifted upward

**Solutions:**
```python
# Check threshold placement
print(f"Min value: {data.min():.2f}")
print(f"Threshold i: {encoder.threshold_i:.2f}")
print(f"Threshold q: {encoder.threshold_q:.2f}")

# Visualize thresholds vs data
import matplotlib.pyplot as plt

plt.hist(data, bins=50, alpha=0.5, label='Data')
plt.axvline(encoder.threshold_i, color='orange', 
            linestyle='--', label='Threshold i')
plt.axvline(encoder.threshold_q, color='red', 
            linestyle='--', label='Threshold q')
plt.legend()
plt.show()
```

### Issue 3: Pipeline Produces NaN or Errors

**Symptoms:**
```python
decisions, states = pipeline.transform(data)
# ValueError: Input contains NaN
```

**Causes:**
- Missing data not handled
- Division by zero in preprocessing
- Invalid threshold calculation

**Solutions:**
```python
# Check for NaN in input
print(f"NaN count: {np.isnan(data).sum()}")

# Add missing data handler
from channelpy.pipeline.preprocessors import MissingDataHandler

pipeline.add_preprocessor(MissingDataHandler(strategy='median'))

# Debug each stage
print("Stage 1: Preprocessing")
features = pipeline._preprocess(data)
print(f"Features shape: {features.shape}")
print(f"Features NaN: {np.isnan(features).sum()}")

print("Stage 2: Encoding")
states = pipeline._encode(features)
print(f"States shape: {len(states)}")

print("Stage 3: Interpretation")
decisions = pipeline._interpret(states)
print(f"Decisions: {decisions[:5]}")
```

### Issue 4: Poor Decision Quality

**Symptoms:**
- Decisions don't match expectations
- Low accuracy on validation data

**Causes:**
- Interpretation rules incorrect
- Thresholds not optimal
- Data distribution mismatch

**Solutions:**
```python
# Analyze state distribution
from channelpy.visualization import plot_state_distribution

plot_state_distribution(states, title="Training States")
plot_state_distribution(test_states, title="Test States")

# Compare distributions - should be similar
train_counts = states.count_by_state()
test_counts = test_states.count_by_state()

print("Training distribution:")
for state, count in train_counts.items():
    print(f"  {state}: {count}")

print("Test distribution:")
for state, count in test_counts.items():
    print(f"  {state}: {count}")

# If distributions differ significantly, investigate why
```

---

## Debugging Tools

### Tool 1: Pipeline Inspector
```python
class PipelineInspector:
    """
    Inspect pipeline internals at each stage
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def inspect_full(self, data, verbose=True):
        """
        Run data through pipeline and capture all intermediate results
        """
        results = {}
        
        # Stage 1: Preprocessing
        if verbose:
            print("=" * 60)
            print("STAGE 1: PREPROCESSING")
            print("=" * 60)
        
        features = self.pipeline._preprocess(data)
        results['features'] = features
        
        if verbose:
            print(f"Input shape: {data.shape}")
            print(f"Output shape: {features.shape}")
            print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"Feature mean: {features.mean():.3f}")
            print(f"Feature std: {features.std():.3f}")
            print(f"NaN count: {np.isnan(features).sum()}")
        
        # Stage 2: Encoding
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: ENCODING")
            print("=" * 60)
        
        states = self.pipeline._encode(features)
        results['states'] = states
        
        if verbose:
            if isinstance(states, list):
                states_combined = states[0]  # First encoder
            else:
                states_combined = states
            
            counts = states_combined.count_by_state()
            print(f"State counts:")
            for state, count in counts.items():
                pct = 100 * count / len(states_combined)
                print(f"  {state}: {count} ({pct:.1f}%)")
        
        # Stage 3: Interpretation
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 3: INTERPRETATION")
            print("=" * 60)
        
        decisions = self.pipeline._interpret(states)
        results['decisions'] = decisions
        
        if verbose:
            if isinstance(decisions, list):
                decisions_arr = np.array(decisions[0])
            else:
                decisions_arr = np.array(decisions)
            
            print(f"Decisions shape: {decisions_arr.shape}")
            print(f"Unique decisions: {np.unique(decisions_arr)}")
            print(f"Decision distribution:")
            unique, counts = np.unique(decisions_arr, return_counts=True)
            for val, count in zip(unique, counts):
                pct = 100 * count / len(decisions_arr)
                print(f"  {val}: {count} ({pct:.1f}%)")
        
        return results
    
    def visualize_pipeline(self, data):
        """
        Create visualization of pipeline flow
        """
        import matplotlib.pyplot as plt
        
        results = self.inspect_full(data, verbose=False)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Features
        ax1 = axes[0]
        features = results['features']
        ax1.hist(features, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('Stage 1: Features')
        ax1.set_xlabel('Feature Value')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: States
        ax2 = axes[1]
        states = results['states']
        if isinstance(states, list):
            states = states[0]
        
        counts = states.count_by_state()
        labels = ['∅', 'δ', 'φ', 'ψ']
        values = [counts[s] for s in [EMPTY, DELTA, PHI, PSI]]
        colors = ['lightgray', 'lightyellow', 'lightblue', 'lightgreen']
        
        ax2.bar(labels, values, color=colors, edgecolor='black')
        ax2.set_title('Stage 2: States')
        ax2.set_xlabel('State')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Decisions
        ax3 = axes[2]
        decisions = results['decisions']
        if isinstance(decisions, list):
            decisions = decisions[0]
        
        decision_arr = np.array(decisions)
        ax3.hist(decision_arr, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_title('Stage 3: Decisions')
        ax3.set_xlabel('Decision')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Usage
inspector = PipelineInspector(pipeline)
results = inspector.inspect_full(test_data)
fig = inspector.visualize_pipeline(test_data)
```

### Tool 2: Threshold Debugger
```python
class ThresholdDebugger:
    """
    Debug threshold-related issues
    """
    
    def __init__(self, encoder):
        self.encoder = encoder
    
    def analyze_thresholds(self, data):
        """
        Analyze how well thresholds are placed
        """
        print("THRESHOLD ANALYSIS")
        print("=" * 60)
        
        # Data statistics
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"Data mean: {data.mean():.3f}")
        print(f"Data std: {data.std():.3f}")
        print(f"Data median: {np.median(data):.3f}")
        print(f"Data IQR: {np.percentile(data, 75) - np.percentile(data, 25):.3f}")
        
        # Threshold values
        print(f"\nThreshold i: {self.encoder.threshold_i:.3f}")
        print(f"Threshold q: {self.encoder.threshold_q:.3f}")
        print(f"Threshold gap: {self.encoder.threshold_q - self.encoder.threshold_i:.3f}")
        
        # Percentile analysis
        i_percentile = (data < self.encoder.threshold_i).sum() / len(data) * 100
        q_percentile = (data < self.encoder.threshold_q).sum() / len(data) * 100
        
        print(f"\nThreshold i at {i_percentile:.1f}th percentile")
        print(f"Threshold q at {q_percentile:.1f}th percentile")
        
        # State distribution preview
        states = self.encoder(data)
        counts = states.count_by_state()
        
        print(f"\nResulting state distribution:")
        for state, count in counts.items():
            pct = 100 * count / len(states)
            print(f"  {state}: {count} ({pct:.1f}%)")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        
        if i_percentile < 25 or i_percentile > 75:
            print(f"  ⚠ Threshold i at extreme percentile ({i_percentile:.1f})")
            print(f"    Consider adjusting to 50th percentile: {np.median(data):.3f}")
        
        if q_percentile < 50 or q_percentile > 90:
            print(f"  ⚠ Threshold q at extreme percentile ({q_percentile:.1f})")
            print(f"    Consider adjusting to 75th percentile: {np.percentile(data, 75):.3f}")
        
        if (self.encoder.threshold_q - self.encoder.threshold_i) < 0.1 * data.std():
            print(f"  ⚠ Thresholds too close together")
            print(f"    Consider increasing gap to ~0.5 * std: {0.5 * data.std():.3f}")
        
        if counts[EMPTY] == len(states) or counts[PSI] == len(states):
            print(f"  ⚠ All states identical - no discrimination!")
            print(f"    Thresholds need adjustment")
    
    def plot_thresholds(self, data):
        """
        Visualize threshold placement
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Histogram with thresholds
        ax1.hist(data, bins=50, alpha=0.5, edgecolor='black', label='Data')
        ax1.axvline(self.encoder.threshold_i, color='orange', 
                   linestyle='--', linewidth=2, label=f'Threshold i = {self.encoder.threshold_i:.3f}')
        ax1.axvline(self.encoder.threshold_q, color='red', 
                   linestyle='--', linewidth=2, label=f'Threshold q = {self.encoder.threshold_q:.3f}')
        ax1.axvline(data.mean(), color='green', 
                   linestyle=':', linewidth=2, alpha=0.5, label=f'Mean = {data.mean():.3f}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Count')
        ax1.set_title('Threshold Placement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative distribution
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax2.plot(sorted_data, cumulative, linewidth=2, label='Cumulative')
        ax2.axvline(self.encoder.threshold_i, color='orange', 
                   linestyle='--', linewidth=2, label='Threshold i')
        ax2.axvline(self.encoder.threshold_q, color='red', 
                   linestyle='--', linewidth=2, label='Threshold q')
        
        # Add percentile lines
        for pct in [25, 50, 75]:
            val = np.percentile(data, pct)
            ax2.axhline(pct/100, color='gray', linestyle=':', alpha=0.3)
            ax2.axvline(val, color='gray', linestyle=':', alpha=0.3)
        
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Usage
debugger = ThresholdDebugger(encoder)
debugger.analyze_thresholds(data)
fig = debugger.plot_thresholds(data)
```

### Tool 3: State Tracer
```python
class StateTracer:
    """
    Trace individual data points through pipeline
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def trace_sample(self, data, index):
        """
        Trace a single sample through entire pipeline
        """
        print(f"TRACING SAMPLE {index}")
        print("=" * 60)
        
        sample = data[index]
        
        # Stage 1: Preprocessing
        print("Stage 1: Preprocessing")
        print(f"  Input: {sample:.3f}")
        
        # Apply each preprocessor
        current_value = sample
        for i, prep in enumerate(self.pipeline.preprocessors):
            prev_value = current_value
            if hasattr(prep, 'transform'):
                current_value = prep.transform(np.array([current_value]))[0]
            else:
                current_value = prep(np.array([current_value]))[0]
            
            print(f"  After preprocessor {i} ({prep.__class__.__name__}): "
                  f"{prev_value:.3f} → {current_value:.3f}")
        
        feature = current_value
        print(f"  Final feature: {feature:.3f}")
        
        # Stage 2: Encoding
        print("\nStage 2: Encoding")
        for i, encoder in enumerate(self.pipeline.encoders):
            if hasattr(encoder, 'threshold_i'):
                print(f"  Encoder {i} thresholds: "
                      f"i={encoder.threshold_i:.3f}, q={encoder.threshold_q:.3f}")
            
            state = encoder(np.array([feature]))
            if hasattr(state, '__getitem__'):
                state = state[0]
            
            print(f"  Encoder {i} result: {state}")
            print(f"    i-bit: {feature:.3f} > {encoder.threshold_i:.3f} = {state.i}")
            print(f"    q-bit: {feature:.3f} > {encoder.threshold_q:.3f} = {state.q}")
        
        # Stage 3: Interpretation
        print("\nStage 3: Interpretation")
        states = self.pipeline._encode(np.array([feature]))
        decisions = self.pipeline._interpret(states)
        
        if isinstance(decisions, list):
            decision = decisions[0]
        else:
            decision = decisions
        
        print(f"  Decision: {decision}")
        
        return {
            'input': sample,
            'feature': feature,
            'state': state,
            'decision': decision
        }
    
    def trace_multiple(self, data, indices):
        """
        Trace multiple samples
        """
        results = []
        for idx in indices:
            result = self.trace_sample(data, idx)
            results.append(result)
            print("\n")
        
        return results

# Usage
tracer = StateTracer(pipeline)

# Trace interesting samples
tracer.trace_sample(data, 0)  # First sample
tracer.trace_sample(data, len(data)//2)  # Middle sample
tracer.trace_sample(data, -1)  # Last sample

# Or trace multiple
results = tracer.trace_multiple(data, [0, 10, 20, 30])
```

---

## Step-by-Step Debugging

### Checklist for Debugging

When your pipeline isn't working:
```python
def debug_pipeline_checklist(pipeline, train_data, test_data):
    """
    Comprehensive pipeline debugging checklist
    """
    
    issues = []
    
    print("PIPELINE DEBUG CHECKLIST")
    print("=" * 60)
    
    # Check 1: Pipeline is fitted
    print("✓ Checking if pipeline is fitted...")
    if not pipeline.is_fitted:
        issues.append("Pipeline not fitted")
        print("  ✗ Pipeline not fitted. Call pipeline.fit(train_data) first.")
    else:
        print("  ✓ Pipeline is fitted")
    
    # Check 2: Data validity
    print("\n✓ Checking data validity...")
    if np.isnan(train_data).any():
        issues.append("Training data contains NaN")
        print(f"  ✗ Training data contains {np.isnan(train_data).sum()} NaN values")
    else:
        print("  ✓ Training data is valid")
    
    if np.isnan(test_data).any():
        issues.append("Test data contains NaN")
        print(f"  ✗ Test data contains {np.isnan(test_data).sum()} NaN values")
    else:
        print("  ✓ Test data is valid")
    
    # Check 3: Data ranges
    print("\n✓ Checking data ranges...")
    train_range = (train_data.min(), train_data.max())
    test_range = (test_data.min(), test_data.max())
    
    print(f"  Training range: [{train_range[0]:.3f}, {train_range[1]:.3f}]")
    print(f"  Test range: [{test_range[0]:.3f}, {test_range[1]:.3f}]")
    
    if test_range[0] < train_range[0] or test_range[1] > train_range[1]:
        issues.append("Test data outside training range")
        print("  ⚠ Test data outside training range - may cause issues")
    else:
        print("  ✓ Test data within training range")
    
    # Check 4: Encoders have reasonable thresholds
    print("\n✓ Checking encoder thresholds...")
    for i, encoder in enumerate(pipeline.encoders):
        if hasattr(encoder, 'threshold_i'):
            print(f"  Encoder {i}:")
            print(f"    Threshold i: {encoder.threshold_i:.3f}")
            print(f"    Threshold q: {encoder.threshold_q:.3f}")
            
            if encoder.threshold_i >= train_range[1]:
                issues.append(f"Encoder {i} threshold_i too high")
                print(f"    ✗ Threshold i >= max training value")
            elif encoder.threshold_i <= train_range[0]:
                issues.append(f"Encoder {i} threshold_i too low")
                print(f"    ✗ Threshold i <= min training value")
            else:
                print(f"    ✓ Threshold i in valid range")
    
    # Check 5: State distribution
    print("\n✓ Checking state distribution...")
    try:
        _, states = pipeline.transform(train_data[:100])
        if isinstance(states, list):
            states = states[0]
        
        counts = states.count_by_state()
        total = sum(counts.values())
        
        print("  State distribution:")
        for state, count in counts.items():
            pct = 100 * count / total
            print(f"    {state}: {pct:.1f}%")
        
        if counts[EMPTY] == total or counts[PSI] == total:
            issues.append("All states identical")
            print("  ✗ All states identical - no discrimination")
        else:
            print("  ✓ States have variety")
    
    except Exception as e:
        issues.append(f"Transform failed: {e}")
        print(f"  ✗ Transform failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print(f"FOUND {len(issues)} ISSUE(S):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✓ ALL CHECKS PASSED")
    
    return issues

# Usage
issues = debug_pipeline_checklist(pipeline, train_data, test_data)
```

---

## Performance Debugging

### Profiling Pipeline Performance
```python
import time

class PipelineProfiler:
    """
    Profile pipeline performance
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.timings = {}
    
    def profile(self, data):
        """
        Profile each stage of pipeline
        """
        print("PIPELINE PERFORMANCE PROFILE")
        print("=" * 60)
        
        # Overall timing
        start_overall = time.time()
        
        # Stage 1: Preprocessing
        start = time.time()
        features = self.pipeline._preprocess(data)
        preprocess_time = time.time() - start
        self.timings['preprocess'] = preprocess_time
        
        # Stage 2: Encoding
        start = time.time()
        states = self.pipeline._encode(features)
        encode_time = time.time() - start
        self.timings['encode'] = encode_time
        
        # Stage 3: Interpretation
        start = time.time()
        decisions = self.pipeline._interpret(states)
        interpret_time = time.time() - start
        self.timings['interpret'] = interpret_time
        
        # Total
        total_time = time.time() - start_overall
        self.timings['total'] = total_time
        
        # Report
        print(f"Data size: {len(data)} samples")
        print(f"\nStage timings:")
        print(f"  Preprocessing: {preprocess_time:.4f}s ({100*preprocess_time/total_time:.1f}%)")
        print(f"  Encoding:      {encode_time:.4f}s ({100*encode_time/total_time:.1f}%)")
        print(f"  Interpretation: {interpret_time:.4f}s ({100*interpret_time/total_time:.1f}%)")
        print(f"  Total:         {total_time:.4f}s")
        
        print(f"\nThroughput: {len(data)/total_time:.0f} samples/second")
        
        return self.timings

# Usage
profiler = PipelineProfiler(pipeline)
timings = profiler.profile(large_dataset)
```

---

## Logging and Monitoring

### Setting Up Pipeline Logging
```python
import logging

def setup_pipeline_logging(pipeline, level=logging.INFO):
    """
    Add logging to pipeline for debugging
    """
    
    logger = logging.getLogger('channelpy')
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Wrap pipeline methods with logging
    original_transform = pipeline.transform
    
    def logged_transform(X):
        logger.info(f"Starting transform on {len(X)} samples")
        
        try:
            result = original_transform(X)
            logger.info("Transform completed successfully")
            return result
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            raise
    
    pipeline.transform = logged_transform
    
    return logger

# Usage
logger = setup_pipeline_logging(pipeline, level=logging.DEBUG)
decisions, states = pipeline.transform(data)
```

---

## Quick Debugging Commands
```python
# Quick inspection
inspector = PipelineInspector(pipeline)
inspector.inspect_full(data)

# Quick threshold check
debugger = ThresholdDebugger(encoder)
debugger.analyze_thresholds(data)

# Quick trace
tracer = StateTracer(pipeline)
tracer.trace_sample(data, 0)

# Quick checklist
issues = debug_pipeline_checklist(pipeline, train_data, test_data)

# Quick performance check
profiler = PipelineProfiler(pipeline)
profiler.profile(data)
```

---

## Common Error Messages

### "Pipeline not fitted"
**Solution:** Call `pipeline.fit(training_data)` before `transform()`

### "Input contains NaN"
**Solution:** Add `MissingDataHandler` preprocessor or filter NaN values

### "Thresholds not set"
**Solution:** Encoder not fitted. Call `encoder.fit(data)` first

### "Shape mismatch"
**Solution:** Check that preprocessors return correct shape

### "All states identical"
**Solution:** Adjust thresholds using `ThresholdDebugger`

---

## Best Practices

1. **Always inspect intermediate results** - Don't just look at final output
2. **Visualize state distributions** - Catches threshold issues early
3. **Test on small data first** - Faster iteration
4. **Use logging in production** - Track issues in deployed systems
5. **Profile before optimizing** - Know where the bottlenecks are
6. **Keep test cases** - Build library of debugging examples

---

## Next Steps

- [Custom Encoder Guide](custom_encoder.md)
- [Custom Interpreter Guide](custom_interpreter.md)
- [API Reference: Pipeline](../api_reference/pipeline.md)