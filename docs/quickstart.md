# ChannelPy Quickstart Guide

Get started with ChannelPy in 5 minutes!

## Installation
```bash
pip install channelpy
```

Or install from source:
```bash
git clone https://github.com/channelalgebra/channelpy.git
cd channelpy
pip install -e .
```

## Core Concepts in 60 Seconds

ChannelPy uses **two bits** to represent complex states:

- **i-bit (presence)**: Is something there?
- **q-bit (membership)**: Does it belong?

This creates **four fundamental states**:

| State | Symbol | i | q | Meaning |
|-------|--------|---|---|---------|
| Empty | ∅ | 0 | 0 | Nothing present |
| Delta | δ | 1 | 0 | Present but doesn't belong (puncture) |
| Phi | φ | 0 | 1 | Not present but expected (hole) |
| Psi | ψ | 1 | 1 | Present and belongs (resonant) |

## Your First Channel
```python
from channelpy import State, EMPTY, DELTA, PHI, PSI

# Create states
state1 = State(i=1, q=0)  # δ (puncture)
state2 = PSI                # ψ (resonant) - using constant

# Check state
print(state1)  # Output: δ
print(state2 == PSI)  # Output: True
```

## Basic Operations
```python
from channelpy import gate, admit, overlay, weave

# Gate: Remove unvalidated elements (if q=0, set i=0)
validated = gate(DELTA)  # δ → ∅
print(validated)  # Output: ∅

# Admit: Grant membership to present elements (if i=1, set q=1)
promoted = admit(DELTA)  # δ → ψ
print(promoted)  # Output: ψ

# Overlay: Combine states (bitwise OR)
combined = overlay(DELTA, PHI)  # δ ⊕ φ → ψ
print(combined)  # Output: ψ

# Weave: Intersection (bitwise AND)
intersect = weave(PSI, DELTA)  # ψ ⊗ δ → δ
print(intersect)  # Output: δ
```

## Building a Simple Pipeline
```python
from channelpy.pipeline import ChannelPipeline
from channelpy.pipeline.encoders import ThresholdEncoder
from channelpy.pipeline.interpreters import RuleBasedInterpreter
import numpy as np

# Create sample data
data = np.random.randn(100)

# Build pipeline
pipeline = ChannelPipeline()

# Add encoder: convert values to states
encoder = ThresholdEncoder(threshold_i=0.0, threshold_q=0.5)
pipeline.add_encoder(encoder)

# Add interpreter: states to decisions
def simple_interpreter(states):
    decisions = []
    for state in states[0]:  # states is a list of encoder outputs
        if state == PSI:
            decisions.append('HIGH')
        elif state == DELTA:
            decisions.append('MEDIUM')
        elif state == PHI:
            decisions.append('LOW')
        else:
            decisions.append('NONE')
    return decisions

pipeline.add_interpreter(simple_interpreter)

# Fit and transform
pipeline.fit(data)
decisions, states = pipeline.transform(data)

print(f"First 5 decisions: {decisions[:5]}")
print(f"First 5 states: {[str(s) for s in states[0][:5]]}")
```

## Adaptive Thresholds

The real power: thresholds that adapt to your data!
```python
from channelpy.adaptive import StreamingAdaptiveThreshold

# Create adaptive threshold
threshold = StreamingAdaptiveThreshold(window_size=100)

# Process stream
stream = np.random.randn(1000)

for value in stream:
    threshold.update(value)
    state = threshold.encode(value)
    
    # Thresholds adapt automatically!
    stats = threshold.get_stats()
    print(f"Value: {value:.2f}, State: {state}, "
          f"Threshold_i: {stats['threshold_i']:.2f}")
```

## Topology-Aware Adaptation 🌟

ChannelPy's **unique feature**: thresholds that understand your data's shape!
```python
from channelpy.adaptive import TopologyAdaptiveThreshold

# Create topology-aware threshold
threshold = TopologyAdaptiveThreshold(window_size=500)

# Bimodal data (two peaks)
bimodal_data = np.concatenate([
    np.random.normal(-2, 0.5, 500),
    np.random.normal(2, 0.5, 500)
])

for value in bimodal_data:
    threshold.update(value)

# Topology is detected automatically
topology = threshold.get_topology()
print(f"Detected {topology.modality} modes")
print(f"Mode locations: {topology.local_maxima}")

# Threshold placed between modes!
thresholds = threshold.get_thresholds()
print(f"Strategy: {thresholds['adaptation_strategy']}")
```

## Complete Example: Trading Signal
```python
from channelpy.applications import TradingChannelSystem
import pandas as pd

# Load price data
prices = pd.Series([100, 102, 101, 105, 103, 108, 110, 107])
volumes = pd.Series([1000, 1200, 900, 1500, 1100, 2000, 1800, 1300])

# Create trading system
system = TradingChannelSystem()
system.fit(prices, volumes)

# Process new tick
signal = system.process_tick(price=112, volume=2200)

print(f"Action: {signal['action']}")
print(f"Confidence: {signal['confidence']:.2f}")
```

## What's Next?

- **[Tutorial 1: States and Operations](tutorials/01_basic_states.md)** - Deep dive into channel algebra
- **[Tutorial 2: Building Pipelines](tutorials/02_building_pipeline.md)** - Create production pipelines
- **[Tutorial 3: Adaptive Thresholds](tutorials/03_adaptive_thresholds.md)** - Master adaptation strategies
- **[Trading Bot Tutorial](tutorials/04_trading_bot.md)** - Build a complete trading system
- **[Medical Diagnosis Tutorial](tutorials/05_medical_diagnosis.md)** - Healthcare application

## Key Resources

- **[API Reference](api_reference/core.md)** - Complete API documentation
- **[How-To Guides](how_to_guides/)** - Practical recipes
- **[GitHub Repository](https://github.com/channelalgebra/channelpy)** - Source code
- **[Paper](https://arxiv.org/abs/...)** - Theoretical foundations

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/channelalgebra/channelpy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/channelalgebra/channelpy/discussions)
- **Email**: support@channelalgebra.org

---

**Ready to build something amazing?** Start with [Tutorial 1](tutorials/01_basic_states.md)!