# Tutorial 1: States and Operations

Learn the fundamentals of channel algebra through hands-on examples.

## Table of Contents

1. [Understanding the Four States](#understanding-the-four-states)
2. [Creating States](#creating-states)
3. [Core Operations](#core-operations)
4. [State Algebra](#state-algebra)
5. [Working with Arrays](#working-with-arrays)
6. [Practical Examples](#practical-examples)

## Understanding the Four States

Channel algebra uses **two bits** to capture complex information:

### The i-bit (Presence)
- **0**: Element is not present
- **1**: Element is present

### The q-bit (Membership)
- **0**: Element does not belong to the set
- **1**: Element belongs to the set

### Four Fundamental States
```python
from channelpy import State, EMPTY, DELTA, PHI, PSI

# Empty (∅): i=0, q=0
# Nothing is present
state_empty = EMPTY
print(f"Empty: {state_empty}")  # ∅

# Delta (δ): i=1, q=0  
# Present but doesn't belong (puncture, outlier)
state_delta = DELTA
print(f"Delta: {state_delta}")  # δ

# Phi (φ): i=0, q=1
# Not present but expected (hole, missing data)
state_phi = PHI
print(f"Phi: {state_phi}")  # φ

# Psi (ψ): i=1, q=1
# Present and belongs (resonant, validated)
state_psi = PSI
print(f"Psi: {state_psi}")  # ψ
```

### Real-World Interpretations

**Customer Validation:**
- ∅ (Empty): No customer data
- δ (Delta): Customer exists but failed validation
- φ (Phi): Expected customer, not yet arrived
- ψ (Psi): Validated, active customer

**Medical Test:**
- ∅ (Empty): No test taken
- δ (Delta): Test result present but abnormal
- φ (Phi): Test scheduled but not completed
- ψ (Psi): Test complete and normal

**Trading Signal:**
- ∅ (Empty): No signal
- δ (Delta): Weak signal (present but not actionable)
- φ (Phi): Expected signal hasn't triggered
- ψ (Psi): Strong confirmed signal

## Creating States
```python
from channelpy import State

# Method 1: Direct construction
state1 = State(i=1, q=0)
print(state1)  # δ

# Method 2: Using constants
state2 = PSI
print(state2)  # ψ

# Method 3: From integer (0-3)
state3 = State.from_int(2)  # 2 = i:1, q:0
print(state3)  # δ

# Method 4: From name
state4 = State.from_name('psi')
print(state4)  # ψ

# Method 5: From name with Unicode
state5 = State.from_name('ψ')
print(state5)  # ψ
```

## Core Operations

### 1. Gate Operation

**Gate removes unvalidated elements** (if q=0, set i=0)

Think of it as a filter: "Only keep what's validated"
```python
from channelpy import gate

# Gate each state
print(f"gate(∅) = {gate(EMPTY)}")    # ∅ → ∅ (nothing to remove)
print(f"gate(δ) = {gate(DELTA)}")    # δ → ∅ (puncture removed)
print(f"gate(φ) = {gate(PHI)}")      # φ → φ (hole preserved)
print(f"gate(ψ) = {gate(PSI)}")      # ψ → ψ (resonant preserved)
```

**Real example: Quality Control**
```python
# Product inspection results
raw_data = [PSI, DELTA, PSI, PHI, DELTA, PSI]
print("Raw:", [str(s) for s in raw_data])

# Apply gate: remove defective products
filtered = [gate(s) for s in raw_data]
print("Filtered:", [str(s) for s in filtered])
# [ψ, ∅, ψ, φ, ∅, ψ] - defects (δ) removed
```

### 2. Admit Operation

**Admit grants membership to present elements** (if i=1, set q=1)

Think of it as validation: "Approve what's present"
```python
from channelpy import admit

# Admit each state
print(f"admit(∅) = {admit(EMPTY)}")    # ∅ → ∅ (nothing to admit)
print(f"admit(δ) = {admit(DELTA)}")    # δ → ψ (puncture validated!)
print(f"admit(φ) = {admit(PHI)}")      # φ → φ (hole unchanged)
print(f"admit(ψ) = {admit(PSI)}")      # ψ → ψ (already validated)
```

**Real example: User Approval**
```python
# New user registrations
pending_users = [DELTA, DELTA, PSI, DELTA]
print("Pending:", [str(s) for s in pending_users])

# Admin approves all present users
approved = [admit(s) for s in pending_users]
print("Approved:", [str(s) for s in approved])
# [ψ, ψ, ψ, ψ] - all present users approved
```

### 3. Overlay Operation

**Overlay combines states** (bitwise OR)

Think of it as union: "Take maximum information from both"
```python
from channelpy import overlay

# Overlay combines information
print(f"overlay(δ, φ) = {overlay(DELTA, PHI)}")  # δ ⊕ φ → ψ
print(f"overlay(∅, ψ) = {overlay(EMPTY, PSI)}")  # ∅ ⊕ ψ → ψ
print(f"overlay(δ, δ) = {overlay(DELTA, DELTA)}")  # δ ⊕ δ → δ
```

**Real example: Multi-Source Data Fusion**
```python
# Sensor 1 readings
sensor1 = [PSI, DELTA, EMPTY, PHI]

# Sensor 2 readings  
sensor2 = [DELTA, PSI, PHI, EMPTY]

# Fuse sensor data
fused = [overlay(s1, s2) for s1, s2 in zip(sensor1, sensor2)]
print("Sensor 1:", [str(s) for s in sensor1])
print("Sensor 2:", [str(s) for s in sensor2])
print("Fused:   ", [str(s) for s in fused])
# [ψ, ψ, φ, φ] - combined information
```

### 4. Weave Operation

**Weave intersects states** (bitwise AND)

Think of it as intersection: "Keep only common information"
```python
from channelpy import weave

# Weave keeps only common bits
print(f"weave(ψ, δ) = {weave(PSI, DELTA)}")  # ψ ⊗ δ → δ
print(f"weave(ψ, φ) = {weave(PSI, PHI)}")    # ψ ⊗ φ → φ
print(f"weave(δ, φ) = {weave(DELTA, PHI)}")  # δ ⊗ φ → ∅
```

**Real example: Feature Agreement**
```python
# Two models' predictions
model1 = [PSI, PSI, DELTA, PHI]
model2 = [PSI, DELTA, DELTA, EMPTY]

# Keep only agreement
agreement = [weave(m1, m2) for m1, m2 in zip(model1, model2)]
print("Model 1:  ", [str(s) for s in model1])
print("Model 2:  ", [str(s) for s in model2])
print("Agreement:", [str(s) for s in agreement])
# [ψ, δ, δ, ∅] - only common information
```

### 5. Complement Operation

**Complement flips both bits**
```python
from channelpy import comp

# Complement inverts state
print(f"comp(∅) = {comp(EMPTY)}")  # ∅ → ψ
print(f"comp(δ) = {comp(DELTA)}")  # δ → φ
print(f"comp(φ) = {comp(PHI)}")    # φ → δ
print(f"comp(ψ) = {comp(PSI)}")    # ψ → ∅
```

## State Algebra

### Composition of Operations

Operations can be composed:
```python
from channelpy import gate, admit, overlay, compose, pipe

# Compose: right to left (mathematical notation)
admit_then_gate = compose(gate, admit)
result = admit_then_gate(DELTA)
print(f"admit then gate on δ: {result}")  # δ → ψ → ψ

# Pipe: left to right (Unix-style)
gate_then_admit = pipe(gate, admit)
result = gate_then_admit(DELTA)
print(f"gate then admit on δ: {result}")  # δ → ∅ → ∅
```

### Idempotence

Some operations are idempotent:
```python
# Gate is idempotent
s = DELTA
print(f"gate(δ) = {gate(s)}")
print(f"gate(gate(δ)) = {gate(gate(s))}")  # Same result

# Admit is idempotent
s = DELTA
print(f"admit(δ) = {admit(s)}")
print(f"admit(admit(δ)) = {admit(admit(s))}")  # Same result
```

### Absorption

Gate and admit have absorption properties:
```python
# gate ∘ admit ∘ gate = gate
s = DELTA
result1 = gate(s)
result2 = gate(admit(gate(s)))
print(f"gate(δ) = {result1}")
print(f"gate(admit(gate(δ))) = {result2}")
print(f"Equal: {result1 == result2}")  # True
```

## Working with Arrays

For efficiency with large datasets, use `StateArray`:
```python
from channelpy import StateArray
import numpy as np

# Create state array from bits
i_bits = np.array([1, 0, 1, 1, 0])
q_bits = np.array([1, 1, 0, 1, 0])
states = StateArray(i_bits, q_bits)

print(f"Length: {len(states)}")
print(f"First state: {states[0]}")
print(f"All states: {states.to_strings()}")

# Apply operations to arrays
gated = gate(states)
print(f"After gate: {gated.to_strings()}")

# Count states
counts = states.count_by_state()
print(f"ψ count: {counts[PSI]}")
print(f"δ count: {counts[DELTA]}")
```

## Practical Examples

### Example 1: Data Quality Assessment
```python
from channelpy import State, gate, admit

def assess_data_quality(values, thresholds):
    """
    Assess data quality using channel states
    
    Present if value exists
    Valid if value within thresholds
    """
    states = []
    
    for value in values:
        if value is None:
            states.append(PHI)  # Missing (expected but not present)
        elif value < thresholds['min'] or value > thresholds['max']:
            states.append(DELTA)  # Outlier (present but invalid)
        else:
            states.append(PSI)  # Good (present and valid)
    
    return states

# Test data
data = [5.2, None, 8.7, 15.3, 6.1, None, 4.8]
thresholds = {'min': 4.0, 'max': 10.0}

states = assess_data_quality(data, thresholds)
print("Data states:", [str(s) for s in states])
# [ψ, φ, ψ, δ, ψ, φ, ψ]

# Filter to valid only
valid_states = [gate(s) for s in states]
print("Valid only: ", [str(s) for s in valid_states])
# [ψ, φ, ψ, ∅, ψ, φ, ψ]
```

### Example 2: Multi-Stage Validation
```python
def multi_stage_validation(data):
    """
    Three-stage validation pipeline
    """
    # Stage 1: Check if data exists
    stage1 = []
    for value in data:
        if value is not None:
            stage1.append(DELTA)  # Present, not yet validated
        else:
            stage1.append(EMPTY)  # Not present
    
    print("After stage 1:", [str(s) for s in stage1])
    
    # Stage 2: Basic validation (admit present elements)
    stage2 = [admit(s) for s in stage1]
    print("After stage 2:", [str(s) for s in stage2])
    
    # Stage 3: Quality check (gate to remove bad data)
    stage3 = []
    for s, value in zip(stage2, data):
        if s == PSI and value < 0:  # Negative values fail
            stage3.append(DELTA)  # Downgrade to puncture
        else:
            stage3.append(s)
    
    stage3_final = [gate(s) for s in stage3]
    print("After stage 3:", [str(s) for s in stage3_final])
    
    return stage3_final

# Test
test_data = [5, None, -2, 8, 3]
result = multi_stage_validation(test_data)
```

### Example 3: Consensus Building
```python
def build_consensus(votes):
    """
    Build consensus from multiple votes
    
    votes: list of lists of states
    """
    consensus = []
    
    for i in range(len(votes[0])):
        # Get all votes for position i
        position_votes = [vote[i] for vote in votes]
        
        # Start with empty
        result = EMPTY
        
        # Overlay all votes (union)
        for vote in position_votes:
            result = overlay(result, vote)
        
        consensus.append(result)
    
    return consensus

# Three voters
voter1 = [PSI, DELTA, EMPTY, PHI]
voter2 = [PSI, PSI, EMPTY, EMPTY]
voter3 = [DELTA, PSI, PHI, PHI]

consensus = build_consensus([voter1, voter2, voter3])
print("Voter 1:  ", [str(s) for s in voter1])
print("Voter 2:  ", [str(s) for s in voter2])
print("Voter 3:  ", [str(s) for s in voter3])
print("Consensus:", [str(s) for s in consensus])
# [ψ, ψ, φ, φ] - maximum information from all votes
```

## Key Takeaways

1. **Four states** capture presence and membership
2. **Gate** filters (removes unvalidated)
3. **Admit** validates (promotes present to validated)
4. **Overlay** combines (union/maximum information)
5. **Weave** intersects (common information only)
6. **Operations compose** naturally
7. **StateArray** for efficient bulk operations

## What's Next?

Now that you understand states and operations, learn to build pipelines:

➡️ **[Tutorial 2: Building Pipelines](02_building_pipeline.md)**

## Exercises

Try these to test your understanding:

1. What happens when you `gate(admit(DELTA))`?
2. What's the difference between `overlay(PSI, EMPTY)` and `weave(PSI, EMPTY)`?
3. Create a function that converts a list of nullable booleans to states
4. Build a consensus function that requires 2/3 majority for PSI

Solutions in the [GitHub repository](https://github.com/channelalgebra/channelpy/tree/main/examples).