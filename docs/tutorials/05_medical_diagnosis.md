# Tutorial 05: Medical Diagnosis System

In this tutorial, we'll build a medical diagnosis support system using ChannelPy. This demonstrates how channel algebra can create safe, interpretable diagnostic aids.

⚠️ **Important Medical Disclaimer**: This is for educational purposes only. This system is NOT a replacement for professional medical diagnosis. Always consult qualified healthcare professionals.

## What We'll Build

A diagnostic system that:
- Encodes multiple test results and symptoms
- Uses parallel channels for different diagnostic dimensions
- Provides confidence-weighted diagnoses
- Explains reasoning transparently
- Handles missing data gracefully

## Prerequisites
```python
import numpy as np
import pandas as pd
from channelpy import State, PSI, DELTA, PHI, EMPTY
from channelpy.core import ParallelChannels, NestedState
from channelpy.applications import MedicalDiagnosisSystem
from channelpy.adaptive import create_medical_scorer
from channelpy.pipeline import ChannelPipeline
```

## Step 1: Define Medical Features
```python
# Example: Cardiovascular disease risk assessment

class CardiovascularFeatures:
    """
    Features for cardiovascular disease assessment
    """
    
    # Laboratory tests (continuous)
    LAB_TESTS = [
        'cholesterol_total',     # mg/dL
        'cholesterol_ldl',       # mg/dL (bad cholesterol)
        'cholesterol_hdl',       # mg/dL (good cholesterol)
        'triglycerides',         # mg/dL
        'blood_glucose',         # mg/dL
        'blood_pressure_sys',    # mmHg (systolic)
        'blood_pressure_dia',    # mmHg (diastolic)
    ]
    
    # Clinical measurements (continuous)
    MEASUREMENTS = [
        'age',                   # years
        'bmi',                   # kg/m²
        'heart_rate',            # bpm
    ]
    
    # Symptoms (binary → will encode directly)
    SYMPTOMS = [
        'chest_pain',
        'shortness_of_breath',
        'fatigue',
        'dizziness',
        'palpitations',
    ]
    
    # Risk factors (binary)
    RISK_FACTORS = [
        'smoking',
        'diabetes',
        'family_history',
        'sedentary_lifestyle',
    ]
    
    @staticmethod
    def normal_ranges():
        """Reference ranges for tests"""
        return {
            'cholesterol_total': (0, 200),      # Normal < 200
            'cholesterol_ldl': (0, 100),        # Optimal < 100
            'cholesterol_hdl': (60, 1000),      # Good > 60
            'triglycerides': (0, 150),          # Normal < 150
            'blood_glucose': (70, 100),         # Normal 70-100
            'blood_pressure_sys': (90, 120),    # Normal < 120
            'blood_pressure_dia': (60, 80),     # Normal < 80
            'age': (0, 65),                     # Risk increases > 65
            'bmi': (18.5, 25),                  # Normal 18.5-25
            'heart_rate': (60, 100),            # Normal 60-100
        }

# Generate synthetic patient data
def generate_patient_data(n_patients=100, risk_level='mixed'):
    """
    Generate synthetic patient data
    
    risk_level: 'low', 'high', 'mixed'
    """
    np.random.seed(42)
    
    patients = []
    
    for i in range(n_patients):
        if risk_level == 'mixed':
            is_high_risk = np.random.random() > 0.5
        elif risk_level == 'high':
            is_high_risk = True
        else:
            is_high_risk = False
        
        if is_high_risk:
            # High risk profile
            patient = {
                'patient_id': f'P{i:04d}',
                'cholesterol_total': np.random.normal(240, 30),
                'cholesterol_ldl': np.random.normal(160, 30),
                'cholesterol_hdl': np.random.normal(35, 10),
                'triglycerides': np.random.normal(200, 50),
                'blood_glucose': np.random.normal(120, 20),
                'blood_pressure_sys': np.random.normal(145, 15),
                'blood_pressure_dia': np.random.normal(95, 10),
                'age': np.random.normal(65, 10),
                'bmi': np.random.normal(32, 5),
                'heart_rate': np.random.normal(85, 15),
                'chest_pain': np.random.random() > 0.4,
                'shortness_of_breath': np.random.random() > 0.5,
                'fatigue': np.random.random() > 0.3,
                'dizziness': np.random.random() > 0.6,
                'palpitations': np.random.random() > 0.5,
                'smoking': np.random.random() > 0.4,
                'diabetes': np.random.random() > 0.5,
                'family_history': np.random.random() > 0.3,
                'sedentary_lifestyle': np.random.random() > 0.4,
                'true_risk': 'high'
            }
        else:
            # Low risk profile
            patient = {
                'patient_id': f'P{i:04d}',
                'cholesterol_total': np.random.normal(180, 20),
                'cholesterol_ldl': np.random.normal(90, 20),
                'cholesterol_hdl': np.random.normal(70, 10),
                'triglycerides': np.random.normal(100, 30),
                'blood_glucose': np.random.normal(85, 10),
                'blood_pressure_sys': np.random.normal(110, 10),
                'blood_pressure_dia': np.random.normal(70, 8),
                'age': np.random.normal(45, 10),
                'bmi': np.random.normal(23, 3),
                'heart_rate': np.random.normal(70, 10),
                'chest_pain': np.random.random() > 0.9,
                'shortness_of_breath': np.random.random() > 0.8,
                'fatigue': np.random.random() > 0.7,
                'dizziness': np.random.random() > 0.9,
                'palpitations': np.random.random() > 0.8,
                'smoking': np.random.random() > 0.8,
                'diabetes': np.random.random() > 0.9,
                'family_history': np.random.random() > 0.7,
                'sedentary_lifestyle': np.random.random() > 0.7,
                'true_risk': 'low'
            }
        
        patients.append(patient)
    
    return pd.DataFrame(patients)

# Generate data
patients_df = generate_patient_data(n_patients=50, risk_level='mixed')
print(f"Generated {len(patients_df)} patient records")
print("\nSample patient:")
print(patients_df.iloc[0])
```

## Step 2: Build Diagnostic Pipeline
```python
from channelpy.applications.medical import MedicalDiagnosisSystem

# Initialize diagnosis system
diagnosis_system = MedicalDiagnosisSystem(
    condition='cardiovascular_risk',
    features=CardiovascularFeatures()
)

# Define encoding strategy for each feature type
diagnosis_system.add_feature_group(
    name='lipid_panel',
    features=['cholesterol_total', 'cholesterol_ldl', 'cholesterol_hdl', 'triglycerides'],
    strategy='reference_range'  # Use clinical reference ranges
)

diagnosis_system.add_feature_group(
    name='blood_pressure',
    features=['blood_pressure_sys', 'blood_pressure_dia'],
    strategy='reference_range'
)

diagnosis_system.add_feature_group(
    name='metabolic',
    features=['blood_glucose', 'bmi'],
    strategy='reference_range'
)

diagnosis_system.add_feature_group(
    name='symptoms',
    features=CardiovascularFeatures.SYMPTOMS,
    strategy='binary'  # Already binary
)

diagnosis_system.add_feature_group(
    name='risk_factors',
    features=CardiovascularFeatures.RISK_FACTORS,
    strategy='binary'
)

print("Diagnosis system configured")
```

## Step 3: Encode Patient Data
```python
# Encode first patient
patient = patients_df.iloc[0]

encoded = diagnosis_system.encode_patient(patient)

print("\n🏥 Patient Encoding:")
print(f"Patient ID: {patient['patient_id']}")
print(f"\nLipid Panel:")
for feature in encoded['lipid_panel']:
    print(f"  {feature['name']:20s}: {feature['value']:6.1f} → {feature['state']}")

print(f"\nBlood Pressure:")
for feature in encoded['blood_pressure']:
    print(f"  {feature['name']:20s}: {feature['value']:6.1f} → {feature['state']}")

print(f"\nSymptoms:")
for feature in encoded['symptoms']:
    print(f"  {feature['name']:20s}: {feature['value']} → {feature['state']}")
```

## Step 4: Generate Diagnosis
```python
# Get diagnosis for patient
diagnosis = diagnosis_system.diagnose(patient)

print(f"\n📋 Diagnosis Report:")
print(f"Patient: {patient['patient_id']}")
print(f"Risk Level: {diagnosis['risk_level']}")
print(f"Confidence: {diagnosis['confidence']:.1%}")
print(f"\n{diagnosis['summary']}")

print(f"\n⚠️ Risk Factors ({len(diagnosis['risk_factors'])}):")
for rf in diagnosis['risk_factors']:
    print(f"  • {rf['factor']}: {rf['severity']} (state: {rf['state']})")

print(f"\n💊 Recommendations:")
for rec in diagnosis['recommendations']:
    print(f"  {rec['priority']}: {rec['action']}")
```

## Step 5: Batch Diagnosis with Analysis
```python
# Diagnose all patients
all_diagnoses = []

for idx, patient in patients_df.iterrows():
    diagnosis = diagnosis_system.diagnose(patient)
    all_diagnoses.append({
        'patient_id': patient['patient_id'],
        'true_risk': patient['true_risk'],
        'predicted_risk': diagnosis['risk_level'],
        'confidence': diagnosis['confidence'],
        'num_risk_factors': len(diagnosis['risk_factors'])
    })

diagnoses_df = pd.DataFrame(all_diagnoses)

# Evaluate accuracy
from sklearn.metrics import confusion_matrix, classification_report

# Convert to binary
y_true = (diagnoses_df['true_risk'] == 'high').astype(int)
y_pred = (diagnoses_df['predicted_risk'] == 'high').astype(int)

print("\n📊 Diagnostic Accuracy:")
print(classification_report(y_true, y_pred, 
                          target_names=['Low Risk', 'High Risk']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print("                 Predicted")
print("                Low    High")
print(f"Actual Low    {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"Actual High   {cm[1,0]:4d}   {cm[1,1]:4d}")
```

## Step 6: Handling Missing Data
```python
# Create patient with missing data
incomplete_patient = patient.copy()
incomplete_patient['cholesterol_ldl'] = np.nan
incomplete_patient['blood_glucose'] = np.nan
incomplete_patient['chest_pain'] = np.nan

print("\n🔍 Handling Missing Data:")
print(f"Missing features: cholesterol_ldl, blood_glucose, chest_pain")

# Diagnose with missing data
diagnosis_incomplete = diagnosis_system.diagnose(
    incomplete_patient,
    handle_missing='skip'  # Options: 'skip', 'impute', 'flag'
)

print(f"\nRisk Level: {diagnosis_incomplete['risk_level']}")
print(f"Confidence: {diagnosis_incomplete['confidence']:.1%}")
print(f"Data Completeness: {diagnosis_incomplete['completeness']:.1%}")

print(f"\n⚠️ Missing Data Warning:")
print(diagnosis_incomplete['missing_data_warning'])
```

## Step 7: Interpretable Explanations
```python
def explain_diagnosis(patient, diagnosis):
    """
    Generate detailed explanation of diagnosis
    """
    print(f"\n{'='*60}")
    print(f"DETAILED DIAGNOSIS EXPLANATION")
    print(f"Patient: {patient['patient_id']}")
    print(f"{'='*60}")
    
    print(f"\n📊 OVERALL ASSESSMENT:")
    print(f"Risk Level: {diagnosis['risk_level'].upper()}")
    print(f"Confidence: {diagnosis['confidence']:.1%}")
    
    print(f"\n🔬 CHANNEL STATE ANALYSIS:")
    
    # Group by channel state
    from collections import defaultdict
    by_state = defaultdict(list)
    
    for group_name, features in diagnosis['encoded'].items():
        for feature in features:
            state = feature['state']
            by_state[str(state)].append({
                'name': feature['name'],
                'value': feature['value'],
                'group': group_name
            })
    
    # Print by state (most concerning first)
    state_order = [PSI, DELTA, PHI, EMPTY]
    state_interpretation = {
        str(PSI): "⚠️ CONCERNING (Present AND Abnormal)",
        str(DELTA): "⚡ BORDERLINE (Present but marginal)",
        str(PHI): "ℹ️ EXPECTED but absent (protective factors missing)",
        str(EMPTY): "✅ NORMAL (Absent and should be)"
    }
    
    for state in state_order:
        state_str = str(state)
        if state_str in by_state:
            print(f"\n{state_interpretation[state_str]}:")
            for item in by_state[state_str]:
                print(f"  • {item['name']:25s} = {item['value']}")
    
    print(f"\n🎯 KEY RISK FACTORS:")
    for i, rf in enumerate(diagnosis['risk_factors'][:5], 1):
        print(f"  {i}. {rf['factor']}: {rf['severity']}")
        print(f"     State: {rf['state']} | Impact: {rf['impact']}")
    
    print(f"\n💊 RECOMMENDED ACTIONS:")
    for i, rec in enumerate(diagnosis['recommendations'], 1):
        print(f"  {i}. [{rec['priority']}] {rec['action']}")

# Explain a high-risk patient
high_risk_patients = diagnoses_df[diagnoses_df['predicted_risk'] == 'high']
if len(high_risk_patients) > 0:
    patient_idx = high_risk_patients.iloc[0]['patient_id']
    patient = patients_df[patients_df['patient_id'] == patient_idx].iloc[0]
    diagnosis = diagnosis_system.diagnose(patient)
    explain_diagnosis(patient, diagnosis)
```

## Step 8: Visualization
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Risk distribution
ax1 = axes[0, 0]
risk_counts = diagnoses_df['predicted_risk'].value_counts()
ax1.bar(risk_counts.index, risk_counts.values, color=['green', 'red'])
ax1.set_ylabel('Count')
ax1.set_title('Predicted Risk Distribution')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Confidence distribution
ax2 = axes[0, 1]
ax2.hist(diagnoses_df['confidence'], bins=20, edgecolor='black')
ax2.axvline(diagnoses_df['confidence'].mean(), 
           color='red', linestyle='--', label='Mean')
ax2.set_xlabel('Confidence')
ax2.set_ylabel('Count')
ax2.set_title('Diagnostic Confidence Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Risk factors vs confidence
ax3 = axes[1, 0]
scatter = ax3.scatter(
    diagnoses_df['num_risk_factors'],
    diagnoses_df['confidence'],
    c=y_pred,
    cmap='RdYlGn_r',
    alpha=0.6,
    edgecolors='black'
)
ax3.set_xlabel('Number of Risk Factors')
ax3.set_ylabel('Confidence')
ax3.set_title('Risk Factors vs Diagnostic Confidence')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Predicted Risk')

# Plot 4: Feature importance (state distribution)
ax4 = axes[1, 1]
# Get state counts across all patients
feature_states = {}
for idx, patient in patients_df.iterrows():
    encoded = diagnosis_system.encode_patient(patient)
    for group_name, features in encoded.items():
        for feature in features:
            fname = feature['name']
            state = str(feature['state'])
            if fname not in feature_states:
                feature_states[fname] = {str(s): 0 for s in [PSI, DELTA, PHI, EMPTY]}
            feature_states[fname][state] += 1

# Plot top features by PSI state (most concerning)
top_features = sorted(
    feature_states.items(),
    key=lambda x: x[1][str(PSI)],
    reverse=True
)[:8]

feature_names = [f[0][:15] for f in top_features]
psi_counts = [f[1][str(PSI)] for f in top_features]

ax4.barh(feature_names, psi_counts, color='red', alpha=0.7)
ax4.set_xlabel('Count of ψ states (concerning)')
ax4.set_title('Most Frequently Abnormal Features')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('medical_diagnosis_results.png', dpi=150)
plt.show()
```

## Key Insights

### Why Channel Algebra for Medical Diagnosis?

1. **Interpretability is Critical**
   - Every diagnosis must be explainable
   - Channel states show exactly which tests are abnormal
   - No "black box" decisions

2. **Handles Missing Data Gracefully**
   - Missing lab → φ state (expected but absent)
   - Doesn't crash or make assumptions
   - Explicitly notes data limitations

3. **Multi-Modal Evidence**
   - Parallel channels for different test types
   - Each provides independent evidence
   - Confidence based on agreement

4. **Safety Through Transparency**
   - Low confidence → flag for human review
   - Borderline cases (δ states) handled carefully
   - Always provides reasoning

### Channel State Meanings in Medicine

**ψ (PSI) - Concerning**
```
Test is present AND abnormal
Example: High cholesterol measured AND above threshold
→ Definite risk factor
```

**δ (DELTA) - Borderline**
```
Test present but borderline abnormal
Example: Slightly elevated blood pressure
→ Monitor closely, may warrant intervention
```

**φ (PHI) - Missing Protection**
```
Expected protective factor is absent
Example: HDL cholesterol below protective threshold
→ Lack of protection, different from active risk
```

**∅ (EMPTY) - Normal**
```
Test absent or normal
Example: No chest pain, normal blood pressure
→ Good sign
```

## Production Considerations

### 1. Clinical Validation
```python
# Validate against clinical guidelines
def validate_against_guidelines(diagnosis, guidelines):
    """
    Ensure diagnosis aligns with clinical guidelines
    """
    for guideline in guidelines:
        if not guideline.check(diagnosis):
            diagnosis['warnings'].append(guideline.warning)
    return diagnosis
```

### 2. Audit Trail
```python
# Log all diagnoses for review
import json
from datetime import datetime

def log_diagnosis(patient, diagnosis):
    """
    Create audit trail
    """
    audit_entry = {
        'timestamp': datetime.now().isoformat(),
        'patient_id': patient['patient_id'],
        'diagnosis': diagnosis,
        'system_version': '1.0.0',
        'encoded_states': diagnosis['encoded']
    }
    
    with open('diagnosis_audit.jsonl', 'a') as f:
        f.write(json.dumps(audit_entry) + '\n')
```

### 3. Confidence Thresholds
```python
# Only act on high-confidence diagnoses
CONFIDENCE_THRESHOLD = 0.75

if diagnosis['confidence'] < CONFIDENCE_THRESHOLD:
    print("⚠️ LOW CONFIDENCE - Flag for physician review")
    diagnosis['requires_human_review'] = True
```

### 4. Differential Diagnosis
```python
# Consider multiple conditions
def differential_diagnosis(patient):
    """
    Evaluate multiple possible conditions
    """
    conditions = [
        'cardiovascular_risk',
        'metabolic_syndrome',
        'hypertension'
    ]
    
    diagnoses = {}
    for condition in conditions:
        system = MedicalDiagnosisSystem(condition=condition)
        diagnoses[condition] = system.diagnose(patient)
    
    return diagnoses
```

## Ethical Considerations

⚠️ **Critical Reminders**:

1. **NOT A REPLACEMENT**: This is a diagnostic *support* tool, not a replacement for physicians
2. **ALWAYS VERIFY**: All diagnoses must be reviewed by qualified healthcare professionals
3. **EXPLAIN LIMITATIONS**: Clearly communicate what the system can and cannot do
4. **PATIENT PRIVACY**: Handle all medical data according to HIPAA/GDPR regulations
5. **BIAS AWARENESS**: Validate system across diverse patient populations
6. **CONTINUOUS VALIDATION**: Regularly audit system performance against clinical outcomes

## Next Steps

- **How-To Guide**: Create custom encoder for your specific medical domain
- **How-To Guide**: Handle missing medical data appropriately
- **API Reference**: Deep dive into MedicalDiagnosisSystem class

## Complete Code

The complete, runnable code is available in `examples/medical_diagnosis.py`.

---

**💡 Key Takeaway**: Channel algebra provides the interpretability and transparency required for medical decision support. Every diagnosis can be traced back to specific test results and clinical states.