"""
Medical diagnosis application

Complete medical diagnosis system using channel algebra for patient assessment
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..core.state import State, EMPTY, DELTA, PHI, PSI
from ..core.parallel import ParallelChannels
from ..pipeline.base import ChannelPipeline
from ..pipeline.interpreters import RuleBasedInterpreter, FSMInterpreter
from ..adaptive.thresholds import DomainThresholds, SupervisedThresholds


class MedicalDiagnosisSystem:
    """
    Medical diagnosis using channel algebra
    
    Uses multiple channels to assess patient condition:
    - Symptoms: Patient-reported symptoms
    - Tests: Laboratory test results  
    - Vitals: Vital signs
    - History: Medical history factors
    
    Examples
    --------
    >>> system = MedicalDiagnosisSystem()
    >>> diagnosis = system.diagnose(patient_data)
    >>> print(diagnosis['recommendation'])
    """
    
    def __init__(self):
        self.pipeline = ChannelPipeline()
        self.symptom_thresholds = {}
        self.test_thresholds = {}
        self.vital_thresholds = {}
        self.history_thresholds = {}
        self.is_fitted = False
        
        # Initialize domain-specific thresholds
        self._setup_medical_thresholds()
    
    def _setup_medical_thresholds(self):
        """Setup medical domain thresholds"""
        # Symptom severity thresholds
        self.symptom_thresholds = {
            'mild': 0.3,
            'severe': 0.7
        }
        
        # Test result thresholds
        self.test_thresholds = {
            'abnormal': 0.3,
            'critical': 0.6
        }
        
        # Vital sign thresholds
        self.vital_thresholds = {
            'temperature': {'normal': 37.0, 'fever': 37.5, 'high_fever': 38.5},
            'heart_rate': {'normal': 70, 'tachycardia': 100, 'severe_tachycardia': 120},
            'systolic_bp': {'normal': 120, 'hypertension': 140, 'severe_hypertension': 160},
            'diastolic_bp': {'normal': 80, 'hypertension': 90, 'severe_hypertension': 100},
            'oxygen_saturation': {'normal': 95, 'low': 90, 'critical': 85}
        }
        
        # Medical history thresholds
        self.history_thresholds = {
            'risk_factors': 0.5,
            'high_risk': 0.8
        }
    
    def fit(self, patient_records: pd.DataFrame, diagnoses: np.ndarray):
        """
        Train on historical patient data
        
        Parameters
        ----------
        patient_records : pd.DataFrame
            Historical patient data with columns for symptoms, tests, vitals
        diagnoses : np.ndarray
            Corresponding diagnoses (0=healthy, 1=disease)
        """
        # Learn thresholds from data
        self._learn_symptom_thresholds(patient_records, diagnoses)
        self._learn_test_thresholds(patient_records, diagnoses)
        self._learn_vital_thresholds(patient_records, diagnoses)
        self._learn_history_thresholds(patient_records, diagnoses)
        
        self.is_fitted = True
        return self
    
    def _learn_symptom_thresholds(self, records: pd.DataFrame, diagnoses: np.ndarray):
        """Learn symptom thresholds from data"""
        if 'symptom_score' in records.columns:
            # Use supervised learning for symptom thresholds
            learner = SupervisedThresholds(metric='mutual_info')
            learner.fit(records['symptom_score'].values, diagnoses)
            thresh_i, thresh_q = learner.get_thresholds()
            
            self.symptom_thresholds['mild'] = thresh_i
            self.symptom_thresholds['severe'] = thresh_q
    
    def _learn_test_thresholds(self, records: pd.DataFrame, diagnoses: np.ndarray):
        """Learn test result thresholds from data"""
        test_columns = [col for col in records.columns if col.startswith('test_')]
        
        for test_col in test_columns:
            if test_col in records.columns:
                learner = SupervisedThresholds(metric='mutual_info')
                learner.fit(records[test_col].values, diagnoses)
                thresh_i, thresh_q = learner.get_thresholds()
                
                self.test_thresholds[test_col] = {
                    'abnormal': thresh_i,
                    'critical': thresh_q
                }
    
    def _learn_vital_thresholds(self, records: pd.DataFrame, diagnoses: np.ndarray):
        """Learn vital sign thresholds from data"""
        vital_columns = ['temperature', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'oxygen_saturation']
        
        for vital in vital_columns:
            if vital in records.columns:
                learner = SupervisedThresholds(metric='mutual_info')
                learner.fit(records[vital].values, diagnoses)
                thresh_i, thresh_q = learner.get_thresholds()
                
                self.vital_thresholds[vital] = {
                    'normal': thresh_i,
                    'abnormal': thresh_q
                }
    
    def _learn_history_thresholds(self, records: pd.DataFrame, diagnoses: np.ndarray):
        """Learn medical history thresholds from data"""
        if 'risk_score' in records.columns:
            learner = SupervisedThresholds(metric='mutual_info')
            learner.fit(records['risk_score'].values, diagnoses)
            thresh_i, thresh_q = learner.get_thresholds()
            
            self.history_thresholds['risk_factors'] = thresh_i
            self.history_thresholds['high_risk'] = thresh_q
    
    def encode_symptoms(self, symptoms: Dict) -> State:
        """
        Encode symptoms channel
        
        i-bit: Symptoms present
        q-bit: Symptoms severe
        
        Parameters
        ----------
        symptoms : dict
            Dictionary of symptom_name -> severity (0-1)
            
        Returns
        -------
        state : State
            Encoded symptom state
        """
        if not symptoms:
            return EMPTY
        
        # Calculate overall symptom score
        symptom_scores = list(symptoms.values())
        symptom_score = np.mean(symptom_scores)
        
        # Apply thresholds
        mild_threshold = self.symptom_thresholds.get('mild', 0.3)
        severe_threshold = self.symptom_thresholds.get('severe', 0.7)
        
        return State(
            i=int(symptom_score > mild_threshold),
            q=int(symptom_score > severe_threshold)
        )
    
    def encode_tests(self, test_results: Dict) -> State:
        """
        Encode test results channel
        
        i-bit: Tests abnormal
        q-bit: Tests critically abnormal
        
        Parameters
        ----------
        test_results : dict
            Dictionary of test_name -> result_value
            
        Returns
        -------
        state : State
            Encoded test state
        """
        if not test_results:
            return EMPTY
        
        # Count abnormal tests
        abnormal_count = 0
        critical_count = 0
        
        for test_name, result in test_results.items():
            if test_name in self.test_thresholds:
                thresholds = self.test_thresholds[test_name]
                if result > thresholds.get('abnormal', 0.5):
                    abnormal_count += 1
                if result > thresholds.get('critical', 0.8):
                    critical_count += 1
        
        abnormal_ratio = abnormal_count / len(test_results)
        critical_ratio = critical_count / len(test_results)
        
        return State(
            i=int(abnormal_ratio > 0.3),
            q=int(critical_ratio > 0.2)
        )
    
    def encode_vitals(self, vitals: Dict) -> State:
        """
        Encode vital signs channel
        
        i-bit: Any vital abnormal
        q-bit: Multiple vitals abnormal
        
        Parameters
        ----------
        vitals : dict
            Dictionary of vital_name -> value
            
        Returns
        -------
        state : State
            Encoded vital state
        """
        if not vitals:
            return EMPTY
        
        abnormal_count = 0
        
        # Check each vital against normal ranges
        for vital_name, value in vitals.items():
            if vital_name in self.vital_thresholds:
                thresholds = self.vital_thresholds[vital_name]
                normal_value = thresholds.get('normal', 0)
                
                # Check if abnormal (simplified logic)
                if abs(value - normal_value) > normal_value * 0.1:  # 10% deviation
                    abnormal_count += 1
        
        return State(
            i=int(abnormal_count > 0),
            q=int(abnormal_count > 1)
        )
    
    def encode_history(self, history: Dict) -> State:
        """
        Encode medical history channel
        
        i-bit: Risk factors present
        q-bit: High risk factors present
        
        Parameters
        ----------
        history : dict
            Dictionary of history factors
            
        Returns
        -------
        state : State
            Encoded history state
        """
        if not history:
            return EMPTY
        
        # Calculate risk score
        risk_factors = sum(history.values())
        risk_score = risk_factors / len(history)
        
        risk_threshold = self.history_thresholds.get('risk_factors', 0.5)
        high_risk_threshold = self.history_thresholds.get('high_risk', 0.8)
        
        return State(
            i=int(risk_score > risk_threshold),
            q=int(risk_score > high_risk_threshold)
        )
    
    def diagnose(self, patient_data: Dict) -> Dict:
        """
        Generate diagnosis from patient data
        
        Parameters
        ----------
        patient_data : dict
            Patient data with keys: symptoms, tests, vitals, history
            
        Returns
        -------
        diagnosis : dict
            Diagnosis with urgency, recommendation, and explanation
        """
        # Encode all channels
        channels = ParallelChannels(
            symptoms=self.encode_symptoms(patient_data.get('symptoms', {})),
            tests=self.encode_tests(patient_data.get('tests', {})),
            vitals=self.encode_vitals(patient_data.get('vitals', {})),
            history=self.encode_history(patient_data.get('history', {}))
        )
        
        # Interpret channels
        return self._interpret_channels(channels)
    
    def _interpret_channels(self, channels: ParallelChannels) -> Dict:
        """Interpret channel states to diagnosis"""
        symptoms = channels['symptoms']
        tests = channels['tests']
        vitals = channels['vitals']
        history = channels['history']
        
        # Count severe channels
        severe_channels = sum(s == PSI for s in [symptoms, tests, vitals, history])
        abnormal_channels = sum(s in [PSI, DELTA] for s in [symptoms, tests, vitals, history])
        
        # Critical: Multiple severe channels
        if severe_channels >= 2:
            return {
                'urgency': 'CRITICAL',
                'recommendation': 'IMMEDIATE_TREATMENT',
                'confidence': 0.95,
                'explanation': f'Multiple severe indicators: {severe_channels} channels at PSI level',
                'channels': {
                    'symptoms': str(symptoms),
                    'tests': str(tests),
                    'vitals': str(vitals),
                    'history': str(history)
                }
            }
        
        # High priority: Tests + symptoms severe
        if tests == PSI and symptoms in [PSI, DELTA]:
            return {
                'urgency': 'HIGH',
                'recommendation': 'TREAT_AND_MONITOR',
                'confidence': 0.85,
                'explanation': 'Severe test results with significant symptoms',
                'channels': {
                    'symptoms': str(symptoms),
                    'tests': str(tests),
                    'vitals': str(vitals),
                    'history': str(history)
                }
            }
        
        # High priority: Vitals + history severe
        if vitals == PSI and history == PSI:
            return {
                'urgency': 'HIGH',
                'recommendation': 'IMMEDIATE_ASSESSMENT',
                'confidence': 0.80,
                'explanation': 'Severe vital signs with high-risk history',
                'channels': {
                    'symptoms': str(symptoms),
                    'tests': str(tests),
                    'vitals': str(vitals),
                    'history': str(history)
                }
            }
        
        # Medium priority: Some abnormalities
        if abnormal_channels >= 2:
            return {
                'urgency': 'MEDIUM',
                'recommendation': 'MONITOR_CLOSELY',
                'confidence': 0.70,
                'explanation': f'Multiple abnormal indicators: {abnormal_channels} channels',
                'channels': {
                    'symptoms': str(symptoms),
                    'tests': str(tests),
                    'vitals': str(vitals),
                    'history': str(history)
                }
            }
        
        # Low priority: Single abnormality
        if abnormal_channels == 1:
            return {
                'urgency': 'LOW',
                'recommendation': 'ROUTINE_MONITORING',
                'confidence': 0.60,
                'explanation': 'Single abnormal indicator detected',
                'channels': {
                    'symptoms': str(symptoms),
                    'tests': str(tests),
                    'vitals': str(vitals),
                    'history': str(history)
                }
            }
        
        # Healthy
        return {
            'urgency': 'LOW',
            'recommendation': 'ROUTINE_CHECKUP',
            'confidence': 0.90,
            'explanation': 'All indicators within normal ranges',
            'channels': {
                'symptoms': str(symptoms),
                'tests': str(tests),
                'vitals': str(vitals),
                'history': str(history)
            }
        }
    
    def batch_diagnose(self, patient_data_list: List[Dict]) -> List[Dict]:
        """
        Diagnose multiple patients
        
        Parameters
        ----------
        patient_data_list : List[Dict]
            List of patient data dictionaries
            
        Returns
        -------
        diagnoses : List[Dict]
            List of diagnosis dictionaries
        """
        return [self.diagnose(patient_data) for patient_data in patient_data_list]
    
    def get_diagnosis_summary(self, diagnoses: List[Dict]) -> Dict:
        """
        Get summary statistics from batch diagnoses
        
        Parameters
        ----------
        diagnoses : List[Dict]
            List of diagnosis dictionaries
            
        Returns
        -------
        summary : dict
            Summary statistics
        """
        if not diagnoses:
            return {}
        
        # Count by urgency
        urgency_counts = {}
        for diagnosis in diagnoses:
            urgency = diagnosis.get('urgency', 'UNKNOWN')
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
        
        # Count by recommendation
        recommendation_counts = {}
        for diagnosis in diagnoses:
            rec = diagnosis.get('recommendation', 'UNKNOWN')
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Average confidence
        confidences = [d.get('confidence', 0) for d in diagnoses]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'total_patients': len(diagnoses),
            'urgency_distribution': urgency_counts,
            'recommendation_distribution': recommendation_counts,
            'average_confidence': avg_confidence,
            'critical_cases': urgency_counts.get('CRITICAL', 0),
            'high_priority_cases': urgency_counts.get('HIGH', 0)
        }


class PatientMonitoringSystem:
    """
    Real-time patient monitoring using channel algebra
    
    Monitors patient state changes over time and alerts on deterioration
    """
    
    def __init__(self):
        self.diagnosis_system = MedicalDiagnosisSystem()
        self.patient_history = {}  # patient_id -> list of states
        self.alert_thresholds = {
            'deterioration': 0.3,  # 30% confidence drop
            'improvement': 0.2,    # 20% confidence increase
            'critical_change': 0.5  # 50% confidence change
        }
    
    def add_patient(self, patient_id: str, initial_data: Dict):
        """Add new patient to monitoring"""
        diagnosis = self.diagnosis_system.diagnose(initial_data)
        self.patient_history[patient_id] = [diagnosis]
        return diagnosis
    
    def update_patient(self, patient_id: str, new_data: Dict) -> Dict:
        """
        Update patient with new data and check for changes
        
        Parameters
        ----------
        patient_id : str
            Patient identifier
        new_data : dict
            New patient data
            
        Returns
        -------
        update : dict
            Update with diagnosis, changes, and alerts
        """
        if patient_id not in self.patient_history:
            return self.add_patient(patient_id, new_data)
        
        # Get new diagnosis
        new_diagnosis = self.diagnosis_system.diagnose(new_data)
        
        # Get previous diagnosis
        previous_diagnosis = self.patient_history[patient_id][-1]
        
        # Check for changes
        changes = self._detect_changes(previous_diagnosis, new_diagnosis)
        
        # Add to history
        self.patient_history[patient_id].append(new_diagnosis)
        
        # Generate alerts
        alerts = self._generate_alerts(changes, new_diagnosis)
        
        return {
            'patient_id': patient_id,
            'diagnosis': new_diagnosis,
            'changes': changes,
            'alerts': alerts,
            'history_length': len(self.patient_history[patient_id])
        }
    
    def _detect_changes(self, previous: Dict, current: Dict) -> Dict:
        """Detect changes between diagnoses"""
        changes = {
            'urgency_changed': previous.get('urgency') != current.get('urgency'),
            'recommendation_changed': previous.get('recommendation') != current.get('recommendation'),
            'confidence_change': current.get('confidence', 0) - previous.get('confidence', 0),
            'channel_changes': {}
        }
        
        # Check channel changes
        prev_channels = previous.get('channels', {})
        curr_channels = current.get('channels', {})
        
        for channel_name in prev_channels:
            if channel_name in curr_channels:
                if prev_channels[channel_name] != curr_channels[channel_name]:
                    changes['channel_changes'][channel_name] = {
                        'from': prev_channels[channel_name],
                        'to': curr_channels[channel_name]
                    }
        
        return changes
    
    def _generate_alerts(self, changes: Dict, diagnosis: Dict) -> List[Dict]:
        """Generate alerts based on changes"""
        alerts = []
        
        # Urgency escalation alert
        if changes.get('urgency_changed'):
            alerts.append({
                'type': 'URGENCY_CHANGE',
                'message': f"Urgency changed to {diagnosis.get('urgency')}",
                'severity': 'HIGH' if diagnosis.get('urgency') in ['CRITICAL', 'HIGH'] else 'MEDIUM'
            })
        
        # Confidence drop alert
        confidence_change = changes.get('confidence_change', 0)
        if confidence_change < -self.alert_thresholds['deterioration']:
            alerts.append({
                'type': 'DETERIORATION',
                'message': f"Patient condition deteriorating (confidence drop: {confidence_change:.2f})",
                'severity': 'HIGH'
            })
        
        # Critical change alert
        if abs(confidence_change) > self.alert_thresholds['critical_change']:
            alerts.append({
                'type': 'CRITICAL_CHANGE',
                'message': f"Critical change in patient condition (confidence change: {confidence_change:.2f})",
                'severity': 'CRITICAL'
            })
        
        # Channel change alerts
        for channel_name, channel_change in changes.get('channel_changes', {}).items():
            alerts.append({
                'type': 'CHANNEL_CHANGE',
                'message': f"{channel_name} channel changed from {channel_change['from']} to {channel_change['to']}",
                'severity': 'MEDIUM'
            })
        
        return alerts
    
    def get_patient_summary(self, patient_id: str) -> Dict:
        """Get summary of patient monitoring history"""
        if patient_id not in self.patient_history:
            return {}
        
        history = self.patient_history[patient_id]
        
        # Analyze trends
        urgencies = [d.get('urgency') for d in history]
        confidences = [d.get('confidence', 0) for d in history]
        
        return {
            'patient_id': patient_id,
            'total_updates': len(history),
            'current_urgency': urgencies[-1] if urgencies else 'UNKNOWN',
            'current_confidence': confidences[-1] if confidences else 0,
            'urgency_trend': self._analyze_trend(urgencies),
            'confidence_trend': self._analyze_trend(confidences),
            'history': history
        }
    
    def _analyze_trend(self, values: List) -> str:
        """Analyze trend in values"""
        if len(values) < 2:
            return 'INSUFFICIENT_DATA'
        
        # Simple trend analysis
        if values[-1] > values[0]:
            return 'IMPROVING'
        elif values[-1] < values[0]:
            return 'DETERIORATING'
        else:
            return 'STABLE'


def create_sample_patient_data() -> Dict:
    """Create sample patient data for testing"""
    return {
        'symptoms': {
            'fever': 0.8,
            'cough': 0.6,
            'fatigue': 0.7,
            'headache': 0.4
        },
        'tests': {
            'white_blood_cells': 0.9,  # High
            'c_reactive_protein': 0.8,  # High
            'blood_sugar': 0.3  # Normal
        },
        'vitals': {
            'temperature': 38.2,  # Fever
            'heart_rate': 95,  # Slightly elevated
            'systolic_bp': 130,  # Normal
            'diastolic_bp': 85,  # Normal
            'oxygen_saturation': 97  # Normal
        },
        'history': {
            'diabetes': 0.0,
            'hypertension': 0.0,
            'smoking': 0.0,
            'age_risk': 0.3  # Moderate age risk
        }
    }


def demonstrate_medical_system():
    """Demonstrate the medical diagnosis system"""
    print("=== Medical Diagnosis System Demo ===")
    
    # Create system
    system = MedicalDiagnosisSystem()
    
    # Create sample patient
    patient_data = create_sample_patient_data()
    print("Sample patient data:")
    for category, data in patient_data.items():
        print(f"  {category}: {data}")
    
    # Diagnose patient
    diagnosis = system.diagnose(patient_data)
    print(f"\nDiagnosis:")
    print(f"  Urgency: {diagnosis['urgency']}")
    print(f"  Recommendation: {diagnosis['recommendation']}")
    print(f"  Confidence: {diagnosis['confidence']:.2f}")
    print(f"  Explanation: {diagnosis['explanation']}")
    print(f"  Channels: {diagnosis['channels']}")
    
    return system, diagnosis


if __name__ == "__main__":
    demonstrate_medical_system()




