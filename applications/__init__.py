"""
Applications module for channel algebra.

Contains domain-specific applications and real-world use cases.
"""

from .trading import (
    TechnicalIndicators, TradingSignalEncoder, TradingStrategy,
    SimpleChannelStrategy, AdaptiveMomentumStrategy, TradingChannelSystem,
    LiveTradingSystem
)
from .medical import (
    MedicalDiagnosisSystem, PatientMonitoringSystem, create_sample_patient_data,
    demonstrate_medical_system
)

__all__ = [
    'TechnicalIndicators', 'TradingSignalEncoder', 'TradingStrategy',
    'SimpleChannelStrategy', 'AdaptiveMomentumStrategy', 'TradingChannelSystem',
    'LiveTradingSystem',
    'MedicalDiagnosisSystem', 'PatientMonitoringSystem', 'create_sample_patient_data',
    'demonstrate_medical_system',
]
