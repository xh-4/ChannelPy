"""
ChannelPy: Channel Algebra for Structured Data Analysis

A production-ready Python library implementing channel algebra concepts
for structured data analysis, adaptive thresholding, and interpretable AI.
"""

__version__ = "0.1.0"
__author__ = "Channel Algebra Team"

# Core exports
from .core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from .core.operations import (
    gate, admit, overlay, weave, comp, neg_i, neg_q,
    compose, pipe
)
from .core.lattice import (
    partial_order, are_comparable, meet, join, lattice_distance,
    complement, is_atom, is_coatom, ChannelLattice, get_lattice,
    lattice_operations
)

# Pipeline exports
from .pipeline.base import BasePipeline, ChannelPipeline
from .pipeline.preprocessors import (
    StandardScaler, RobustScaler, MissingDataHandler, OutlierDetector,
    TimeSeriesFeatureExtractor, StatisticalFeatureExtractor, normalize, standardize
)
from .pipeline.interpreters import (
    RuleBasedInterpreter, LookupTableInterpreter, FSMInterpreter,
    PatternMatcher, ThresholdBasedInterpreter, quick_interpret
)

# Adaptive exports
from .adaptive.streaming import StreamingAdaptiveThreshold
from .adaptive.thresholds import (
    BaseThresholdLearner, StatisticalThresholds, SupervisedThresholds,
    DomainThresholds, OptimalThresholds, MultiFeatureThresholdLearner,
    ThresholdStabilityAnalyzer
)
from .adaptive.scoring import (
    FeatureScorer, ScoreDimension, relevance_scorer, confidence_scorer,
    freshness_scorer, stability_scorer, density_scorer,
    create_trading_scorer, create_medical_scorer, create_signal_scorer
)
from .adaptive.topology_adaptive import (
    TopologyFeatures, TopologyAnalyzer, TopologyAdaptiveThreshold
)
from .adaptive.multiscale import (
    RegimeType, RegimeChange, MultiScaleAdaptiveThreshold
)

# Visualization exports
from .visualization.states import plot_states, plot_state_distribution

# Examples exports
from .examples.datasets import (
    make_classification_data, make_time_series_data, make_regime_change_data,
    make_trading_data, make_medical_data, make_state_sequence,
    load_example_dataset, generate_streaming_data
)

# Applications exports
from .applications.trading import (
    TechnicalIndicators, TradingSignalEncoder, TradingStrategy,
    SimpleChannelStrategy, AdaptiveMomentumStrategy, TradingChannelSystem,
    LiveTradingSystem
)
from .applications.medical import (
    MedicalDiagnosisSystem, PatientMonitoringSystem, create_sample_patient_data,
    demonstrate_medical_system
)

# Topology exports
from .topology.persistence import (
    PersistenceDiagram, compute_betti_numbers, compute_persistence_entropy,
    compute_total_persistence, compare_persistence, plot_comparison
)

# Metrics exports
from .metrics.quality import (
    encoding_accuracy, state_distribution_quality, discrimination_power,
    threshold_stability, information_content, encoding_consistency,
    state_transition_quality, channel_correlation, encoding_robustness,
    comprehensive_quality_report, compare_encodings
)

__all__ = [
    # Core
    'State', 'StateArray', 'EMPTY', 'DELTA', 'PHI', 'PSI',
    'gate', 'admit', 'overlay', 'weave', 'comp', 'neg_i', 'neg_q',
    'compose', 'pipe',
    'partial_order', 'are_comparable', 'meet', 'join', 'lattice_distance',
    'complement', 'is_atom', 'is_coatom', 'ChannelLattice', 'get_lattice',
    'lattice_operations',
    
    # Pipeline
    'BasePipeline', 'ChannelPipeline',
    'StandardScaler', 'RobustScaler', 'MissingDataHandler', 'OutlierDetector',
    'TimeSeriesFeatureExtractor', 'StatisticalFeatureExtractor', 'normalize', 'standardize',
    'RuleBasedInterpreter', 'LookupTableInterpreter', 'FSMInterpreter',
    'PatternMatcher', 'ThresholdBasedInterpreter', 'quick_interpret',
    
    # Adaptive
    'StreamingAdaptiveThreshold',
    'BaseThresholdLearner', 'StatisticalThresholds', 'SupervisedThresholds',
    'DomainThresholds', 'OptimalThresholds', 'MultiFeatureThresholdLearner',
    'ThresholdStabilityAnalyzer',
    'FeatureScorer', 'ScoreDimension', 'relevance_scorer', 'confidence_scorer',
    'freshness_scorer', 'stability_scorer', 'density_scorer',
    'create_trading_scorer', 'create_medical_scorer', 'create_signal_scorer',
    'TopologyFeatures', 'TopologyAnalyzer', 'TopologyAdaptiveThreshold',
    'RegimeType', 'RegimeChange', 'MultiScaleAdaptiveThreshold',
    
    # Visualization
    'plot_states', 'plot_state_distribution',
    
    # Examples
    'make_classification_data', 'make_time_series_data', 'make_regime_change_data',
    'make_trading_data', 'make_medical_data', 'make_state_sequence',
    'load_example_dataset', 'generate_streaming_data',
    
    # Applications
    'TechnicalIndicators', 'TradingSignalEncoder', 'TradingStrategy',
    'SimpleChannelStrategy', 'AdaptiveMomentumStrategy', 'TradingChannelSystem',
    'LiveTradingSystem',
    'MedicalDiagnosisSystem', 'PatientMonitoringSystem', 'create_sample_patient_data',
    'demonstrate_medical_system',
    
    # Topology
    'PersistenceDiagram', 'compute_betti_numbers', 'compute_persistence_entropy',
    'compute_total_persistence', 'compare_persistence', 'plot_comparison',
    
    # Metrics
    'encoding_accuracy', 'state_distribution_quality', 'discrimination_power',
    'threshold_stability', 'information_content', 'encoding_consistency',
    'state_transition_quality', 'channel_correlation', 'encoding_robustness',
    'comprehensive_quality_report', 'compare_encodings',
]
