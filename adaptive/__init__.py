"""
Adaptive module for channel algebra.

Contains adaptive thresholding, streaming capabilities, and scoring systems.
"""

from .streaming import StreamingAdaptiveThreshold
from .thresholds import (
    BaseThresholdLearner, StatisticalThresholds, SupervisedThresholds,
    DomainThresholds, OptimalThresholds, MultiFeatureThresholdLearner,
    ThresholdStabilityAnalyzer
)
from .scoring import (
    FeatureScorer, ScoreDimension, relevance_scorer, confidence_scorer,
    freshness_scorer, stability_scorer, density_scorer,
    create_trading_scorer, create_medical_scorer, create_signal_scorer
)
from .topology_adaptive import (
    TopologyFeatures, TopologyAnalyzer, TopologyAdaptiveThreshold
)
from .multiscale import (
    RegimeType, RegimeChange, MultiScaleAdaptiveThreshold
)

__all__ = [
    'StreamingAdaptiveThreshold',
    'BaseThresholdLearner', 'StatisticalThresholds', 'SupervisedThresholds',
    'DomainThresholds', 'OptimalThresholds', 'MultiFeatureThresholdLearner',
    'ThresholdStabilityAnalyzer',
    'FeatureScorer', 'ScoreDimension', 'relevance_scorer', 'confidence_scorer',
    'freshness_scorer', 'stability_scorer', 'density_scorer',
    'create_trading_scorer', 'create_medical_scorer', 'create_signal_scorer',
    'TopologyFeatures', 'TopologyAnalyzer', 'TopologyAdaptiveThreshold',
    'RegimeType', 'RegimeChange', 'MultiScaleAdaptiveThreshold',
]
