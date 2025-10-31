"""
Metrics for evaluating ChannelPy systems

This module provides three categories of metrics:

1. Quality Metrics (quality.py):
   - Encoding quality and information content
   - Discrimination power
   - State distribution balance
   
2. Explainability Metrics (explainability.py):
   - Decision interpretability
   - Rule complexity
   - Explanation completeness
   
3. Stability Metrics (stability.py):
   - Threshold stability over time
   - Distribution drift detection
   - Regime consistency

Examples
--------
>>> from channelpy.metrics import (
...     encoding_quality,
...     decision_clarity,
...     threshold_stability
... )
>>> 
>>> # Evaluate encoding quality
>>> quality = encoding_quality(states, labels)
>>> print(f"Accuracy: {quality['accuracy']:.3f}")
>>> 
>>> # Measure interpretability
>>> clarity = decision_clarity(interpreter)
>>> print(f"Rule complexity: {clarity['complexity']:.1f}")
>>> 
>>> # Check threshold stability
>>> stability = threshold_stability(threshold_history)
>>> print(f"Drift rate: {stability['drift_rate']:.4f}")
"""

from .quality import (
    encoding_accuracy,
    state_distribution_balance,
    discrimination_power,
    information_content,
    encoding_quality_report,
    compare_encoders
)

from .explainability import (
    decision_clarity,
    rule_complexity,
    explanation_completeness,
    interpretation_consistency,
    explainability_score,
    generate_explanation
)

from .stability import (
    threshold_stability,
    drift_detection,
    regime_consistency,
    adaptation_quality,
    stability_report
)

__all__ = [
    # Quality metrics
    'encoding_accuracy',
    'state_distribution_balance',
    'discrimination_power',
    'information_content',
    'encoding_quality_report',
    'compare_encoders',
    
    # Explainability metrics
    'decision_clarity',
    'rule_complexity',
    'explanation_completeness',
    'interpretation_consistency',
    'explainability_score',
    'generate_explanation',
    
    # Stability metrics
    'threshold_stability',
    'drift_detection',
    'regime_consistency',
    'adaptation_quality',
    'stability_report'
]