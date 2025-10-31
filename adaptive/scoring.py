"""
Multi-dimensional feature scoring system

This module provides sophisticated feature scoring across multiple dimensions,
enabling context-aware threshold adaptation.
"""

from typing import Dict, List, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ScoreDimension:
    """
    A single dimension for scoring features
    
    Attributes
    ----------
    name : str
        Dimension name (e.g., 'relevance', 'confidence')
    scorer : Callable
        Function that computes score for this dimension
    weight : float
        Weight for aggregation (default 1.0)
    description : str
        Human-readable description
    """
    name: str
    scorer: Callable[[float, Dict[str, Any]], float]
    weight: float = 1.0
    description: str = ""
    
    def score(self, value: float, context: Dict[str, Any]) -> float:
        """Compute score for this dimension"""
        return self.scorer(value, context)


class FeatureScorer:
    """
    Score features across multiple dimensions
    
    Aggregates evidence from different perspectives to determine
    optimal thresholds and encoding strategies.
    
    Examples
    --------
    >>> scorer = FeatureScorer()
    >>> scorer.add_dimension('relevance', relevance_scorer, weight=2.0)
    >>> scorer.add_dimension('confidence', confidence_scorer, weight=1.5)
    >>> scorer.add_dimension('freshness', freshness_scorer, weight=1.0)
    >>> 
    >>> scores = scorer.score_feature(0.75, context)
    >>> aggregate = scorer.aggregate_scores(scores)
    >>> print(f"Overall score: {aggregate:.3f}")
    """
    
    def __init__(self):
        self.dimensions: Dict[str, ScoreDimension] = {}
        self.history: List[Dict[str, Any]] = []
    
    def add_dimension(
        self, 
        name: str, 
        scorer: Callable[[float, Dict[str, Any]], float],
        weight: float = 1.0,
        description: str = ""
    ):
        """
        Add a scoring dimension
        
        Parameters
        ----------
        name : str
            Dimension name
        scorer : Callable
            Function(value, context) -> score
        weight : float
            Dimension weight for aggregation
        description : str
            Human-readable description
        """
        self.dimensions[name] = ScoreDimension(
            name=name,
            scorer=scorer,
            weight=weight,
            description=description
        )
    
    def score_feature(
        self, 
        value: float, 
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Score feature across all dimensions
        
        Parameters
        ----------
        value : float
            Feature value to score
        context : Dict
            Context dictionary with additional information
            
        Returns
        -------
        scores : Dict[str, float]
            Score for each dimension
        """
        scores = {}
        for name, dimension in self.dimensions.items():
            try:
                score = dimension.score(value, context)
                scores[name] = score
            except Exception as e:
                # Graceful degradation
                scores[name] = 0.5  # Neutral score
                print(f"Warning: Scoring failed for dimension '{name}': {e}")
        
        return scores
    
    def aggregate_scores(
        self, 
        dimension_scores: Dict[str, float],
        method: str = 'weighted_average'
    ) -> float:
        """
        Aggregate dimension scores to overall score
        
        Parameters
        ----------
        dimension_scores : Dict[str, float]
            Scores for each dimension
        method : str
            Aggregation method:
            - 'weighted_average': Weighted average (default)
            - 'min': Minimum score (conservative)
            - 'max': Maximum score (optimistic)
            - 'product': Product of scores
            
        Returns
        -------
        score : float
            Aggregated score in [0, 1]
        """
        if not dimension_scores:
            return 0.5  # Neutral
        
        if method == 'weighted_average':
            weighted_sum = 0.0
            total_weight = 0.0
            
            for name, score in dimension_scores.items():
                if name in self.dimensions:
                    weight = self.dimensions[name].weight
                    weighted_sum += weight * score
                    total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.5
        
        elif method == 'min':
            return min(dimension_scores.values())
        
        elif method == 'max':
            return max(dimension_scores.values())
        
        elif method == 'product':
            result = 1.0
            for score in dimension_scores.values():
                result *= score
            return result
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def score_and_aggregate(
        self, 
        value: float, 
        context: Dict[str, Any],
        method: str = 'weighted_average'
    ) -> tuple[float, Dict[str, float]]:
        """
        Convenience method: score and aggregate in one call
        
        Returns
        -------
        aggregate : float
            Overall aggregated score
        dimension_scores : Dict[str, float]
            Individual dimension scores
        """
        dimension_scores = self.score_feature(value, context)
        aggregate = self.aggregate_scores(dimension_scores, method)
        return aggregate, dimension_scores
    
    def explain_score(
        self, 
        value: float, 
        context: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation of score
        
        Parameters
        ----------
        value : float
            Feature value
        context : Dict
            Context dictionary
            
        Returns
        -------
        explanation : str
            Multi-line explanation
        """
        aggregate, dimension_scores = self.score_and_aggregate(value, context)
        
        lines = [
            f"Feature value: {value:.3f}",
            f"Overall score: {aggregate:.3f}",
            "",
            "Dimension breakdown:"
        ]
        
        # Sort by score (highest first)
        sorted_scores = sorted(
            dimension_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for name, score in sorted_scores:
            dimension = self.dimensions[name]
            weight = dimension.weight
            contribution = weight * score
            
            lines.append(
                f"  {name:15s}: {score:.3f} "
                f"(weight={weight:.1f}, contribution={contribution:.3f})"
            )
        
        return "\n".join(lines)
    
    def record_score(
        self, 
        value: float, 
        context: Dict[str, Any],
        outcome: Optional[Any] = None
    ):
        """
        Record score for analysis
        
        Parameters
        ----------
        value : float
            Feature value
        context : Dict
            Context used for scoring
        outcome : Any, optional
            Actual outcome (for validation)
        """
        aggregate, dimension_scores = self.score_and_aggregate(value, context)
        
        self.history.append({
            'value': value,
            'aggregate_score': aggregate,
            'dimension_scores': dimension_scores.copy(),
            'context': context.copy(),
            'outcome': outcome
        })
    
    def get_dimension_statistics(self, dimension_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific dimension from history
        
        Returns mean, std, min, max of dimension scores
        """
        if not self.history:
            return {}
        
        scores = [
            record['dimension_scores'].get(dimension_name, 0.5)
            for record in self.history
        ]
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores)
        }


# ============================================================================
# Standard Scorers
# ============================================================================

def relevance_scorer(value: float, context: Dict[str, Any]) -> float:
    """
    Score based on correlation with outcomes
    
    Measures how predictive this feature value is
    
    Context keys expected:
    - 'historical_values': array of historical values
    - 'historical_outcomes': array of historical outcomes
    """
    if 'historical_values' not in context or 'historical_outcomes' not in context:
        return 0.5  # Neutral if no history
    
    historical_values = np.array(context['historical_values'])
    historical_outcomes = np.array(context['historical_outcomes'])
    
    if len(historical_values) < 10:
        return 0.5  # Need enough history
    
    # Find similar historical values
    similarity_threshold = context.get('similarity_threshold', 0.1)
    similar_mask = np.abs(historical_values - value) < similarity_threshold
    
    if similar_mask.sum() < 5:
        return 0.5  # Not enough similar cases
    
    # Average outcome for similar values
    similar_outcomes = historical_outcomes[similar_mask]
    
    # Score based on outcome strength
    # Assumes outcomes are binary or normalized to [-1, 1]
    avg_outcome = np.mean(similar_outcomes)
    score = (avg_outcome + 1) / 2  # Map [-1, 1] to [0, 1]
    
    return np.clip(score, 0, 1)


def confidence_scorer(value: float, context: Dict[str, Any]) -> float:
    """
    Score based on data quality and sample size
    
    Context keys expected:
    - 'sample_size': number of samples
    - 'missing_rate': fraction of missing data
    - 'noise_level': estimated noise standard deviation
    """
    score = 1.0
    
    # Penalize small sample size
    sample_size = context.get('sample_size', 100)
    if sample_size < 30:
        score *= sample_size / 30
    
    # Penalize high missing rate
    missing_rate = context.get('missing_rate', 0.0)
    score *= (1 - missing_rate)
    
    # Penalize high noise
    noise_level = context.get('noise_level', 0.1)
    signal_to_noise = 1.0 / (1.0 + noise_level)
    score *= signal_to_noise
    
    return np.clip(score, 0, 1)


def freshness_scorer(value: float, context: Dict[str, Any]) -> float:
    """
    Score based on data recency
    
    Context keys expected:
    - 'age_seconds': age of data in seconds
    - 'half_life_seconds': half-life for exponential decay
    """
    age = context.get('age_seconds', 0)
    half_life = context.get('half_life_seconds', 3600)  # 1 hour default
    
    # Exponential decay
    decay = 0.5 ** (age / half_life)
    
    return np.clip(decay, 0, 1)


def stability_scorer(value: float, context: Dict[str, Any]) -> float:
    """
    Score based on feature stability (low variance is good)
    
    Context keys expected:
    - 'recent_values': array of recent values
    - 'typical_std': typical standard deviation
    """
    recent_values = context.get('recent_values', [value])
    recent_values = np.array(recent_values)
    
    if len(recent_values) < 3:
        return 0.5  # Neutral if not enough history
    
    # Compute coefficient of variation
    std = np.std(recent_values)
    mean = np.mean(recent_values)
    
    if mean == 0:
        return 0.5
    
    cv = std / abs(mean)
    
    # Lower CV = higher stability score
    # Map CV to [0, 1]: CV=0 → score=1, CV=1 → score=0.5, CV=2 → score=0
    score = 1.0 / (1.0 + cv)
    
    return np.clip(score, 0, 1)


def density_scorer(value: float, context: Dict[str, Any]) -> float:
    """
    Score based on local data density
    
    Higher density = more reliable estimate
    
    Context keys expected:
    - 'all_values': array of all values
    - 'bandwidth': kernel bandwidth for density estimation
    """
    all_values = context.get('all_values', [value])
    all_values = np.array(all_values)
    
    if len(all_values) < 10:
        return 0.5
    
    # Count values within bandwidth
    bandwidth = context.get('bandwidth', 0.1)
    nearby_mask = np.abs(all_values - value) < bandwidth
    density = nearby_mask.sum() / len(all_values)
    
    # Normalize to [0, 1]
    # Assume max density is 0.2 (20% of points nearby)
    score = min(density / 0.2, 1.0)
    
    return score


# ============================================================================
# Pre-configured Scorers
# ============================================================================

def create_trading_scorer() -> FeatureScorer:
    """
    Create scorer configured for trading signals
    
    Examples
    --------
    >>> scorer = create_trading_scorer()
    >>> context = {
    ...     'historical_values': recent_signals,
    ...     'historical_outcomes': recent_returns,
    ...     'sample_size': len(recent_signals),
    ...     'age_seconds': 0
    ... }
    >>> score = scorer.score_and_aggregate(signal_value, context)[0]
    """
    scorer = FeatureScorer()
    
    scorer.add_dimension(
        'strength',
        relevance_scorer,
        weight=2.0,
        description="Signal strength and predictive power"
    )
    
    scorer.add_dimension(
        'confidence',
        confidence_scorer,
        weight=1.5,
        description="Data quality and sample size"
    )
    
    scorer.add_dimension(
        'timeliness',
        freshness_scorer,
        weight=1.5,
        description="Signal freshness"
    )
    
    scorer.add_dimension(
        'stability',
        stability_scorer,
        weight=1.0,
        description="Signal stability over time"
    )
    
    return scorer


def create_medical_scorer() -> FeatureScorer:
    """
    Create scorer configured for medical diagnosis
    
    Emphasizes confidence and stability over freshness
    """
    scorer = FeatureScorer()
    
    scorer.add_dimension(
        'diagnostic_value',
        relevance_scorer,
        weight=2.5,
        description="Correlation with diagnosis"
    )
    
    scorer.add_dimension(
        'measurement_quality',
        confidence_scorer,
        weight=2.0,
        description="Test reliability and precision"
    )
    
    scorer.add_dimension(
        'consistency',
        stability_scorer,
        weight=1.5,
        description="Consistency across measurements"
    )
    
    return scorer


def create_signal_scorer() -> FeatureScorer:
    """
    Create scorer configured for signal processing
    
    Emphasizes stability and density
    """
    scorer = FeatureScorer()
    
    scorer.add_dimension(
        'signal_strength',
        relevance_scorer,
        weight=2.0,
        description="Signal power"
    )
    
    scorer.add_dimension(
        'noise_level',
        confidence_scorer,
        weight=1.5,
        description="Signal-to-noise ratio"
    )
    
    scorer.add_dimension(
        'temporal_stability',
        stability_scorer,
        weight=1.5,
        description="Stability over time"
    )
    
    scorer.add_dimension(
        'local_density',
        density_scorer,
        weight=1.0,
        description="Local data density"
    )
    
    return scorer




