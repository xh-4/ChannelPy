"""
Complete integrated pipelines with builder pattern and auto-configuration
"""

from typing import Optional, List, Dict, Any
import numpy as np

from .base import ChannelPipeline, PipelineStage
from .preprocessors import (
    StandardScaler, RobustScaler, MissingDataHandler, 
    OutlierDetector, TimeSeriesFeatureExtractor
)
from .encoders import (
    LearnedThresholdEncoder, AdaptiveEncoder, TopologyAwareEncoder
)
from .interpreters import (
    RuleBasedInterpreter, LookupTableInterpreter, ScoreBasedInterpreter
)


class FullChannelPipeline(ChannelPipeline):
    """
    Pre-configured complete pipeline
    
    Automatically sets up preprocessing, encoding, and interpretation
    
    Examples
    --------
    >>> pipeline = FullChannelPipeline(
    ...     scaling='standard',
    ...     encoding='adaptive',
    ...     interpretation='rule_based'
    ... )
    >>> pipeline.fit(X_train, y_train)
    >>> predictions, states = pipeline.transform(X_test)
    """
    
    def __init__(
        self,
        scaling: str = 'standard',
        handle_missing: bool = True,
        remove_outliers: bool = False,
        encoding: str = 'learned',
        interpretation: str = 'lookup',
        verbose: bool = False
    ):
        """
        Parameters
        ----------
        scaling : str
            'standard', 'robust', 'minmax', or 'none'
        handle_missing : bool
            Whether to handle missing data
        remove_outliers : bool
            Whether to remove outliers
        encoding : str
            'learned', 'adaptive', 'topology'
        interpretation : str
            'lookup', 'rule_based', 'score_based'
        verbose : bool
            Print progress
        """
        super().__init__(verbose=verbose)
        
        # Add preprocessors
        if handle_missing:
            self.add_preprocessor(MissingDataHandler(strategy='median'))
        
        if remove_outliers:
            self.add_preprocessor(OutlierDetector(method='clip'))
        
        if scaling == 'standard':
            self.add_preprocessor(StandardScaler())
        elif scaling == 'robust':
            self.add_preprocessor(RobustScaler())
        elif scaling != 'none':
            raise ValueError(f"Unknown scaling: {scaling}")
        
        # Add encoder
        if encoding == 'learned':
            self.add_encoder(LearnedThresholdEncoder())
        elif encoding == 'adaptive':
            self.add_encoder(AdaptiveEncoder())
        elif encoding == 'topology':
            self.add_encoder(TopologyAwareEncoder())
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        
        # Add interpreter
        if interpretation == 'lookup':
            self.add_interpreter(LookupTableInterpreter())
        elif interpretation == 'rule_based':
            self.add_interpreter(RuleBasedInterpreter())
        elif interpretation == 'score_based':
            self.add_interpreter(ScoreBasedInterpreter())
        else:
            raise ValueError(f"Unknown interpretation: {interpretation}")


class PipelineBuilder:
    """
    Fluent interface for building pipelines
    
    Examples
    --------
    >>> pipeline = (PipelineBuilder()
    ...     .with_standard_scaling()
    ...     .with_missing_data_handling()
    ...     .with_adaptive_encoding()
    ...     .with_rule_based_interpretation({PSI: 'buy', EMPTY: 'sell'})
    ...     .build()
    ... )
    >>> pipeline.fit(X_train, y_train)
    """
    
    def __init__(self):
        self.pipeline = ChannelPipeline()
    
    def with_standard_scaling(self):
        """Add standard scaling"""
        self.pipeline.add_preprocessor(StandardScaler())
        return self
    
    def with_robust_scaling(self):
        """Add robust scaling"""
        self.pipeline.add_preprocessor(RobustScaler())
        return self
    
    def with_missing_data_handling(self, strategy: str = 'median'):
        """Add missing data handler"""
        self.pipeline.add_preprocessor(MissingDataHandler(strategy=strategy))
        return self
    
    def with_outlier_removal(self, method: str = 'clip'):
        """Add outlier detection"""
        self.pipeline.add_preprocessor(OutlierDetector(method=method))
        return self
    
    def with_time_series_features(self, window_size: int = 10):
        """Add time series feature extraction"""
        self.pipeline.add_preprocessor(TimeSeriesFeatureExtractor(window_size=window_size))
        return self
    
    def with_learned_encoding(self, method: str = 'percentile'):
        """Add learned threshold encoder"""
        self.pipeline.add_encoder(LearnedThresholdEncoder(method=method))
        return self
    
    def with_adaptive_encoding(self, use_topology: bool = False):
        """Add adaptive encoder"""
        self.pipeline.add_encoder(AdaptiveEncoder(use_topology=use_topology))
        return self
    
    def with_topology_encoding(self):
        """Add topology-aware encoder"""
        self.pipeline.add_encoder(TopologyAwareEncoder())
        return self
    
    def with_rule_based_interpretation(self, rules: Optional[Dict] = None):
        """Add rule-based interpreter"""
        self.pipeline.add_interpreter(RuleBasedInterpreter(rules=rules))
        return self
    
    def with_lookup_interpretation(self):
        """Add lookup table interpreter"""
        self.pipeline.add_interpreter(LookupTableInterpreter())
        return self
    
    def with_score_based_interpretation(self):
        """Add score-based interpreter"""
        self.pipeline.add_interpreter(ScoreBasedInterpreter())
        return self
    
    def verbose(self, verbose: bool = True):
        """Set verbose mode"""
        self.pipeline.verbose = verbose
        return self
    
    def build(self) -> ChannelPipeline:
        """Return built pipeline"""
        return self.pipeline


class AutoPipeline:
    """
    Automatically configure pipeline based on data characteristics
    
    Examples
    --------
    >>> auto = AutoPipeline()
    >>> pipeline = auto.create_pipeline(X_train, y_train)
    >>> predictions, states = pipeline.transform(X_test)
    """
    
    def create_pipeline(
        self, 
        X, 
        y=None,
        verbose: bool = False
    ) -> ChannelPipeline:
        """
        Auto-configure pipeline
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like, optional
            Training labels
        verbose : bool
            Print configuration decisions
            
        Returns
        -------
        pipeline : ChannelPipeline
            Configured pipeline
        """
        X = np.asarray(X)
        
        builder = PipelineBuilder()
        
        if verbose:
            print("Auto-configuring pipeline...")
        
        # Check for missing data
        if np.any(np.isnan(X)):
            if verbose:
                print("  ✓ Missing data detected → Adding MissingDataHandler")
            builder.with_missing_data_handling()
        
        # Check for outliers
        q99 = np.nanpercentile(X, 99)
        q01 = np.nanpercentile(X, 1)
        if (q99 - q01) > 10 * (np.nanpercentile(X, 75) - np.nanpercentile(X, 25)):
            if verbose:
                print("  ✓ Outliers detected → Adding OutlierDetector")
            builder.with_outlier_removal()
        
        # Always scale
        if verbose:
            print("  ✓ Adding StandardScaler")
        builder.with_standard_scaling()
        
        # Choose encoder
        if len(X) > 10000:
            if verbose:
                print("  ✓ Large dataset → Using TopologyAwareEncoder")
            builder.with_topology_encoding()
        elif len(X) > 1000:
            if verbose:
                print("  ✓ Medium dataset → Using AdaptiveEncoder")
            builder.with_adaptive_encoding()
        else:
            if verbose:
                print("  ✓ Small dataset → Using LearnedThresholdEncoder")
            builder.with_learned_encoding()
        
        # Choose interpreter
        if y is not None:
            if verbose:
                print("  ✓ Labels provided → Using LookupTableInterpreter")
            builder.with_lookup_interpretation()
        else:
            if verbose:
                print("  ✓ No labels → Using RuleBasedInterpreter")
            builder.with_rule_based_interpretation()
        
        if verbose:
            print("\nPipeline configuration complete!")
        
        return builder.build()