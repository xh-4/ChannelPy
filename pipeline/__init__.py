"""
Pipeline module for ChannelPy

Three-stage data processing architecture:
1. Preprocess: Raw data → Clean features
2. Encode: Features → Channel states
3. Interpret: States → Decisions

Examples
--------
>>> from channelpy.pipeline import ChannelPipeline
>>> from channelpy.pipeline.preprocessors import StandardScaler
>>> from channelpy.pipeline.encoders import ThresholdEncoder
>>> from channelpy.pipeline.interpreters import RuleBasedInterpreter
>>> 
>>> pipeline = ChannelPipeline()
>>> pipeline.add_preprocessor(StandardScaler())
>>> pipeline.add_encoder(ThresholdEncoder())
>>> pipeline.add_interpreter(RuleBasedInterpreter())
>>> 
>>> pipeline.fit(X_train, y_train)
>>> decisions, states = pipeline.transform(X_test)
"""

from .base import (
    BasePipeline,
    ChannelPipeline,
    PipelineStage
)

from .preprocessors import (
    Preprocessor,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MissingDataHandler,
    OutlierDetector,
    FeatureExtractor,
    TimeSeriesFeatureExtractor,
    StatisticalFeatureExtractor
)

from .encoders import (
    Encoder,
    ThresholdEncoder,
    LearnedThresholdEncoder,
    DualFeatureEncoder,
    AdaptiveEncoder,
    MultiFeatureEncoder,
    TopologyAwareEncoder
)

from .interpreters import (
    Interpreter,
    RuleBasedInterpreter,
    LookupTableInterpreter,
    FSMInterpreter,
    PatternMatcher,
    ScoreBasedInterpreter,
    EnsembleInterpreter
)

from .full_pipeline import (
    FullChannelPipeline,
    PipelineBuilder,
    AutoPipeline
)

__all__ = [
    # Base classes
    'BasePipeline',
    'ChannelPipeline',
    'PipelineStage',
    
    # Preprocessors
    'Preprocessor',
    'StandardScaler',
    'RobustScaler',
    'MinMaxScaler',
    'MissingDataHandler',
    'OutlierDetector',
    'FeatureExtractor',
    'TimeSeriesFeatureExtractor',
    'StatisticalFeatureExtractor',
    
    # Encoders
    'Encoder',
    'ThresholdEncoder',
    'LearnedThresholdEncoder',
    'DualFeatureEncoder',
    'AdaptiveEncoder',
    'MultiFeatureEncoder',
    'TopologyAwareEncoder',
    
    # Interpreters
    'Interpreter',
    'RuleBasedInterpreter',
    'LookupTableInterpreter',
    'FSMInterpreter',
    'PatternMatcher',
    'ScoreBasedInterpreter',
    'EnsembleInterpreter',
    
    # Full pipelines
    'FullChannelPipeline',
    'PipelineBuilder',
    'AutoPipeline'
]