"""
Base pipeline architecture for ChannelPy

Defines the three-stage processing model:
Stage 1 (Preprocess): Raw data → Clean features
Stage 2 (Encode): Features → Channel states  
Stage 3 (Interpret): States → Decisions
"""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import warnings

from ..core.state import State, StateArray


class PipelineStage(Enum):
    """Pipeline processing stages"""
    PREPROCESS = "preprocess"
    ENCODE = "encode"
    INTERPRET = "interpret"


class BasePipeline(ABC):
    """
    Base class for channel pipelines
    
    All pipelines follow the three-stage architecture:
    1. Preprocess: Clean and prepare data
    2. Encode: Convert to channel states
    3. Interpret: Make decisions from states
    
    Examples
    --------
    >>> class MyPipeline(BasePipeline):
    ...     def fit(self, X, y=None):
    ...         # Learn from data
    ...         return self
    ...     
    ...     def transform(self, X):
    ...         # Process data
    ...         return decisions, states
    """
    
    def __init__(self):
        self.is_fitted = False
        self.fit_metadata = {}
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit pipeline on training data
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target labels
            
        Returns
        -------
        self : BasePipeline
            Fitted pipeline
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """
        Transform data through pipeline
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        decisions : array-like
            Pipeline output (decisions)
        states : StateArray or List[State]
            Intermediate channel states
        """
        pass
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one call
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like, optional
            Target labels
            
        Returns
        -------
        decisions : array-like
            Transformed output
        states : StateArray or List[State]
            Channel states
        """
        self.fit(X, y)
        return self.transform(X)
    
    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}({fitted_str})"


class ChannelPipeline(BasePipeline):
    """
    Concrete implementation of channel pipeline
    
    Maintains ordered lists of preprocessors, encoders, and interpreters.
    Data flows through stages sequentially.
    
    Examples
    --------
    >>> from channelpy.pipeline import ChannelPipeline
    >>> from channelpy.pipeline.preprocessors import StandardScaler
    >>> from channelpy.pipeline.encoders import ThresholdEncoder
    >>> from channelpy.pipeline.interpreters import RuleBasedInterpreter
    >>> 
    >>> pipeline = ChannelPipeline()
    >>> pipeline.add_preprocessor(StandardScaler())
    >>> pipeline.add_encoder(ThresholdEncoder(threshold_i=0.5, threshold_q=0.75))
    >>> pipeline.add_interpreter(RuleBasedInterpreter())
    >>> 
    >>> pipeline.fit(X_train, y_train)
    >>> decisions, states = pipeline.transform(X_test)
    """
    
    def __init__(self, verbose: bool = False):
        """
        Parameters
        ----------
        verbose : bool
            Print progress messages
        """
        super().__init__()
        self.preprocessors = []
        self.encoders = []
        self.interpreters = []
        self.verbose = verbose
        
        # Store intermediate results for debugging
        self.last_features = None
        self.last_states = None
        self.last_decisions = None
    
    def add_preprocessor(self, preprocessor):
        """
        Add preprocessing step
        
        Parameters
        ----------
        preprocessor : Preprocessor
            Preprocessor instance with fit() and transform() methods
        """
        self.preprocessors.append(preprocessor)
        if self.verbose:
            print(f"Added preprocessor: {preprocessor.__class__.__name__}")
    
    def add_encoder(self, encoder):
        """
        Add encoding step
        
        Parameters
        ----------
        encoder : Encoder
            Encoder instance that converts features to states
        """
        self.encoders.append(encoder)
        if self.verbose:
            print(f"Added encoder: {encoder.__class__.__name__}")
    
    def add_interpreter(self, interpreter):
        """
        Add interpretation step
        
        Parameters
        ----------
        interpreter : Interpreter
            Interpreter instance that converts states to decisions
        """
        self.interpreters.append(interpreter)
        if self.verbose:
            print(f"Added interpreter: {interpreter.__class__.__name__}")
    
    def fit(self, X, y=None):
        """
        Fit all pipeline components
        
        Parameters
        ----------
        X : array-like
            Training data
        y : array-like, optional
            Training labels
            
        Returns
        -------
        self : ChannelPipeline
            Fitted pipeline
        """
        if self.verbose:
            print("=" * 60)
            print("FITTING PIPELINE")
            print("=" * 60)
        
        # Convert to numpy array
        X = np.asarray(X)
        
        # Stage 1: Fit preprocessors
        if self.verbose:
            print(f"\nStage 1: Fitting {len(self.preprocessors)} preprocessors...")
        
        features = X
        for i, preprocessor in enumerate(self.preprocessors):
            if self.verbose:
                print(f"  [{i+1}/{len(self.preprocessors)}] {preprocessor.__class__.__name__}...", end=" ")
            
            if hasattr(preprocessor, 'fit'):
                preprocessor.fit(features, y)
            
            if hasattr(preprocessor, 'transform'):
                features = preprocessor.transform(features)
            elif callable(preprocessor):
                features = preprocessor(features)
            
            if self.verbose:
                print(f"✓ (shape: {features.shape})")
        
        # Stage 2: Fit encoders
        if self.verbose:
            print(f"\nStage 2: Fitting {len(self.encoders)} encoders...")
        
        for i, encoder in enumerate(self.encoders):
            if self.verbose:
                print(f"  [{i+1}/{len(self.encoders)}] {encoder.__class__.__name__}...", end=" ")
            
            if hasattr(encoder, 'fit'):
                encoder.fit(features, y)
            
            if self.verbose:
                print("✓")
        
        # Stage 3: Fit interpreters
        if y is not None:
            if self.verbose:
                print(f"\nStage 3: Fitting {len(self.interpreters)} interpreters...")
            
            # Encode features to get states for interpreter training
            states = self._encode(features)
            
            for i, interpreter in enumerate(self.interpreters):
                if self.verbose:
                    print(f"  [{i+1}/{len(self.interpreters)}] {interpreter.__class__.__name__}...", end=" ")
                
                if hasattr(interpreter, 'fit'):
                    interpreter.fit(states, y)
                
                if self.verbose:
                    print("✓")
        
        self.is_fitted = True
        self.fit_metadata = {
            'n_samples': len(X),
            'n_features': X.shape[1] if X.ndim > 1 else 1,
            'n_preprocessors': len(self.preprocessors),
            'n_encoders': len(self.encoders),
            'n_interpreters': len(self.interpreters)
        }
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("PIPELINE FITTED SUCCESSFULLY")
            print("=" * 60)
        
        return self
    
    def transform(self, X):
        """
        Transform data through complete pipeline
        
        Parameters
        ----------
        X : array-like
            Input data
            
        Returns
        -------
        decisions : array-like
            Final decisions
        states : StateArray or list
            Intermediate channel states (for debugging)
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        X = np.asarray(X)
        
        # Stage 1: Preprocess
        features = self._preprocess(X)
        self.last_features = features
        
        # Stage 2: Encode
        states = self._encode(features)
        self.last_states = states
        
        # Stage 3: Interpret
        decisions = self._interpret(states)
        self.last_decisions = decisions
        
        return decisions, states
    
    def _preprocess(self, X):
        """Apply all preprocessors"""
        result = X
        for preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'transform'):
                result = preprocessor.transform(result)
            elif callable(preprocessor):
                result = preprocessor(result)
        return result
    
    def _encode(self, features):
        """Apply all encoders"""
        if len(self.encoders) == 0:
            warnings.warn("No encoders in pipeline. Returning features as-is.")
            return features
        
        # If single encoder, apply directly
        if len(self.encoders) == 1:
            return self.encoders[0](features)
        
        # Multiple encoders: apply each and collect states
        # For now, use first encoder (could be extended to multi-channel)
        return self.encoders[0](features)
    
    def _interpret(self, states):
        """Apply all interpreters"""
        if len(self.interpreters) == 0:
            warnings.warn("No interpreters in pipeline. Returning states as-is.")
            return states
        
        # Apply interpreters in sequence
        result = states
        for interpreter in self.interpreters:
            result = interpreter(result)
        
        return result
    
    def get_stage_output(self, X, stage: PipelineStage):
        """
        Get output at specific pipeline stage (for debugging)
        
        Parameters
        ----------
        X : array-like
            Input data
        stage : PipelineStage
            Which stage to stop at
            
        Returns
        -------
        output : array-like
            Output at specified stage
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        X = np.asarray(X)
        
        if stage == PipelineStage.PREPROCESS:
            return self._preprocess(X)
        
        elif stage == PipelineStage.ENCODE:
            features = self._preprocess(X)
            return self._encode(features)
        
        elif stage == PipelineStage.INTERPRET:
            return self.transform(X)
        
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def summary(self) -> str:
        """
        Generate pipeline summary
        
        Returns
        -------
        summary : str
            Multi-line description of pipeline
        """
        lines = [
            "=" * 60,
            "CHANNEL PIPELINE SUMMARY",
            "=" * 60,
            f"Status: {'FITTED' if self.is_fitted else 'NOT FITTED'}",
            ""
        ]
        
        if self.is_fitted:
            meta = self.fit_metadata
            lines.extend([
                f"Training samples: {meta['n_samples']}",
                f"Input features: {meta['n_features']}",
                ""
            ])
        
        lines.append("STAGE 1: PREPROCESSING")
        lines.append("-" * 60)
        if self.preprocessors:
            for i, prep in enumerate(self.preprocessors, 1):
                lines.append(f"  {i}. {prep.__class__.__name__}")
        else:
            lines.append("  (none)")
        lines.append("")
        
        lines.append("STAGE 2: ENCODING")
        lines.append("-" * 60)
        if self.encoders:
            for i, enc in enumerate(self.encoders, 1):
                lines.append(f"  {i}. {enc.__class__.__name__}")
        else:
            lines.append("  (none)")
        lines.append("")
        
        lines.append("STAGE 3: INTERPRETATION")
        lines.append("-" * 60)
        if self.interpreters:
            for i, interp in enumerate(self.interpreters, 1):
                lines.append(f"  {i}. {interp.__class__.__name__}")
        else:
            lines.append("  (none)")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.summary()