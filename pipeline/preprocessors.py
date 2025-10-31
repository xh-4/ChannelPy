"""
Preprocessors: Raw data → Clean features

Stage 1 of the pipeline: prepare data for encoding
"""

from typing import Optional, Union, Callable, List
import numpy as np
from abc import ABC, abstractmethod
import warnings


class Preprocessor(ABC):
    """
    Base class for preprocessors
    
    All preprocessors should implement fit() and transform()
    """
    
    @abstractmethod
    def fit(self, X, y=None):
        """Learn from data"""
        pass
    
    @abstractmethod
    def transform(self, X):
        """Transform data"""
        pass
    
    def fit_transform(self, X, y=None):
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)


class StandardScaler(Preprocessor):
    """
    Standardize features by removing mean and scaling to unit variance
    
    z = (x - mean) / std
    
    Examples
    --------
    >>> scaler = StandardScaler()
    >>> X_train_scaled = scaler.fit_transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Compute mean and std"""
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Standardize data"""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X)
        return (X - self.mean_) / self.std_


class RobustScaler(Preprocessor):
    """
    Scale features using median and IQR (robust to outliers)
    
    z = (x - median) / IQR
    
    Examples
    --------
    >>> scaler = RobustScaler()
    >>> X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(self):
        self.median_ = None
        self.iqr_ = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Compute median and IQR"""
        X = np.asarray(X)
        self.median_ = np.median(X, axis=0)
        
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        
        # Avoid division by zero
        self.iqr_[self.iqr_ == 0] = 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Scale data"""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X)
        return (X - self.median_) / self.iqr_


class MinMaxScaler(Preprocessor):
    """
    Scale features to [0, 1] range
    
    z = (x - min) / (max - min)
    
    Examples
    --------
    >>> scaler = MinMaxScaler()
    >>> X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Compute min and max"""
        X = np.asarray(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        
        # Avoid division by zero
        range_ = self.max_ - self.min_
        range_[range_ == 0] = 1.0
        self.max_[range_ == 0] = self.min_[range_ == 0] + 1.0
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Scale to range"""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X)
        X_std = (X - self.min_) / (self.max_ - self.min_)
        
        min_val, max_val = self.feature_range
        return X_std * (max_val - min_val) + min_val


class MissingDataHandler(Preprocessor):
    """
    Handle missing data (NaN values)
    
    Strategies:
    - 'mean': Replace with mean
    - 'median': Replace with median
    - 'forward_fill': Use last valid value
    - 'drop': Remove rows with missing data
    
    Examples
    --------
    >>> handler = MissingDataHandler(strategy='median')
    >>> X_clean = handler.fit_transform(X)
    """
    
    def __init__(self, strategy: str = 'median'):
        """
        Parameters
        ----------
        strategy : str
            How to handle missing data
        """
        self.strategy = strategy
        self.fill_value_ = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Learn fill values"""
        X = np.asarray(X, dtype=float)
        
        if self.strategy == 'mean':
            self.fill_value_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.fill_value_ = np.nanmedian(X, axis=0)
        elif self.strategy in ['forward_fill', 'drop']:
            # No learning needed
            self.fill_value_ = None
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Handle missing data"""
        if not self.is_fitted:
            raise RuntimeError("Handler not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=float).copy()
        
        if self.strategy == 'mean' or self.strategy == 'median':
            # Replace NaN with fill value
            mask = np.isnan(X)
            if X.ndim == 1:
                X[mask] = self.fill_value_
            else:
                for i in range(X.shape[1]):
                    X[mask[:, i], i] = self.fill_value_[i]
        
        elif self.strategy == 'forward_fill':
            # Forward fill
            if X.ndim == 1:
                X = self._forward_fill_1d(X)
            else:
                for i in range(X.shape[1]):
                    X[:, i] = self._forward_fill_1d(X[:, i])
        
        elif self.strategy == 'drop':
            # Remove rows with NaN
            mask = ~np.isnan(X).any(axis=1 if X.ndim > 1 else 0)
            X = X[mask]
        
        return X
    
    def _forward_fill_1d(self, x):
        """Forward fill 1D array"""
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, out=idx)
        return x[idx]


class OutlierDetector(Preprocessor):
    """
    Detect and handle outliers
    
    Methods:
    - 'clip': Clip to percentile range
    - 'remove': Remove outliers
    - 'winsorize': Replace with percentile values
    
    Examples
    --------
    >>> detector = OutlierDetector(method='clip', percentile_range=(1, 99))
    >>> X_clean = detector.fit_transform(X)
    """
    
    def __init__(self, method: str = 'clip', percentile_range: tuple = (1, 99)):
        """
        Parameters
        ----------
        method : str
            How to handle outliers
        percentile_range : tuple
            (lower, upper) percentiles for defining outliers
        """
        self.method = method
        self.percentile_range = percentile_range
        self.lower_bound_ = None
        self.upper_bound_ = None
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Learn outlier bounds"""
        X = np.asarray(X)
        
        lower_pct, upper_pct = self.percentile_range
        self.lower_bound_ = np.percentile(X, lower_pct, axis=0)
        self.upper_bound_ = np.percentile(X, upper_pct, axis=0)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Handle outliers"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        X = np.asarray(X).copy()
        
        if self.method == 'clip':
            # Clip to bounds
            X = np.clip(X, self.lower_bound_, self.upper_bound_)
        
        elif self.method == 'remove':
            # Remove outliers
            if X.ndim == 1:
                mask = (X >= self.lower_bound_) & (X <= self.upper_bound_)
            else:
                mask = np.all(
                    (X >= self.lower_bound_) & (X <= self.upper_bound_),
                    axis=1
                )
            X = X[mask]
        
        elif self.method == 'winsorize':
            # Replace with bound values
            if X.ndim == 1:
                X[X < self.lower_bound_] = self.lower_bound_
                X[X > self.upper_bound_] = self.upper_bound_
            else:
                for i in range(X.shape[1]):
                    X[X[:, i] < self.lower_bound_[i], i] = self.lower_bound_[i]
                    X[X[:, i] > self.upper_bound_[i], i] = self.upper_bound_[i]
        
        return X


class FeatureExtractor(Preprocessor):
    """
    Base class for feature extraction
    
    Subclass and implement extract_features()
    """
    
    @abstractmethod
    def extract_features(self, X):
        """Extract features from raw data"""
        pass
    
    def fit(self, X, y=None):
        """Feature extractors typically don't need fitting"""
        return self
    
    def transform(self, X):
        """Extract features"""
        return self.extract_features(X)


class TimeSeriesFeatureExtractor(FeatureExtractor):
    """
    Extract features from time series data
    
    Features:
    - Mean, std, min, max
    - Trend (linear regression slope)
    - Volatility (rolling std)
    - Momentum (rate of change)
    
    Examples
    --------
    >>> extractor = TimeSeriesFeatureExtractor(window_size=10)
    >>> features = extractor.transform(time_series_data)
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    def extract_features(self, X):
        """Extract time series features"""
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize feature array
        features = []
        
        for i in range(n_features):
            series = X[:, i]
            
            # Basic statistics
            mean = np.mean(series)
            std = np.std(series)
            minimum = np.min(series)
            maximum = np.max(series)
            
            # Trend (slope)
            if len(series) > 1:
                x = np.arange(len(series))
                trend = np.polyfit(x, series, 1)[0]
            else:
                trend = 0
            
            # Volatility (rolling std)
            if len(series) >= self.window_size:
                volatility = np.std([
                    np.std(series[i:i+self.window_size])
                    for i in range(len(series) - self.window_size + 1)
                ])
            else:
                volatility = std
            
            # Momentum (rate of change)
            if len(series) > 1:
                momentum = (series[-1] - series[0]) / len(series)
            else:
                momentum = 0
            
            features.extend([mean, std, minimum, maximum, trend, volatility, momentum])
        
        return np.array(features).reshape(1, -1) if n_samples == 1 else np.array(features)


class StatisticalFeatureExtractor(FeatureExtractor):
    """
    Extract statistical features
    
    Features:
    - Mean, median, std
    - Skewness, kurtosis
    - Percentiles (25th, 50th, 75th)
    - Range, IQR
    
    Examples
    --------
    >>> extractor = StatisticalFeatureExtractor()
    >>> features = extractor.transform(data)
    """
    
    def extract_features(self, X):
        """Extract statistical features"""
        X = np.asarray(X)
        
        if X.ndim == 1:
            return self._extract_single(X)
        else:
            # Extract for each column
            features = [self._extract_single(X[:, i]) for i in range(X.shape[1])]
            return np.concatenate(features)
    
    def _extract_single(self, x):
        """Extract features from single array"""
        try:
            from scipy import stats
            skewness = stats.skew(x)
            kurtosis_val = stats.kurtosis(x)
        except ImportError:
            skewness = self._skewness(x)
            kurtosis_val = self._kurtosis(x)
        
        features = [
            np.mean(x),
            np.median(x),
            np.std(x),
            skewness,
            kurtosis_val,
            np.percentile(x, 25),
            np.percentile(x, 50),
            np.percentile(x, 75),
            np.ptp(x),  # range
            np.percentile(x, 75) - np.percentile(x, 25)  # IQR
        ]
        
        return np.array(features)
    
    def _skewness(self, x):
        """Compute skewness (fallback)"""
        mean = np.mean(x)
        std = np.std(x)
        return np.mean(((x - mean) / std) ** 3) if std > 0 else 0
    
    def _kurtosis(self, x):
        """Compute kurtosis (fallback)"""
        mean = np.mean(x)
        std = np.std(x)
        return np.mean(((x - mean) / std) ** 4) - 3 if std > 0 else 0