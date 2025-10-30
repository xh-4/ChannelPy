# How to Handle Missing Data

This guide shows strategies for handling missing, invalid, or incomplete data in channel pipelines.

## Table of Contents
- [Types of Missing Data](#types-of-missing-data)
- [Detection Strategies](#detection-strategies)
- [Handling Strategies](#handling-strategies)
- [Custom Missing Data Handlers](#custom-missing-data-handlers)
- [Integration with Pipelines](#integration-with-pipelines)
- [Best Practices](#best-practices)

---

## Types of Missing Data

### 1. **Explicitly Missing** (NaN, None, null)
```python
import numpy as np

data = [1.0, 2.0, np.nan, 4.0, None, 6.0]
```

### 2. **Sentinel Values** (e.g., -999, 0)
```python
data = [1.0, 2.0, -999, 4.0, -999, 6.0]  # -999 = missing
```

### 3. **Structural Missing** (missing features)
```python
# Some samples have fewer features
incomplete_sample = {'feature1': 1.0}  # Missing feature2
complete_sample = {'feature1': 1.0, 'feature2': 2.0}
```

### 4. **Temporal Missing** (gaps in time series)
```python
timestamps = [0, 1, 2, 5, 6]  # Missing t=3, t=4
values = [1.0, 2.0, 3.0, 6.0, 7.0]
```

---

## Detection Strategies

### Strategy 1: Simple Detection
```python
import numpy as np

def detect_missing(data):
    """
    Detect various types of missing data
    
    Returns
    -------
    mask : np.ndarray
        Boolean array: True where data is missing
    """
    data = np.asarray(data, dtype=float)
    
    # Check for NaN
    nan_mask = np.isnan(data)
    
    # Check for None (converted to NaN)
    # Already captured by isnan
    
    # Check for sentinel values
    sentinel_mask = np.abs(data) > 1e10  # Very large values
    
    # Combine
    missing_mask = nan_mask | sentinel_mask
    
    return missing_mask

# Usage
data = [1.0, 2.0, np.nan, 4.0, 1e12, 6.0]
missing = detect_missing(data)
print(f"Missing indices: {np.where(missing)[0]}")  # [2, 4]
```

### Strategy 2: Domain-Specific Detection
```python
class MissingDataDetector:
    """
    Flexible missing data detection
    
    Examples
    --------
    >>> detector = MissingDataDetector()
    >>> detector.add_rule(lambda x: np.isnan(x), "NaN")
    >>> detector.add_rule(lambda x: x == -999, "Sentinel -999")
    >>> detector.add_rule(lambda x: x < 0, "Negative (invalid)")
    >>> 
    >>> data = [1.0, -999, np.nan, -1.0, 5.0]
    >>> mask, reasons = detector.detect(data)
    """
    
    def __init__(self):
        self.rules = []  # List of (check_func, reason)
    
    def add_rule(self, check_func, reason: str):
        """
        Add detection rule
        
        Parameters
        ----------
        check_func : Callable
            Function that returns True for missing data
        reason : str
            Description of why data is considered missing
        """
        self.rules.append((check_func, reason))
    
    def detect(self, data):
        """
        Detect missing data using all rules
        
        Returns
        -------
        mask : np.ndarray
            Boolean mask
        reasons : dict
            {index: reason} for each missing value
        """
        data = np.asarray(data, dtype=float)
        mask = np.zeros(len(data), dtype=bool)
        reasons = {}
        
        for i, value in enumerate(data):
            for check_func, reason in self.rules:
                try:
                    if check_func(value):
                        mask[i] = True
                        reasons[i] = reason
                        break  # First matching rule
                except:
                    pass  # Ignore errors in check functions
        
        return mask, reasons
```

---

## Handling Strategies

### Strategy 1: Deletion (Drop Missing Values)
```python
def drop_missing(data, mask=None):
    """
    Remove missing values
    
    Simple but loses data
    """
    data = np.asarray(data, dtype=float)
    
    if mask is None:
        mask = np.isnan(data)
    
    return data[~mask]

# Usage
data = [1.0, 2.0, np.nan, 4.0, np.nan, 6.0]
clean_data = drop_missing(data)
print(clean_data)  # [1. 2. 4. 6.]
```

**Pros:** Simple, no assumptions
**Cons:** Loses data, can't handle test data with missing values

### Strategy 2: Imputation (Fill Missing Values)
```python
from typing import Literal

class SimpleImputer:
    """
    Fill missing values with statistics
    
    Examples
    --------
    >>> imputer = SimpleImputer(strategy='mean')
    >>> imputer.fit([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
    >>> filled = imputer.transform([np.nan, 2.0, np.nan])
    >>> print(filled)  # [3.25, 2.0, 3.25]  (mean of 1,2,4,6)
    """
    
    def __init__(self, strategy: Literal['mean', 'median', 'mode', 'constant'] = 'mean'):
        self.strategy = strategy
        self.fill_value = None
    
    def fit(self, data):
        """Learn fill value from data"""
        data = np.asarray(data, dtype=float)
        valid_data = data[~np.isnan(data)]
        
        if len(valid_data) == 0:
            self.fill_value = 0.0
            return self
        
        if self.strategy == 'mean':
            self.fill_value = np.mean(valid_data)
        elif self.strategy == 'median':
            self.fill_value = np.median(valid_data)
        elif self.strategy == 'mode':
            from scipy import stats
            self.fill_value = stats.mode(valid_data, keepdims=True)[0][0]
        
        return self
    
    def transform(self, data):
        """Fill missing values"""
        if self.fill_value is None:
            raise RuntimeError("Imputer not fitted")
        
        data = np.asarray(data, dtype=float).copy()
        missing_mask = np.isnan(data)
        data[missing_mask] = self.fill_value
        
        return data
    
    def fit_transform(self, data):
        """Fit and transform"""
        return self.fit(data).transform(data)
```

### Strategy 3: Forward/Backward Fill (Time Series)
```python
class TimeSeriesImputer:
    """
    Fill missing values in time series
    
    Examples
    --------
    >>> imputer = TimeSeriesImputer(method='forward')
    >>> data = [1.0, 2.0, np.nan, np.nan, 5.0, np.nan, 7.0]
    >>> filled = imputer.transform(data)
    >>> print(filled)  # [1. 2. 2. 2. 5. 5. 7.]
    """
    
    def __init__(self, method: Literal['forward', 'backward', 'both'] = 'forward'):
        self.method = method
    
    def transform(self, data):
        """Fill missing values"""
        data = np.asarray(data, dtype=float).copy()
        
        if self.method == 'forward' or self.method == 'both':
            # Forward fill
            last_valid = None
            for i in range(len(data)):
                if np.isnan(data[i]):
                    if last_valid is not None:
                        data[i] = last_valid
                else:
                    last_valid = data[i]
        
        if self.method == 'backward' or self.method == 'both':
            # Backward fill
            next_valid = None
            for i in range(len(data) - 1, -1, -1):
                if np.isnan(data[i]):
                    if next_valid is not None:
                        data[i] = next_valid
                else:
                    next_valid = data[i]
        
        return data
```

### Strategy 4: Interpolation
```python
class Interpolator:
    """
    Interpolate missing values
    
    Examples
    --------
    >>> interpolator = Interpolator(method='linear')
    >>> data = [1.0, 2.0, np.nan, np.nan, 5.0]
    >>> filled = interpolator.transform(data)
    >>> print(filled)  # [1. 2. 3. 4. 5.]
    """
    
    def __init__(self, method: Literal['linear', 'polynomial', 'spline'] = 'linear'):
        self.method = method
    
    def transform(self, data):
        """Interpolate missing values"""
        from scipy import interpolate
        
        data = np.asarray(data, dtype=float).copy()
        
        # Find valid data points
        valid_mask = ~np.isnan(data)
        valid_indices = np.where(valid_mask)[0]
        valid_values = data[valid_mask]
        
        if len(valid_indices) < 2:
            # Not enough points to interpolate
            return data
        
        # Create interpolation function
        if self.method == 'linear':
            f = interpolate.interp1d(
                valid_indices, 
                valid_values, 
                kind='linear',
                fill_value='extrapolate'
            )
        elif self.method == 'polynomial':
            f = interpolate.interp1d(
                valid_indices, 
                valid_values, 
                kind='cubic',
                fill_value='extrapolate'
            )
        
        # Fill missing values
        missing_indices = np.where(~valid_mask)[0]
        data[missing_indices] = f(missing_indices)
        
        return data
```

### Strategy 5: Predictive Imputation (ML-based)
```python
from sklearn.linear_model import LinearRegression

class PredictiveImputer:
    """
    Use ML to predict missing values
    
    Examples
    --------
    >>> # Multivariate data
    >>> data = np.array([
    ...     [1.0, 2.0, 3.0],
    ...     [2.0, np.nan, 5.0],
    ...     [3.0, 4.0, 7.0],
    ...     [4.0, np.nan, 9.0]
    ... ])
    >>> imputer = PredictiveImputer()
    >>> filled = imputer.fit_transform(data, target_col=1)
    """
    
    def __init__(self, model=None):
        self.model = model or LinearRegression()
    
    def fit_transform(self, data, target_col):
        """
        Predict missing values in target column using other columns
        
        Parameters
        ----------
        data : np.ndarray
            2D array with shape (n_samples, n_features)
        target_col : int
            Column index to impute
        """
        data = np.asarray(data, dtype=float).copy()
        
        # Identify complete rows
        complete_mask = ~np.isnan(data).any(axis=1)
        
        if complete_mask.sum() < 2:
            # Not enough complete rows
            # Fallback to mean imputation
            col_mean = np.nanmean(data[:, target_col])
            data[np.isnan(data[:, target_col]), target_col] = col_mean
            return data
        
        # Train on complete rows
        X_train = np.delete(data[complete_mask], target_col, axis=1)
        y_train = data[complete_mask, target_col]
        
        self.model.fit(X_train, y_train)
        
        # Predict for rows with missing target
        missing_mask = np.isnan(data[:, target_col])
        if missing_mask.any():
            X_pred = np.delete(data[missing_mask], target_col, axis=1)
            
            # Check if X_pred has any NaN
            if not np.isnan(X_pred).any():
                predictions = self.model.predict(X_pred)
                data[missing_mask, target_col] = predictions
        
        return data
```

---

## Custom Missing Data Handlers

### Complete Handler Class
```python
from channelpy.core import State, EMPTY, DELTA, PHI, PSI

class MissingDataHandler:
    """
    Complete missing data handling pipeline
    
    Detects and handles missing data with multiple strategies
    
    Examples
    --------
    >>> handler = MissingDataHandler(
    ...     detect_method='auto',
    ...     handle_method='impute',
    ...     impute_strategy='mean'
    ... )
    >>> 
    >>> # Fit on training data
    >>> handler.fit(train_data)
    >>> 
    >>> # Transform test data
    >>> clean_data, info = handler.transform(test_data)
    >>> print(f"Handled {info['num_missing']} missing values")
    """
    
    def __init__(
        self,
        detect_method: str = 'auto',
        handle_method: str = 'impute',
        impute_strategy: str = 'mean',
        sentinel_values: List[float] = None
    ):
        self.detect_method = detect_method
        self.handle_method = handle_method
        self.impute_strategy = impute_strategy
        self.sentinel_values = sentinel_values or []
        
        # Will be set during fit
        self.imputer = None
        self.is_fitted = False
    
    def fit(self, data):
        """Learn parameters from data"""
        data = np.asarray(data, dtype=float)
        
        # Set up imputer if needed
        if self.handle_method == 'impute':
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
            self.imputer.fit(data)
        
        self.is_fitted = True
        return self
    
    def detect(self, data):
        """
        Detect missing values
        
        Returns
        -------
        mask : np.ndarray
            Boolean mask: True where missing
        reasons : dict
            Reasons for each missing value
        """
        data = np.asarray(data, dtype=float)
        mask = np.zeros(len(data), dtype=bool)
        reasons = {}
        
        for i, value in enumerate(data):
            # Check NaN
            if np.isnan(value):
                mask[i] = True
                reasons[i] = "NaN"
                continue
            
            # Check sentinel values
            for sentinel in self.sentinel_values:
                if value == sentinel:
                    mask[i] = True
                    reasons[i] = f"Sentinel value ({sentinel})"
                    break
        
        return mask, reasons
    
    def handle(self, data, mask):
        """
        Handle missing values
        
        Returns
        -------
        handled_data : np.ndarray
            Data with missing values handled
        method_used : str
            Method that was used
        """
        data = np.asarray(data, dtype=float).copy()
        
        if not mask.any():
            return data, "none_missing"
        
        if self.handle_method == 'drop':
            return data[~mask], "dropped"
        
        elif self.handle_method == 'impute':
            if not self.is_fitted:
                raise RuntimeError("Handler not fitted")
            return self.imputer.transform(data), f"imputed_{self.impute_strategy}"
        
        elif self.handle_method == 'mark':
            # Keep missing as is but mark with flag
            # Return both data and mask
            return data, "marked"
        
        else:
            raise ValueError(f"Unknown handle method: {self.handle_method}")
    
    def transform(self, data):
        """
        Complete transformation: detect + handle
        
        Returns
        -------
        clean_data : np.ndarray
            Transformed data
        info : dict
            Information about transformation
        """
        mask, reasons = self.detect(data)
        handled_data, method = self.handle(data, mask)
        
        info = {
            'num_missing': mask.sum(),
            'missing_indices': np.where(mask)[0].tolist(),
            'reasons': reasons,
            'method_used': method,
            'original_length': len(data),
            'final_length': len(handled_data)
        }
        
        return handled_data, info
    
    def fit_transform(self, data):
        """Fit and transform"""
        return self.fit(data).transform(data)
```

---

## Integration with Pipelines

### Example: Complete Pipeline with Missing Data Handling
```python
from channelpy.pipeline import ChannelPipeline
from channelpy.pipeline.encoders import ThresholdEncoder

# Create handler
missing_handler = MissingDataHandler(
    detect_method='auto',
    handle_method='impute',
    impute_strategy='median'
)

# Create pipeline
pipeline = ChannelPipeline()

# Add as preprocessor
def handle_missing_wrapper(data):
    """Wrapper to use with pipeline"""
    clean_data, info = missing_handler.transform(data)
    if info['num_missing'] > 0:
        print(f"Handled {info['num_missing']} missing values using {info['method_used']}")
    return clean_data

pipeline.add_preprocessor(missing_handler.fit_transform)  # Fit on first call
pipeline.add_encoder(ThresholdEncoder())

# Use pipeline
pipeline.fit(train_data)
decisions, states = pipeline.transform(test_data)
```

### Example: Mark Missing as Special State
```python
from channelpy.core import State, EMPTY

def encode_with_missing_awareness(value, threshold_i, threshold_q):
    """
    Encode value, treating missing as EMPTY state
    
    This makes missing data explicit in the channel representation
    """
    # Check if missing
    if np.isnan(value) or value == -999:
        return EMPTY  # Missing → ∅
    
    # Normal encoding
    return State(
        i=int(value > threshold_i),
        q=int(value > threshold_q)
    )

# Usage in pipeline
def create_missing_aware_encoder(threshold_i=0.5, threshold_q=0.75):
    """Factory for missing-aware encoder"""
    def encoder(value):
        return encode_with_missing_awareness(value, threshold_i, threshold_q)
    return encoder

encoder = create_missing_aware_encoder()
```

---

## Best Practices

### 1. **Understand Your Missing Data**

Analyze patterns before choosing strategy:
```python
def analyze_missing(data):
    """Comprehensive missing data analysis"""
    data = np.asarray(data, dtype=float)
    mask = np.isnan(data)
    
    analysis = {
        'total_samples': len(data),
        'missing_count': mask.sum(),
        'missing_percentage': 100 * mask.sum() / len(data),
        'missing_indices': np.where(mask)[0].tolist(),
        'longest_gap': find_longest_gap(mask),
        'pattern': detect_pattern(mask)  # Random, systematic, etc.
    }
    
    return analysis

def find_longest_gap(mask):
    """Find longest consecutive missing sequence"""
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i, is_missing in enumerate(mask):
        if is_missing and not in_gap:
            gap_start = i
            in_gap = True
        elif not is_missing and in_gap:
            gaps.append(i - gap_start)
            in_gap = False
    
    return max(gaps) if gaps else 0
```

### 2. **Choose Strategy Based on Impact**

| Missing % | Recommended Strategy |
|-----------|---------------------|
| < 5%      | Drop or simple imputation |
| 5-20%     | Mean/median imputation |
| 20-40%    | Forward fill (time series) or ML imputation |
| > 40%     | Collect more data! Or mark explicitly |

### 3. **Validate Imputation Quality**
```python
def validate_imputation(original, imputed):
    """
    Test if imputation makes sense
    """
    # Check if imputed values are within reasonable range
    original_clean = original[~np.isnan(original)]
    
    valid_min = np.percentile(original_clean, 5)
    valid_max = np.percentile(original_clean, 95)
    
    imputed_mask = np.isnan(original)
    imputed_values = imputed[imputed_mask]
    
    out_of_range = (imputed_values < valid_min) | (imputed_values > valid_max)
    
    print(f"Imputed {len(imputed_values)} values")
    print(f"Out of range: {out_of_range.sum()} ({100*out_of_range.mean():.1f}%)")
    
    return out_of_range.mean() < 0.1  # Less than 10% out of range
```

### 4. **Track Missing Data in Production**
```python
class ProductionMissingDataHandler(MissingDataHandler):
    """Handler with logging and monitoring"""
    
    def __init__(self, *args, alert_threshold=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.alert_threshold = alert_threshold
        self.stats = []
    
    def transform(self, data):
        """Transform with monitoring"""
        clean_data, info = super().transform(data)
        
        # Record statistics
        self.stats.append({
            'timestamp': time.time(),
            'missing_rate': info['num_missing'] / info['original_length'],
            'method': info['method_used']
        })
        
        # Alert if missing rate too high
        if info['num_missing'] / info['original_length'] > self.alert_threshold:
            self._send_alert(info)
        
        return clean_data, info
    
    def _send_alert(self, info):
        """Send alert about high missing rate"""
        print(f"⚠️  HIGH MISSING DATA RATE: {info['num_missing']}/{info['original_length']} "
              f"({100*info['num_missing']/info['original_length']:.1f}%)")
```

### 5. **Test on Edge Cases**
```python
def test_missing_handler():
    """Test handler on edge cases"""
    handler = MissingDataHandler(handle_method='impute', impute_strategy='mean')
    
    # Test 1: All missing
    all_missing = np.array([np.nan] * 10)
    result, info = handler.fit_transform(all_missing)
    assert not np.any(np.isnan(result)), "Failed on all missing"
    
    # Test 2: No missing
    no_missing = np.array([1.0, 2.0, 3.0, 4.0])
    result, info = handler.transform(no_missing)
    assert info['num_missing'] == 0
    assert np.array_equal(result, no_missing)
    
    # Test 3: Single valid value
    one_valid = np.array([np.nan, 5.0, np.nan])
    result, info = handler.transform(one_valid)
    assert np.all(result == 5.0), "Should fill with only valid value"
    
    print("All edge case tests passed!")
```

---

## Summary

You've learned:
- ✅ Types of missing data (explicit, sentinel, structural, temporal)
- ✅ Detection strategies (simple, domain-specific)
- ✅ Handling strategies (drop, impute, interpolate, predict)
- ✅ Custom handlers with full control
- ✅ Integration with channel pipelines
- ✅ Best practices for production use

**Next Steps:**
- See [Debug Pipeline](debug_pipeline.md) for troubleshooting
- See [Custom Encoder](custom_encoder.md) for encoding strategies
- Check [API Reference](../api_reference/pipeline.md) for full details