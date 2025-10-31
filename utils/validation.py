"""
Input validation utilities

Decorators and functions to validate inputs to channel algebra operations
"""
from typing import Any, Callable, Union, get_type_hints
import numpy as np
from functools import wraps
import inspect


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


def validate_state_bits(i: int, q: int):
    """
    Validate that i and q are valid bits (0 or 1)
    
    Parameters
    ----------
    i : int
        Presence bit
    q : int
        Membership bit
        
    Raises
    ------
    ValidationError
        If bits are invalid
    """
    if i not in (0, 1):
        raise ValidationError(f"i-bit must be 0 or 1, got {i}")
    if q not in (0, 1):
        raise ValidationError(f"q-bit must be 0 or 1, got {q}")


def validate_threshold(threshold: float, name: str = "threshold"):
    """
    Validate threshold value
    
    Parameters
    ----------
    threshold : float
        Threshold value
    name : str
        Name of threshold (for error messages)
        
    Raises
    ------
    ValidationError
        If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError(
            f"{name} must be numeric, got {type(threshold)}"
        )
    
    if not np.isfinite(threshold):
        raise ValidationError(
            f"{name} must be finite, got {threshold}"
        )


def validate_threshold_pair(threshold_i: float, threshold_q: float):
    """
    Validate threshold pair
    
    For proper encoding, should have threshold_i <= threshold_q
    
    Parameters
    ----------
    threshold_i : float
        Threshold for i-bit
    threshold_q : float
        Threshold for q-bit
        
    Raises
    ------
    ValidationError
        If thresholds are invalid or inconsistent
    """
    validate_threshold(threshold_i, "threshold_i")
    validate_threshold(threshold_q, "threshold_q")
    
    if threshold_i > threshold_q:
        raise ValidationError(
            f"threshold_i ({threshold_i}) should be <= threshold_q ({threshold_q})"
        )


def validate_array_shape(arr: np.ndarray, expected_shape: tuple = None,
                        min_dims: int = None, max_dims: int = None):
    """
    Validate numpy array shape
    
    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    expected_shape : tuple, optional
        Expected shape (None for any dimension)
    min_dims : int, optional
        Minimum number of dimensions
    max_dims : int, optional
        Maximum number of dimensions
        
    Raises
    ------
    ValidationError
        If shape is invalid
    """
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"Expected numpy array, got {type(arr)}")
    
    if expected_shape is not None:
        if len(expected_shape) != arr.ndim:
            raise ValidationError(
                f"Expected {len(expected_shape)}D array, got {arr.ndim}D"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, arr.shape)):
            if expected is not None and expected != actual:
                raise ValidationError(
                    f"Dimension {i}: expected {expected}, got {actual}"
                )
    
    if min_dims is not None and arr.ndim < min_dims:
        raise ValidationError(
            f"Array must have at least {min_dims} dimensions, got {arr.ndim}"
        )
    
    if max_dims is not None and arr.ndim > max_dims:
        raise ValidationError(
            f"Array must have at most {max_dims} dimensions, got {arr.ndim}"
        )


def validate_probability(p: float, name: str = "probability"):
    """
    Validate probability value (must be in [0, 1])
    
    Parameters
    ----------
    p : float
        Probability value
    name : str
        Name for error messages
        
    Raises
    ------
    ValidationError
        If probability is invalid
    """
    if not isinstance(p, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(p)}")
    
    if not (0 <= p <= 1):
        raise ValidationError(f"{name} must be in [0, 1], got {p}")


def validate_window_size(window_size: int):
    """
    Validate window size for rolling computations
    
    Parameters
    ----------
    window_size : int
        Window size
        
    Raises
    ------
    ValidationError
        If window size is invalid
    """
    if not isinstance(window_size, int):
        raise ValidationError(
            f"window_size must be integer, got {type(window_size)}"
        )
    
    if window_size < 1:
        raise ValidationError(
            f"window_size must be positive, got {window_size}"
        )


def validate_positive(value: Union[int, float], name: str = "value"):
    """
    Validate that value is positive
    
    Parameters
    ----------
    value : numeric
        Value to validate
    name : str
        Name for error messages
        
    Raises
    ------
    ValidationError
        If value is not positive
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(value: Union[int, float], name: str = "value"):
    """
    Validate that value is non-negative
    
    Parameters
    ----------
    value : numeric
        Value to validate
    name : str
        Name for error messages
        
    Raises
    ------
    ValidationError
        If value is negative
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_fitted(obj, method_name: str = "transform"):
    """
    Validate that object has been fitted
    
    Parameters
    ----------
    obj : object
        Object to check
    method_name : str
        Name of method being called
        
    Raises
    ------
    ValidationError
        If object not fitted
    """
    if not hasattr(obj, 'is_fitted') or not obj.is_fitted:
        raise ValidationError(
            f"Object must be fitted before calling {method_name}(). "
            f"Call fit() first."
        )


def validate_same_length(*arrays, names: list = None):
    """
    Validate that arrays have same length
    
    Parameters
    ----------
    *arrays : array-like
        Arrays to check
    names : list, optional
        Names of arrays for error messages
        
    Raises
    ------
    ValidationError
        If arrays have different lengths
    """
    if not arrays:
        return
    
    lengths = [len(arr) for arr in arrays]
    
    if len(set(lengths)) > 1:
        if names is None:
            names = [f"array_{i}" for i in range(len(arrays))]
        
        length_info = ", ".join(
            f"{name}={length}" 
            for name, length in zip(names, lengths)
        )
        raise ValidationError(
            f"Arrays must have same length, got {length_info}"
        )


# Decorators for automatic validation

def validate_input(**validators):
    """
    Decorator to validate function inputs
    
    Examples
    --------
    >>> @validate_input(threshold=validate_threshold, 
    ...                  window_size=validate_window_size)
    ... def my_function(threshold, window_size):
    ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each argument
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    validator(value)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_fitted(method):
    """
    Decorator to require that object is fitted
    
    Examples
    --------
    >>> class MyEncoder:
    ...     @require_fitted
    ...     def transform(self, X):
    ...         pass
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        validate_fitted(self, method.__name__)
        return method(self, *args, **kwargs)
    
    return wrapper


def validate_state_input(func):
    """
    Decorator to validate State inputs
    
    Checks that State objects have valid bits
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from ..core.state import State, StateArray
        
        # Check all State arguments
        for arg in args:
            if isinstance(arg, State):
                validate_state_bits(arg.i, arg.q)
        
        for value in kwargs.values():
            if isinstance(value, State):
                validate_state_bits(value.i, value.q)
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_array_input(min_dims: int = None, max_dims: int = None):
    """
    Decorator to validate array inputs
    
    Examples
    --------
    >>> @validate_array_input(min_dims=1, max_dims=2)
    ... def my_function(X):
    ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check array arguments
            for arg in args:
                if isinstance(arg, np.ndarray):
                    validate_array_shape(arg, min_dims=min_dims, max_dims=max_dims)
            
            for value in kwargs.values():
                if isinstance(value, np.ndarray):
                    validate_array_shape(value, min_dims=min_dims, max_dims=max_dims)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Context managers for validation

class ValidationContext:
    """
    Context manager for validation
    
    Examples
    --------
    >>> with ValidationContext("Processing data"):
    ...     validate_threshold(threshold)
    ...     validate_window_size(window)
    """
    
    def __init__(self, context_name: str = "Validation"):
        self.context_name = context_name
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ValidationError:
            # Re-raise with context
            raise ValidationError(
                f"{self.context_name}: {exc_val}"
            ) from exc_val
        return False


# Batch validation utilities

def validate_pipeline_data(X, y=None):
    """
    Validate data for pipeline processing
    
    Parameters
    ----------
    X : array-like
        Feature data
    y : array-like, optional
        Target data
        
    Raises
    ------
    ValidationError
        If data is invalid
    """
    # Convert to numpy
    X = np.asarray(X)
    
    # Check for NaN/inf
    if not np.all(np.isfinite(X)):
        raise ValidationError("X contains NaN or inf values")
    
    # Check shape
    if X.ndim < 1:
        raise ValidationError("X must be at least 1D")
    
    # Validate y if provided
    if y is not None:
        y = np.asarray(y)
        
        if not np.all(np.isfinite(y)):
            raise ValidationError("y contains NaN or inf values")
        
        # Check length match
        if len(X) != len(y):
            raise ValidationError(
                f"X and y must have same length: {len(X)} vs {len(y)}"
            )


def validate_encoder_thresholds(threshold_i, threshold_q):
    """
    Validate encoder thresholds
    
    Convenience function for common validation
    """
    with ValidationContext("Encoder threshold validation"):
        validate_threshold_pair(threshold_i, threshold_q)




