"""
Example datasets and data generators

Provides synthetic and real datasets for testing and tutorials
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from ..core.state import State, StateArray


def make_classification_data(n_samples: int = 1000, n_features: int = 2,
                             n_classes: int = 2, noise: float = 0.1,
                             random_state: Optional[int] = None) -> Tuple:
    """
    Generate synthetic classification data for channel encoding
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    noise : float
        Noise level (0 to 1)
    random_state : int, optional
        Random seed
        
    Returns
    -------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    
    Examples
    --------
    >>> X, y = make_classification_data(n_samples=500, n_features=3)
    >>> print(X.shape, y.shape)
    (500, 3) (500,)
    """
    rng = np.random.RandomState(random_state)
    
    # Generate centers for each class
    centers = rng.randn(n_classes, n_features) * 3
    
    # Generate samples
    samples_per_class = n_samples // n_classes
    X = []
    y = []
    
    for class_idx in range(n_classes):
        # Generate samples around center
        class_samples = rng.randn(samples_per_class, n_features) + centers[class_idx]
        # Add noise
        class_samples += rng.randn(samples_per_class, n_features) * noise
        
        X.append(class_samples)
        y.append(np.full(samples_per_class, class_idx))
    
    X = np.vstack(X)
    y = np.concatenate(y)
    
    # Shuffle
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


def make_time_series_data(n_samples: int = 1000, trend: float = 0.001,
                          seasonality: float = 0.1, noise: float = 0.05,
                          random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate synthetic time series data
    
    Parameters
    ----------
    n_samples : int
        Length of time series
    trend : float
        Linear trend coefficient
    seasonality : float
        Seasonal component amplitude
    noise : float
        Random noise standard deviation
    random_state : int, optional
        Random seed
        
    Returns
    -------
    ts : np.ndarray
        Time series values (n_samples,)
        
    Examples
    --------
    >>> ts = make_time_series_data(n_samples=1000, trend=0.01)
    >>> print(ts.shape)
    (1000,)
    """
    rng = np.random.RandomState(random_state)
    
    t = np.arange(n_samples)
    
    # Trend component
    trend_component = trend * t
    
    # Seasonal component
    seasonal_component = seasonality * np.sin(2 * np.pi * t / 50)
    
    # Noise
    noise_component = rng.randn(n_samples) * noise
    
    # Combine
    ts = 10 + trend_component + seasonal_component + noise_component
    
    return ts


def make_regime_change_data(n_samples: int = 1000, n_regimes: int = 3,
                            random_state: Optional[int] = None) -> Tuple:
    """
    Generate time series with regime changes
    
    Useful for testing adaptive thresholds
    
    Parameters
    ----------
    n_samples : int
        Total length
    n_regimes : int
        Number of regimes
    random_state : int, optional
        Random seed
        
    Returns
    -------
    ts : np.ndarray
        Time series (n_samples,)
    regimes : np.ndarray
        Regime labels (n_samples,)
    change_points : list
        Indices of regime changes
        
    Examples
    --------
    >>> ts, regimes, changes = make_regime_change_data(n_samples=1000)
    >>> print(f"Regimes change at: {changes}")
    """
    rng = np.random.RandomState(random_state)
    
    # Divide into regimes
    regime_length = n_samples // n_regimes
    
    ts = []
    regimes = []
    change_points = []
    
    for regime_idx in range(n_regimes):
        # Each regime has different mean and std
        mean = rng.randn() * 5 + 10
        std = rng.rand() * 2 + 0.5
        
        regime_data = rng.randn(regime_length) * std + mean
        regime_labels = np.full(regime_length, regime_idx)
        
        ts.append(regime_data)
        regimes.append(regime_labels)
        
        if regime_idx > 0:
            change_points.append(regime_idx * regime_length)
    
    ts = np.concatenate(ts)
    regimes = np.concatenate(regimes)
    
    return ts, regimes, change_points


def make_trading_data(n_samples: int = 1000, volatility: float = 0.02,
                     drift: float = 0.0005, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic trading data (OHLCV)
    
    Parameters
    ----------
    n_samples : int
        Number of time bars
    volatility : float
        Price volatility
    drift : float
        Average return per bar
    random_state : int, optional
        Random seed
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: open, high, low, close, volume
        
    Examples
    --------
    >>> df = make_trading_data(n_samples=500)
    >>> print(df.head())
    """
    rng = np.random.RandomState(random_state)
    
    # Generate price series using geometric Brownian motion
    returns = rng.randn(n_samples) * volatility + drift
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from prices
    df = pd.DataFrame()
    df['close'] = prices
    
    # Open is close shifted
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    # High and low
    intrabar_range = rng.rand(n_samples) * volatility * prices
    df['high'] = np.maximum(df['open'], df['close']) + intrabar_range / 2
    df['low'] = np.minimum(df['open'], df['close']) - intrabar_range / 2
    
    # Volume (log-normal distribution)
    df['volume'] = np.exp(rng.randn(n_samples) * 0.5 + 15)
    
    return df


def make_medical_data(n_samples: int = 500, disease_prevalence: float = 0.2,
                     test_sensitivity: float = 0.9, test_specificity: float = 0.95,
                     random_state: Optional[int] = None) -> Tuple:
    """
    Generate synthetic medical diagnosis data
    
    Parameters
    ----------
    n_samples : int
        Number of patients
    disease_prevalence : float
        Proportion with disease
    test_sensitivity : float
        True positive rate
    test_specificity : float
        True negative rate
    random_state : int, optional
        Random seed
        
    Returns
    -------
    symptoms : np.ndarray
        Symptom severity scores (n_samples, 3)
    test_results : np.ndarray
        Test results (n_samples,)
    true_labels : np.ndarray
        True disease status (n_samples,)
        
    Examples
    --------
    >>> symptoms, tests, labels = make_medical_data(n_samples=500)
    >>> print(f"Prevalence: {labels.mean():.2f}")
    """
    rng = np.random.RandomState(random_state)
    
    # True disease status
    true_labels = rng.rand(n_samples) < disease_prevalence
    
    # Symptoms (higher when diseased)
    symptoms = np.zeros((n_samples, 3))
    for i in range(n_samples):
        if true_labels[i]:
            # Diseased: higher symptom scores
            symptoms[i] = rng.rand(3) * 0.5 + 0.5
        else:
            # Healthy: lower symptom scores
            symptoms[i] = rng.rand(3) * 0.4
    
    # Test results (with sensitivity and specificity)
    test_results = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        if true_labels[i]:
            # True positive
            test_results[i] = rng.rand() < test_sensitivity
        else:
            # False positive
            test_results[i] = rng.rand() > test_specificity
    
    return symptoms, test_results, true_labels


def make_state_sequence(length: int = 100, transition_probs: dict = None,
                       initial_state: State = None, 
                       random_state: Optional[int] = None) -> StateArray:
    """
    Generate random sequence of states with Markov transitions
    
    Parameters
    ----------
    length : int
        Length of sequence
    transition_probs : dict, optional
        Transition probabilities {from_state: {to_state: prob}}
    initial_state : State, optional
        Starting state
    random_state : int, optional
        Random seed
        
    Returns
    -------
    states : StateArray
        Sequence of states
        
    Examples
    --------
    >>> states = make_state_sequence(length=100)
    >>> print(states.count_by_state())
    """
    from ..core.state import EMPTY, DELTA, PHI, PSI
    
    rng = np.random.RandomState(random_state)
    
    # Default transition probabilities (slightly persistent)
    if transition_probs is None:
        transition_probs = {
            EMPTY: {EMPTY: 0.7, PHI: 0.15, DELTA: 0.1, PSI: 0.05},
            PHI: {EMPTY: 0.1, PHI: 0.6, DELTA: 0.1, PSI: 0.2},
            DELTA: {EMPTY: 0.2, PHI: 0.1, DELTA: 0.6, PSI: 0.1},
            PSI: {EMPTY: 0.05, PHI: 0.15, DELTA: 0.1, PSI: 0.7}
        }
    
    # Initial state
    if initial_state is None:
        initial_state = PSI
    
    # Generate sequence
    states = [initial_state]
    current = initial_state
    
    for _ in range(length - 1):
        # Get transition probabilities
        probs = transition_probs[current]
        next_states = list(probs.keys())
        next_probs = list(probs.values())
        
        # Sample next state
        current = rng.choice(next_states, p=next_probs)
        states.append(current)
    
    return StateArray.from_states(states)


def load_example_dataset(name: str) -> Tuple:
    """
    Load pre-defined example dataset
    
    Parameters
    ----------
    name : str
        Dataset name:
        - 'iris_simple': Simplified iris for quick testing
        - 'trading_sample': Sample trading data
        - 'medical_sample': Sample medical data
        
    Returns
    -------
    data : tuple
        Dataset (format depends on dataset)
        
    Examples
    --------
    >>> X, y = load_example_dataset('iris_simple')
    """
    if name == 'iris_simple':
        # Simplified iris dataset (2 features, 2 classes)
        return make_classification_data(
            n_samples=150, 
            n_features=2,
            n_classes=2,
            noise=0.2,
            random_state=42
        )
    
    elif name == 'trading_sample':
        return make_trading_data(
            n_samples=252,  # One trading year
            volatility=0.02,
            drift=0.0005,
            random_state=42
        )
    
    elif name == 'medical_sample':
        return make_medical_data(
            n_samples=500,
            disease_prevalence=0.2,
            test_sensitivity=0.9,
            test_specificity=0.95,
            random_state=42
        )
    
    else:
        raise ValueError(f"Unknown dataset: {name}")


def generate_streaming_data(base_value: float = 10.0, volatility: float = 0.1,
                           drift: float = 0.0):
    """
    Infinite generator of streaming data
    
    Parameters
    ----------
    base_value : float
        Starting value
    volatility : float
        Random volatility
    drift : float
        Upward/downward drift
        
    Yields
    ------
    value : float
        Next data point
        
    Examples
    --------
    >>> stream = generate_streaming_data(base_value=100, volatility=0.02)
    >>> for i, value in enumerate(stream):
    ...     if i >= 1000:
    ...         break
    ...     process(value)
    """
    current = base_value
    
    while True:
        # Random walk with drift
        change = np.random.randn() * volatility + drift
        current = current * (1 + change)
        
        yield current




