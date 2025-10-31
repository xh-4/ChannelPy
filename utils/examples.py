"""
Example datasets and utilities for ChannelPy

Provides sample data for testing and demonstration
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI


def generate_sample_trading_data(n_samples: int = 1000, 
                                trend_strength: float = 0.1,
                                noise_level: float = 0.05,
                                volatility_clusters: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample trading data (prices and volumes)
    
    Parameters
    ----------
    n_samples : int
        Number of data points
    trend_strength : float
        Strength of price trend
    noise_level : float
        Level of random noise
    volatility_clusters : bool
        Whether to include volatility clustering
        
    Returns
    -------
    prices : np.ndarray
        Price series
    volumes : np.ndarray
        Volume series
    """
    np.random.seed(42)
    
    # Generate price trend
    trend = np.linspace(100, 100 + trend_strength * n_samples, n_samples)
    
    # Add noise
    noise = np.random.normal(0, noise_level * 100, n_samples)
    
    # Add volatility clustering if requested
    if volatility_clusters:
        # GARCH-like volatility clustering
        volatility = np.zeros(n_samples)
        volatility[0] = noise_level
        for i in range(1, n_samples):
            volatility[i] = 0.1 + 0.8 * volatility[i-1] + 0.1 * np.random.normal(0, 0.1)
        noise *= volatility
    
    prices = trend + noise
    
    # Generate volume (correlated with price changes)
    price_changes = np.abs(np.diff(prices, prepend=prices[0]))
    volume_base = np.random.exponential(1000, n_samples)
    volume_spike = price_changes * 50
    volumes = volume_base + volume_spike
    
    return prices, volumes


def generate_sample_medical_data(n_patients: int = 500,
                                n_features: int = 10,
                                disease_prevalence: float = 0.3) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate sample medical diagnosis data
    
    Parameters
    ----------
    n_patients : int
        Number of patients
    n_features : int
        Number of medical features
    disease_prevalence : float
        Prevalence of disease (0-1)
        
    Returns
    -------
    X : np.ndarray
        Feature matrix (patients x features)
    y : np.ndarray
        Disease labels (0=healthy, 1=diseased)
    feature_names : list
        Names of medical features
    """
    np.random.seed(42)
    
    # Feature names
    feature_names = [
        'age', 'blood_pressure', 'heart_rate', 'cholesterol', 'glucose',
        'weight', 'height', 'temperature', 'oxygen_saturation', 'pain_level'
    ][:n_features]
    
    # Generate features
    X = np.random.normal(0, 1, (n_patients, n_features))
    
    # Add some structure
    X[:, 0] += np.random.normal(50, 15, n_patients)  # Age
    X[:, 1] += np.random.normal(120, 20, n_patients)  # Blood pressure
    X[:, 2] += np.random.normal(70, 15, n_patients)   # Heart rate
    
    # Generate disease labels
    disease_prob = 1 / (1 + np.exp(-np.sum(X[:, :5], axis=1)))  # Logistic function
    y = (disease_prob > (1 - disease_prevalence)).astype(int)
    
    # Make diseased patients have more extreme values
    diseased_mask = y == 1
    X[diseased_mask] += np.random.normal(0, 0.5, (np.sum(diseased_mask), n_features))
    
    return X, y, feature_names


def generate_sample_signal_data(n_samples: int = 1000,
                               n_frequencies: int = 3,
                               sampling_rate: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample signal processing data
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_frequencies : int
        Number of frequency components
    sampling_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    time : np.ndarray
        Time vector
    signal : np.ndarray
        Signal values
    """
    np.random.seed(42)
    
    # Time vector
    time = np.linspace(0, n_samples / sampling_rate, n_samples)
    
    # Generate multi-frequency signal
    signal = np.zeros(n_samples)
    frequencies = np.random.uniform(1, 10, n_frequencies)
    amplitudes = np.random.uniform(0.5, 2.0, n_frequencies)
    phases = np.random.uniform(0, 2*np.pi, n_frequencies)
    
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * freq * time + phase)
    
    # Add noise
    noise = np.random.normal(0, 0.1, n_samples)
    signal += noise
    
    return time, signal


def generate_sample_text_data(n_documents: int = 100,
                             n_words_per_doc: int = 50,
                             vocabulary_size: int = 1000) -> Tuple[List[str], List[int]]:
    """
    Generate sample text classification data
    
    Parameters
    ----------
    n_documents : int
        Number of documents
    n_words_per_doc : int
        Words per document
    vocabulary_size : int
        Size of vocabulary
        
    Returns
    -------
    documents : list
        List of document texts
    labels : list
        Document labels (0=negative, 1=positive)
    """
    np.random.seed(42)
    
    # Generate vocabulary
    vocabulary = [f"word_{i}" for i in range(vocabulary_size)]
    
    # Generate documents
    documents = []
    labels = []
    
    for i in range(n_documents):
        # Random label
        label = np.random.randint(0, 2)
        
        # Generate words (biased by label)
        if label == 1:  # Positive documents
            # More likely to use words from first half of vocabulary
            word_probs = np.ones(vocabulary_size)
            word_probs[:vocabulary_size//2] *= 2
        else:  # Negative documents
            # More likely to use words from second half of vocabulary
            word_probs = np.ones(vocabulary_size)
            word_probs[vocabulary_size//2:] *= 2
        
        word_probs /= word_probs.sum()
        words = np.random.choice(vocabulary, n_words_per_doc, p=word_probs)
        
        documents.append(" ".join(words))
        labels.append(label)
    
    return documents, labels


def generate_sample_state_sequence(n_states: int = 100,
                                  transition_prob: float = 0.3) -> StateArray:
    """
    Generate sample state sequence with transitions
    
    Parameters
    ----------
    n_states : int
        Number of states to generate
    transition_prob : float
        Probability of state transition
        
    Returns
    -------
    StateArray
        Sequence of states
    """
    np.random.seed(42)
    
    states = [PSI]  # Start with PSI
    current_state = PSI
    
    for _ in range(n_states - 1):
        if np.random.random() < transition_prob:
            # Transition to different state
            if current_state == PSI:
                current_state = DELTA
            elif current_state == DELTA:
                current_state = PHI
            elif current_state == PHI:
                current_state = EMPTY
            else:  # EMPTY
                current_state = PSI
        
        states.append(current_state)
    
    return StateArray.from_states(states)


def generate_sample_nested_states(n_sequences: int = 50,
                                 max_depth: int = 3) -> List[NestedState]:
    """
    Generate sample nested states
    
    Parameters
    ----------
    n_sequences : int
        Number of nested state sequences
    max_depth : int
        Maximum nesting depth
        
    Returns
    -------
    list
        List of NestedState objects
    """
    np.random.seed(42)
    
    all_states = [EMPTY, DELTA, PHI, PSI]
    nested_states = []
    
    for _ in range(n_sequences):
        depth = np.random.randint(1, max_depth + 1)
        levels = {}
        
        for i in range(depth):
            state = np.random.choice(all_states)
            levels[f'level{i}'] = state
        
        nested_states.append(NestedState(**levels))
    
    return nested_states


def generate_sample_parallel_channels(n_sequences: int = 50,
                                     channel_names: List[str] = None) -> List[ParallelChannels]:
    """
    Generate sample parallel channel data
    
    Parameters
    ----------
    n_sequences : int
        Number of channel sequences
    channel_names : list, optional
        Names of channels
        
    Returns
    -------
    list
        List of ParallelChannels objects
    """
    if channel_names is None:
        channel_names = ['technical', 'business', 'team', 'market']
    
    np.random.seed(42)
    all_states = [EMPTY, DELTA, PHI, PSI]
    channel_sequences = []
    
    for _ in range(n_sequences):
        channels = {}
        for name in channel_names:
            state = np.random.choice(all_states)
            channels[name] = state
        
        channel_sequences.append(ParallelChannels(**channels))
    
    return channel_sequences


def create_sample_pipeline_data() -> Dict[str, Any]:
    """
    Create comprehensive sample data for pipeline testing
    
    Returns
    -------
    dict
        Dictionary containing various sample datasets
    """
    return {
        'trading': {
            'prices': generate_sample_trading_data()[0],
            'volumes': generate_sample_trading_data()[1]
        },
        'medical': {
            'X': generate_sample_medical_data()[0],
            'y': generate_sample_medical_data()[1],
            'feature_names': generate_sample_medical_data()[2]
        },
        'signal': {
            'time': generate_sample_signal_data()[0],
            'signal': generate_sample_signal_data()[1]
        },
        'text': {
            'documents': generate_sample_text_data()[0],
            'labels': generate_sample_text_data()[1]
        },
        'states': {
            'sequence': generate_sample_state_sequence(),
            'nested': generate_sample_nested_states(),
            'parallel': generate_sample_parallel_channels()
        }
    }


# Utility functions for data analysis

def analyze_state_distribution(states: StateArray) -> Dict[str, Any]:
    """
    Analyze distribution of states in StateArray
    
    Parameters
    ----------
    states : StateArray
        States to analyze
        
    Returns
    -------
    dict
        Analysis results
    """
    counts = states.count_by_state()
    total = len(states)
    
    analysis = {
        'total_states': total,
        'counts': {str(k): v for k, v in counts.items()},
        'percentages': {str(k): 100 * v / total for k, v in counts.items()},
        'most_common': max(counts.items(), key=lambda x: x[1]),
        'least_common': min(counts.items(), key=lambda x: x[1])
    }
    
    return analysis


def find_state_patterns(states: StateArray, pattern_length: int = 3) -> Dict[tuple, int]:
    """
    Find common patterns in state sequence
    
    Parameters
    ----------
    states : StateArray
        State sequence
    pattern_length : int
        Length of patterns to find
        
    Returns
    -------
    dict
        Pattern counts
    """
    patterns = {}
    
    for i in range(len(states) - pattern_length + 1):
        pattern = tuple(states[i:i+pattern_length])
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    return patterns


def calculate_state_transitions(states: StateArray) -> Dict[tuple, int]:
    """
    Calculate state transition counts
    
    Parameters
    ----------
    states : StateArray
        State sequence
        
    Returns
    -------
    dict
        Transition counts
    """
    transitions = {}
    
    for i in range(len(states) - 1):
        transition = (states[i], states[i+1])
        transitions[transition] = transitions.get(transition, 0) + 1
    
    return transitions




