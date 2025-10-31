"""
Utilities module for channel algebra.

Contains validation, serialization, and helper utilities.
"""

from .validation import (
    ValidationError, validate_state_bits, validate_threshold, validate_threshold_pair,
    validate_array_shape, validate_probability, validate_window_size, validate_positive,
    validate_non_negative, validate_fitted, validate_same_length, validate_input,
    require_fitted, validate_state_input, validate_array_input, ValidationContext,
    validate_pipeline_data, validate_encoder_thresholds
)
from .serialization import (
    ChannelPyEncoder, channelpy_object_hook, save_state, load_state,
    save_state_array, load_state_array, save_pipeline, load_pipeline,
    save_nested_state, load_nested_state, save_parallel_channels, load_parallel_channels,
    to_dict, from_dict, save, load
)
from .examples import (
    generate_sample_trading_data, generate_sample_medical_data, generate_sample_signal_data,
    generate_sample_text_data, generate_sample_state_sequence, generate_sample_nested_states,
    generate_sample_parallel_channels, create_sample_pipeline_data, analyze_state_distribution,
    find_state_patterns, calculate_state_transitions
)

__all__ = [
    # Validation
    'ValidationError', 'validate_state_bits', 'validate_threshold', 'validate_threshold_pair',
    'validate_array_shape', 'validate_probability', 'validate_window_size', 'validate_positive',
    'validate_non_negative', 'validate_fitted', 'validate_same_length', 'validate_input',
    'require_fitted', 'validate_state_input', 'validate_array_input', 'ValidationContext',
    'validate_pipeline_data', 'validate_encoder_thresholds',
    
    # Serialization
    'ChannelPyEncoder', 'channelpy_object_hook', 'save_state', 'load_state',
    'save_state_array', 'load_state_array', 'save_pipeline', 'load_pipeline',
    'save_nested_state', 'load_nested_state', 'save_parallel_channels', 'load_parallel_channels',
    'to_dict', 'from_dict', 'save', 'load',
    
    # Examples
    'generate_sample_trading_data', 'generate_sample_medical_data', 'generate_sample_signal_data',
    'generate_sample_text_data', 'generate_sample_state_sequence', 'generate_sample_nested_states',
    'generate_sample_parallel_channels', 'create_sample_pipeline_data', 'analyze_state_distribution',
    'find_state_patterns', 'calculate_state_transitions',
]
