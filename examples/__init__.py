"""
Examples module for channel algebra.

Contains example datasets, data generators, and tutorial utilities.
"""

from .datasets import (
    make_classification_data, make_time_series_data, make_regime_change_data,
    make_trading_data, make_medical_data, make_state_sequence,
    load_example_dataset, generate_streaming_data
)

__all__ = [
    'make_classification_data', 'make_time_series_data', 'make_regime_change_data',
    'make_trading_data', 'make_medical_data', 'make_state_sequence',
    'load_example_dataset', 'generate_streaming_data',
]




