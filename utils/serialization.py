"""
Serialization utilities for saving and loading ChannelPy objects

Supports multiple formats:
- Pickle: Full Python object serialization
- JSON: Human-readable, portable format
- Custom: Optimized binary format for large datasets

Handles:
- Individual states and state arrays
- Complete pipelines
- Threshold learners and adaptive systems
- Version compatibility
"""

import pickle
import json
import gzip
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
from datetime import datetime
import warnings

from ..core.state import State, StateArray, EMPTY, DELTA, PHI, PSI
from ..core.nested import NestedState
from ..core.parallel import ParallelChannels


# Version tracking for backwards compatibility
SERIALIZATION_VERSION = "0.1.0"


class SerializationError(Exception):
    """Raised when serialization/deserialization fails"""
    pass


# ============================================================================
# Core State Serialization
# ============================================================================

def state_to_dict(state: State) -> Dict:
    """
    Convert State to dictionary
    
    Parameters
    ----------
    state : State
        State to serialize
        
    Returns
    -------
    data : Dict
        Dictionary representation
    """
    return {
        'type': 'State',
        'i': int(state.i),
        'q': int(state.q),
        'version': SERIALIZATION_VERSION
    }


def dict_to_state(data: Dict) -> State:
    """
    Convert dictionary to State
    
    Parameters
    ----------
    data : Dict
        Dictionary representation
        
    Returns
    -------
    state : State
        Reconstructed state
    """
    if data['type'] != 'State':
        raise SerializationError(f"Expected State, got {data['type']}")
    
    return State(i=data['i'], q=data['q'])


def state_array_to_dict(states: StateArray) -> Dict:
    """
    Convert StateArray to dictionary
    
    Parameters
    ----------
    states : StateArray
        State array to serialize
        
    Returns
    -------
    data : Dict
        Dictionary representation
    """
    return {
        'type': 'StateArray',
        'i': states.i.tolist(),
        'q': states.q.tolist(),
        'shape': states.i.shape,
        'version': SERIALIZATION_VERSION
    }


def dict_to_state_array(data: Dict) -> StateArray:
    """
    Convert dictionary to StateArray
    
    Parameters
    ----------
    data : Dict
        Dictionary representation
        
    Returns
    -------
    states : StateArray
        Reconstructed state array
    """
    if data['type'] != 'StateArray':
        raise SerializationError(f"Expected StateArray, got {data['type']}")
    
    i = np.array(data['i'], dtype=np.int8)
    q = np.array(data['q'], dtype=np.int8)
    
    return StateArray(i=i, q=q)


def nested_state_to_dict(nested: NestedState) -> Dict:
    """
    Convert NestedState to dictionary
    
    Parameters
    ----------
    nested : NestedState
        Nested state to serialize
        
    Returns
    -------
    data : Dict
        Dictionary representation
    """
    levels = {}
    for i in range(nested.num_levels):
        level_state = nested.get_level(i)
        levels[f'level{i}'] = state_to_dict(level_state)
    
    return {
        'type': 'NestedState',
        'levels': levels,
        'depth': nested.depth,
        'version': SERIALIZATION_VERSION
    }


def dict_to_nested_state(data: Dict) -> NestedState:
    """
    Convert dictionary to NestedState
    
    Parameters
    ----------
    data : Dict
        Dictionary representation
        
    Returns
    -------
    nested : NestedState
        Reconstructed nested state
    """
    if data['type'] != 'NestedState':
        raise SerializationError(f"Expected NestedState, got {data['type']}")
    
    levels = {}
    for key, level_data in data['levels'].items():
        levels[key] = dict_to_state(level_data)
    
    return NestedState(**levels)


def parallel_channels_to_dict(parallel: ParallelChannels) -> Dict:
    """
    Convert ParallelChannels to dictionary
    
    Parameters
    ----------
    parallel : ParallelChannels
        Parallel channels to serialize
        
    Returns
    -------
    data : Dict
        Dictionary representation
    """
    channels = {}
    for name, state in parallel.to_dict().items():
        channels[name] = state_to_dict(state)
    
    return {
        'type': 'ParallelChannels',
        'channels': channels,
        'version': SERIALIZATION_VERSION
    }


def dict_to_parallel_channels(data: Dict) -> ParallelChannels:
    """
    Convert dictionary to ParallelChannels
    
    Parameters
    ----------
    data : Dict
        Dictionary representation
        
    Returns
    -------
    parallel : ParallelChannels
        Reconstructed parallel channels
    """
    if data['type'] != 'ParallelChannels':
        raise SerializationError(f"Expected ParallelChannels, got {data['type']}")
    
    channels = {}
    for name, state_data in data['channels'].items():
        channels[name] = dict_to_state(state_data)
    
    return ParallelChannels(**channels)


# ============================================================================
# Generic Object Serialization
# ============================================================================

def to_dict(obj: Any) -> Dict:
    """
    Convert any ChannelPy object to dictionary
    
    Parameters
    ----------
    obj : Any
        Object to serialize
        
    Returns
    -------
    data : Dict
        Dictionary representation
    """
    if isinstance(obj, State):
        return state_to_dict(obj)
    elif isinstance(obj, StateArray):
        return state_array_to_dict(obj)
    elif isinstance(obj, NestedState):
        return nested_state_to_dict(obj)
    elif isinstance(obj, ParallelChannels):
        return parallel_channels_to_dict(obj)
    elif hasattr(obj, 'to_dict'):
        # Object has its own serialization method
        return obj.to_dict()
    else:
        raise SerializationError(f"Cannot serialize object of type {type(obj)}")


def from_dict(data: Dict) -> Any:
    """
    Reconstruct object from dictionary
    
    Parameters
    ----------
    data : Dict
        Dictionary representation
        
    Returns
    -------
    obj : Any
        Reconstructed object
    """
    obj_type = data.get('type')
    
    if obj_type == 'State':
        return dict_to_state(data)
    elif obj_type == 'StateArray':
        return dict_to_state_array(data)
    elif obj_type == 'NestedState':
        return dict_to_nested_state(data)
    elif obj_type == 'ParallelChannels':
        return dict_to_parallel_channels(data)
    else:
        raise SerializationError(f"Unknown object type: {obj_type}")


# ============================================================================
# File I/O
# ============================================================================

def save_json(
    obj: Any,
    filepath: Union[str, Path],
    compress: bool = False,
    indent: int = 2
):
    """
    Save object to JSON file
    
    Parameters
    ----------
    obj : Any
        Object to save
    filepath : str or Path
        Output file path
    compress : bool
        Whether to gzip compress
    indent : int
        JSON indentation for readability
        
    Examples
    --------
    >>> state = State(1, 1)
    >>> save_json(state, 'state.json')
    >>> loaded = load_json('state.json')
    """
    filepath = Path(filepath)
    
    # Convert to dictionary
    data = to_dict(obj)
    
    # Add metadata
    data['_metadata'] = {
        'saved_at': datetime.now().isoformat(),
        'serialization_version': SERIALIZATION_VERSION
    }
    
    # Write
    json_str = json.dumps(data, indent=indent)
    
    if compress:
        if not filepath.suffix == '.gz':
            filepath = filepath.with_suffix(filepath.suffix + '.gz')
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            f.write(json_str)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load object from JSON file
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    obj : Any
        Loaded object
    """
    filepath = Path(filepath)
    
    # Read
    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Check version compatibility
    if '_metadata' in data:
        saved_version = data['_metadata'].get('serialization_version')
        if saved_version != SERIALIZATION_VERSION:
            warnings.warn(
                f"File saved with version {saved_version}, "
                f"loading with version {SERIALIZATION_VERSION}. "
                f"Compatibility not guaranteed."
            )
    
    # Remove metadata before reconstruction
    data.pop('_metadata', None)
    
    # Reconstruct
    return from_dict(data)


def save_pickle(
    obj: Any,
    filepath: Union[str, Path],
    compress: bool = True
):
    """
    Save object using pickle (most flexible)
    
    Parameters
    ----------
    obj : Any
        Object to save
    filepath : str or Path
        Output file path
    compress : bool
        Whether to gzip compress
        
    Examples
    --------
    >>> pipeline = ChannelPipeline()
    >>> # ... configure pipeline ...
    >>> save_pickle(pipeline, 'pipeline.pkl')
    >>> loaded = load_pickle('pipeline.pkl')
    """
    filepath = Path(filepath)
    
    # Add metadata
    save_data = {
        'object': obj,
        'metadata': {
            'saved_at': datetime.now().isoformat(),
            'serialization_version': SERIALIZATION_VERSION,
            'object_type': type(obj).__name__
        }
    }
    
    # Write
    if compress:
        if not filepath.suffix == '.gz':
            filepath = filepath.with_suffix(filepath.suffix + '.gz')
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    obj : Any
        Loaded object
    """
    filepath = Path(filepath)
    
    # Read
    if filepath.suffix == '.gz':
        with gzip.open(filepath, 'rb') as f:
            save_data = pickle.load(f)
    else:
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
    
    # Check version if metadata present
    if isinstance(save_data, dict) and 'metadata' in save_data:
        saved_version = save_data['metadata'].get('serialization_version')
        if saved_version != SERIALIZATION_VERSION:
            warnings.warn(
                f"File saved with version {saved_version}, "
                f"loading with version {SERIALIZATION_VERSION}. "
                f"Compatibility not guaranteed."
            )
        return save_data['object']
    else:
        # Old format without metadata
        return save_data


# ============================================================================
# Convenience Functions
# ============================================================================

def save(
    obj: Any,
    filepath: Union[str, Path],
    format: str = 'auto',
    **kwargs
):
    """
    Save object (auto-detect format from extension)
    
    Parameters
    ----------
    obj : Any
        Object to save
    filepath : str or Path
        Output file path
    format : str
        Format: 'auto', 'json', 'pickle'
    **kwargs
        Additional arguments for specific formats
        
    Examples
    --------
    >>> state = State(1, 1)
    >>> save(state, 'state.json')  # Auto-detects JSON
    >>> save(state, 'state.pkl')   # Auto-detects pickle
    """
    filepath = Path(filepath)
    
    # Auto-detect format
    if format == 'auto':
        suffix = filepath.suffix.lower()
        if suffix in ['.json', '.json.gz']:
            format = 'json'
        elif suffix in ['.pkl', '.pickle', '.pkl.gz', '.pickle.gz']:
            format = 'pickle'
        else:
            # Default to pickle
            format = 'pickle'
            warnings.warn(
                f"Unknown file extension '{suffix}', defaulting to pickle format"
            )
    
    # Save
    if format == 'json':
        save_json(obj, filepath, **kwargs)
    elif format == 'pickle':
        save_pickle(obj, filepath, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")


def load(filepath: Union[str, Path], format: str = 'auto') -> Any:
    """
    Load object (auto-detect format from extension)
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
    format : str
        Format: 'auto', 'json', 'pickle'
        
    Returns
    -------
    obj : Any
        Loaded object
        
    Examples
    --------
    >>> state = load('state.json')
    >>> pipeline = load('pipeline.pkl')
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format
    if format == 'auto':
        suffix = filepath.suffix.lower()
        if suffix in ['.json', '.json.gz']:
            format = 'json'
        elif suffix in ['.pkl', '.pickle', '.pkl.gz', '.pickle.gz']:
            format = 'pickle'
        else:
            # Try pickle first, then JSON
            try:
                return load_pickle(filepath)
            except:
                try:
                    return load_json(filepath)
                except:
                    raise SerializationError(
                        f"Could not load file: {filepath}"
                    )
    
    # Load
    if format == 'json':
        return load_json(filepath)
    elif format == 'pickle':
        return load_pickle(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


# ============================================================================
# State History Serialization
# ============================================================================

def save_state_history(
    states: Union[List[State], StateArray],
    filepath: Union[str, Path],
    metadata: Optional[Dict] = None,
    compress: bool = True
):
    """
    Save sequence of states efficiently
    
    Parameters
    ----------
    states : List[State] or StateArray
        State sequence
    filepath : str or Path
        Output file path
    metadata : Dict, optional
        Additional metadata
    compress : bool
        Whether to compress
        
    Examples
    --------
    >>> states = [State(1,0), State(1,1), State(0,1)]
    >>> save_state_history(states, 'states.npz', 
    ...                     metadata={'description': 'Trading signals'})
    """
    filepath = Path(filepath)
    
    # Convert to StateArray if needed
    if isinstance(states, list):
        states = StateArray.from_states(states)
    
    # Prepare data
    save_data = {
        'i': states.i,
        'q': states.q,
        'version': np.array([SERIALIZATION_VERSION]),
    }
    
    # Add metadata
    if metadata:
        # Convert metadata to JSON string for storage
        save_data['metadata'] = np.array([json.dumps(metadata)])
    
    # Save
    if compress:
        np.savez_compressed(filepath, **save_data)
    else:
        np.savez(filepath, **save_data)


def load_state_history(
    filepath: Union[str, Path]
) -> Tuple[StateArray, Optional[Dict]]:
    """
    Load state sequence
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    states : StateArray
        State sequence
    metadata : Dict or None
        Metadata if present
    """
    filepath = Path(filepath)
    
    # Load
    data = np.load(filepath, allow_pickle=True)
    
    # Extract states
    i = data['i']
    q = data['q']
    states = StateArray(i=i, q=q)
    
    # Extract metadata if present
    metadata = None
    if 'metadata' in data:
        metadata_str = str(data['metadata'][0])
        metadata = json.loads(metadata_str)
    
    return states, metadata


# ============================================================================
# Batch Operations
# ============================================================================

def save_batch(
    objects: Dict[str, Any],
    directory: Union[str, Path],
    format: str = 'pickle'
):
    """
    Save multiple objects to a directory
    
    Parameters
    ----------
    objects : Dict[str, Any]
        Dictionary mapping names to objects
    directory : str or Path
        Output directory
    format : str
        Format for all objects
        
    Examples
    --------
    >>> objects = {
    ...     'encoder': my_encoder,
    ...     'interpreter': my_interpreter,
    ...     'states': my_states
    ... }
    >>> save_batch(objects, 'my_model/', format='pickle')
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    manifest = {
        'files': list(objects.keys()),
        'format': format,
        'saved_at': datetime.now().isoformat(),
        'version': SERIALIZATION_VERSION
    }
    
    with open(directory / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Save each object
    for name, obj in objects.items():
        ext = '.pkl' if format == 'pickle' else '.json'
        filepath = directory / f"{name}{ext}"
        save(obj, filepath, format=format)


def load_batch(
    directory: Union[str, Path]
) -> Dict[str, Any]:
    """
    Load multiple objects from a directory
    
    Parameters
    ----------
    directory : str or Path
        Input directory
        
    Returns
    -------
    objects : Dict[str, Any]
        Dictionary mapping names to objects
        
    Examples
    --------
    >>> objects = load_batch('my_model/')
    >>> encoder = objects['encoder']
    >>> interpreter = objects['interpreter']
    """
    directory = Path(directory)
    
    # Load manifest
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        raise SerializationError(f"No manifest found in {directory}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Check version
    saved_version = manifest.get('version')
    if saved_version != SERIALIZATION_VERSION:
        warnings.warn(
            f"Directory saved with version {saved_version}, "
            f"loading with version {SERIALIZATION_VERSION}"
        )
    
    # Load each object
    objects = {}
    format_type = manifest.get('format', 'pickle')
    ext = '.pkl' if format_type == 'pickle' else '.json'
    
    for name in manifest['files']:
        filepath = directory / f"{name}{ext}"
        if filepath.exists():
            objects[name] = load(filepath, format=format_type)
        else:
            warnings.warn(f"File not found: {filepath}")
    
    return objects