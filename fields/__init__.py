"""
Spatial channel fields

This module enables channel algebra over spatial domains (2D images, 3D volumes, etc.)
Each point in space has a channel state, enabling:
- Spatial pattern detection
- Image segmentation
- Volumetric analysis
- Spatial topology computation

Examples
--------
>>> from channelpy.fields import ChannelField2D, map_field, convolve_field
>>> 
>>> # Create a 2D field
>>> field = ChannelField2D.from_function(
...     width=100, height=100,
...     func=lambda x, y: State(i=int(x > 50), q=int(y > 50))
... )
>>> 
>>> # Apply operations
>>> filtered = convolve_field(field, kernel='gaussian')
>>> patterns = find_connected_components(field)
"""

from .field import (
    ChannelField,
    ChannelField1D,
    ChannelField2D,
    ChannelField3D,
    BoundaryCondition
)

from .operations import (
    map_field,
    reduce_field,
    convolve_field,
    gradient_field,
    find_connected_components,
    detect_patterns,
    field_topology,
    field_distance
)

from .lazy_field import (
    LazyChannelField,
    LazyChannelField2D,
    LazyChannelField3D,
    materialize_region,
    stream_field
)

__all__ = [
    # Core field classes
    'ChannelField',
    'ChannelField1D',
    'ChannelField2D',
    'ChannelField3D',
    'BoundaryCondition',
    
    # Operations
    'map_field',
    'reduce_field',
    'convolve_field',
    'gradient_field',
    'find_connected_components',
    'detect_patterns',
    'field_topology',
    'field_distance',
    
    # Lazy evaluation
    'LazyChannelField',
    'LazyChannelField2D',
    'LazyChannelField3D',
    'materialize_region',
    'stream_field',
]