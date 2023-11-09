'''
DeepModelZoo is a gallary of deep learning models for spatiotemporal dynamics.
* The default data shape is [Time, Batch, [Depth, Height], Width, Channel] 
    (`[T,B,D,H,W,C]`)
'''
import os

__all__ = [
    'backend',
]


backEnds = ['Torch', 'JaxHK']

DEFAULT_BACKEND = 'Torch'

backend = os.environ.get('DMZ_BACKEND')
    

if backend is None: 
    backend = DEFAULT_BACKEND
    print(f'\033[93mWarning, using default backend: \'{backend}\'\033[0m')


assert backend in backEnds, f'backend must be one of {backEnds}'

from . import temporal
from . import spatial
from . import paramEC
from . import utility