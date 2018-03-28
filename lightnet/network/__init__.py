"""Lightnet Network Module.

|br|
This module contains classes and functions to create deep neural network with pytorch_.
It is mostly targeted at networks from the darknet_ framework, but can be used to create and CNN.
"""

from .network import *  # NOQA
from .loss import *  # NOQA
from .weight import *  # NOQA

from . import layer

__all__ = ['Darknet', 'RegionLoss', 'layer']
