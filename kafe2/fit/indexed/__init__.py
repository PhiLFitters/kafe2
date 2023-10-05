"""This submodule provides the necessary objects for parameter estimation using data consisting of an indexed series of
measurements. This can be useful for calculating weighted mean values or template fits.

:synopsis: This submodule provides the necessary objects for parameter estimation using data consisting of an indexed
    series of measurements.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

# flake8: noqa F401, F403 (imported but unused, used but unable to detect undefined names)

from .container import *
from .cost import *
from .fit import *
from .format import *
from .model import *
from .plot import *
