"""This submodule provides the necessary objects for parameter estimation using data consisting of ordered *xy* pairs.
This fit type is used for most cases e.g. when performing fits for the first time or in physics laboratory courses.

:synopsis: This submodule provides the necessary objects for parameter estimation using data consisting of ordered *xy*
    pairs.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

# flake8: noqa F401, F403 (imported but unused, used but unable to detect undefined names)

from .container import *
from .cost import *
from .ensemble import *
from .fit import *
from .model import *
from .plot import *
