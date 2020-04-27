"""This submodule provides the necessary objects for parameter estimation from histograms. Currently a histogram needs
to be filled with all individual data points. A function for setting the bin heights is available but not recommended,
as saving and loading those to and from a file is not yet supported.

:synopsis: This submodule provides the necessary objects for parameter estimation from histograms.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

from .container import *
from .cost import *
from .fit import *
from .model import *
from .plot import *
