"""
.. module:: kafe.fit.xy
    :platform: Unix
    :synopsis: This submodule provides the necessary objects for parameter estimation
               using data consisting of ordered *xy* pairs.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

from .container import XYContainer
from .cost import (XYCostFunction_UserDefined, XYCostFunction_Chi2, XYCostFunction_NegLogLikelihood)
from .ensemble import XYFitEnsemble
from .fit import XYFit
from .format import XYModelFunctionFormatter
from .model import XYParametricModel, XYModelFunction
from .plot import XYPlot, XYPlotContainer
