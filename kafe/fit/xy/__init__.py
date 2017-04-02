"""
.. module:: kafe.fit.xy
    :platform: Unix
    :synopsis: This submodule provides the necessary objects for parameter estimation
               using data consisting of ordered *xy* pairs.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

from .container import XYContainer
from .cost import (XYCostFunction_UserDefined, XYCostFunction_Chi2_CovarianceMatrix_Y,
                   XYCostFunction_Chi2_NoErrors_Y, XYCostFunction_Chi2_PointwiseErrors_Y,
                   XYCostFunction_NegLogLikelihood_Gaussian_Y)
from .fit import XYFit
from .format import XYModelFunctionFormatter
from .model import XYParametricModel, XYModelFunction
from .plot import XYPlot, XYPlotContainer
