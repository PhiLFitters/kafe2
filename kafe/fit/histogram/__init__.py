"""
.. module:: kafe.fit.histogram
    :platform: Unix
    :synopsis: This submodule provides the necessary objects for parameter estimation
               from histograms.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

from .container import HistContainer
from .cost import (HistCostFunction_UserDefined, HistCostFunction_Chi2_CovarianceMatrix,
                   HistCostFunction_Chi2_NoErrors, HistCostFunction_Chi2_PointwiseErrors,
                   HistCostFunction_NegLogLikelihood_Gaussian, HistCostFunction_NegLogLikelihood_Poisson)
from .fit import HistFit
from .format import HistModelDensityFunctionFormatter
from .model import HistParametricModel, HistModelFunction
from .plot import HistPlot, HistPlotContainer
