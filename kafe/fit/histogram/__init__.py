"""
.. module:: kafe.fit.histogram
    :platform: Unix
    :synopsis: This submodule provides the necessary objects for parameter estimation
               from histograms.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

from .container import HistContainer
from .cost import HistCostFunction_UserDefined, HistCostFunction_Chi2, HistCostFunction_NegLogLikelihood, HistCostFunction_NegLogLikelihoodRatio
from .fit import HistFit
from .format import HistModelDensityFunctionFormatter
from .model import HistParametricModel, HistModelFunction
from .plot import HistPlot, HistPlotContainer
