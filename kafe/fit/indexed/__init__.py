"""
.. module:: kafe.fit.indexed
    :platform: Unix
    :synopsis: This submodule provides the necessary objects for parameter estimation
               using data consisting of an indexed series of measurements.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

from .container import IndexedContainer
from .cost import (IndexedCostFunction_UserDefined, IndexedCostFunction_Chi2_CovarianceMatrix,
                   IndexedCostFunction_Chi2_NoErrors, IndexedCostFunction_Chi2_PointwiseErrors,
                   IndexedCostFunction_NegLogLikelihood_Gaussian)
from .fit import IndexedFit
from .format import IndexedModelFunctionFormatter
from .model import IndexedParametricModel, IndexedModelFunction
from .plot import IndexedPlot, IndexedPlotContainer
