import numpy as np
from types import FunctionType

from .._base import CostFunction, CostFunctionException
from ..util import function_library

__all__ = [
    "UnbinnedCostFunction_NegLogLikelihood",
]


class UnbinnedCostFunction_NegLogLikelihood(CostFunction):
    def __init__(self):
        r"""
        Built-in negative log-likelihood cost function for *Unbinned* data.

        When using an unbinned dataset, the negative log-likelihood is the best method to fit a probability density
        funtion *pdf* to the density of the datapoints

        In general, a negative log-likelihood cost function is defined as the double negative logarithm of the
        product of the individual likelihoods of the data points.
        """
        super(UnbinnedCostFunction_NegLogLikelihood, self).__init__(cost_function=self.nll)
        self._needs_errors = False
        self._formatter.latex_name = "-2\\ln\\mathcal{L}"
        self._formatter.name = "nll"
        self._formatter.description = "negative log-likelihood"

    # model is the pdf already evaluated at all x-points with the given params, as far as I understand.
    # so there's only need to evaluate the model in the nll calculations?
    @staticmethod
    def nll(model):
        _total_log_likelihood = np.sum(np.log(model))
        # guard against returning NaN
        if np.isnan(_total_log_likelihood):
            return np.inf
        return -2.0 * _total_log_likelihood


STRING_TO_COST_FUNCTION = {
    'nll': UnbinnedCostFunction_NegLogLikelihood,
    'negloglikelihood': UnbinnedCostFunction_NegLogLikelihood,
    'neg_log_likelihood': UnbinnedCostFunction_NegLogLikelihood,
}
