import numpy as np
from types import FunctionType

from .._base import CostFunctionBase, CostFunctionException
from ..util import function_library

__all__ = [
    "UnbinnedCostFunction_UserDefined",
    "UnbinnedCostFunction_NegLogLikelihood"
]


class UnbinnedCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to histograms.
        The function handle must be provided by the user.

        :param user_defined_cost_function: function handle

        .. note::
            The names of the function arguments must be valid reserved
            names for the associated fit type (:py:obj:`~kafe2.fit.HistFit`)!
        """
        super(UnbinnedCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class UnbinnedCostFunction_NegLogLikelihood(CostFunctionBase):
    def __init__(self):
        r"""
        Built-in negative log-likelihood cost function for *Hist* data.

        In addition to the measurement data and model predictions, likelihood-fits require a
        probability distribution describing how the measurements are distributed around the model
        predictions.
        This built-in cost function supports two such distributions: the *Poisson* and *Gaussian* (normal)
        distributions.

        In general, a negative log-likelihood cost function is defined as the double negative logarithm of the
        product of the individual likelihoods of the data points.

        :param data_point_distribution: which type of statistics to use for modelling the distribution of individual data points
        :type data_point_distribution: ``'poisson'`` or ``'gaussian'``
        """
        super(UnbinnedCostFunction_NegLogLikelihood, self).__init__(cost_function=self.nll)
        self._needs_errors = False

    # model is the pdf already evaluated at all x-points with the given params, as far as I understand.
    # so there's only need to evaluate the model in the nll calculations?
    @staticmethod
    def nll(model):
        return -np.sum(np.log(model))

STRING_TO_COST_FUNCTION = {
    'nll': UnbinnedCostFunction_NegLogLikelihood,
    'negloglikelihood': UnbinnedCostFunction_NegLogLikelihood,
    'neg_log_likelihood': UnbinnedCostFunction_NegLogLikelihood,
}
