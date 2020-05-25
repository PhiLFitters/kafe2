from .._base import CostFunctionBase, CostFunctionBase_Chi2, CostFunctionBase_NegLogLikelihood, \
    CostFunctionBase_NegLogLikelihoodRatio


__all__ = [
    "IndexedCostFunction_UserDefined",
    "IndexedCostFunction_Chi2",
    "IndexedCostFunction_NegLogLikelihood",
    "IndexedCostFunction_NegLogLikelihoodRatio"
]



class IndexedCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to series of indexed measurements.
        The function handle must be provided by the user.

        :param user_defined_cost_function: function handle

        .. note::
            The names of the function arguments must be valid reserved
            names for the associated fit type (:py:obj:`~kafe2.fit.IndexedFit`)!
        """
        super(IndexedCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class IndexedCostFunction_Chi2(CostFunctionBase_Chi2):
    def __init__(self, errors_to_use='covariance', fallback_on_singular=True):
        """
        Built-in least-squares cost function for histogram data.

        :param errors_to_use: which erros to use when calculating :math:`\chi^2`
        :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
        """
        super(IndexedCostFunction_Chi2, self).__init__(errors_to_use=errors_to_use, fallback_on_singular=fallback_on_singular)


class IndexedCostFunction_NegLogLikelihood(CostFunctionBase_NegLogLikelihood):
    def __init__(self, data_point_distribution='poisson'):
        r"""
        Built-in negative log-likelihood cost function for *indexed* data.

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
        super(IndexedCostFunction_NegLogLikelihood, self).__init__(data_point_distribution=data_point_distribution)


class IndexedCostFunction_NegLogLikelihoodRatio(CostFunctionBase_NegLogLikelihoodRatio):
    def __init__(self, data_point_distribution='poisson'):
        r"""
        Built-in negative log-likelihood cost function for *indexed* data.

        .. warning:: This cost function has not yet been properly tested and should not
                  be used yet!

        In addition to the measurement data and model predictions, likelihood-fits require a
        probability distribution describing how the measurements are distributed around the model
        predictions.
        This built-in cost function supports two such distributions: the *Poisson* and *Gaussian* (normal)
        distributions.

        The likelihood ratio is defined as ratio of the likelihood function for each individual
        observation, divided by the so-called *marginal likelihood*.

        .. TODO:: Explain the above in detail.

        :param data_point_distribution: which type of statistics to use for modelling the distribution of individual data points
        :type data_point_distribution: ``'poisson'`` or ``'gaussian'``
        """
        super(IndexedCostFunction_NegLogLikelihoodRatio, self).__init__(data_point_distribution=data_point_distribution)


STRING_TO_COST_FUNCTION = {
    'chi2': IndexedCostFunction_Chi2,
    'chi_2': IndexedCostFunction_Chi2,
    'chisquared': IndexedCostFunction_Chi2,
    'chi_squared': IndexedCostFunction_Chi2,
    'nll': IndexedCostFunction_NegLogLikelihood,
    'negloglikelihood': IndexedCostFunction_NegLogLikelihood,
    'neg_log_likelihood': IndexedCostFunction_NegLogLikelihood,
    'nllr': IndexedCostFunction_NegLogLikelihoodRatio,
    'negloglikelihoodratio': IndexedCostFunction_NegLogLikelihoodRatio,
    'neg_log_likelihood_ratio': IndexedCostFunction_NegLogLikelihoodRatio,
}
