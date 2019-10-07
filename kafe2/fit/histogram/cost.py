from .._base import CostFunctionBase, CostFunctionBase_Chi2, CostFunctionBase_NegLogLikelihood, CostFunctionBase_NegLogLikelihoodRatio, CostFunctionException

__all__ = [
    "HistCostFunction_UserDefined",
    "HistCostFunction_Chi2",
    "HistCostFunction_NegLogLikelihood",
    "HistCostFunction_NegLogLikelihoodRatio",
    "STRING_TO_COST_FUNCTION"
]


class HistCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to histograms.
        The function handle must be provided by the user.

        :param user_defined_cost_function: function handle

        .. note::
            The names of the function arguments must be valid reserved
            names for the associated fit type (:py:obj:`~kafe2.fit.HistFit`)!
        """
        super(HistCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class HistCostFunction_Chi2(CostFunctionBase_Chi2):
    def __init__(self, errors_to_use='covariance', fallback_on_singular=True):
        """
        Built-in least-squares cost function for histogram data.

        :param errors_to_use: which erros to use when calculating :math:`\chi^2`
        :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
        """
        super(HistCostFunction_Chi2, self).__init__(errors_to_use=errors_to_use, fallback_on_singular=fallback_on_singular)

    @staticmethod
    def chi2_no_errors(data, model, parameter_values, parameter_constraints):
        r"""A least-squares cost function calculated from `y` data and model values,
        without considering uncertainties:

        .. math::
            C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})\cdot({\bf d} - {\bf m})
                +
                C({\bf p})

        In the above, :math:`{\bf d}` are the measurements and :math:`{\bf m}` are the model
        predictions,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        return CostFunctionBase_Chi2.chi2_no_errors(data=data, model=model, parameter_values=parameter_values,
                                                    parameter_constraints=parameter_constraints)

    @staticmethod
    def chi2_covariance(data, model, total_cov_mat_inverse, parameter_values, parameter_constraints):
        r"""A least-squares cost function calculated from `y` data and model values,
        considering the covariance matrix of the `y` measurements.

        .. math::
            C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})^{\top}\,{{\bf V}^{-1}}\,({\bf d} - {\bf m})
                +
                C({\bf p})

        In the above, :math:`{\bf d}` are the measurements, :math:`{\bf m}` are the model
        predictions, and :math:`{{\bf V}^{-1}}` is the inverse of the total covariance matrix,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_cov_mat_inverse: inverse of the total covariance matrix :math:`{\bf V}^{-1}`
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        return CostFunctionBase_Chi2.chi2_covariance(data=data, model=model,
                                                     total_cov_mat_inverse=total_cov_mat_inverse,
                                                     parameter_values=parameter_values,
                                                     parameter_constraints=parameter_constraints)

    @staticmethod
    def chi2_pointwise_errors(data, model, total_error, parameter_values, parameter_constraints):
        r"""A least-squares cost function calculated from `y` data and model values,
        considering pointwise (uncorrelated) uncertainties for each data point:

        .. math::
            C = \chi^2({\bf d}, {\bf m}, {\bf \sigma}) = \sum_k \left(\frac{d_k - m_k}{\sigma_k}\right)^2
                +
                C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        :math:`{\bf \sigma}` are the pointwise total uncertainties,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_error: total error vector :math:`{\bf \sigma}`
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        return CostFunctionBase_Chi2.chi2_pointwise_errors(data=data, model=model, total_error=total_error,
                                                           parameter_values=parameter_values,
                                                           parameter_constraints=parameter_constraints)


class HistCostFunction_NegLogLikelihood(CostFunctionBase_NegLogLikelihood):
    def __init__(self, data_point_distribution='poisson'):
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
        super(HistCostFunction_NegLogLikelihood, self).__init__(data_point_distribution=data_point_distribution)

    @staticmethod
    def nll_gaussian(data, model, total_error, parameter_values, parameter_constraints):
        r"""A negative log-likelihood function assuming Gaussian statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}, {\bf \sigma}) = -2 \ln \prod_j \mathcal{L}_{\rm Gaussian} (x=d_j, \mu=m_j, \sigma=\sigma_j)
                +
                C({\bf p})

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{1}{\sqrt{2{\sigma_j}^2\pi}} \exp{\left(-\frac{ (d_j-m_j)^2 }{ {\sigma_j}^2}\right)}
                            +
                            C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        :math:`{\bf \sigma}` are the pointwise total uncertainties,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_error: total error vector :math:`{\bf \sigma}`
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        # "translate" the argument names
        return CostFunctionBase_NegLogLikelihood.nll_gaussian(data=data, model=model, total_error=total_error,
                                                              parameter_values=parameter_values,
                                                              parameter_constraints=parameter_constraints)

    @staticmethod
    def nll_poisson(data, model, parameter_values, parameter_constraints):
        r"""A negative log-likelihood function assuming Poisson statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}) = -2 \ln \prod_j \mathcal{L}_{\rm Poisson} (k=d_j, \lambda=m_j)
                +
                C({\bf p})

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{{m_j}^{d_j} \exp(-m_j)}{d_j!}
                            +
                            C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        # "translate" the argument names
        return CostFunctionBase_NegLogLikelihood.nll_poisson(data=data, model=model,
                                                             parameter_values=parameter_values,
                                                             parameter_constraints=parameter_constraints)


class HistCostFunction_NegLogLikelihoodRatio(CostFunctionBase_NegLogLikelihoodRatio):
    def __init__(self, data_point_distribution='poisson'):
        r"""
        Built-in negative log-likelihood ratio cost function for histograms.

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
        super(HistCostFunction_NegLogLikelihoodRatio, self).__init__(data_point_distribution=data_point_distribution)

    @staticmethod
    def nllr_gaussian(data, model, total_error, parameter_values, parameter_constraints):
        r"""A negative log-likelihood function assuming Gaussian statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}, {\bf \sigma}) = -2 \ln \prod_j \mathcal{L}_{\rm Gaussian} (x=d_j, \mu=m_j, \sigma=\sigma_j)
                +
                C({\bf p})

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{1}{\sqrt{2{\sigma_j}^2\pi}} \exp{\left(-\frac{ (d_j-m_j)^2 }{ {\sigma_j}^2}\right)}
                            +
                            C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        :math:`{\bf \sigma}` are the pointwise total uncertainties,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_error: total *y* uncertainties for data
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        # "translate" the argument names
        return CostFunctionBase_NegLogLikelihoodRatio.nllr_gaussian(
            data=data, model=model, total_error=total_error, parameter_values=parameter_values,
            parameter_constraints=parameter_constraints)

    @staticmethod
    def nllr_poisson(data, model, parameter_values, parameter_constraints):
        r"""A negative log-likelihood function assuming Poisson statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}) = -2 \ln \prod_j \mathcal{L}_{\rm Poisson} (k=d_j, \lambda=m_j)
                +
                C({\bf p})

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{{m_j}^{d_j} \exp(-m_j)}{d_j!}
                            +
                            C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        # "translate" the argument names
        return CostFunctionBase_NegLogLikelihoodRatio.nllr_poisson(
            data=data, model=model, parameter_values=parameter_values, parameter_constraints=parameter_constraints)


STRING_TO_COST_FUNCTION = {
    'chi2': HistCostFunction_Chi2,
    'chi_2': HistCostFunction_Chi2,
    'chisquared': HistCostFunction_Chi2,
    'chi_squared': HistCostFunction_Chi2,
    'nll': HistCostFunction_NegLogLikelihood,
    'negloglikelihood': HistCostFunction_NegLogLikelihood,
    'neg_log_likelihood': HistCostFunction_NegLogLikelihood,
    'nllr': HistCostFunction_NegLogLikelihoodRatio,
    'negloglikelihoodratio': HistCostFunction_NegLogLikelihoodRatio,
    'neg_log_likelihood_ratio': HistCostFunction_NegLogLikelihoodRatio,
}
