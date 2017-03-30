import numpy as np

from scipy.stats import poisson, norm

from .._base import CostFunctionBase, CostFunctionException


class HistCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to histograms.
        The function handle must be provided by the user.
        """
        super(HistCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class HistCostFunction_Chi2_NoErrors(CostFunctionBase):
    def __init__(self):
        """
        Built-in least-squares cost function calculated from data and model values,
        without considering uncertainties.
        """
        super(HistCostFunction_Chi2_NoErrors, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(data, model):
        return np.sum((data - model)**2)


class HistCostFunction_Chi2_PointwiseErrors(CostFunctionBase):
    def __init__(self):
        """
        Built-in least-squares cost function calculated from data and model values,
        considering pointwise (uncorrelated) uncertainties for each data point.
        """
        super(HistCostFunction_Chi2_PointwiseErrors, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(data, model, total_error):
        return np.sum((data - model)**2/total_error**2)


class HistCostFunction_Chi2_CovarianceMatrix(CostFunctionBase):
    def __init__(self):
        """
        Built-in least-squares cost function calculated from data and model values,
        considering the covariance matrix of the measurements.
        """
        super(HistCostFunction_Chi2_CovarianceMatrix, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(data, model, total_cov_mat_inverse):
        _res = (data - model)
        return _res.dot(total_cov_mat_inverse).dot(_res)[0, 0]


class HistCostFunction_NegLogLikelihood_Poisson(CostFunctionBase):
    def __init__(self):
        """
        Built-in negative log-likelihood cost function calculated from data and model values,
        assuming Poisson statistics for the individual bin contents.
        """
        super(HistCostFunction_NegLogLikelihood_Poisson, self).__init__(cost_function=self.nll)

    @staticmethod
    def nll(data, model):
        _per_point_likelihoods = poisson.pmf(data, mu=model, loc=0.0)
        _total_likelihood = np.prod(_per_point_likelihoods)
        return -2.0 * np.log(_total_likelihood)


class HistCostFunction_NegLogLikelihood_Gaussian(CostFunctionBase):
    def __init__(self):
        """
        Built-in negative log-likelihood cost function calculated from data and model values,
        assuming Poisson statistics for the individual bin contents.
        """
        super(HistCostFunction_NegLogLikelihood_Gaussian, self).__init__(cost_function=self.nll)

    @staticmethod
    def nll(data, model, total_error):
        _per_point_likelihoods = norm.pdf(data, loc=model, scale=total_error)
        _total_likelihood = np.prod(_per_point_likelihoods)
        return -2.0 * np.log(_total_likelihood)