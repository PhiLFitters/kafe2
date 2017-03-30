import numpy as np

from scipy.stats import poisson, norm

from .._base import CostFunctionBase, CostFunctionException


class XYCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to xy data.
        The function handle must be provided by the user.
        """
        super(XYCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class XYCostFunction_Chi2_NoErrors_Y(CostFunctionBase):
    def __init__(self):
        """
        Built-in least-squares cost function calculated from 'y' data and model values,
        without considering uncertainties.
        """
        super(XYCostFunction_Chi2_NoErrors_Y, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(y_data, y_model):
        return np.sum((y_data - y_model)**2)


class XYCostFunction_Chi2_PointwiseErrors_Y(CostFunctionBase):
    def __init__(self):
        """
        Built-in least-squares cost function calculated from 'y' data and model values,
        considering pointwise (uncorrelated) uncertainties for each data point.
        """
        super(XYCostFunction_Chi2_PointwiseErrors_Y, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(y_data, y_model, y_total_error):
        return np.sum((y_data - y_model)**2/y_total_error**2)


class XYCostFunction_Chi2_CovarianceMatrix_Y(CostFunctionBase):
    def __init__(self):
        """
        Built-in least-squares cost function calculated from 'y' data and model values,
        considering the covariance matrix of the 'y' measurements.
        """
        super(XYCostFunction_Chi2_CovarianceMatrix_Y, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(y_data, y_model, y_total_cov_mat_inverse):
        _res = (y_data - y_model)
        return _res.dot(y_total_cov_mat_inverse).dot(_res)[0, 0]


class XYCostFunction_NegLogLikelihood_Poisson_Y(CostFunctionBase):
    def __init__(self):
        """
        Built-in negative log-likelihood cost function calculated from 'y' data and model values,
        assuming Poisson statistics for the individual bin contents.
        """
        super(XYCostFunction_NegLogLikelihood_Poisson_Y, self).__init__(cost_function=self.nll)

    @staticmethod
    def nll(y_data, y_model):
        _per_point_likelihoods = poisson.pmf(y_data, mu=y_model, loc=0.0)
        _total_likelihood = np.prod(_per_point_likelihoods)
        return -2.0 * np.log(_total_likelihood)


class XYCostFunction_NegLogLikelihood_Gaussian_Y(CostFunctionBase):
    def __init__(self):
        """
        Built-in negative log-likelihood cost function calculated from 'y' data and model values,
        assuming Gaussian statistics for the individual bin contents (sigma=uncertainty).
        """
        super(XYCostFunction_NegLogLikelihood_Gaussian_Y, self).__init__(cost_function=self.nll)

    @staticmethod
    def nll(y_data, y_model, y_total_error):
        _per_point_likelihoods = norm.pdf(y_data, loc=y_model, scale=y_total_error)
        _total_likelihood = np.prod(_per_point_likelihoods)
        return -2.0 * np.log(_total_likelihood)