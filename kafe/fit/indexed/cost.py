import numpy as np

from scipy.stats import poisson, norm

from .._base import CostFunctionBase, CostFunctionException


class IndexedCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to *indexed* data.
        The function handle must be provided by the user.

        :param user_defined_cost_function: function handle

        .. note::
            The names of the function arguments must be valid reserved
            names for the associated fit type (:py:obj:`~kafe.fit.IndexedFit`)!
        """
        super(IndexedCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class IndexedCostFunction_Chi2_NoErrors(CostFunctionBase):
    def __init__(self):
        r"""
        Built-in least-squares cost function calculated from data and model values,
        without considering uncertainties:

        .. math::
            C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})\cdot({\bf d} - {\bf m})

        In the above, :math:`{\bf d}` are the measurements and :math:`{\bf m}` are the model
        predictions.
        """
        super(IndexedCostFunction_Chi2_NoErrors, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(data, model):
        """Static method for calculating the :math:`\chi^2` cost function.

        :param data: measurement data
        :param model: model predictions
        :return: value of the cost function
        """
        return np.sum((data - model)**2)


class IndexedCostFunction_Chi2_PointwiseErrors(CostFunctionBase):
    def __init__(self):
        r"""
        Built-in least-squares cost function calculated from data and model values,
        considering pointwise (uncorrelated) uncertainties for each data point:

        .. math::
            C = \chi^2({\bf d}, {\bf m}, {\bf \sigma}) = \sum_k \frac{d_k - m_k}{\sigma_k}

        In the above, :math:`{\bf d}` are the measurements, :math:`{\bf m}` are the model
        predictions, and :math:`{\bf \sigma}` are the pointwise total uncertainties.
        """
        super(IndexedCostFunction_Chi2_PointwiseErrors, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(data, model, total_error):
        """Static method for calculating the :math:`\chi^2` cost function.

        :param data: measurement data
        :param model: model predictions
        :param total_error: total uncertainties of the data points
        :return: value of the cost function
        """
        return np.sum((data - model)**2/total_error**2)


class IndexedCostFunction_Chi2_CovarianceMatrix(CostFunctionBase):
    def __init__(self):
        r"""
        Built-in least-squares cost function calculated from data and model values,
        considering the covariance matrix of the measurements:

        .. math::
            C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})^{\top}\,{{\bf V}^{-1}}\,({\bf d} - {\bf m})

        In the above, :math:`{\bf d}` are the measurements, :math:`{\bf m}` are the model
        predictions, and :math:`{{\bf V}^{-1}}` is the inverse of the total covariance matrix.
        """
        super(IndexedCostFunction_Chi2_CovarianceMatrix, self).__init__(cost_function=self.chi2)

    @staticmethod
    def chi2(data, model, total_cov_mat_inverse):
        """Static method for calculating the :math:`\chi^2` cost function.

        :param data: measurement data
        :param model: model predictions
        :param total_cov_mat_inverse: inverse of the total covariance matrix
        :return: value of the cost function
        """
        _res = (data - model)
        if total_cov_mat_inverse is None:
            raise np.linalg.LinAlgError("Total covariance matrix is singular!")
        return _res.dot(total_cov_mat_inverse).dot(_res)[0, 0]


class IndexedCostFunction_NegLogLikelihood_Poisson(CostFunctionBase):
    def __init__(self):
        r"""
        Built-in negative log-likelihood cost function calculated from data and model values,
        assuming Poisson statistics for the individual bin contents:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}) = -2 \ln \prod_j \mathcal{L}_{\rm Poisson} (k=d_j, \lambda=m_j)

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{{m_j}^{d_j} \exp(-m_j)}{d_j!}

        In the above, :math:`{\bf d}` are the measurements and :math:`{\bf m}` are the model
        predictions.
        """
        super(IndexedCostFunction_NegLogLikelihood_Poisson, self).__init__(cost_function=self.nll)

    @staticmethod
    def nll(data, model):
        """Static method for calculating the negative log-likelihood cost function.

        :param data: measurement data
        :param model: model predictions
        :return: value of the cost function
        """
        _per_point_likelihoods = poisson.pmf(data, mu=model, loc=0.0)
        _total_likelihood = np.prod(_per_point_likelihoods)
        return -2.0 * np.log(_total_likelihood)


class IndexedCostFunction_NegLogLikelihood_Gaussian(CostFunctionBase):
    def __init__(self):
        r"""
        Built-in negative log-likelihood cost function calculated from data and model values,
        assuming Poisson statistics for the individual bin contents:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}, {\bf \sigma}) = -2 \ln \prod_j \mathcal{L}_{\rm Gaussian} (x=d_j, \mu=m_j, \sigma=\sigma_j)

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{1}{\sqrt{2{\sigma_j}^2\pi}} \exp{\left(-\frac{ (d_j-m_j)^2 }{ {\sigma_j}^2}\right)}

        In the above, :math:`{\bf d}` are the measurements, :math:`{\bf m}` are the model
        predictions, and :math:`{\bf \sigma}` are the pointwise total uncertainties.
        """
        super(IndexedCostFunction_NegLogLikelihood_Gaussian, self).__init__(cost_function=self.nll)

    @staticmethod
    def nll(data, model, total_error):
        """Static method for calculating the negative log-likelihood cost function.

        :param data: measurement data
        :param model: model predictions
        :param total_error: total uncertainties of the data points
        :return: value of the cost function
        """
        _per_point_likelihoods = norm.pdf(data, loc=model, scale=total_error)
        _total_likelihood = np.prod(_per_point_likelihoods)
        return -2.0 * np.log(_total_likelihood)