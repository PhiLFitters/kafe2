from .._base import CostFunction_Chi2, CostFunction_NegLogLikelihood, CostFunctionException

__all__ = [
    "XYCostFunction_Chi2",
    "XYCostFunction_NegLogLikelihood",
    "XYCostFunction_GaussApproximation"
]


class XYCostFunction_Chi2(CostFunction_Chi2):
    def __init__(
            self, errors_to_use='covariance', fallback_on_singular=True, axes_to_use='xy',
            add_constraint_cost=True, add_determinant_cost=True):
        """Built-in least-squares cost function for *xy* data.

        :param errors_to_use: Which errors to use when calculating :math:`\\chi^2`. This is either
            `'covariance'``, ``'pointwise'`` or :py:obj:`None`.
        :type errors_to_use: str or None
        :param axes_to_use: The errors for the given axes are taken into account when calculating
            :math:`\\chi^2`. Either ``'y'`` or ``'xy'``
        :param bool add_constraint_cost: If :py:obj:`True`, automatically add the cost for kafe2
            constraints.
        :param bool add_determinant_cost: If :py:obj:`True`, automatically increase the cost
            function value by the logarithm of the determinant of the covariance matrix to reduce
            bias.
        """
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        if axes_to_use.lower() == 'y':
            self._COV_MAT_CHOLESKY_NAME = "y_total_cov_mat_cholesky"
            self._ERROR_NAME = "y_total_error"
        elif axes_to_use.lower() == 'xy':
            self._COV_MAT_CHOLESKY_NAME = "total_cov_mat_cholesky"
            self._ERROR_NAME = "total_error"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y')")
        super(XYCostFunction_Chi2, self).__init__(
            errors_to_use=errors_to_use, fallback_on_singular=fallback_on_singular,
            add_constraint_cost=add_constraint_cost, add_determinant_cost=add_determinant_cost)


class XYCostFunction_NegLogLikelihood(CostFunction_NegLogLikelihood):
    def __init__(self, data_point_distribution="poisson", ratio=False, axes_to_use="xy"):
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        if axes_to_use.lower() == 'y':
            self._ERROR_NAME = "y_total_error"
        elif axes_to_use.lower() == 'xy':
            self._ERROR_NAME = "total_error"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y')")
        super(XYCostFunction_NegLogLikelihood, self).__init__(
            data_point_distribution=data_point_distribution, ratio=ratio)


class XYCostFunction_GaussApproximation(CostFunction_Chi2):
    def __init__(
            self, errors_to_use='covariance', axes_to_use='xy',
            add_constraint_cost=True, add_determinant_cost=True):
        """
        Built-in Gaussian approximation of the Poisson negative log-likelihood cost function for
        *xy* data.

        :param errors_to_use: Which errors to use when calculating :math:`\\chi^2`. This is either
            `'covariance'``, ``'pointwise'``.
        :type errors_to_use: str
        :param axes_to_use: The errors for the given axes are taken into account when calculating
            :math:`\\chi^2`. Either ``'y'`` or ``'xy'``
        :param bool add_constraint_cost: If :py:obj:`True`, automatically add the cost for kafe2
            constraints.
        :param bool add_determinant_cost: If :py:obj:`True`, automatically increase the cost
            function value by the logarithm of the determinant of the covariance matrix to reduce
            bias.
        """
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        if axes_to_use.lower() == 'y':
            self._COV_MAT_CHOLESKY_NAME = "y_total_cov_mat_cholesky"
            self._ERROR_NAME = "y_total_error"
        elif axes_to_use.lower() == 'xy':
            self._COV_MAT_CHOLESKY_NAME = "total_cov_mat_cholesky"
            self._ERROR_NAME = "total_error"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y')")
        super().__init__(
            errors_to_use=errors_to_use, add_constraint_cost=add_constraint_cost,
            add_determinant_cost=add_determinant_cost
        )


STRING_TO_COST_FUNCTION = {
    'chi2': (XYCostFunction_Chi2, {}),
    'chi_2': (XYCostFunction_Chi2, {}),
    'chisquared': (XYCostFunction_Chi2, {}),
    'chi_squared': (XYCostFunction_Chi2, {}),
    'chi2_no_errors': (XYCostFunction_Chi2, {"errors_to_use": None}),
    'chi2_pointwise': (XYCostFunction_Chi2, {"errors_to_use": "pointwise"}),
    'chi2_pointwise_errors': (XYCostFunction_Chi2, {"errors_to_use": "pointwise"}),
    'chi2_covariance': (XYCostFunction_Chi2, {"errors_to_use": "covariance"}),
    'nll': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll-poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll_poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nllpoisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll-gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'nll_gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'nllgaussiann': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'negloglikelihood': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'neg_log_likelihood': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'nllr': (CostFunction_NegLogLikelihood, {"ratio": True}),
    'nllr-poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllr_poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllrpoisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllr-gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'nllr_gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'nllrgaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'negloglikelihoodratio': (XYCostFunction_NegLogLikelihood, {"ratio": True}),
    'neg_log_likelihood_ratio': (XYCostFunction_NegLogLikelihood, {"ratio": True}),
    'gauss-approximation': (XYCostFunction_GaussApproximation, {}),
    'gauss_approximation': (XYCostFunction_GaussApproximation, {}),
    'gauss_approximation_covariance': (
        XYCostFunction_GaussApproximation, {"errors_to_use": "covariance"}),
    'gauss_approximation_pointwise': (
        XYCostFunction_GaussApproximation, {"errors_to_use": "pointwise"}),
    'gauss_approximation_pointwise_errors': (
        XYCostFunction_GaussApproximation, {"errors_to_use": "pointwise"}),
}
