import numpy as np
import six
import warnings

from scipy.stats import poisson, norm, chi2
from ..io.file import FileIOMixin
from .format import ParameterFormatter, CostFunctionFormatter


if six.PY2:
    from funcsigs import signature
else:
    from inspect import signature


__all__ = ["CostFunction",
           "CostFunction_Chi2",
           "CostFunction_NegLogLikelihood",
           "CostFunctionException"]


class CostFunctionException(Exception):
    pass


class CostFunction(FileIOMixin, object):
    """
    Base class for cost functions. Built from a Python function with some extra functionality used
    by Fit objects.

    Any Python function returning a ``float`` can be used as a cost function,
    although a number of common cost functions are provided as built-ins for
    all fit types.

    In order to be used as a model function, a native Python function must be wrapped
    by an object whose class derives from this base class.
    There is a dedicated :py:class:`CostFunction` specialization for each type of
    fit.

    This class provides the basic functionality used by all :py:class:`CostFunction` objects.
    These use introspection (:py:mod:`inspect`) for determining the parameter structure of the
    cost function and to ensure the function can be used as a cost function (validation).
    """

    EXCEPTION_TYPE = CostFunctionException
    _DATA_NAME = "data"
    _MODEL_NAME = "model"
    _COV_MAT_INVERSE_NAME = "total_cov_mat_inverse"
    _ERROR_NAME = "total_error"

    def __init__(self, cost_function, arg_names=None, add_constraint_cost=True):
        """
        Construct :py:class:`CostFunction` object (a wrapper for a native Python function):

        :param typing.Callable cost_function: function handle
        :param typing.Iterable[str] arg_names: the names to use for the cost function arguments.
            If None, detect from function signature.
        :param bool add_constraint_cost: If :py:obj:`True`, automatically add the cost for kafe2
            constraints.
        """
        self._cost_function_handle = cost_function
        _signature = signature(self._cost_function_handle)
        if arg_names is None:
            self._arg_names = list(_signature.parameters.keys())
            for _par in _signature.parameters.values():
                if _par.kind == _par.VAR_POSITIONAL:
                    raise self.__class__.EXCEPTION_TYPE(
                        "Cost function '{}' with variable number of positional arguments "
                        "(*{}) needs explicit argument names.".format(
                            self._cost_function_handle.__name__, _par.name))
        else:
            self._arg_names = list(arg_names)
        if "cost" in self._arg_names:
            raise self.__class__.EXCEPTION_TYPE(
                "The alias 'cost' for the cost function value cannot be used as a name for one of"
                "the cost function arguments!")
        for _par in _signature.parameters.values():
            if _par.kind == _par.VAR_KEYWORD:
                raise self.__class__.EXCEPTION_TYPE(
                    "Cost function '{}' with variable number of keyword arguments "
                    "(**{}) is not supported.".format(
                        self._cost_function_handle.__name__, _par.name, ))
        self._arg_count = len(self._arg_names)
        self._formatter = CostFunctionFormatter(
            name=self.name,
            arg_formatters=[ParameterFormatter(_arg_name, value=0, error=None)
                            for _arg_name in self._arg_names])

        self._add_constraint_cost = add_constraint_cost
        if self._add_constraint_cost:
            self._arg_names += ["parameter_values", "parameter_constraints"]
            self._arg_count += 2

        self._needs_errors = True
        self._is_chi2 = False
        self._saturated = False
        super(CostFunction, self).__init__()

    @classmethod
    def _get_base_class(cls):
        return CostFunction

    @classmethod
    def _get_object_type_name(cls):
        return 'cost_function'

    def __call__(self, *args):
        additional_cost = 0.0
        if self._add_constraint_cost:
            _par_constraints = args[-1]
            _par_vals = args[-2]
            args = args[:-2]
            if _par_constraints is not None:
                for _par_constraint in _par_constraints:
                    additional_cost += _par_constraint.cost(_par_vals)
        return self._cost_function_handle(*args) + additional_cost

    @property
    def name(self):
        """The cost function name (a valid Python identifier)"""
        return self._cost_function_handle.__name__

    @property
    def func(self):
        """The cost function handle"""
        return self._cost_function_handle

    @property
    def arg_names(self):
        """The names of the cost function arguments."""
        return self._arg_names

    @property
    def formatter(self):
        """The :py:obj:`Formatter` object for this function"""
        return self._formatter

    @property
    def argument_formatters(self):
        """The :py:obj:`Formatter` objects for the function arguments"""
        return self._formatter.arg_formatters

    @property
    def needs_errors(self):
        """Whether the cost function needs errors for a meaningful result"""
        return self._needs_errors

    @property
    def is_chi2(self):
        """Whether the cost function is a chi2 cost function."""
        return self._is_chi2

    @property
    def saturated(self):
        """Whether the cost function value is calculated from a saturated likelihood."""
        return self._saturated

    def goodness_of_fit(self, *args):
        """How well the model agrees with the data."""
        try:
            _index_data = self._arg_names.index(self._DATA_NAME)
            _index_model = self._arg_names.index(self._MODEL_NAME)
        except ValueError:
            return None
        _cost = self(*args)
        args = list(args)
        if self._add_constraint_cost:
            args = args[:-2]
        args[_index_model] = args[_index_data]
        _saturated_cost = self._cost_function_handle(*args)
        return _cost - _saturated_cost

    def chi2_probability(self, cost_function_value, ndf):
        """The chi2 probability associated with this cost function, None for non-chi2 cost
        functions.

        :param float cost_function_value: the associated cost function value.
        :param int ndf: the associated number of degrees of freedom.
        :returns: the associated chi2 probability.
        :rtype: float or None
        """
        return 1.0 - chi2.cdf(cost_function_value, ndf) if self.is_chi2 else None

    def get_uncertainty_gaussian_approximation(self, data):
        """Get the gaussian approximation of the uncertainty inherent to the cost function, returns
        0 by default.

        :param data: the fit data
        :return: the approximated gaussian uncertainty given the fit data
        """
        return 0

    def is_data_compatible(self, data):
        """Tests if model data is compatible with cost function

        :param data: the fit data
        :type data: numpy.ndarray
        :return: if the data is compatible, and if not a reason for the incompatibility
        :rtype: (boo, str)
        """
        return True, None


class CostFunction_Chi2(CostFunction):
    def __init__(
            self, errors_to_use='covariance', fallback_on_singular=True, add_constraint_cost=True):
        """Base class for built-in least-squares cost function.

        :param errors_to_use: Which errors to use when calculating :math:`\\chi^2`.
            Either ``'covariance'``, ``'pointwise'`` or ``None``.
        :type errors_to_use: str or None
        :param bool fallback_on_singular: If :py:obj:`True` and the covariance matrix is singular
            (or the errors are zero), calculate :math:`\\chi^2` as with ``errors_to_use=None``
        :param bool add_constraint_cost: If :py:obj:`True`, automatically add the cost for kafe2
            constraints.
        """

        _cost_function_description = "chi-square"
        if errors_to_use is None:
            _chi2_func = self.chi2_no_errors
            _arg_names = [self._DATA_NAME, self._MODEL_NAME]
            self._fail_on_no_matrix = False
            self._fail_on_no_errors = False
            _cost_function_description += " (no uncertainties)"
        elif errors_to_use.lower() == 'covariance':
            _chi2_func = self.chi2_covariance
            _arg_names = [self._DATA_NAME, self._MODEL_NAME, self._COV_MAT_INVERSE_NAME]
            self._fail_on_no_matrix = not fallback_on_singular
            self._fail_on_no_errors = True
            _cost_function_description += " (with covariance matrix)"
        elif errors_to_use.lower() == 'pointwise':
            _chi2_func = self.chi2_pointwise_errors
            _arg_names = [self._DATA_NAME, self._MODEL_NAME, self._ERROR_NAME]
            self._fail_on_no_matrix = False
            self._fail_on_no_errors = not fallback_on_singular
            _cost_function_description += " (with pointwise errors)"
        else:
            raise ValueError(
                "Unknown value '%s' for 'errors_to_use': must be one of "
                "('covariance', 'pointwise', None)")

        super(CostFunction_Chi2, self).__init__(
            cost_function=_chi2_func,
            arg_names=_arg_names,
            add_constraint_cost=add_constraint_cost
        )

        self._formatter.latex_name = "\\chi^2"
        self._formatter.name = "chi2"
        self._formatter.description = _cost_function_description
        self._needs_errors = errors_to_use is not None
        self._is_chi2 = True
        self._saturated = True

    def _chi2(self, data, model, cov_mat_inverse=None, err=None):
        data = np.asarray(data)
        model = np.asarray(model)

        if model.shape != data.shape:
            raise ValueError(
                "'data' and 'model' must have the same shape! Got %r and %r..."
                % (data.shape, model.shape))

        _res = (data - model)

        # if a covariance matrix inverse is given, use it
        if cov_mat_inverse is not None:
            _cost = _res.dot(cov_mat_inverse).dot(_res)
            if np.isnan(_cost):
                _cost = np.inf
            return _cost

        if self._fail_on_no_matrix:
            raise CostFunctionException("Covariance matrix is singular!")

        # otherwise, if an array of point-wise errors is given, use that
        if err is not None:
            err = np.asarray(err)
            if np.any(err == 0.0):
                if self._fail_on_no_errors:
                    raise CostFunctionException("'err' must not contain any zero values!")
            else:
                _res = _res / err

        if self.needs_errors:
            # There are other warnings that notify the user about singular cov mat, etc.
            warnings.warn("Setting all data errors to 1 as a fallback.")
        # return sum of squared residuals
        _cost = np.sum(_res ** 2)
        if np.isnan(_cost):
            _cost = np.inf
        return _cost

    def chi2_no_errors(self, data, model):
        r"""A least-squares cost function calculated from `y` data and model values,
        without considering uncertainties:

        .. math::
            C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})\cdot({\bf d} - {\bf m})
                +
                C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`

        :return: cost function value
        """
        return self._chi2(data=data, model=model)

    def chi2_covariance(self, data, model, total_cov_mat_inverse):
        r"""A least-squares cost function calculated from `y` data and model values,
        considering the covariance matrix of the `y` measurements.

        .. math::
            C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})^{\top}\,{{\bf V}^{-1}}\,({\bf d} - {\bf m})
                +
                C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        :math:`{{\bf V}^{-1}}` is the inverse of the total covariance matrix,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_cov_mat_inverse: inverse of the total covariance matrix :math:`{\bf V}^{-1}`

        :return: cost function value
        """
        return self._chi2(data=data, model=model, cov_mat_inverse=total_cov_mat_inverse)

    def chi2_pointwise_errors(self, data, model, total_error):
        r"""A least-squares cost function calculated from `y` data and model values,
        considering pointwise (uncorrelated) uncertainties for each data point:

        .. math::
            C = \chi^2({\bf d}, {\bf m}, {\bf \sigma}) = \sum_k \frac{d_k - m_k}{\sigma_k}
                +
                C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        :math:`{\bf \sigma}` are the pointwise total uncertainties,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_error: total error vector :math:`{\bf \sigma}`

        :return: cost function value
        """
        return self._chi2(data=data, model=model, err=total_error)


class CostFunction_NegLogLikelihood(CostFunction):
    def __init__(self, data_point_distribution='poisson', ratio=False):
        r"""
        Base class for built-in negative log-likelihood cost function.

        In addition to the measurement data and model predictions, likelihood-fits require a
        probability distribution describing how the measurements are distributed around the model
        predictions.
        This built-in cost function supports two such distributions: the *Poisson* and *Gaussian*
        (normal) distributions.

        In general, a negative log-likelihood cost function is defined as the double negative
        logarithm of the product of the individual likelihoods of the data points.

        The likelihood ratio is defined as ratio of the likelihood function for each individual
        observation, divided by the so-called *marginal likelihood*.

        :param data_point_distribution: Which type of statistics to use for modelling the
            distribution of individual data points. Either ``'poisson'`` or ``'gaussian'``.
        :type data_point_distribution: str
        :param ratio: If :py:obj:`True`, divide the likelihood by the marginal likelihood.
        :type ratio: bool
        """

        _cost_function_description = "negative log-likelihood"
        if data_point_distribution.lower() == "gaussian":
            if ratio:
                _nll_func = self.nllr_gaussian
                _cost_function_description += " ratio"
            else:
                _nll_func = self.nll_gaussian
            _cost_function_description += " (Gaussian uncertainties)"
            _arg_names = [self._DATA_NAME, self._MODEL_NAME, self._ERROR_NAME]
        elif data_point_distribution.lower() == "poisson":
            if ratio:
                _nll_func = self.nllr_poisson
                _cost_function_description += " ratio"
            else:
                _nll_func = self.nll_poisson
            _cost_function_description += " (Poisson uncertainties)"
            _arg_names = [self._DATA_NAME, self._MODEL_NAME]
        else:
            raise ValueError("Unknown value '%s' for 'data_point_distribution': "
                             "must be one of ('gaussian', 'poisson')!")

        super(CostFunction_NegLogLikelihood, self).__init__(
            cost_function=_nll_func,
            arg_names=_arg_names
        )

        if ratio:
            self._formatter.latex_name = r"-2\ln\mathcal{L}_{\rm R}"
            self._formatter.name = "nllr"
        else:
            self._formatter.latex_name = r"-2\ln\mathcal{L}"
            self._formatter.latex_name_saturated = r"-2\ln\mathcal{L}_{\rm R}"
            self._formatter.name = "nll"
            self._formatter.name_saturated = "nllr"
        self._formatter.description = _cost_function_description
        self._needs_errors = _nll_func in [self.nll_gaussian, self.nllr_gaussian]
        self._saturated = ratio

    @staticmethod
    def nll_gaussian(data, model, total_error):
        r"""A negative log-likelihood function assuming Gaussian statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}, {\bf \sigma})
              = -2 \ln \prod_j \mathcal{L}_{\rm Gaussian} (x=d_j, \mu=m_j, \sigma=\sigma_j)
                + C({\bf p})

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{1}{\sqrt{2{\sigma_j}^2\pi}}
                            \exp{\left(-\frac{ (d_j-m_j)^2 }{ {\sigma_j}^2}\right)}
                            +
                            C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        :math:`{\bf \sigma}` are the pointwise total uncertainties,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_error: total error vector :math:`{\bf \sigma}`

        :return: cost function value
        """

        _total_log_likelihood = np.sum(norm.logpdf(data, loc=model, scale=total_error))
        # guard against returning NaN
        if np.isnan(_total_log_likelihood):
            return np.inf
        return -2.0 * _total_log_likelihood

    @staticmethod
    def nll_poisson(data, model):
        r"""A negative log-likelihood function assuming Poisson statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m})
              = -2 \ln \prod_j \mathcal{L}_{\rm Poisson} (k=d_j, \lambda=m_j)
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

        :return: cost function value
        """

        _total_log_likelihood = np.sum(poisson.logpmf(data, mu=model, loc=0.0))
        # guard against returning NaN
        if np.isnan(_total_log_likelihood):
            return np.inf
        return -2.0 * _total_log_likelihood

    @staticmethod
    def nllr_gaussian(data, model, total_error):
        _total_log_likelihood = np.sum(norm.logpdf(data, loc=model, scale=total_error))
        _saturated_log_likelihood = np.sum(norm.logpdf(x=data, loc=data, scale=total_error))
        _log_likelihood_ratio = _total_log_likelihood - _saturated_log_likelihood
        # guard against returning NaN
        if np.isnan(_log_likelihood_ratio):
            return np.inf
        return -2.0 * _log_likelihood_ratio

    @staticmethod
    def nllr_poisson(data, model):
        _total_log_likelihood = np.sum(poisson.logpmf(k=data, mu=model, loc=0.0))
        _saturated_log_likelihood = np.sum(poisson.logpmf(k=data, mu=data, loc=0.0))
        _log_likelihood_ratio = _total_log_likelihood - _saturated_log_likelihood
        # guard against returning NaN
        if np.isnan(_log_likelihood_ratio):
            return np.inf
        return -2.0 * _log_likelihood_ratio

    def is_data_compatible(self, data):
        if self._cost_function_handle in [self.nll_poisson, self.nllr_poisson] \
                and (np.count_nonzero(data % 1) > 0 or np.any(data < 0)):
            return False, "poisson distribution can only have non-negative integers as y data."
        return True, None

    def get_uncertainty_gaussian_approximation(self, data):
        if self._cost_function_handle in [self.nll_poisson, self.nllr_poisson]:
            return np.sqrt(data)
        return 0


STRING_TO_COST_FUNCTION = {
    'chi2': (CostFunction_Chi2, {}),
    'chi_2': (CostFunction_Chi2, {}),
    'chisquared': (CostFunction_Chi2, {}),
    'chi_squared': (CostFunction_Chi2, {}),
    'nll': (CostFunction_NegLogLikelihood, {"ratio": False}),
    'nll-poisson': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll_poisson': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nllpoisson': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll-gaussian': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'nll_gaussian': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'nllgaussiann': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'negloglikelihood': (CostFunction_NegLogLikelihood, {"ratio": False}),
    'neg_log_likelihood': (CostFunction_NegLogLikelihood, {"ratio": False}),
    'nllr': (CostFunction_NegLogLikelihood, {"ratio": True}),
    'nllr-poisson': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllr_poisson': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllrpoisson': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllr-gaussian': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'nllr_gaussian': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'nllrgaussian': (CostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'negloglikelihoodratio': (CostFunction_NegLogLikelihood, {"ratio": True}),
    'neg_log_likelihood_ratio': (CostFunction_NegLogLikelihood, {"ratio": True}),
}
