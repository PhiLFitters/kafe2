import abc
import numpy as np
import six

from scipy.stats import poisson, norm
from ..io.file import FileIOMixin
from .format import ParameterFormatter, CostFunctionFormatter


if six.PY2:
    from funcsigs import signature
else:
    from inspect import signature


__all__ = ["CostFunctionBase",
           "CostFunctionBase_Chi2",
           "CostFunctionBase_NegLogLikelihood",
           "CostFunctionBase_NegLogLikelihoodRatio",
           "CostFunctionException"]


def _generic_chi2(data, model,
                  cov_mat_inverse=None,
                  err=None, err_relative_to=None,
                  fail_on_no_matrix=False,
                  fail_on_zero_errors=False,
                  parameter_values=None,
                  parameter_constraints=None):

    data = np.asarray(data)
    model = np.asarray(model)

    if model.shape != data.shape:
        raise CostFunctionException("'data' and 'model' must have the same shape! Got %r and %r..."
                                    % (data.shape, model.shape))

    _par_cost = 0.0
    if parameter_constraints is not None:
        for _par_constraint in parameter_constraints:
            _par_cost += _par_constraint.cost(parameter_values)

    _res = (data - model)

    # if a covariance matrix inverse is given, use it
    if cov_mat_inverse is not None:
        return _res.dot(cov_mat_inverse).dot(_res) + _par_cost

    if fail_on_no_matrix:
        raise np.linalg.LinAlgError("Covariance matrix is singular!")

    # otherwise, if an array of point-wise errors is given, use that
    if err is not None:
        err = np.asarray(err)
        if err.shape != data.shape:
            raise CostFunctionException("'err' must have the same shape as 'data'! Got %r and %r..."
                                        % (err.shape, data.shape))

        if err_relative_to == 'data':
            err *= data
        elif err_relative_to == 'model':
            err *= model
        elif err_relative_to is not None:
            raise CostFunctionException("'err_relative_to' must be either 'data', 'model' or None")

        if np.any(err == 0.0):
            if fail_on_zero_errors:
                raise CostFunctionException("'err' must not contain any zero values!")
        else:
            _res = _res/err

    # return sum of squared residuals
    if np.isnan(_res).any():
        return np.inf
    return np.sum(_res ** 2) + _par_cost


def _generic_chi2_nuisance(data, model,
                           nuisance_vector=np.array([]),
                           nuisance_cor_design_mat=None,
                           uncor_cov_mat_inverse=None,
                           fail_on_no_matrix=False,
                           parameter_values=None, parameter_constraints=None):

    data = np.asarray(data)
    model = np.asarray(model)

    if model.shape != data.shape:
        raise CostFunctionException("'data' and 'model' must have the same shape! Got %r and %r..."
                                    % (data.shape, model.shape))

    _par_cost = 0.0
    if parameter_constraints is not None:
        for _par_constraint in parameter_constraints:
            _par_cost += _par_constraint.cost(parameter_values)

    # if an uncorrelated cov-mat is given use it
    if uncor_cov_mat_inverse is not None:
        _inner_sum = np.squeeze(np.asarray(nuisance_vector.dot(nuisance_cor_design_mat)))
        _nuisance_penalties = nuisance_vector.dot(nuisance_vector)
        _chisquare = (data - model - _inner_sum).dot(uncor_cov_mat_inverse).dot(data - model - _inner_sum)
        if np.isnan(_chisquare):
            return np.inf
        return _chisquare + _nuisance_penalties + _par_cost

    # raise if uncorrelated matrix is None and the correlated is not None
    if nuisance_vector.all() == 0.0:
        raise CostFunctionException('y-errors must not be fully correlated!')

    if fail_on_no_matrix:
        raise np.linalg.LinAlgError("Uncorrelated Covariance matrix is singular!")
    # chisquare without errors
    _chisquare = (data -model).dot(data-model)
    if np.isnan(_chisquare):
        return np.inf
    return _chisquare + _par_cost


class CostFunctionException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class CostFunctionBase(FileIOMixin, object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    cost functions.

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
    FORMATTER_TYPE = CostFunctionFormatter

    def __init__(self, cost_function):
        """
        Construct :py:class:`CostFunction` object (a wrapper for a native Python function):

        :param cost_function: function handle
        """
        self._cost_function_handle = cost_function
        self._cost_function_signature = signature(self._cost_function_handle)
        self._cost_function_argcount = self._cost_function_handle.__code__.co_argcount
        self._validate_cost_function_raise()
        self._assign_function_formatter()

        self._flags = {}
        self._ndf = None
        self._needs_errors = True
        self._is_chi2 = False
        self._no_errors_warning_printed = False
        super(CostFunctionBase, self).__init__()

    @classmethod
    def _get_base_class(cls):
        return CostFunctionBase

    @classmethod
    def _get_object_type_name(cls):
        return 'cost_function'

    def _validate_cost_function_raise(self):
        self._cost_func_signature = signature(self._cost_function_handle)
        if 'cost' in self._cost_func_signature.parameters:
            raise self.__class__.EXCEPTION_TYPE(
                "The alias 'cost' for the cost function value cannot be used as an argument to the cost function!")

        # evaluate general cost function requirements
        for _par in self._cost_func_signature.parameters.values():
            if _par.kind == _par.VAR_POSITIONAL:
                raise self.__class__.EXCEPTION_TYPE("Cost function '{}' with variable number of positional arguments "
                                                    "(*{}) is not supported".format(self._cost_function_handle.__name__,
                                                                                    _par.name))
            if _par.kind == _par.VAR_KEYWORD:
                raise self.__class__.EXCEPTION_TYPE("Cost function '{}' with variable number of keyword arguments "
                                                    "(**{}) is not supported".format(self._cost_function_handle.__name__,
                                                                                     _par.name,))

        # TODO: fail if cost function does not depend on data or model

    def _get_parameter_formatters(self):
        return [ParameterFormatter(name=_pn, value=_pv, error=None)
                for _pn, _pv in zip(self.signature.parameters, self.argvals)]

    def _assign_function_formatter(self):
        self._formatter = self.__class__.FORMATTER_TYPE(
            self.name, arg_formatters=self._get_parameter_formatters())

    def __call__(self, *args, **kwargs):
        return self._cost_function_handle(*args, **kwargs)

    @property
    def name(self):
        """The cost function name (a valid Python identifier)"""
        return self._cost_function_handle.__name__

    @property
    def func(self):
        """The cost function handle"""
        return self._cost_function_handle

    @property
    def signature(self):
        """The model function argument specification, as returned by :py:meth:`inspect.signature`"""
        return self._cost_function_signature

    @property
    def argcount(self):
        """The number of arguments the model function accepts
        (including any independent variables which are not parameters)"""
        return self._cost_function_argcount

    @property
    def argvals(self):
        """The current values of the function arguments (**not implemented**, returns an array of zeros)"""
        # NOTE: only exists because needed by formatter (FIXME?)
        return [0.0] * (self.argcount)

    @property
    def formatter(self):
        """The :py:obj:`Formatter` object for this function"""
        return self._formatter

    @property
    def argument_formatters(self):
        """The :py:obj:`Formatter` objects for the function arguments"""
        return self._formatter.arg_formatters

    @property
    def ndf(self):
        """The number of degrees of freedom of this cost function"""
        return self._ndf

    @ndf.setter
    def ndf(self, new_ndf):
        """The number of degrees of freedom of this cost function"""
        assert new_ndf > 0  # ndf must be positive
        assert new_ndf == int(new_ndf)  # ndf must be integer
        self._ndf = new_ndf

    @property
    def needs_errors(self):
        """Whether the cost function needs errors for a meaningful result"""
        return self._needs_errors

    @property
    def is_chi2(self):
        """Whether the cost function is a chi2 cost function."""
        return self._is_chi2

    def get_uncertainty_gaussian_approximation(self, data):
        """
        Get the gaussian approximation of the uncertainty inherent to the cost function, returns 0 by default.
        :param data: the fit data
        :return: the approximated gaussian uncertainty given the fit data
        """
        return 0

    def set_flag(self, name, value):
        self._flags[name] = value

    def get_flag(self, name):
        return self._flags.get(name, None)

    def is_data_compatible(self, data):
        """
        Tests if model data is compatible with cost function
        :param data: the fit data
        :type data: numpy.ndarray
        :return: if the data is compatible, and if not a reason for the incompatibility
        :rtype: (boo, str)
        """
        return True, None

    def on_no_errors(self):
        if not self._no_errors_warning_printed:
            print('WARNING: No data errors were specified. The fit results may be wrong.')
            self._no_errors_warning_printed = True


class CostFunctionBase_Chi2(CostFunctionBase):
    def __init__(self, errors_to_use='covariance', fallback_on_singular=True):
        """
        Base class for built-in least-squares cost function.

        :param errors_to_use: which errors to use when calculating :math:`\chi^2`
        :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
        :param fallback_on_singular: if ``True`` and the covariance matrix is singular (or the errors are zero),
                                     calculate :math:`\chi^2` as with ``errors_to_use=None``
        :type fallback_on_singular: bool
        """

        _cost_function_description = "chi-square"
        if errors_to_use is None:
            _chi2_func = self.chi2_no_errors
            _cost_function_description += " (no uncertainties)"
        elif errors_to_use.lower() == 'covariance':
            if fallback_on_singular:
                _chi2_func = self.chi2_covariance_fallback
            else:
                _chi2_func = self.chi2_covariance
            _cost_function_description += " (with covariance matrix)"
        elif errors_to_use.lower() == 'pointwise':
            if fallback_on_singular:
                _chi2_func = self.chi2_pointwise_errors_fallback
            else:
                _chi2_func = self.chi2_pointwise_errors
            _cost_function_description += " (with pointwise errors)"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'errors_to_use': must be one of ('covariance', 'pointwise', None)")

        super(CostFunctionBase_Chi2, self).__init__(cost_function=_chi2_func)

        self._formatter.latex_name = "\\chi^2"
        self._formatter.name = "chi2"
        self._formatter.description = _cost_function_description
        self._needs_errors = _chi2_func is not self.chi2_no_errors
        self._is_chi2 = True

    def on_no_errors(self):
        if not self._no_errors_warning_printed:
            if (self._cost_function_handle is self.chi2_covariance_fallback
                    or self._cost_function_handle is self.chi2_pointwise_errors_fallback):
                print('WARNING: No data errors were specified. Setting data errors to 1.')
            else:
                print('WARNING: No data errors were specified. The fit results may be wrong.')
            self._no_errors_warning_printed = True

    @staticmethod
    def chi2_no_errors(data, model, parameter_values, parameter_constraints):
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
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        return _generic_chi2(data=data, model=model, cov_mat_inverse=None, fail_on_no_matrix=False,
                             parameter_values=parameter_values, parameter_constraints=parameter_constraints)

    @staticmethod
    def chi2_covariance(data, model, total_cov_mat_inverse, parameter_values, parameter_constraints):
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
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        return _generic_chi2(data=data, model=model, cov_mat_inverse=total_cov_mat_inverse, fail_on_no_matrix=True,
                             parameter_values=parameter_values, parameter_constraints=parameter_constraints)

    @staticmethod
    def chi2_pointwise_errors(data, model, total_error, parameter_values, parameter_constraints):
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
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        return _generic_chi2(data=data, model=model, cov_mat_inverse=None, err=total_error, fail_on_zero_errors=True,
                             parameter_values=parameter_values, parameter_constraints=parameter_constraints)

    @staticmethod
    def chi2_covariance_fallback(data, model, total_cov_mat_inverse, parameter_values, parameter_constraints):
        return _generic_chi2(
            data=data, model=model, cov_mat_inverse=total_cov_mat_inverse, parameter_values=parameter_values,
            parameter_constraints=parameter_constraints, fail_on_no_matrix=False
        )

    @staticmethod
    def chi2_pointwise_errors_fallback(data, model, total_error, parameter_values, parameter_constraints):
        return _generic_chi2(data=data, model=model, cov_mat_inverse=None, err=total_error, fail_on_zero_errors=False,
                             parameter_values=parameter_values, parameter_constraints=parameter_constraints)


class CostFunctionBase_NegLogLikelihood(CostFunctionBase):
    def __init__(self, data_point_distribution='poisson'):
        r"""
        Base class for built-in negative log-likelihood cost function.

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

        _cost_function_description = "negative log-likelihood"
        if data_point_distribution.lower() == 'gaussian':
            _nll_func = self.nll_gaussian
            _cost_function_description += " (Gaussian uncertainties)"
        elif data_point_distribution.lower() == 'poisson':
            _nll_func = self.nll_poisson
            _cost_function_description += " (Poisson uncertainties)"
        else:
            raise CostFunctionException("Unknown value '%s' for 'data_point_distribution': must be one of ('gaussian', 'poisson')!")

        super(CostFunctionBase_NegLogLikelihood, self).__init__(cost_function=_nll_func)

        self._formatter.latex_name = "-2\\ln\\mathcal{L}"
        self._formatter.name = "nll"
        self._formatter.description = _cost_function_description
        self._needs_errors = _nll_func is self.nll_gaussian

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
        _par_cost = 0.0
        if parameter_constraints is not None:
            for _par_constraint in parameter_constraints:
                _par_cost += _par_constraint.cost(parameter_values)

        _total_log_likelihood = np.sum(norm.logpdf(data, loc=model, scale=total_error))
        # guard against returning NaN
        if np.isnan(_total_log_likelihood):
            return np.inf
        return -2.0 * _total_log_likelihood + _par_cost

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
        _par_cost = 0.0
        if parameter_constraints is not None:
            for _par_constraint in parameter_constraints:
                _par_cost += _par_constraint.cost(parameter_values)

        _total_log_likelihood = np.sum(poisson.logpmf(data, mu=model, loc=0.0))
        # guard against returning NaN
        if np.isnan(_total_log_likelihood):
            return np.inf
        return -2.0 * _total_log_likelihood + _par_cost

    def is_data_compatible(self, data):
        if self._cost_function_handle is self.nll_poisson and (np.count_nonzero(data % 1) > 0 or np.any(data < 0)):
            return False, "poisson distribution can only have non-negative integers as y data."
        return True, None

    def get_uncertainty_gaussian_approximation(self, data):
        if self._cost_function_handle is self.nll_poisson:
            return np.sqrt(data)
        return 0


class CostFunctionBase_NegLogLikelihoodRatio(CostFunctionBase):
    def __init__(self, data_point_distribution='poisson'):
        r"""
        Base class for built-in negative log-likelihood ratio cost function.

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

        _cost_function_description = "negative log-likelihood ratio"
        if data_point_distribution.lower() == 'gaussian':
            _nll_func = self.nllr_gaussian
            _cost_function_description += " (Gaussian uncertainties)"
        elif data_point_distribution.lower() == 'poisson':
            _nll_func = self.nllr_poisson
            _cost_function_description += " (Poisson uncertainties)"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'data_point_distribution': must be one of ('gaussian', 'poisson')!")

        super(CostFunctionBase_NegLogLikelihoodRatio, self).__init__(cost_function=_nll_func)

        self._formatter.latex_name = r"-2\ln\mathcal{L}_{\rm R}"
        self._formatter.name = "nllr"
        self._formatter.description = _cost_function_description
        self._needs_errors = _nll_func is self.nllr_gaussian

    @staticmethod
    def nllr_gaussian(data, model, total_error, parameter_values, parameter_constraints):
        r"""A negative log-likelihood ratio function assuming Gaussian statistics for each measurement.

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
        _par_cost = 0.0
        if parameter_constraints is not None:
            for _parameter_constraint in parameter_constraints:
                _par_cost += _parameter_constraint.cost(parameter_values)
        _total_log_likelihood = np.sum(norm.logpdf(data, loc=model, scale=total_error))
        _saturated_log_likelihood = np.sum(norm.logpdf(x=data, loc=data, scale=total_error))
        _log_likelihood_ratio = _total_log_likelihood - _saturated_log_likelihood
        # guard against returning NaN
        if np.isnan(_log_likelihood_ratio):
            return np.inf
        return -2.0 * _log_likelihood_ratio + _par_cost

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
        _par_cost = 0.0
        if parameter_constraints is not None:
            for _parameter_constraint in parameter_constraints:
                _par_cost += _parameter_constraint.cost(parameter_values)
        _total_log_likelihood = np.sum(poisson.logpmf(k=data, mu=model, loc=0.0))
        _saturated_log_likelihood = np.sum(poisson.logpmf(k=data, mu=data, loc=0.0))
        _log_likelihood_ratio = _total_log_likelihood - _saturated_log_likelihood
        # guard against returning NaN
        if np.isnan(_log_likelihood_ratio):
            return np.inf
        return -2.0 * _log_likelihood_ratio + _par_cost

    def is_data_compatible(self, data):
        if self._cost_function_handle is self.nllr_poisson and (np.count_nonzero(data % 1) > 0 or np.any(data < 0)):
            return False, "poisson distribution can only have non-negative integers as y data."
        return True, None

    def get_uncertainty_gaussian_approximation(self, data):
        if self._cost_function_handle is self.nllr_poisson:
            return np.sqrt(data)
        return 0


class CostFunctionBase_Chi2_Nuisance(CostFunctionBase_Chi2):

    def __init__(self, errors_to_use='covariance', fall_back_on_singular=True):
        """
        Base class for built-in least-squares cost function, which uses nuisance-parameters.

          :param errors_to_use: which errors to use when calculating :math:`\chi^2`
          :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
          :param fallback_on_singular: if ``True`` and the covariance matrix is singular (or the errors are zero),
                                                 calculate :math:`\chi^2` as with ``errors_to_use=None``
          :type fallback_on_singular: bool
        """

        if errors_to_use is None:
            _chi2_nui = self.chi2_no_errors  # inherited from CostFunctionBase_Chi2
        elif errors_to_use == 'covariance':
            if fall_back_on_singular:
                _chi2_nui = self.chi2_nui_cov_fallback
            else:
                _chi2_nui = self.chi2_nui_cov
        elif errors_to_use == 'pointwise':
            if fall_back_on_singular:
                _chi2_nui = self.chi2_nui_pointwise_fallback
            else:
                _chi2_nui = self.chi2_nui_pointwise
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'errors_to_use': must be one of ('covariance', 'pointwise', None)")

        super(CostFunctionBase_Chi2, self).__init__(cost_function=_chi2_nui)
        #set flag for creating nuisance-parameters
        self.set_flag("need_nuisance", True)

        self._formatter.latex_name = r"\chi^{2}_{nui}"
        self._formatter.name = "chi2_nui"
        self._formatter.description = "chi-square (with nuisance parameters for correlated uncertainties)"

    @staticmethod
    def chi2_nui_cov(data, model, total_uncor_cov_mat_inverse, total_nuisance_cor_design_mat, nuisance_vector,
                     parameter_values, parameter_constraints):
        r"""A least-squares cost function that accounts for correlated uncertainties through nuisance parameters. The nuisance parameters are fitted.

        The cost function is given by:

        .. math::
            C = \chi^2 =
                ({\bf d} - {\bf m} - {\bf G}{\bf b})^{\top}
                ({\bf V}^{\mathrm{uncor}})^{-1}
                ({\bf d} - {\bf m} - {\bf G}{\bf b})
                +
                {\bf b}^2
                +
                C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        :math:{\bf cor_cov} is the total nuisance correlated covariance matrix,
        :math:{\bf uncor_cov}} is the inverse of the total uncorrelated covariance matrix,
        :math:{\bf nui} is the Nuisance-vector,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_uncor_cov_mat_inverse: inverse :math:`({\bf V}^{\mathrm{uncor}})^{-1}` of the uncorrelated part of the total covariance matrix
        :param total_nuisance_cor_design_mat: design matrix :math:`{\bf G}` containing correlated uncertainties
        :param nuisance_vector: nuisance parameter vector :math:`{\bf b}`
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints
        :return: cost function value
        """
        return _generic_chi2_nuisance(
            data=data, model=model, uncor_cov_mat_inverse=total_uncor_cov_mat_inverse,
            nuisance_cor_design_mat=total_nuisance_cor_design_mat, nuisance_vector=nuisance_vector,
            fail_on_no_matrix=True, parameter_values=parameter_values, parameter_constraints=parameter_constraints)

    @staticmethod
    def chi2_nui_cov_fallback(data, model, total_uncor_cov_mat_inverse, total_nuisance_cor_design_mat, nuisance_vector,
                              parameter_values, parameter_constraints):
        r"""A least-squares cost function that accounts for correlated uncertainties through nuisance parameters. The nuisance parameters are fitted.

        .. TODO: describe fallback behavior

        The cost function is given by:

        .. math::
            C = \chi^2 =
                ({\bf d} - {\bf m} - {\bf G}{\bf b})^{\top}
                ({\bf V}^{\mathrm{uncor}})^{-1}
                ({\bf d} - {\bf m} - {\bf G}{\bf b})
                +
                {\bf b}^2
                +
                C({\bf p})

        In the above, :math:`{\bf d}` are the measurements,
        :math:`{\bf m}` are the model predictions,
        :math:{\bf cor_cov} is the total nuisance correlated covariance matrix,
        :math:{\bf uncor_cov}} is the inverse of the total uncorrelated covariance matrix,
        :math:{\bf nui} is the Nuisance-vector,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param data: measurement data :math:`{\bf d}`
        :param model: model predictions :math:`{\bf m}`
        :param total_uncor_cov_mat_inverse: inverse :math:`({\bf V}^{\mathrm{uncor}})^{-1}` of the uncorrelated part of the total covariance matrix
        :param total_nuisance_cor_design_mat: design matrix :math:`{\bf G}` containing correlated uncertainties
        :param nuisance_vector: nuisance parameter vector :math:`{\bf b}`
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return: cost function value
        """
        return _generic_chi2_nuisance(
            data=data, model=model, uncor_cov_mat_inverse=total_uncor_cov_mat_inverse,
            nuisance_cor_design_mat=total_nuisance_cor_design_mat, nuisance_vector=nuisance_vector,
            fail_on_no_matrix=False, parameter_values=parameter_values, parameter_constraints=parameter_constraints)

    @staticmethod
    def chi2_nui_pointwise(data, model, total_error, parameter_values, parameter_constraints):
        r"""A least-squares cost function calculated from the data and model values,
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
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return cost function value
        """
        return CostFunctionBase_Chi2.chi2_pointwise_errors(
            data=data, model=model, total_error=total_error, parameter_values=parameter_values,
            parameter_constraints=parameter_constraints)

    @staticmethod
    def chi2_nui_pointwise_fallback(data, model, total_error, parameter_values, parameter_constraints):
        r"""A least-squares cost function calculated from the data and model values,
              considering pointwise (uncorrelated) uncertainties for each data point:

        .. TODO: describe fallback behavior

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
        :param parameter_values: vector of parameters :math:`{\bf p}`
        :param parameter_constraints: list of fit parameter constraints

        :return cost function value
        """
        return CostFunctionBase_Chi2.chi2_pointwise_errors_fallback(
            data=data, model=model, total_error=total_error, parameter_values=parameter_values,
            parameter_constraints=parameter_constraints)
