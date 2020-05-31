from copy import copy
import six
from abc import ABCMeta, abstractmethod
import numpy as np
import numdifftools as nd
from scipy.optimize import brentq

from ..error import CovMat


class MinimizerException(Exception):
    pass


@six.add_metaclass(ABCMeta)
class MinimizerBase(object):

    ERRORDEF_CHI2 = 1.0
    ERRORDEF_NLL = 0.5

    def __init__(
            self, parameter_names, parameter_values, parameter_errors, function_to_minimize,
            tolerance=1e-6, errordef=ERRORDEF_CHI2):
        """
        :param parameter_names: the names of the parameters to vary during minimization.
        :type parameter_names: iterable of str
        :param parameter_values: the initial values of the parameters.
        :type parameter_values: iterable of float
        :param parameter_errors: the initial parameter errors, used to determine step size during
        minimization.
        :type parameter_errors: iterable of float
        :param function_to_minimize: the cost function to minimize.
        :type function_to_minimize: callable that returns a float
        :param tolerance: tolerance for convergence. Specifics depend on the used backend.
        :type tolerance: float
        :param errordef: difference in cost equivalent to the standard error.
        :type errordef: float
        """
        assert len(parameter_names) == len(parameter_values) == len(parameter_errors)
        self._invalidate_cache()  # initializes caches with None
        self.errordef = errordef
        self.tolerance = tolerance
        self._func_handle = function_to_minimize
        self._par_names = list(parameter_names)
        self.parameter_values = parameter_values
        self.parameter_errors = parameter_errors
        self._save_state_dict = dict()
        self._did_fit = False
        self._printed_inf_cost_warning = False

    def _invalidate_cache(self):
        """
        Invalidate the cache outside the used backend. Called at the end of `minimize` or `reset`.
        Parameter errors are **not** cleared because they need to be consitent.
        """
        self._fval = None
        self._par_asymm_err = None
        self._hessian = None
        self._hessian_inv = None
        self._par_cov_mat = None
        self._par_cor_mat = None

    def _save_state(self):
        """
        Save the state of this object including the state of the used backend, if any.
        """
        if self._par_asymm_err is None:
            self._save_state_dict['asymmetric_parameter_error'] = self._par_asymm_err
        else:
            self._save_state_dict['asymmetric_parameter_error'] = np.array(self._par_asymm_err)
        if self._hessian is None:
            self._save_state_dict['hessian'] = self._hessian
        else:
            self._save_state_dict['hessian'] = np.array(self._hessian)
        if self._hessian_inv is None:
            self._save_state_dict['hessian_inv'] = self._hessian_inv
        else:
            self._save_state_dict['hessian_inv'] = np.array(self._hessian_inv)
        if self._par_cov_mat is None:
            self._save_state_dict['par_cov_mat'] = self._par_cov_mat
        else:
            self._save_state_dict['par_cov_mat'] = np.array(self._par_cov_mat)
        if self._par_cor_mat is None:
            self._save_state_dict['par_cor_mat'] = self._par_cor_mat
        else:
            self._save_state_dict['par_cor_mat'] = np.array(self._par_cor_mat)
        self._save_state_dict['did_fit'] = self._did_fit

    def _load_state(self):
        """
        Load the state of this object including the state of the used backend, if any.
        """
        self._par_asymm_err = self._save_state_dict['asymmetric_parameter_error']
        if self._par_asymm_err is not None:
            self._par_asymm_err = np.array(self._par_asymm_err)
        self._hessian = self._save_state_dict['hessian']
        if self._hessian is not None:
            self._hessian = np.array(self._hessian)
        self._hessian_inv = self._save_state_dict['hessian_inv']
        if self._hessian_inv is not None:
            self._hessian_inv = np.array(self._hessian_inv)
        self._par_cov_mat = self._save_state_dict['par_cov_mat']
        if self._par_cov_mat is not None:
            self._par_cov_mat = np.array(self._par_cov_mat)
        self._par_cor_mat = self._save_state_dict['par_cor_mat']
        if self._par_cor_mat is not None:
            self._par_cor_mat = np.array(self._par_cor_mat)
        # Write back parameter values to nexus parameter nodes:
        self._func_wrapper_unpack_args(self.parameter_values)
        self._did_fit = self._save_state_dict['did_fit']

    def _func_wrapper(self, *args):
        """
        Wrapper for the cost function.
        :param args: cost function arguments (usually just the model function arguments) as varargs.
        :return: the cost function value.
        :rtype: float
        """
        _fval = self._func_handle(*args)
        if not self._printed_inf_cost_warning and np.isinf(_fval):
            print('Warning: the cost function has been evaluated as infinite. '
                  'The fit might not converge correctly.')
            self._printed_inf_cost_warning = True
        return _fval

    def _func_wrapper_unpack_args(self, args):
        """
        Wrapper for the cost function.
        :param args: cost function arguments (usually just the model function arguments) as an
        iterable.
        :type args: iterable of float
        :return: the cost function value.
        :rtype: float
        """
        return self._func_wrapper(*args)

    def _calculate_asymmetric_parameter_errors(self):  # TODO max calls
        """
        Calculate the asymmetric parameter errors. Works independently of the used backend, but
        might be overridden by a backend-specific implementation. Cost function must be negative log
        likelihood or chi squared to produce meaningful results. If no fit has been performed, all
        values in the return array are nan.
        :return: the asymmetric parameter errors for all parameters.
        :rtype numpy.ndarray of shape (num_pars, 2)
        """
        self.minimize()
        _ = self.parameter_errors  # call par error property so they're initialized for _save_state
        self._save_state()
        _asymm_par_errs = np.zeros(shape=self.parameter_values.shape + (2,))
        for _par_index, _par_name in enumerate(self.parameter_names):
            if self.is_fixed(_par_name):
                _asymm_par_errs[_par_index, :] = 0
            else:
                _target_chi_2 = self.function_value + 1.0
                _min_parameters = self.parameter_values

                _par_min = self.parameter_values[_par_index]
                _par_err = self.parameter_errors[_par_index]

                _cut_dn = self._find_cost_cut(
                    _par_name, _par_min - 2 * _par_err, _par_min, _target_chi_2, _min_parameters)
                _asymm_par_errs[_par_index, 0] = _cut_dn - _par_min

                _cut_up = self._find_cost_cut(
                    _par_name, _par_min, _par_min + 2 * _par_err, _target_chi_2, _min_parameters)
                _asymm_par_errs[_par_index, 1] = _cut_up - _par_min
                self._load_state()
        return _asymm_par_errs

    def _find_cost_cut(self, parameter_name, low, high, target_cost, min_parameters):
        """
        Utility function that finds the parameter value for a single parameter at which the cost
        function reaches a given value. The other parameters are **not** fixed. Instead the profile
        likelihood method is used.
        :param parameter_name: the name of the parameter to vary.
        :param low: lower bound of the parameter.
        :param high: upper bound of the parameter.
        :param target_cost: cost function value to find the cut for.
        :param min_parameters: parameter values at the cost function minimum.
        :return: the parameter value where the cost function value has the given value.
        :rtype: float
        """
        _all_pars_would_be_fixed = True
        for _par_name in self._par_names:
            if _par_name != parameter_name and not self.is_fixed(_par_name):
                _all_pars_would_be_fixed = False
                break

        def _profile(parameter_value):
            self.set_several(self.parameter_names, min_parameters)
            self.set(parameter_name, parameter_value)
            self._fval = None  # Clear fval cache
            if not _all_pars_would_be_fixed:
                self.fix(parameter_name)
                self.minimize()
                self.release(parameter_name)
            return self.function_value - target_cost

        return brentq(f=_profile, a=low, b=high, xtol=self.tolerance)

    def _remove_zeroes_for_fixed(self, matrix):
        """
        Takes a full error matrix and removes the rows and
        columns that corresponding to fixed parameters.
        """
        assert (matrix.shape == (self.num_pars, self.num_pars))

        _fixed_par_indices = [_i for _i, _par_name_i in enumerate(self._par_names)
                              if self.is_fixed(_par_name_i)]
        _submat = np.delete(
            np.delete(matrix, _fixed_par_indices, axis=0), _fixed_par_indices, axis=1)

        return _submat

    def _fill_in_zeroes_for_fixed(self, submatrix):
        """
        Takes the partial error matrix (submatrix) and adds
        rows and columns with 0.0 where the fixed
        parameters should go.
        """
        _mat = submatrix

        _fixed_par_indices = [_i for _i, _par_name_i in enumerate(self._par_names)
                              if self.is_fixed(_par_name_i)]
        for _id in _fixed_par_indices:
            _mat = np.insert(np.insert(_mat, _id, 0., axis=0), _id, 0., axis=1)

        assert (_mat.shape == (self.num_pars, self.num_pars))

        return _mat

    @property
    def did_fit(self):
        """
        :return: True if a fit has been performed after relevant input data has changed.
        """
        return self._did_fit

    @property
    def errordef(self):
        """
        :return: Difference in cost equivalent to the standard error.
        :rtype: float
        """
        return self._err_def

    @errordef.setter
    def errordef(self, err_def):
        assert err_def > 0
        self._err_def = err_def
        self.reset()

    @property
    def function_to_minimize(self):
        """
        :return: the cost function to minimize.
        :rtype: callable that returns a float
        """
        return self._func_handle

    @property
    def function_value(self):
        """
        :return: the cost function value using the current parameter values.
        :rtype: float
        """
        if self._fval is None:
            self._fval = self._func_handle(*self.parameter_values)
        if not self._printed_inf_cost_warning and np.isinf(self._fval):
            print('Warning: the cost function has been evaluated as infinite. '
                  'The fit might not converge correctly.')
            self._printed_inf_cost_warning = True
        return self._fval

    @property
    def num_pars(self):
        """
        :return: number of parameters to minimize the cost function value for.
        :rtype: int
        """
        return len(self._par_names)

    @property
    @abstractmethod
    def parameter_values(self):
        """
        :return: the current parameter values.
        :rtype: numpy.ndarray of shape (num_pars,)
        """

    @parameter_values.setter
    @abstractmethod
    def parameter_values(self, new_values):
        pass

    @property
    @abstractmethod
    def parameter_errors(self):
        """
        :return: the current parabolic parameter errors.
        :rtype: numpy.ndarray of shape (num_pars,)
        """

    @parameter_errors.setter
    @abstractmethod
    def parameter_errors(self, new_errors):
        pass

    @property
    def asymmetric_parameter_errors(self):
        """
        :return: the current asymmetric parameter errors derived from the likelihood.
        :rtype: numpy.ndarray of shape (num_pars, 2)
        """
        if not self.did_fit:
            return None
        if self._par_asymm_err is None:
            self._par_asymm_err = self._calculate_asymmetric_parameter_errors()
        return self._par_asymm_err.copy()

    @property
    def asymmetric_parameter_errors_if_calculated(self):
        """
        As asymmetric_parameter_errors, but return None if the asymmetric parameter errors have not
        been calculated so far.
        :return: the current asymmetric parameter errors derived from the likelihood or None.
        :rtype: numpy.ndarray of shape (num_pars, 2) or None
        """
        return None if self._par_asymm_err is None else self._par_asymm_err.copy()

    @property
    def parameter_names(self):
        """
        :return: the names of the parameters.
        :rtype: list of str
        """
        return copy(self._par_names)

    @property
    def tolerance(self):
        """
        :return: tolerance to use during minimization. Details depend on the used backend.
        :rtype: float
        """
        return self._tol

    @tolerance.setter
    def tolerance(self, tolerance):
        assert tolerance > 0
        self._tol = tolerance
        self.reset()

    @property
    def hessian(self):
        """
        :return: the Hessian matrix calculated at the current parameter values.
        :rtype: numpy.ndarray of shape (num_pars, num_pars)
        """
        if not self.did_fit:
            return None
        if self._hessian is None:
            self._hessian = nd.Hessian(self._func_wrapper_unpack_args)(self.parameter_values)
            assert(np.all(self._hessian == self._hessian.T))
            # Write back parameter values to nexus parameter nodes:
            self._func_wrapper_unpack_args(self.parameter_values)
        return self._hessian.copy()

    @property
    def hessian_inv(self):
        """
        :return: the inverse of the Hessian matrix calculated at the current parameter values.
        :rtype: numpy.ndarray of shape (num_pars, num_pars)
        """
        if not self.did_fit:
            return None
        if self._hessian_inv is None:
            _hessian = self.hessian  # including zeroes for fixed parameter
            _subhessian = self._remove_zeroes_for_fixed(_hessian)
            _subhessian_inv = np.linalg.inv(_subhessian)
            self._hessian_inv = self._fill_in_zeroes_for_fixed(_subhessian_inv)
            # ensure symmetric
            self._hessian_inv = 0.5 * (self._hessian_inv + self._hessian_inv.T)
            assert (np.all(self._hessian_inv == self._hessian_inv.T))
        return self._hessian_inv.copy()

    @property
    def cov_mat(self):
        """
        :return: the parameter covariance matrix calculated at the current parameter values.
        :rtype: numpy.ndarray of shape (num_pars, num_pars)
        """
        if not self.did_fit:
            return None
        if self._par_cov_mat is None:
            self._par_cov_mat = self.hessian_inv * 2.0 * self.errordef
        return self._par_cov_mat.copy()

    @property
    def cor_mat(self):
        """
        :return: the parameter correlation matrix calculated at the current parameter values.
        :rtype: numpy.ndarray of shape (num_pars, num_pars)
        """
        if not self.did_fit:
            return None
        if self._par_cor_mat is None:
            _subcov_mat = self._remove_zeroes_for_fixed(self.cov_mat)
            _subcor_mat = CovMat(_subcov_mat).cor_mat
            self._par_cor_mat = self._fill_in_zeroes_for_fixed(_subcor_mat)
        return self._par_cor_mat.copy()

    def reset(self):
        """Clears caches and resets the internal state of the used backend (if any)."""
        self._invalidate_cache()
        self._did_fit = False

    @abstractmethod
    def set(self, parameter_name, parameter_value):
        """
        Sets one parameter to the given value.
        :param parameter_name: the name of the parameter.
        :type parameter_name: str
        :param parameter_value: the value to set the parameter to.
        :type parameter_value: float
        """

    def set_several(self, parameter_names, parameter_values):
        """
        Sets multiple parameters to the given values.
        :param parameter_names: the names of the parameter.
        :type parameter_names: iterable of str
        :param parameter_values: the values to set the parameters to.
        :type parameter_values: iterable of float
        """
        for _pn, _pv in zip(parameter_names, parameter_values):
            self.set(_pn, _pv)

    @abstractmethod
    def limit(self, parameter_name, parameter_bounds):
        """
        Limit a parameter value to a given closed interval during minimization.
        :param parameter_name: the name of the parameter to limit.
        :type parameter_name: str
        :param parameter_bounds: the lower and upper bounds of the parameter.
        :type parameter_bounds: tuple consisting of two floats: (lower_bound, upper_bound)
        """

    @abstractmethod
    def unlimit(self, parameter_name):
        """
        Reverse the limitation of a parameter.
        :param parameter_name: the name of the parameter to unlimit.
        :type parameter_name: str
        """

    @abstractmethod
    def fix(self, parameter_name):
        """
        Fix a parameter to its current value. It will be treated as constant during minimization.
        :param parameter_name: the name of the parameter to fix.
        :type parameter_name: str
        """

    def fix_several(self, parameter_names):
        """
        Fix several parameters to their current values. They will be treated as constant during
        minimization.
        :param parameter_names: the names of the parameters to fix.
        :type parameter_names: iterable of str
        """
        for _pn in parameter_names:
            self.fix(_pn)

    @abstractmethod
    def is_fixed(self, parameter_name):
        """
        Check if the given parameter is currently fixed.
        :param parameter_name: the name of the parameter to check fixed status for.
        :type parameter_name: str
        :return: True if the parameter is currently fixed.
        :rtype: bool
        """

    @abstractmethod
    def release(self, parameter_name):
        """
        Revert the fixation of a parameter. It will no longer be treated as constant during
        minimization.
        :param parameter_name: the name of the parameter to release.
        :type parameter_name: str
        """

    def release_several(self, parameter_names):
        """
        Revert the fixation of several parameters. They will no longer be treated as constant during
        minimization.
        :param parameter_names: the names of the parameter to release.
        :type parameter_names: iterable of str
        """
        for _pn in parameter_names:
            self.release(_pn)

    @abstractmethod
    def minimize(self):  # TODO max calls
        """
        Minimize the cost function with the used backend.
        """

    @abstractmethod
    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, **minimizer_contour_kwargs):
        """
        Calculate a 2D contour using the profile likelihood method: two parameters are fixed while
        the rest are varied to minimize the cost function. The parameter values of the fixed
        parameters at which the cost function value has a given offset relative to the cost function
        minimum constitute the contour.
        :param parameter_name_1: the name of the first parameter to fix.
        :type parameter_name_1: str
        :param parameter_name_2:
        :type parameter_name_2: str
        :param sigma: difference between the cost function minimum and the contour.
        :type sigma: float
        :param minimizer_contour_kwargs: backend-specific kwargs.
        :return: the calculated contour
        :rtype: kafe2.core.contour.Contour
        """

    @abstractmethod
    def profile(self, parameter_name, bins=20, bound=2, subtract_min=False):
        """
        Calculate a 1D profile using the profile likelihood method: a single parameter is fixed
        while the rest are varied to minimize the cost function. The mapping of parameter value to
        cost function value around the minimum is the profile.
        The interval [par_min - par_err * bound, par_min + par_err * bound] is profiled by this
        method.
        :param parameter_name: the name of the parameter to fix.
        :type parameter_name: str
        :param bins: the number of points to evaluate the cost function at.
        :type bins: int
        :param bound: parameter determining the size of the interval to profile (see above).
        :type bound: float
        :param subtract_min: if True, subtract the cost function value of the minimum from the cost
        function values of the profile.
        :type subtract_min: bool
        :return: the parameter values of the fixed parameter and the corresponding cost function
        values.
        :rtype: numpy.ndarray of shape (2, bins)
        """
