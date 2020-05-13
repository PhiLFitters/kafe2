import numpy as np
import numdifftools as nd
from scipy.optimize import brentq

from ..error import CovMat


class MinimizerBase(object):

    def __init__(self, function_to_minimize):
        self._func_handle = function_to_minimize
        self._invalidate_cache()  # initializes caches with None
        self._save_state_dict = dict()
        self._printed_inf_cost_warning = False

    def _invalidate_cache(self):
        self._fval = None
        self._par_asymm_err = None
        self._hessian = None
        self._hessian_inv = None
        self._par_cov_mat = None
        self._par_cor_mat = None

    def _save_state(self):
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

    def _load_state(self):
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

    def _func_wrapper(self, *args):
        _fval = self._func_handle(*args)
        if not self._printed_inf_cost_warning and np.isinf(_fval):
            print('Warning: the cost function has been evaluated as infinite. The fit might not converge correctly.')
            self._printed_inf_cost_warning = True
        return _fval

    def _func_wrapper_unpack_args(self, args):
        return self._func_wrapper(*args)

    def _calculate_asymmetric_parameter_errors(self):  # TODO max calls
        self.minimize()
        self.parameter_errors  # call par error property so they're initialized for _save_state
        self._save_state()
        _asymm_par_errs = np.zeros(shape=self.parameter_values.shape + (2,))
        for _par_index, _par_name in enumerate(self.parameter_names):
            if self.is_fixed(_par_name):
                _asymm_par_errs[_par_index, 0] = np.nan
                _asymm_par_errs[_par_index, 1] = np.nan
            else:
                _target_chi_2 = self.function_value + 1.0
                _min_parameters = self.parameter_values

                _par_min = self.parameter_values[_par_index]
                _par_err = self.parameter_errors[_par_index]

                _cut_dn = self._find_chi_2_cut(_par_name, _par_min - 2 * _par_err, _par_min, _target_chi_2,
                                               _min_parameters)
                _asymm_par_errs[_par_index, 0] = _cut_dn - _par_min

                _cut_up = self._find_chi_2_cut(_par_name, _par_min, _par_min + 2 * _par_err, _target_chi_2,
                                               _min_parameters)
                _asymm_par_errs[_par_index, 1] = _cut_up - _par_min
                self._load_state()
        return _asymm_par_errs

    def _find_chi_2_cut(self, parameter_name, low, high, target_chi_2, min_parameters):
        def _profile(parameter_value):
            self.set_several(self.parameter_names, min_parameters)
            self.set(parameter_name, parameter_value)
            self.fix(parameter_name)
            self.minimize()
            _fval = self.function_value
            self.release(parameter_name)
            return _fval - target_chi_2

        return brentq(f=_profile, a=low, b=high, xtol=self.tolerance)

    @property
    def function_to_minimize(self):
        return self._func_handle

    @property
    def function_value(self):
        if self._fval is None:
            self._fval = self._func_handle(*self.parameter_values)
        if not self._printed_inf_cost_warning and np.isinf(self._fval):
            print('Warning: the cost function has been evaluated as infinite. The fit might not converge correctly.')
            self._printed_inf_cost_warning = True
        return self._fval

    @property
    def num_pars(self):
        raise NotImplementedError()

    @property
    def parameter_values(self):
        raise NotImplementedError()

    @property
    def parameter_errors(self):
        raise NotImplementedError()

    @property
    def asymmetric_parameter_errors(self):
        if self._par_asymm_err is None:
            self._par_asymm_err = self._calculate_asymmetric_parameter_errors()
        return self._par_asymm_err

    @property
    def asymmetric_parameter_errors_if_calculated(self):
        return self._par_asymm_err

    @property
    def parameter_names(self):
        raise NotImplementedError()

    @property
    def tolerance(self):
        raise NotImplementedError()

    @tolerance.setter
    def tolerance(self, new_tol):
        raise NotImplementedError()

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian = nd.Hessian(self._func_wrapper_unpack_args)(self.parameter_values)
            assert(np.all(self._hessian == self._hessian.T))
        # Write back parameter values to nexus parameter nodes:
        self._func_wrapper_unpack_args(self.parameter_values)
        return self._hessian

    @property
    def hessian_inv(self):
        if self._hessian_inv is None:
            self._hessian_inv = np.linalg.inv(self.hessian)
            self._hessian_inv = 0.5 * (self._hessian_inv + self._hessian_inv.T)
        assert (np.all(self._hessian_inv == self._hessian_inv.T))
        return self._hessian_inv

    @property
    def cov_mat(self):
        if self._par_cov_mat is None:
            self._par_cov_mat = self.hessian_inv * 2.0 * self._err_def
        return np.asarray(self._par_cov_mat)

    @property
    def cor_mat(self):
        if self._par_cor_mat is None:
            self._par_cor_mat = CovMat(self.cov_mat).cor_mat
        return self._par_cor_mat

    def reset(self):
        """Clears caches and resets the internal state of used backends."""
        self._invalidate_cache()

    def set(self, parameter_name, parameter_value):
        raise NotImplementedError()

    def set_several(self, parameter_names, parameter_values):
        raise NotImplementedError()

    def limit(self, parameter_name, parameter_value):
        raise NotImplementedError()

    def unlimit(self, parameter_name):
        raise NotImplementedError()

    def fix(self, parameter_name):
        raise NotImplementedError()

    def is_fixed(self, parameter_name):
        raise NotImplementedError()

    def release(self, parameter_name):
        raise NotImplementedError()

    def minimize(self):  # TODO max calls
        raise NotImplementedError()
