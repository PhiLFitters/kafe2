import logging
from copy import deepcopy

from .minimizer_base import MinimizerBase, MinimizerException
from ..contour import ContourFactory

try:
    import iminuit
except (ImportError, SyntaxError):
    # TODO: handle importing nonexistent minimizer
    raise

import numpy as np


class MinimizerIMinuitException(MinimizerException):
    pass


class MinimizerIMinuit(MinimizerBase):
    def __init__(self,
                 parameter_names, parameter_values, parameter_errors,
                 function_to_minimize, tolerance=1e-6, errordef=iminuit.Minuit.LEAST_SQUARES,
                 strategy=1):
        self._strategy = strategy

        # initialize the minimizer parameter specification
        self._minimizer_param_dict = {}
        for _pn in parameter_names:
            self._minimizer_param_dict["fix_" + _pn] = False
            self._minimizer_param_dict["limit_" + _pn] = None

        self.reset()  # sets self.__iminuit and caches to None
        super(MinimizerIMinuit, self).__init__(
            parameter_names=parameter_names, parameter_values=parameter_values,
            parameter_errors=parameter_errors, function_to_minimize=function_to_minimize,
            tolerance=tolerance, errordef=errordef
        )

    # -- private methods

    def _invalidate_cache(self):
        self._par_val = None
        self._par_err = None
        self._fmin_struct = None
        super(MinimizerIMinuit, self)._invalidate_cache()

    def _save_state(self):
        if self._par_val is None:
            self._save_state_dict["par_val"] = self._par_val
        else:
            self._save_state_dict["par_val"] = np.array(self._par_val)
        if self._par_err is None:
            self._save_state_dict["par_err"] = self._par_err
        else:
            self._save_state_dict["par_err"] = np.array(self._par_err)
        self._save_state_dict["fmin_struct"] = deepcopy(self._fmin_struct)
        self._save_state_dict['minimizer_param_dict'] = self._minimizer_param_dict
        self._save_state_dict['iminuit'] = self.__iminuit
        super(MinimizerIMinuit, self)._save_state()

    def _load_state(self):
        self.reset()
        self._par_val = self._save_state_dict["par_val"]
        if self._par_val is not None:
            self._par_val = np.array(self._par_val)
        self._par_err = self._save_state_dict["par_err"]
        if self._par_err is not None:
            self._par_err = np.array(self._par_err)
        self._fmin_struct = deepcopy(self._save_state_dict["fmin_struct"])
        self._minimizer_param_dict = self._save_state_dict['minimizer_param_dict']
        self.__iminuit = self._save_state_dict['iminuit']
        self._func_handle(*self.parameter_values)  # call the function to propagate the changes to the nexus
        super(MinimizerIMinuit, self)._load_state()

    def _get_fmin_struct(self):
        if self._fmin_struct is None:
            # raise MinimizerIMinuitException("Cannot get requested information: No fit performed!")
            self._fmin_struct = self._get_iminuit().get_fmin()
        return self._fmin_struct

    def _get_iminuit(self):
        if self.__iminuit is None:
            self.__iminuit = iminuit.Minuit(self._func_wrapper,
                                            forced_parameters=self.parameter_names,
                                            errordef=self.errordef,
                                            **self._minimizer_param_dict)
            # set logging level in iminuit arcording to the root logger
            if logging.root.level <= logging.DEBUG:
                self.__iminuit.print_level = 2
            elif logging.root.level <= logging.INFO:
                self.__iminuit.print_level = 1
            else:  # default is logging.WARN, show nothing
                self.__iminuit.print_level = 0
            self.__iminuit.strategy = self._strategy
            self.__iminuit.tol = self.tolerance
        return self.__iminuit

    def _calculate_asymmetric_parameter_errors(self):
        try:
            _minos_result_dict = self._get_iminuit().minos()
        except RuntimeError:
            return None
        _asymm_par_errs = np.zeros(shape=(self.num_pars, 2))
        for _par_name in self.parameter_names:
            _index = self.parameter_names.index(_par_name)
            if _par_name in _minos_result_dict:
                _asymm_par_errs[_index, 0] = _minos_result_dict[_par_name]['lower']
                _asymm_par_errs[_index, 1] = _minos_result_dict[_par_name]['upper']
            else:
                _asymm_par_errs[_index, :] = 0
        self.minimize()
        return _asymm_par_errs

    # -- public properties

    @property
    def hessian(self):
        if not self.did_fit:
            return None
        if self._hessian is None:
            _cov_mat = self.cov_mat
            if _cov_mat is None:
                self._hessian = None
            else:
                _submat = self._remove_zeroes_for_fixed(_cov_mat)
                _submat_inv = 2.0 * self.errordef * np.linalg.inv(_submat)
                self._hessian = self._fill_in_zeroes_for_fixed(_submat_inv)
        return None if self._hessian is None else self._hessian.copy()

    @property
    def cov_mat(self):
        if not self.did_fit:
            return None
        if self._par_cov_mat is None:
            self._save_state()
            try:
                self._get_iminuit().hesse()
                # FIX_UPSTREAM we need skip_fixed=False, but this is unsupported
                # _mat = self._get_iminuit().matrix(correlation, skip_fixed=False)

                # ... so use skip_fixed=True instead and fill in the gaps
                _mat = self._get_iminuit().matrix(correlation=False, skip_fixed=True)
                _mat = np.asarray(_mat)  # reshape into numpy matrix
                _mat = self._fill_in_zeroes_for_fixed(_mat)  # fill in fixed par 'gaps'
                self._func_wrapper_unpack_args(self.parameter_values)
            except RuntimeError:
                _mat = None
            self._load_state()
            self._par_cov_mat = _mat
        return None if self._par_cov_mat is None else self._par_cov_mat.copy()

    @property
    def cor_mat(self):
        if not self.did_fit:
            return None
        if self._par_cor_mat is None:
            self._save_state()
            try:
                self._get_iminuit().hesse()
                # FIX_UPSTREAM we need skip_fixed=False, but this is unsupported
                # _mat = self._get_iminuit().matrix(correlation, skip_fixed=False)

                # ... so use skip_fixed=True instead and fill in the gaps
                _mat = self._get_iminuit().matrix(correlation=True, skip_fixed=True)
                _mat = np.asarray(_mat)  # reshape into numpy matrix
                _mat = self._fill_in_zeroes_for_fixed(_mat)  # fill in fixed par 'gaps'
                self._func_wrapper_unpack_args(self.parameter_values)
            except RuntimeError:
                _mat = None
            self._load_state()
            self._par_cor_mat = _mat
        return None if self._par_cor_mat is None else self._par_cor_mat.copy()

    @property
    def hessian_inv(self):
        if not self.did_fit:
            return None
        if self._hessian_inv is None:
            self._hessian_inv = self.cov_mat / (2.0 * self.errordef)
        return None if self._hessian_inv is None else self._hessian_inv.copy()

    @property
    def parameter_values(self):
        if self._par_val is None:
            _m = self._get_iminuit()
            if not _m.is_clean_state():
                # if the fit has been performed at least once
                _param_struct = _m.get_param_states()
                _pvals = [p.value for p in _param_struct]
            else:
                # need to hack to get initial parameter values
                _v = _m.values
                _pvals = [_v[pname] for pname in self.parameter_names]
            self._par_val = np.array(_pvals)
        return self._par_val.copy()

    @parameter_values.setter
    def parameter_values(self, new_values):
        for _pn, _pv, in zip(self._par_names, new_values):
            self._minimizer_param_dict[_pn] = _pv
        self.reset()

    @property
    def parameter_errors(self):
        if self._par_err is None:
            if not self._get_iminuit().is_clean_state():
                # if the fit has been performed at least once
                _param_struct = self._get_iminuit().get_param_states()
                self._par_err = np.array(
                    [p.error if not self._minimizer_param_dict["fix_%s" % pname] else 0.0
                     for p, pname in zip(_param_struct, self.parameter_names)])
            else:
                # need to hack to get initial parameter errors
                _e = self._get_iminuit().errors
                self._par_err = np.array(
                    [_e[pname] if not self._minimizer_param_dict["fix_%s" % pname] else 0.0
                     for pname in _e])
        return self._par_err.copy()

    @parameter_errors.setter
    def parameter_errors(self, new_errors):
        if not np.all(np.array(new_errors) > 0):
            raise ValueError("All parameter errors must be > 0! Received: %s" % new_errors)
        for _pn, _pe in zip(self._par_names, new_errors):
            self._minimizer_param_dict["error_" + _pn] = _pe
        self.reset()

    # -- private "properties"

    @property
    def _fmin_edm(self):
        _fmin = self._get_fmin_struct()
        return _fmin.edm

    @property
    def _fmin_up(self):
        _fmin = self._get_fmin_struct()
        return _fmin.up

    @property
    def _fmin_has_covariance(self):
        _fmin = self._get_fmin_struct()
        return _fmin.has_covariance

    @property
    def _fmin_covariance_is_pos_def(self):
        _fmin = self._get_fmin_struct()
        return _fmin.has_made_posdef_covar

    @property
    def _fmin_covariance_is_accurate(self):
        _fmin = self._get_fmin_struct()
        return _fmin.has_accurate_covar

    # -- public methods

    def reset(self):
        super(MinimizerIMinuit, self).reset()
        self.__iminuit = None

    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, **minimizer_contour_kwargs):
        if not self.did_fit:
            raise MinimizerIMinuitException("Need to perform a fit before calling contour()!")
        _numpoints = minimizer_contour_kwargs.pop("numpoints", 100)
        if minimizer_contour_kwargs:
            raise MinimizerIMinuitException(
                "Unknown keyword arguments for contour(): {}".format(minimizer_contour_kwargs.keys()))
        _x_errs, _y_errs, _contour_line = self.__iminuit.mncontour(parameter_name_1, parameter_name_2,
                                                                   numpoints=_numpoints, sigma=sigma)
        self.minimize()  # return to minimum
        if len(_contour_line) == 0:
            return None  # failed to find any point on contour
        return ContourFactory.create_xy_contour(np.array(_contour_line), sigma)

    def profile(self, parameter_name, bins=20, bound=2, subtract_min=False):
        if not self.did_fit:
            raise MinimizerIMinuitException("Need to perform a fit before calling profile()!")
        _bins, _vals, _statuses = self.__iminuit.mnprofile(parameter_name, bins=bins, bound=bound,
                                                           subtract_min=subtract_min)
        # TODO: check statuses (?)
        self.minimize()  # return to minimum
        return np.array([_bins, _vals])

    def set(self, parameter_name, parameter_value):
        if parameter_name not in self._minimizer_param_dict:
            raise ValueError("No parameter named '%s'!" % (parameter_name,))
        self._minimizer_param_dict[parameter_name] = parameter_value
        self.reset()

    def fix(self, parameter_name):
        self._minimizer_param_dict["fix_" + parameter_name] = True
        self._get_iminuit().fixed[parameter_name] = True
        self._invalidate_cache()

    def is_fixed(self, parameter_name):
        return self._minimizer_param_dict["fix_%s" % parameter_name]

    def release(self, parameter_name):
        self._minimizer_param_dict["fix_" + parameter_name] = False
        self._get_iminuit().fixed[parameter_name] = False
        self._invalidate_cache()

    def limit(self, parameter_name, parameter_bounds):
        assert len(parameter_bounds) == 2
        self._minimizer_param_dict["limit_" + parameter_name] = (parameter_bounds[0], parameter_bounds[1])
        self.reset()

    def unlimit(self, parameter_name):
        self._minimizer_param_dict["limit_" + parameter_name] = None
        self.reset()

    def minimize(self, max_calls=6000):
        if np.all([self.is_fixed(_par_name) for _par_name in self.parameter_names]):
            raise MinimizerIMinuitException("Cannot perform a fit if all parameters are fixed!")

        self._get_iminuit().migrad(ncall=max_calls)

        for (_pn, _pv, _pe) in zip(
                self.parameter_names, self.parameter_values, self.parameter_errors):
            self._minimizer_param_dict[_pn] = _pv
            self._minimizer_param_dict["error_" + _pn] = _pe

        # invalidate cache
        self._did_fit = True
        self._invalidate_cache()
