from kafe.core.contour import ContourFactory
try:
    import iminuit
except ImportError:
    # TODO: handle importing nonexistent minimizer
    raise

import numpy as np

class MinimizerIMinuitException(Exception):
    pass

class MinimizerIMinuit(object):
    def __init__(self,
                 parameter_names, parameter_values, parameter_errors,
                 function_to_minimize, strategy = 1):
        self._par_names = parameter_names
        self._strategy = strategy
        
        self._func_handle = function_to_minimize
        self._err_def = 1.0
        self._tol = 0.001

        # initialize the minimizer parameter specification
        self._minimizer_param_dict = {}
        assert len(parameter_names) == len(parameter_values) == len(parameter_errors)
        for _pn, _pv, _pe in zip(parameter_names, parameter_values, parameter_errors):
            self._minimizer_param_dict[_pn] = _pv
            self._minimizer_param_dict["error_" + _pn] = _pe
            self._minimizer_param_dict["fix_" + _pn] = False
            self._minimizer_param_dict["limit_" + _pn] = None

        self.__iminuit = None

        # cache for calculations
        self._invalidate_cache()  # also initializes cache member variables

    # -- private methods

    def _invalidate_cache(self):
        self._par_val = None
        self._par_err = None
        self._hessian = None
        self._hessian_inv = None
        self._fval = None
        self._par_cov_mat = None
        self._par_cor_mat = None
        self._par_asymm_err_dn = None
        self._par_asymm_err_up = None
        self._fmin_struct = None
        self._pars_contour = None

    def _get_fmin_struct(self):
        if self._fmin_struct is None:
            # raise MinimizerIMinuitException("Cannot get requested information: No fit performed!")
            self._fmin_struct = self._get_iminuit().get_fmin()
        return self._fmin_struct

    def _get_iminuit(self):
        if self.__iminuit is None:
            self.__iminuit = iminuit.Minuit(self._func_handle,
                                        forced_parameters=self._par_names,
                                        errordef=self._err_def,
                                        **self._minimizer_param_dict)
            self.__iminuit.set_print_level(-1)
            self.__iminuit.set_strategy(self._strategy)
        return self.__iminuit

    def _fill_in_zeroes_for_fixed(self, submatrix):
        """
        Takes the partial error matrix (submatrix) and adds
        rows and columns with 0.0 where the fixed
        parameters should go.
        """
        _mat = submatrix

        _fparams = self._get_iminuit().list_of_fixed_param()
        _fparam_ids = map(lambda k: self.parameter_names.index(k), _fparams)
        for _id in _fparam_ids:
            _mat = np.insert(np.insert(_mat, _id, 0., axis=0), _id, 0., axis=1)

        return _mat

    # -- public properties

    @property
    def errordef(self):
        return self._err_def

    @errordef.setter
    def errordef(self, err_def):
        assert err_def > 0
        self._err_def = err_def
        self._get_iminuit().set_errordef(err_def)

        # invalidate cache
        self._invalidate_cache()

    @property
    def tolerance(self):
        return self._tol

    @tolerance.setter
    def tolerance(self, tolerance):
        assert tolerance > 0
        self._tol = tolerance
        self._get_iminuit().tol = tolerance

        # invalidate cache
        self._invalidate_cache()


    @property
    def hessian(self):
        # TODO: cache this
        return 2.0 * self.errordef * self.cov_mat.I

    @property
    def cov_mat(self):
        if self._par_cov_mat is None:
            try:
                self._get_iminuit().hesse()
                # FIX_UPSTREAM we need skip_fixed=False, but this is unsupported
                # _mat = self._get_iminuit().matrix(correlation, skip_fixed=False)

                # ... so use skip_fixed=True instead and fill in the gaps
                _mat = self._get_iminuit().matrix(correlation=False, skip_fixed=True)
                _mat = np.asmatrix(_mat)  # reshape into numpy matrix
                _mat = self._fill_in_zeroes_for_fixed(_mat)  # fill in fixed par 'gaps'
                self._par_cov_mat = _mat
            except RuntimeError:
                pass
        return self._par_cov_mat

    @property
    def cor_mat(self):
        if self._par_cor_mat is None:
            try:
                self._get_iminuit().hesse()
                # FIX_UPSTREAM we need skip_fixed=False, but this is unsupported
                #_mat = self._get_iminuit().matrix(correlation, skip_fixed=False)

                # ... so use skip_fixed=True instead and fill in the gaps
                _mat = self._get_iminuit().matrix(correlation=True, skip_fixed=True)
                _mat = np.asmatrix(_mat)  # reshape into numpy matrix
                _mat = self._fill_in_zeroes_for_fixed(_mat)  # fill in fixed par 'gaps'
                self._par_cor_mat = _mat
            except RuntimeError:
                pass
        return self._par_cor_mat

    @property
    def hessian_inv(self):
        return self.cov_mat / 2.0 / self.errordef

    @property
    def function_value(self):
        if self._fval is None:
            self._fval = self._func_handle(*self.parameter_values)
        return self._fval
        # if self._fmin_struct is None:
        #     self._fmin_struct = self._get_iminuit().get_fmin()
        #
        # return self._fmin_struct.fval

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
            self._par_val = tuple(_pvals)
        return self._par_val

    @property
    def parameter_errors(self):
        if self._par_err is None:
            if not self._get_iminuit().is_clean_state():
                # if the fit has been performed at least once
                _param_struct = self._get_iminuit().get_param_states()
                _perrs = [p.error for p in _param_struct]
            else:
                # need to hack to get initial parameter errors
                _e = self._get_iminuit().errors
                _perrs = [_e[pname] for pname in _e]
            self._par_err = tuple(_perrs)
        return self._par_err

    @property
    def parameter_names(self):
        return self._par_names

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

    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, **minimizer_contour_kwargs):
        if self.__iminuit is None:
            raise MinimizerIMinuitException("Need to perform a fit before calling contour()!")
        _numpoints = minimizer_contour_kwargs.pop("numpoints", 100)
        if minimizer_contour_kwargs:
            raise MinimizerIMinuitException("Unknown keyword arguments for contour(): {}".format(minimizer_contour_kwargs.keys()))
        _x_errs, _y_errs, _contour_line = self.__iminuit.mncontour(parameter_name_1, parameter_name_2, 
                                                                   numpoints=_numpoints, sigma=sigma)
        self.minimize()  # return to minimum
        if len(_contour_line) == 0:
            return None  # failed to find any point on contour
        return ContourFactory.create_xy_contour(np.array(_contour_line), sigma)

    def profile(self, parameter_name, bins=20, bound=2, args=None, subtract_min=False):
        if self.__iminuit is None:
            raise MinimizerIMinuitException("Need to perform a fit before calling profile()!")
        _bins, _vals, _statuses = self.__iminuit.mnprofile(parameter_name, bins=bins, bound=bound, subtract_min=subtract_min)
        # TODO: check statuses (?)
        self.minimize()  # return to minimum
        return np.array([_bins, _vals])

    def set(self, parameter_name, parameter_value):
        if parameter_name not in self._minimizer_param_dict:
            raise MinimizerIMinuitException("No parameter named '%s'!" % (parameter_name,))
        self.__iminuit = None  # delete curent iminuit instance
        self._minimizer_param_dict[parameter_name] = parameter_value
        self._invalidate_cache()

    def set_several(self, parameter_names, parameter_values):
        for _pn, _pv in zip(parameter_names,parameter_values):
            self.set(_pn, _pv)

    def fix(self, parameter_name):
        self.__iminuit = None  # delete curent iminuit instance
        self._minimizer_param_dict["fix_" + parameter_name] = True

    def fix_several(self, parameter_names):
        for _pn in parameter_names:
            self.fix(_pn)

    def release(self, parameter_name):
        self.__iminuit = None  # delete curent iminuit instance
        self._minimizer_param_dict["fix_" + parameter_name] = False

    def release_several(self, parameter_names):
        for _pn in parameter_names:
            self.release(_pn)

    def limit(self, parameter_name, parameter_bounds):
        self.__iminuit = None  # delete curent iminuit instance
        assert len(parameter_bounds) == 2
        self._minimizer_param_dict["limit_" + parameter_name] = (parameter_bounds[0], parameter_bounds[1])

    def unlimit(self, parameter_name):
        self.__iminuit = None  # delete curent iminuit instance
        self._minimizer_param_dict["limit_" + parameter_name] = None

    def minimize(self, max_calls=6000):
        self._get_iminuit().migrad(ncall=max_calls)

        # invalidate cache
        self._invalidate_cache()