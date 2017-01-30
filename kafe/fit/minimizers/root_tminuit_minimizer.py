try:
    from ROOT import TMinuit, Double, Long
    from ROOT import TMath  # for using ROOT's chi2prob function
except ImportError:
    # TODO: handle importing nonexistent minimizer
    raise

from array import array as arr  # array needed for TMinuit arguments

import numpy as np

class MinimizerROOTTMinuitException(Exception):
    pass

class MinimizerROOTTMinuit(object):
    def __init__(self,
                 parameter_names, parameter_values, parameter_errors,
                 function_to_minimize):
        self._par_names = parameter_names

        self._func_handle = function_to_minimize
        self._err_def = 1.0
        self._tol = 0.001

        # initialize the minimizer parameter specification
        self._minimizer_param_dict = {}
        assert len(parameter_names) == len(parameter_values) == len(parameter_errors)
        self._par_val = []
        self._par_err = []
        self._par_fixed_mask = []
        self._par_limits = []
        for _pn, _pv, _pe in zip(parameter_names, parameter_values, parameter_errors):
            self._par_val.append(_pv)
            self._par_err.append(_pe)
            self._par_fixed_mask.append(False)
            self._par_limits.append(None)
            # TODO: limits/fixed parameters

        self.__gMinuit = None

        # cache for calculations
        self._hessian = None
        self._hessian_inv = None
        self._fval = None
        self._par_cov_mat = None
        self._par_cor_mat = None
        self._par_asymm_err_dn = None
        self._par_asymm_err_up = None
        self._fmin_struct = None
        self._pars_contour = None

        self._min_result_stale = True

    # -- private methods

    # def _invalidate_cache(self):
    #     self._par_val = None
    #     self._par_err = None
    #     self._hessian = None
    #     self._hessian_inv = None
    #     self._fval = None
    #     self._par_cov_mat = None
    #     self._par_cor_mat = None
    #     self._par_asymm_err_dn = None
    #     self._par_asymm_err_up = None
    #     self._fmin_struct = None
    #     self._pars_contour = None

    def _recreate_gMinuit(self):
        self.__gMinuit = TMinuit(self.n_pars)
        self.__gMinuit.SetPrintLevel(-1)
        self.__gMinuit.SetFCN(self._minuit_fcn)
        self.__gMinuit.SetErrorDef(self._err_def)

        # set gMinuit parameters
        error_code = Long(0)
        for _pid, (_pn, _pv, _pe) in enumerate(zip(self._par_names, self._par_val, self._par_err)):
            self.__gMinuit.mnparm(_pid,
                                  _pn,
                                  _pv,
                                  0.1 * _pe,
                                  0, 0, error_code)

        err_code = Long(0)
        # set fixed parameters
        for _par_id, _pf in enumerate(self._par_fixed_mask):
            if _pf:
                self.__gMinuit.mnfixp(_par_id, err_code)

        # set parameter limits
        for _par_id, _pl in enumerate(self._par_limits):
            if _pl is not None:
                _lo_lim, _up_lim = _pl
                self.__gMinuit.mnexcm("SET LIM",
                                      arr('d', [_par_id + 1, _lo_lim, _up_lim]), 3, error_code)

    def _get_gMinuit(self):
        if self.__gMinuit is None:
            self._recreate_gMinuit()
        return self.__gMinuit

    def _migrad(self, max_calls=6000):
        # need to set the FCN explicitly before every call
        self._get_gMinuit().SetFCN(self._minuit_fcn)
        error_code = Long(0)
        self._get_gMinuit().mnexcm("MIGRAD",
                                    arr('d', [max_calls, self.tolerance]),
                                    2, error_code)

    def _hesse(self, max_calls=6000):
        # need to set the FCN explicitly before every call
        self._get_gMinuit().SetFCN(self._minuit_fcn)
        error_code = Long(0)
        self._get_gMinuit().mnexcm("HESSE", arr('d', [max_calls]), 1, error_code)

    def _minuit_fcn(self,
                    number_of_parameters, derivatives, f, parameters, internal_flag):
        '''
        This is actually a function called in *ROOT* and acting as a C wrapper
        for our `FCN`, which is implemented in Python.

        This function is called by `Minuit` several times during a fit. It
        doesn't return anything but modifies one of its arguments (*f*).
        This is *ugly*, but it's how *ROOT*'s ``TMinuit`` works. Its argument
        structure is fixed and determined by `Minuit`:

        **number_of_parameters** : int
            The number of parameters of the current fit

        **derivatives** : C array
            If the user chooses to calculate the first derivative of the
            function inside the `FCN`, this value should be written here. This
            interface to `Minuit` ignores this derivative, however, so
            calculating this inside the `FCN` has no effect (yet).

        **f** : C array
            The desired function value is in f[0] after execution.

        **parameters** : C array
            A C array of parameters. Is cast to a Python list

        **internal_flag** : int
            A flag allowing for different behaviour of the function.
            Can be any integer from 1 (initial run) to 4(normal run). See
            `Minuit`'s specification.
        '''

        # Retrieve the parameters from the C side of ROOT and
        # store them in a Python list -- resource-intensive
        # for many calls, but can't be improved (yet?)
        parameter_list = np.frombuffer(parameters,
                                       dtype=float,
                                       count=self.n_pars)

        # call the Python implementation of FCN.
        f[0] = self._func_handle(*parameter_list)

    def _insert_zeros_for_fixed(self, submatrix):
        '''
        Takes the partial error matrix (submatrix) and adds
        rows and columns with 0.0 where the fixed
        parameters should go.
        '''
        _mat = submatrix

        # reduce the matrix before inserting zeros
        _n_pars_free = self.n_pars_free
        _mat = _mat[0:_n_pars_free,0:_n_pars_free]

        _fparam_ids = [_par_id for _par_id, _p in enumerate(self._par_fixed_mask) if _p]
        for _id in _fparam_ids:
            _mat = np.insert(np.insert(_mat, _id, 0., axis=0), _id, 0., axis=1)

        return _mat

    # -- public properties

    @property
    def n_pars(self):
        return len(self.parameter_names)

    @property
    def n_pars_free(self):
        return len([_p for _p in self._par_fixed_mask if not _p])

    @property
    def errordef(self):
        return self._err_def

    @errordef.setter
    def errordef(self, err_def):
        assert err_def > 0
        self._err_def = err_def
        if self.__gMinuit is not None:
            self.__gMinuit.set_errordef(err_def)
            self._min_result_stale = True

    @property
    def tolerance(self):
        return self._tol

    @tolerance.setter
    def tolerance(self, tolerance):
        assert tolerance > 0
        self._tol = tolerance
        self._min_result_stale = True

    @property
    def hessian(self):
        # TODO: cache this
        return 2.0 * self.errordef * self.cov_mat.I

    @property
    def cov_mat(self):
        if self._min_result_stale:
            raise MinimizerROOTTMinuitException("Cannot get cov_mat: Minimizer result is outdated.")
        if self._par_cov_mat is None:
            _n_pars_total = self.n_pars
            _n_pars_free = self.n_pars_free
            _tmp_mat_array = arr('d', [0.0]*(_n_pars_total**2))
            # get parameter covariance matrix from TMinuit
            self.__gMinuit.mnemat(_tmp_mat_array, _n_pars_total)
            # reshape into 2D array
            _sub_cov_mat = np.asmatrix(
                np.reshape(
                    _tmp_mat_array,
                    (_n_pars_total, _n_pars_total)
                )
            )
            self._par_cov_mat = self._insert_zeros_for_fixed(_sub_cov_mat)
        return self._par_cov_mat

    @property
    def cor_mat(self):
        if self._min_result_stale:
            raise MinimizerROOTTMinuitException("Cannot get cor_mat: Minimizer result is outdated.")
        if self._par_cor_mat is None:
            _cov_mat = self.cov_mat
            # TODO: use CovMat object!
            # Note: for zeros on cov_mat diagonals (which occur for fixed parameters) -> overwrite with 1.0
            _sqrt_diag = np.array([_err if _err>0 else 1.0 for _err in np.sqrt(np.diag(_cov_mat))])
            self._par_cor_mat = np.asmatrix(np.asarray(_cov_mat) / np.outer(_sqrt_diag, _sqrt_diag))
        return self._par_cor_mat

    @property
    def hessian_inv(self):
        return self.cov_mat / 2.0 / self.errordef

    @property
    def function_value(self):
        if self._fval is None:
            self._fval = self._func_handle(*self.parameter_values)
        return self._fval

    @property
    def parameter_values(self):
        return self._par_val

    @property
    def parameter_errors(self):
        return self._par_err

    @property
    def parameter_names(self):
        return self._par_names

    # -- private "properties"


    # -- public methods

    def fix(self, parameter_name):
        # set local flag
        _par_id = self.parameter_names.index(parameter_name)
        if self._par_fixed_mask[_par_id]:
            return  # par is already fixed
        self._par_fixed_mask[_par_id] = True
        if self.__gMinuit is not None:
            # also update Minuit instance
            err_code = Long(0)
            self.__gMinuit.mnfixp(_par_id, err_code)
            # self.__gMinuit.mnexcm("FIX",
            #                   arr('d', [_par_id+1]), 1, error_code)
            self._min_result_stale = True

    def fix_several(self, parameter_names):
        for _pn in parameter_names:
            self.fix(self, _pn)

    def release(self, parameter_name):
        # set local flag
        _par_id = self.parameter_names.index(parameter_name)
        if not self._par_fixed_mask[_par_id]:
            return  # par is already released
        self._par_fixed_mask[_par_id] = False
        if self.__gMinuit is not None:
            # also update Minuit instance
            self.__gMinuit.mnfree(-_par_id-1)
            # self.__gMinuit.mnexcm("RELEASE",
            #                   arr('d', [_par_id+1]), 1, error_code)
            self._min_result_stale = True

    def release_several(self, parameter_names):
        for _pn in parameter_names:
            self.release(self, _pn)

    def limit(self, parameter_name, parameter_bounds):
        assert len(parameter_bounds) == 2
        # set local flag
        _par_id = self.parameter_names.index(parameter_name)
        if self._par_limits[_par_id] == parameter_bounds:
            return  # same limits already set
        self._par_limits[_par_id] = parameter_bounds
        if self.__gMinuit is not None:
            _lo_lim, _up_lim = self._par_limits[_par_id]
            # also update Minuit instance
            error_code = Long(0)
            self.__gMinuit.mnexcm("SET LIM",
                     arr('d', [_par_id+1, _lo_lim, _up_lim]), 3, error_code)
            self._min_result_stale = True

    def unlimit(self, parameter_name):
        # set local flag
        _par_id = self.parameter_names.index(parameter_name)
        if self._par_limits[_par_id] is None:
            return  # parameter is already unlimited
        self._par_limits[_par_id] = None
        if self.__gMinuit is not None:
            # also update Minuit instance
            error_code = Long(0)
            self.__gMinuit.mnexcm("SET LIM",
                     arr('d', [_par_id+1]), 1, error_code)
            self._min_result_stale = True

    def minimize(self, max_calls=6000):
        self._migrad(max_calls=max_calls)

        # retrieve fit parameters
        self._par_val = []
        self._par_err = []
        _pv, _pe = Double(0), Double(0)
        for _par_id in xrange(0, self.n_pars):
            self.__gMinuit.GetParameter(_par_id, _pv, _pe)  # retrieve fitresult
            self._par_val.append(float(_pv))
            self._par_err.append(float(_pe))

        self._min_result_stale = False