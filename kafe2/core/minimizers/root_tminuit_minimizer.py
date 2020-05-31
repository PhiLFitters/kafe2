from __future__ import print_function
import six
import ctypes
from .minimizer_base import MinimizerBase, MinimizerException
from ..contour import ContourFactory
try:
    from ROOT import TMinuit, Double, Long
    from ROOT import TMath  # for using ROOT's chi2prob function
except ImportError:
    # TODO: handle importing nonexistent minimizer
    raise

from array import array as arr  # array needed for TMinuit arguments

import numpy as np


class MinimizerROOTTMinuitException(MinimizerException):
    pass


class MinimizerROOTTMinuit(MinimizerBase):
    def __init__(self,
                 parameter_names, parameter_values, parameter_errors,
                 function_to_minimize, tolerance=1e-9, errordef=MinimizerBase.ERRORDEF_CHI2,
                 strategy=1):
        self._strategy = strategy

        self._par_bounds = np.array([None] * len(parameter_names))
        self._par_fixed = np.array([False] * len(parameter_names))

        self.reset()  # sets self.__gMinuit and caches to None
        super(MinimizerROOTTMinuit, self).__init__(
            parameter_names=parameter_names, parameter_values=parameter_values,
            parameter_errors=parameter_errors, function_to_minimize=function_to_minimize,
            tolerance=tolerance, errordef=errordef
        )

    # -- private methods

    def _save_state(self):
        if self._par_val is None:
            self._save_state_dict["par_val"] = self._par_val
        else:
            self._save_state_dict["par_val"] = np.array(self._par_val)
        if self._par_err is None:
            self._save_state_dict["par_err"] = self._par_err
        else:
            self._save_state_dict["par_err"] = np.array(self._par_err)
        self._save_state_dict['par_fixed'] = np.array(self._par_fixed)
        self._save_state_dict['gMinuit'] = self.__gMinuit
        super(MinimizerROOTTMinuit, self)._save_state()

    def _load_state(self):
        self.reset()
        self._par_val = self._save_state_dict["par_val"]
        if self._par_val is not None:
            self._par_val = np.array(self._par_val)
        self._par_err = self._save_state_dict["par_err"]
        if self._par_err is not None:
            self._par_err = np.array(self._par_err)
        self._par_fixed = np.array(self._save_state_dict['par_fixed'])
        self.__gMinuit = self._save_state_dict['gMinuit']
        # call the function to propagate the changes to the nexus:
        self._func_handle(*self.parameter_values)
        super(MinimizerROOTTMinuit, self)._load_state()

    def _recreate_gMinuit(self):
        self.__gMinuit = TMinuit(self.num_pars)
        self.__gMinuit.SetPrintLevel(-1)
        self.__gMinuit.mncomd("SET STRATEGY {}".format(self._strategy), ctypes.c_int(0))
        self.__gMinuit.SetFCN(self._minuit_fcn)
        self.__gMinuit.SetErrorDef(self._err_def)

        # set gMinuit parameters
        error_code = ctypes.c_int(0)
        for _pid, (_pn, _pv, _pe) in enumerate(zip(self._par_names, self._par_val, self._par_err)):
            self.__gMinuit.mnparm(_pid,
                                  _pn,
                                  _pv,
                                  0.1 * _pe,
                                  0, 0, error_code)

        err_code = ctypes.c_int(0)
        # set fixed parameters
        for _par_id, _pf in enumerate(self._par_fixed):
            if _pf:
                self.__gMinuit.mnfixp(_par_id, err_code)

        # set parameter limits
        for _par_id, _pb in enumerate(self._par_bounds):
            if _pb is not None:
                _lo_lim, _up_lim = _pb
                self.__gMinuit.mnexcm("SET LIM",
                                      arr('d', [_par_id + 1, _lo_lim, _up_lim]), 3, error_code)

    def _get_gMinuit(self):
        if self.__gMinuit is None:
            self._recreate_gMinuit()
        return self.__gMinuit

    def _migrad(self, max_calls=6000):
        # need to set the FCN explicitly before every call
        self._get_gMinuit().SetFCN(self._minuit_fcn)
        error_code = ctypes.c_int(0)
        self._get_gMinuit().mnexcm("MIGRAD",
                                    arr('d', [max_calls, self.tolerance]),
                                    2, error_code)

    def _minuit_fcn(self,
                    number_of_parameters, derivatives, f, parameters, internal_flag):
        """
        This is actually a function called in *ROOT* and acting as a C wrapper
        for our `FCN`, which is implemented in Python.

        This function is called by `Minuit` several times during a fitters. It
        doesn't return anything but modifies one of its arguments (*f*).
        This is *ugly*, but it's how *ROOT*'s ``TMinuit`` works. Its argument
        structure is fixed and determined by `Minuit`:

        **number_of_parameters** : int
            The number of parameters of the current fitters

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
        """

        # Retrieve the parameters from the C side of ROOT and
        # store them in a Python list -- resource-intensive
        # for many calls, but can't be improved (yet?)
        parameter_list = np.frombuffer(parameters,
                                       dtype=float,
                                       count=self.num_pars)

        # call the Python implementation of FCN.
        f[0] = self._func_wrapper(*parameter_list)

    def _calculate_asymmetric_parameter_errors(self):
        self._get_gMinuit().mnmnos()
        _asymm_par_errs = np.zeros(shape=(self.num_pars, 2))
        for _n in range(self.num_pars):
            _number = Long(_n)
            _eplus = ctypes.c_double(0)
            _eminus = ctypes.c_double(0)
            _eparab = ctypes.c_double(0)
            _gcc = ctypes.c_double(0)
            self._get_gMinuit().mnerrs(_number, _eplus, _eminus, _eparab, _gcc)
            _asymm_par_errs[_n, 0] = _eminus.value
            _asymm_par_errs[_n, 1] = _eplus.value
        self.minimize()
        return _asymm_par_errs

    def _get_fit_info(self, info):
        '''Retrieves other info from `Minuit`.
        **info** : string
            Information about the fit to retrieve.
            This can be any of the following:
              - ``'fcn'``: `FCN` value at minimum,
              - ``'edm'``: estimated distance to minimum
              - ``'err_def'``: `Minuit` error matrix status code
              - ``'status_code'``: `Minuit` general status code
        '''

        # declare vars in which to retrieve other info
        fcn_at_min = ctypes.c_double(0)
        edm = ctypes.c_double(0)
        err_def = ctypes.c_double(0)
        n_var_param = ctypes.c_int(0)
        n_tot_param = ctypes.c_int(0)
        status_code = ctypes.c_int(0)

        # Tell TMinuit to update the variables declared above
        self.__gMinuit.mnstat(fcn_at_min, edm, err_def, n_var_param, n_tot_param, status_code)

        if info == 'fcn':
            return fcn_at_min.value
        elif info == 'edm':
            return edm.value
        elif info == 'err_def':
            return err_def.value
        elif info == 'status_code':
            return status_code.value
        else:
            raise ValueError("Unknown fit info: %s" % info)

    # -- public properties

    @property
    def hessian(self):
        if not self.did_fit:
            return None
        if self._hessian is None:
            _submat = self._remove_zeroes_for_fixed(self.cov_mat)
            _submat_inv = 2.0 * self.errordef * np.linalg.inv(_submat)
            self._hessian = self._fill_in_zeroes_for_fixed(_submat_inv)
        return self._hessian.copy()

    @property
    def cov_mat(self):
        if not self.did_fit:
            return None
        if self._par_cov_mat is None:
            _n_pars_total = self.num_pars
            _tmp_mat_array = arr('d', [0.0]*(_n_pars_total**2))
            # get parameter covariance matrix from TMinuit
            self._get_gMinuit().mnemat(_tmp_mat_array, _n_pars_total)
            # reshape into 2D array
            _sub_cov_mat = np.asarray(
                np.reshape(
                    _tmp_mat_array,
                    (_n_pars_total, _n_pars_total)
                ),
                dtype=np.float
            )
            _num_pars_free = np.sum(np.invert(self._par_fixed))
            _sub_cov_mat = _sub_cov_mat[:_num_pars_free, :_num_pars_free]
            self._par_cov_mat = self._fill_in_zeroes_for_fixed(_sub_cov_mat)
        return self._par_cov_mat.copy()

    @property
    def hessian_inv(self):
        if not self.did_fit:
            return None
        if self._hessian_inv is None:
            self._hessian_inv = self.cov_mat / (2.0 * self.errordef)
        return self._hessian_inv.copy()

    @property
    def parameter_values(self):
        return self._par_val.copy()

    @parameter_values.setter
    def parameter_values(self, new_values):
        self._par_val = np.array(new_values)
        self.reset()

    @property
    def parameter_errors(self):
        return self._par_err.copy()

    @parameter_errors.setter
    def parameter_errors(self, new_errors):
        _err_array = np.array(new_errors)
        if not np.all(_err_array > 0):
            raise ValueError("All parameter errors must be > 0! Received: %s" % new_errors)
        self._par_err = _err_array
        self.reset()

    # -- private "properties"

    # -- public methods

    def reset(self):
        super(MinimizerROOTTMinuit, self).reset()
        self.__gMinuit = None

    def set(self, parameter_name, parameter_value):
        if parameter_name not in self._par_names:
            raise ValueError("No parameter named '%s'!" % (parameter_name,))
        _par_id = self.parameter_names.index(parameter_name)
        self._par_val[_par_id] = parameter_value
        self.reset()

    def fix(self, parameter_name):
        # set local flag
        _par_id = self.parameter_names.index(parameter_name)
        if self._par_fixed[_par_id]:
            return  # par is already fixed
        self._par_fixed[_par_id] = True
        if self.__gMinuit is not None:
            # also update Minuit instance
            err_code = ctypes.c_int(0)
            self.__gMinuit.mnfixp(_par_id, err_code)
            # self.__gMinuit.mnexcm("FIX",
            #                   arr('d', [_par_id+1]), 1, error_code)
        self._invalidate_cache()

    def is_fixed(self, parameter_name):
        _par_id = self.parameter_names.index(parameter_name)
        return self._par_fixed[_par_id]

    def release(self, parameter_name):
        # set local flag
        _par_id = self.parameter_names.index(parameter_name)
        if not self._par_fixed[_par_id]:
            return  # par is already released
        self._par_fixed[_par_id] = False
        if self.__gMinuit is not None:
            # also update Minuit instance
            self.__gMinuit.mnfree(-_par_id-1)
            # self.__gMinuit.mnexcm("RELEASE",
            #                   arr('d', [_par_id+1]), 1, error_code)
        self._invalidate_cache()

    def limit(self, parameter_name, parameter_bounds):
        assert len(parameter_bounds) == 2
        if parameter_bounds[0] is None or parameter_bounds[1] is None:
            raise MinimizerROOTTMinuitException(
                "Cannot define one-sided parameter limits when using the ROOT TMinuit Minimizer.")
        # set local flag
        _par_id = self.parameter_names.index(parameter_name)
        if self._par_bounds[_par_id] == parameter_bounds:
            return  # same limits already set
        if self._par_val[_par_id] < parameter_bounds[0]:
            self.set(parameter_name, parameter_bounds[0])
        elif self._par_val[_par_id] > parameter_bounds[1]:
            self.set(parameter_name, parameter_bounds[1])
        self._par_bounds[_par_id] = parameter_bounds
        if self.__gMinuit is not None:
            _lo_lim, _up_lim = self._par_bounds[_par_id]
            # also update Minuit instance
            error_code = ctypes.c_int(0)
            self.__gMinuit.mnexcm("SET LIM",
                     arr('d', [_par_id+1, _lo_lim, _up_lim]), 3, error_code)
            self._did_fit = False
        self._invalidate_cache()

    def unlimit(self, parameter_name):
        # set local flag
        _par_id = self.parameter_names.index(parameter_name)
        if self._par_bounds[_par_id] is None:
            return  # parameter is already unlimited
        self._par_bounds[_par_id] = None
        if self.__gMinuit is not None:
            # also update Minuit instance
            error_code = ctypes.c_int(0)
            self.__gMinuit.mnexcm("SET LIM",
                     arr('d', [_par_id+1]), 1, error_code)
            self._did_fit = False
        self._invalidate_cache()

    def minimize(self, max_calls=6000):
        if np.all(self._par_fixed):
            raise MinimizerROOTTMinuitException("Cannot perform a fit if all parameters are fixed!")
        self._migrad(max_calls=max_calls)

        # retrieve fitters parameters
        self._par_val = np.zeros(self.num_pars)
        self._par_err = np.zeros(self.num_pars)
        _pv, _pe = ctypes.c_double(0), ctypes.c_double(0)
        for _par_id in six.moves.range(0, self.num_pars):
            self.__gMinuit.GetParameter(_par_id, _pv, _pe)  # retrieve fit result
            self._par_val[_par_id] = _pv.value
            self._par_err[_par_id] = _pe.value

        self._did_fit = True
        
    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, **minimizer_contour_kwargs):
        if not self.did_fit:
            raise MinimizerROOTTMinuitException("Need to perform a fit before calling contour()!")
        _numpoints = minimizer_contour_kwargs.pop("numpoints", 100)
        if minimizer_contour_kwargs:
            raise MinimizerROOTTMinuitException("Unknown parameters: {}".format(minimizer_contour_kwargs))
        _id_1 = self.parameter_names.index(parameter_name_1)
        _id_2 = self.parameter_names.index(parameter_name_2)
        self.__gMinuit.SetErrorDef(sigma ** 2)
        _t_graph = self.__gMinuit.Contour(_numpoints, _id_1, _id_2)
        self.__gMinuit.SetErrorDef(self._err_def)
        
        _x_buffer, _y_buffer = _t_graph.GetX(), _t_graph.GetY()
        _N = _t_graph.GetN()
        
        _x = np.frombuffer(_x_buffer, dtype=float, count=_N)
        _y = np.frombuffer(_y_buffer, dtype=float, count=_N)
        self._func_handle(*self.parameter_values)
        return ContourFactory.create_xy_contour((_x, _y), sigma)
    
    def profile(self, parameter_name, bins=21, bound=2, args=None, subtract_min=False):
        if not self.did_fit:
            raise MinimizerROOTTMinuitException("Need to perform a fit before calling profile()!")
        
        MAX_ITERATIONS = 6000
        
        _error_code = ctypes.c_int(0)
        _minuit_id = Long(self.parameter_names.index(parameter_name) + 1)

        _par_min = ctypes.c_double(0)
        _par_err = ctypes.c_double(0)
        self.__gMinuit.GetParameter(_minuit_id - 1, _par_min, _par_err)
        _par_min = _par_min.value
        _par_err = _par_err.value

        _x = np.linspace(start=_par_min - bound * _par_err, stop=_par_min + bound * _par_err, num=bins, endpoint=True)

        self.__gMinuit.mnexcm("FIX", arr('d', [_minuit_id]), 1, _error_code)

        _y = np.zeros(bins)
        for i in range(bins):
            self.__gMinuit.mnexcm("SET PAR", arr('d', [_minuit_id, Double(_x[i])]), 2, _error_code)
            self.__gMinuit.mnexcm("MIGRAD", arr('d', [MAX_ITERATIONS, self.tolerance]), 2, _error_code)
            _y[i] = self._get_fit_info("fcn")

        self.__gMinuit.mnexcm("RELEASE", arr('d', [_minuit_id]), 1, _error_code)
        self._migrad()
        self.__gMinuit.mnexcm("SET PAR", arr('d', [_minuit_id, Double(_par_min)]), 2, _error_code)

        if subtract_min:
            _y -= self.function_value

        return np.asarray((_x, _y))
