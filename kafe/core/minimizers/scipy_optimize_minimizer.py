try:
    import scipy.optimize as opt
except ImportError:
    # TODO: handle importing nonexistent minimizer
    raise

import numpy as np

class MinimizerScipyOptimizeException(Exception):
    pass

class MinimizerScipyOptimize(object):
    def __init__(self,
                 parameter_names, parameter_values, parameter_errors,
                 function_to_minimize):
        self._par_names = parameter_names
        self._par_val = parameter_values
        self._par_err = parameter_errors
        self._par_bounds = None
        #self._par_bounds = [(None, None) for _pn in self._par_names]
        self._par_fixed = [False] * len(parameter_names)
        self._par_constraints = []
        """
        # for fixing:
        dict(type='eq', fun=lambda: _const, jac=lambda: 0.)
        """

        self._func_handle = function_to_minimize
        self._err_def = 1.0
        self._tol = 0.001

        # cache for calculations
        self._hessian = None
        self._hessian_inv = None
        self._fval = None
        self._par_cov_mat = None
        self._par_cor_mat = None
        self._par_asymm_err_dn = None
        self._par_asymm_err_up = None
        self._pars_contour = None

        self._opt_result = None

    # -- private methods

    def _get_opt_result(self):
        if self._opt_result is None:
            raise MinimizerScipyOptimizeException("Cannot get requested information: No fitters performed!")
        return self._opt_result

    # -- public properties

    @property
    def errordef(self):
        return self._err_def

    @errordef.setter
    def errordef(self, err_def):
        assert err_def > 0
        self._err_def = err_def


    @property
    def tolerance(self):
        return self._tol

    @tolerance.setter
    def tolerance(self, tolerance):
        assert tolerance > 0
        self._tol = tolerance




    @property
    def hessian(self):
        # TODO: cache this
        return self._hessian_inv.I

    @property
    def cov_mat(self):
        return self._par_cov_mat

    @property
    def cor_mat(self):
        raise NotImplementedError
        return self._par_cor_mat

    @property
    def hessian_inv(self):
        return self._hessian_inv

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
        raise NotImplementedError
        _par_id = self._par_names.index(parameter_name)
        _pv = self._par_val[_par_id]
        self._par_fixed[_par_id] = True


    def fix_several(self, parameter_names):
        for _pn in parameter_names:
            self.fix(_pn)

    def release(self, parameter_name):
        raise NotImplementedError
        _par_id = self._par_names.index(parameter_name)
        self._par_fixed[_par_id] = False

    def release_several(self, parameter_names):
        for _pn in parameter_names:
            self.release(_pn)

    def limit(self, parameter_name, parameter_bounds):
        assert len(parameter_bounds) == 2
        _par_id = self._par_names.index(parameter_name)
        if self._par_bounds is None:
            self._par_bounds = [(None, None) for _pn in self._par_names]
        self._par_bounds[_par_id] = parameter_bounds

    def unlimit(self, parameter_name):
        _par_id = self._par_names.index(parameter_name)
        self._par_bounds[_par_id] = (None, None)

    def _func_wrapper_unpack_args(self, args):
        return self._func_handle(*args)

    def minimize(self, max_calls=6000, method='slsqp'):
        self._par_constraints = []
        for _par_id, (_pf, _pv) in enumerate(zip(self._par_fixed, self._par_val)):
            if _pf:
                self._par_constraints.append(
                    dict(type='eq', fun=lambda x: x[_par_id] - _pv, jac=lambda x: 0.)
                )


        self._opt_result = opt.minimize(self._func_wrapper_unpack_args,
                                        self._par_val,
                                        args=(),
                                        method=method,
                                        jac=None,
                                        hess=None, hessp=None,
                                        bounds=self._par_bounds,
                                        constraints=self._par_constraints,
                                        tol=self.tolerance,
                                        callback=None,
                                        options=dict(maxiter=max_calls, disp=False))


        self._par_val = self._opt_result.x

        _hi = self._hessian_inv
        if _hi is not None:
            try:
                self._hessian_inv = np.asmatrix(_hi.todense())
            except AttributeError:
                self._hessian_inv = np.asmatrix(_hi)

        if self._hessian_inv is not None:
            self._par_cov_mat = self._hessian_inv * 2.0 * self._err_def
            self._par_err = np.sqrt(np.diag(self._par_cov_mat))

        self._fval = self._opt_result.fun
