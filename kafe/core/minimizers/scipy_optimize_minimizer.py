try:
    import scipy.optimize as opt
except ImportError:
    # TODO: handle importing nonexistent minimizer
    raise

import numpy as np
import numdifftools as nd

class MinimizerScipyOptimizeException(Exception):
    pass

class MinimizerScipyOptimize(object):
    def __init__(self,
                 parameter_names, parameter_values, parameter_errors,
                 function_to_minimize, method="slsqp"):
        self._par_names = parameter_names
        self._par_val = parameter_values
        self._par_err = parameter_errors
        self._method = method
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

    def minimize(self, max_calls=6000):
        self._par_constraints = []
        for _par_id, (_pf, _pv) in enumerate(zip(self._par_fixed, self._par_val)):
            if _pf:
                self._par_constraints.append(
                    dict(type='eq', fun=lambda x: x[_par_id] - _pv, jac=lambda x: 0.)
                )
                
                
        self._opt_result = opt.minimize(self._func_wrapper_unpack_args,
                                        self._par_val,
                                        args=(),
                                        method=self._method,
                                        jac=None,
                                        hess=None, hessp=None,
                                        bounds=self._par_bounds,
                                        constraints=self._par_constraints,
                                        tol=self.tolerance,
                                        callback=None,
                                        options=dict(maxiter=max_calls, disp=False))

        self._par_val = self._opt_result.x

        self._hessian_inv = np.asmatrix(nd.Hessian(self._func_wrapper_unpack_args)(self._par_val)).I

        if self._hessian_inv is not None:
            self._par_cov_mat = self._hessian_inv * 2.0 * self._err_def
            self._par_err = np.sqrt(np.diag(self._par_cov_mat))

        self._fval = self._opt_result.fun


    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, step_1=0.01, step_2=0.01, numpoints = 20, strategy=1):
        if strategy == 0:
            _fraction = 0.08
            _bias = 0.1
        elif strategy == 1:
            _fraction = 0.04
            _bias = 1
        elif strategy == 2:
            _fraction = 0.01
            _bias = 1
            
        _contour_fun = self.function_value + sigma ** 2
        _ids = (self._par_names.index(parameter_name_1), self._par_names.index(parameter_name_2))
        _minimum = np.asarray([self._par_val[_ids[0]], self._par_val[_ids[1]]])
        _coords = (0, 0)
#         step_1 *= self._par_err[_ids[0]]
#         step_2 *= self._par_err[_ids[1]]
        _x_err, _y_err = 0.27, 0.1
        step_1, step_2 = _x_err * _fraction, _y_err * _fraction
        _x_vector = np.asarray([step_1, 0])
        _y_vector = np.asarray([0, step_2])
        _steps = np.asarray([[0, step_2], [step_1, 0], [0, -step_2], [-step_1, 0]])
        _fun_distance = sigma ** 2
        _adjacent_funs = np.zeros(4)
        _last_direction = -1
        
        _contour_coords = []
        _explored_coords = set()
        _explored_coords.add((0,0))
        _log_points = False
        _termination_coords = None
        _first_lap = True

        _loops = 0
        
        while True:
            if _coords == _termination_coords:
                if not _first_lap:
                    break
                else:
                    _first_lap = False
            _adjacent_coords = self._get_adjacent_coords(_coords)
            for i in range(4):
                if _adjacent_coords[i] in _explored_coords or i == _last_direction:
                    _adjacent_funs[i] = 0
                elif _adjacent_coords[i] == _termination_coords:
                    _adjacent_funs[i] = _contour_fun
                else:
                    _adjacent_funs[i] = self._calc_fun_at_point(_minimum + _adjacent_coords[i][0] * _x_vector + _adjacent_coords[i][1] * _y_vector, _ids)
            _distances = _contour_fun - _adjacent_funs
            for i in range(4):
                if _distances[i] < 0:
                    _distances[i] *= -_bias
            _adjacent_funs_best_distance = np.min(_distances)
            _min_index = np.argmin(_distances)
            _new_coords = _adjacent_coords[_min_index]
            
            for i in range(4):
                if i != _last_direction:
                    _explored_coords.add(_adjacent_coords[i])
            if _fun_distance < _adjacent_funs_best_distance and not _log_points:
                _log_points = True
#                 print "found contour"
                _termination_coords = _new_coords
                _explored_coords.clear()
            _coords = _new_coords
            _fun_distance = _adjacent_funs_best_distance
            _last_direction = (np.argmin(_distances) + 2) % 4
            if _log_points:
                _contour_coords.append(_coords)
            if _loops < 10000:
                _loops += 1
            else:
                break 
#         print "contour"
#         print _contour_coords
        _contour_array = np.asarray(_contour_coords, dtype=float).T
        _contour_array[0] = _contour_array[0] * step_1 + _minimum[0]
        _contour_array[1] = _contour_array[1] * step_2 + _minimum[1]
        #function call needed to reset the nexus cache to minimal values
        self._func_wrapper_unpack_args(self._par_val)
        return _contour_array
    
    def _get_adjacent_coords(self, central_coords):
        return [(central_coords[0], central_coords[1] + 1),
                (central_coords[0] + 1, central_coords[1]),
                (central_coords[0], central_coords[1] - 1),
                (central_coords[0] - 1, central_coords[1])]
    
    def _calc_fun_at_point(self, values, ids):
        _local_constraints = self._par_constraints + [{'type' : 'eq', 'fun' : lambda x: x[ids[0]] - values[0]},
                                                      {'type' : 'eq', 'fun' : lambda x: x[ids[1]] - values[1]}]
        _result = opt.minimize(self._func_wrapper_unpack_args,
                                        self._par_val,
                                        args=(),
                                        method="slsqp",
                                        jac=None,
                                        bounds=self._par_bounds,
                                        constraints=_local_constraints,
                                        tol=self.tolerance,
                                        callback=None,
                                        options=dict(maxiter=6000, disp=False))
        return _result.fun
        
    def profile(self, parameter_name, bins=20, bound=2, args=None, subtract_min=False):
        return np.array([[0,1,1.5],[1,2,3.25]])