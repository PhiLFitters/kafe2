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


    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, numpoints = 20, strategy=1):
        print sigma
            
        _fraction = 0.02
        _contour_fun = self.function_value + sigma ** 2
        _ids = (self._par_names.index(parameter_name_1), self._par_names.index(parameter_name_2))
        _minimum = np.asarray([self._par_val[_ids[0]], self._par_val[_ids[1]]])
        _err = np.asarray([self._par_err[_ids[0]], self._par_err[_ids[1]]])
        
        _angles = []

        CONTOUR_ELLIPSE_POINTS = 21
        CONTOUR_STRETCHING = 4.0
        _unstretched_angles = np.linspace(-np.pi/2, np.pi/2, CONTOUR_ELLIPSE_POINTS, endpoint=True)
        _contour_search_ellipse = np.empty((2, CONTOUR_ELLIPSE_POINTS))
        _contour_search_ellipse[0] = sigma * _fraction * np.sin(_unstretched_angles)
        _contour_search_ellipse[1] = sigma * CONTOUR_STRETCHING * _fraction * np.cos(_unstretched_angles)
        _stretched_absolute_angles = np.abs(np.arctan(np.tan(_unstretched_angles) / CONTOUR_STRETCHING))
        _curvature_adjustion_factors = 1 + 0.01 * (10 - _stretched_absolute_angles * 180 / np.pi)
        _curvature_adjustion_factors = np.where(_curvature_adjustion_factors >= 0.25, _curvature_adjustion_factors, 0.25)
        
        _termination_distance = (6 * _fraction) ** 2
        
        _meta_cost_function = lambda z: (_contour_fun - self._calc_fun_with_constraints([{'type' : 'eq', 'fun' : lambda x: x[_ids[0]] - (_minimum[0] + _err[0] * z)},
                                                                                         {'type' : 'eq', 'fun' : lambda x: x[_ids[1]] - _minimum[1]}]))


        
        _start_x = opt.brentq(_meta_cost_function, 0, 2 * sigma, maxiter=1000)
        _start_point = np.asarray([_start_x, 0.0])
        
        _phi = self._calculate_tangential_angle(_start_point, _ids)
        _coords = _start_point
        _curvature_adjustion = 1.0
        _last_backtrack = 0

        _loops = 0
        
        _contour_coords = [_start_point]
        
        while(True):
            _transformed_search_ellipse = self._rotate_clockwise(_contour_search_ellipse * _curvature_adjustion, _phi)
            _transformed_search_ellipse[0] += _coords[0]
            _transformed_search_ellipse[1] += _coords[1]
            _transformed_search_ellipse = _transformed_search_ellipse.T
            _ellipse_fun_values = np.empty(CONTOUR_ELLIPSE_POINTS)
            for i in range(CONTOUR_ELLIPSE_POINTS):
                _ellipse_coords = _transformed_search_ellipse[i]
                _transformed_coords = self._transform_coordinates(_minimum, _ellipse_coords, _err)
                _point_constraints = [{"type" : "eq", "fun" : lambda x: x[_ids[0]] - _transformed_coords[0]},
                                      {"type" : "eq", "fun" : lambda x: x[_ids[1]] - _transformed_coords[1]}]
                _ellipse_fun_values[i] = self._calc_fun_with_constraints(_point_constraints)
            _min_index = np.argmin(np.abs(_ellipse_fun_values - _contour_fun))
            _new_coords = _transformed_search_ellipse[_min_index]

            _curvature_adjustion *= _curvature_adjustion_factors[_min_index]
            if _curvature_adjustion > 1.0:
                _curvature_adjustion = 1.0

            
            if _stretched_absolute_angles[_min_index] > 0.349111 and _last_backtrack > 3:
                print "BACKTRACK"
                _last_backtrack = 0
                _contour_coords = _contour_coords[0:-1]
            else:
                _contour_coords.append(_new_coords)
                _last_backtrack += 1
            
            _delta = _contour_coords[-1] - _contour_coords[-2]
            _phi = np.arctan2(_delta[0], _delta[1])
            _coords = _contour_coords[-1]
#             _phi = self._calculate_tangential_angle(_coords, _ids)
            
            if np.sum((_coords - _start_point) ** 2) < _termination_distance and _loops > 10:
                break
            
            if _loops < 100:
                _loops += 1
            else:
                break
        self._func_wrapper_unpack_args(self._par_val)
        return np.asarray(self._transform_contour(_minimum, _contour_coords, _err)).T
    
    @staticmethod
    def _transform_coordinates(minimum, sigma_coordinates, errors):
        return minimum + sigma_coordinates * errors
    
    @staticmethod
    def _transform_contour(minimum, sigma_contour, errors):
        _transformed_contour = []
        for _coords in sigma_contour:
            _transformed_contour.append(MinimizerScipyOptimize._transform_coordinates(minimum, _coords, errors))
        return _transformed_contour
        
    @staticmethod
    def _rotate_clockwise(xy_values, phi):
        _rotated_xy_values = np.empty(shape=xy_values.shape)
        _rotated_xy_values[0] =  np.cos(phi) * xy_values[0] + np.sin(phi) * xy_values[1]
        _rotated_xy_values[1] = -np.sin(phi) * xy_values[0] + np.cos(phi) * xy_values[1]
        return _rotated_xy_values
    
    def _calculate_tangential_angle(self, coords, ids):
        _meta_cost_function_gradient = lambda pars: self._calc_fun_with_constraints([{'type' : 'eq', 'fun' : lambda x: x[ids[0]] - (self._par_val[ids[0]] + self._par_err[ids[0]] * pars[0])},
                                                                                     {'type' : 'eq', 'fun' : lambda x: x[ids[1]] - (self._par_val[ids[1]] + self._par_err[ids[1]] * pars[1])}])
        _grad = nd.Gradient(_meta_cost_function_gradient)(coords)
        return np.arctan2(_grad[0], _grad[1]) + np.pi / 2

        
    def _calc_fun_with_constraints(self, additional_constraints):
        _local_constraints = self._par_constraints + additional_constraints
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
        
    def profile(self, parameter_name, bins=21, bound=2, args=None, subtract_min=False):
        _par_id = self._par_names.index(parameter_name)
        _par_err = self._par_err[_par_id]
        _par_min = self._par_val[_par_id]
        _par = np.linspace(start=_par_min - bound * _par_err, stop=_par_min + bound * _par_err, num=bins, endpoint=True)
        _y_offset = self.function_value if subtract_min else 0
        
        _y = np.empty(bins)
        for i in range(bins):
            _y[i] = self._calc_fun_with_constraints([{"type" : "eq", "fun" : lambda x: x[_par_id] - _par[i]}])
        self._func_wrapper_unpack_args(self._par_val)
        return np.asarray([_par, _y - _y_offset])