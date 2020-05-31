from __future__ import print_function

import logging

from .minimizer_base import MinimizerBase, MinimizerException
from ..contour import ContourFactory

try:
    import scipy.optimize as opt
except ImportError:
    # TODO: handle importing nonexistent minimizer
    raise

import numpy as np
import numdifftools as nd


class MinimizerScipyOptimizeException(MinimizerException):
    pass


class MinimizerScipyOptimize(MinimizerBase):
    def __init__(self,
                 parameter_names, parameter_values, parameter_errors,
                 function_to_minimize, tolerance=1e-6, errordef=MinimizerBase.ERRORDEF_CHI2,
                 method=None):
        self._method = method
        self._par_bounds = None
        self._par_fixed = np.array([False] * len(parameter_names))
        self._par_constraints = []

        self._opt_result = None
        self._x0 = None  # Stores initial value for x0 when profiling a parameter
        super(MinimizerScipyOptimize, self).__init__(
            parameter_names=parameter_names, parameter_values=parameter_values,
            parameter_errors=parameter_errors, function_to_minimize=function_to_minimize,
            tolerance=tolerance, errordef=errordef
        )

    # -- private methods

    def _save_state(self):
        if self._par_val is None:
            self._save_state_dict['parameter_values'] = self._par_val
        else:
            self._save_state_dict['parameter_values'] = np.array(self._par_val)
        if self._par_err is None:
            self._save_state_dict['parameter_errors'] = self._par_err
        else:
            self._save_state_dict['parameter_errors'] = np.array(self._par_err)
        if self._par_bounds is None:
            self._save_state_dict['parameter_bounds'] = self._par_bounds
        else:
            self._save_state_dict['parameter_bounds'] = np.array(self._par_bounds)
        self._save_state_dict['function_value'] = self._fval
        self._save_state_dict['par_fixed'] = np.array(self._par_fixed)
        self._save_state_dict['opt_result'] = self._opt_result
        super(MinimizerScipyOptimize, self)._save_state()

    def _load_state(self):
        self._par_val = self._save_state_dict['parameter_values']
        if self._par_val is not None:
            self._par_val = np.array(self._par_val)
        self._par_err = self._save_state_dict['parameter_errors']
        if self._par_err is not None:
            self._par_err = np.array(self._par_err)
        self._par_bounds = self._save_state_dict['parameter_bounds']
        if self._par_bounds is not None:
            self._par_bounds = np.array(self._par_bounds)
        self._fval = self._save_state_dict['function_value']
        self._par_fixed = np.array(self._save_state_dict['par_fixed'])
        self._opt_result = self._save_state_dict['opt_result']
        super(MinimizerScipyOptimize, self)._load_state()

    # -- public properties

    @property
    def parameter_values(self):
        return np.array(self._par_val)

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

    def set(self, parameter_name, parameter_value):
        if parameter_name not in self._par_names:
            raise ValueError("No parameter named '%s'!" % (parameter_name,))
        _par_id = self._par_names.index(parameter_name)
        self._par_val[_par_id] = parameter_value
        self.reset()

    def fix(self, parameter_name):
        _par_id = self._par_names.index(parameter_name)
        self._par_fixed[_par_id] = True
        self._invalidate_cache()

    def is_fixed(self, parameter_name):
        _par_id = self._par_names.index(parameter_name)
        return self._par_fixed[_par_id]

    def release(self, parameter_name):
        _par_id = self._par_names.index(parameter_name)
        self._par_fixed[_par_id] = False
        self._invalidate_cache()

    def limit(self, parameter_name, parameter_bounds):
        assert len(parameter_bounds) == 2
        _par_id = self._par_names.index(parameter_name)
        if parameter_bounds[0] is not None and self._par_val[_par_id] < parameter_bounds[0]:
            self.set(parameter_name, parameter_bounds[0])
        elif parameter_bounds[1] is not None and self._par_val[_par_id] > parameter_bounds[1]:
            self.set(parameter_name, parameter_bounds[1])
        if self._par_bounds is None:
            self._par_bounds = [(None, None) for _pn in self._par_names]
        self._par_bounds[_par_id] = parameter_bounds

    def unlimit(self, parameter_name):
        _par_id = self._par_names.index(parameter_name)
        self._par_bounds[_par_id] = (None, None)
        _all_pars_unbounded = True
        for _par_bound in self._par_bounds:
            if _par_bound != (None, None):
                _all_pars_unbounded = False
                break
        if _all_pars_unbounded:
            self._par_bounds = None

    def minimize(self, max_calls=6000):
        if np.all(self._par_fixed):
            raise MinimizerScipyOptimizeException(
                "Cannot perform a fit if all parameters are fixed!")
        if np.any(self._par_fixed):
            # if pars are fixed arg list becomes shorter -> pick and insert fixed pars from self.parameter_values
            _par_fixed_indices = np.array(self._par_fixed, dtype=int)  # 1 if fixed, 0 otherwise
            # keep track of the arg positions within the fixed/dynamic lists:
            _position_indices = np.zeros_like(self.parameter_values, dtype=int)
            _n_fixed_parameters = 0
            _par_vals = []
            for _par_index, _par_fixed in enumerate(self._par_fixed):
                if _par_fixed:
                    _position_indices[_par_index] = _par_index
                    _n_fixed_parameters += 1
                else:
                    _position_indices[_par_index] = _par_index - _n_fixed_parameters
                    _par_vals.append(self.parameter_values[_par_index])
            _dyn_and_fixed_args = np.zeros(shape=(2,) + self.parameter_values.shape)
            _dyn_and_fixed_args[1] = self.parameter_values

            def _func(args):
                _dyn_and_fixed_args[0, 0:-_n_fixed_parameters] = args
                _selected_values = _dyn_and_fixed_args[_par_fixed_indices, _position_indices]
                return self._func_wrapper_unpack_args(_selected_values)

            if self._par_bounds is None:
                _par_bounds = None
            else:
                _par_bounds = []
                for _par_index, _par_fixed in enumerate(self._par_fixed):
                    if not _par_fixed:
                        _par_bounds.append(self._par_bounds[_par_index])

        else:
            _func = self._func_wrapper_unpack_args
            _par_vals = self.parameter_values
            _par_bounds = self._par_bounds

        disp = False
        if logging.root.level <= logging.INFO:
            disp = True

        self._opt_result = opt.minimize(_func,
                                        _par_vals,
                                        args=(),
                                        method=self._method,
                                        jac=None,
                                        hess=None, hessp=None,
                                        bounds=_par_bounds,
                                        constraints=self._par_constraints,
                                        tol=self.tolerance,
                                        callback=None,
                                        options=dict(maxiter=max_calls, disp=disp))
        self._did_fit = True
        self._invalidate_cache()

        if np.any(self._par_fixed):
            _dyn_and_fixed_args[0, 0:-_n_fixed_parameters] = self._opt_result.x
            self._par_val = _dyn_and_fixed_args[_par_fixed_indices, _position_indices]
        else:
            self._par_val = self._opt_result.x

        self._fval = self._opt_result.fun

        # Write back parameter values to nexus parameter nodes:
        self._func_wrapper_unpack_args(self.parameter_values)

        # Update parameter errors.
        # This is not done lazily because parameter errors need to be persistent.
        self._par_err = np.sqrt(np.diag(self.cov_mat))

    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, **minimizer_contour_kwargs):
        if not self.did_fit:
            raise MinimizerScipyOptimizeException("Need to perform a fit before calling contour()!")
        _algorithm = minimizer_contour_kwargs.pop("algorithm", "heuristic_grid")

        if _algorithm == "beacon":
            pass
        elif _algorithm == "heuristic_grid":
            _initial_points = minimizer_contour_kwargs.pop("initial_points", 1)
            _iterations = minimizer_contour_kwargs.pop("iterations", 5)
            _area_scale_factor = minimizer_contour_kwargs.pop("area_scale_factor", 1.5)
        else:
            raise MinimizerScipyOptimizeException("Unknown algorithm: {}".format(_algorithm))

        if minimizer_contour_kwargs:
            raise MinimizerScipyOptimizeException(
                "Unknown parameters for {}: {}".format(_algorithm, minimizer_contour_kwargs.keys()))

        if _algorithm == "beacon":
            return self._contour_beacon(parameter_name_1, parameter_name_2, sigma=sigma)
        if _algorithm == "heuristic_grid":
            return self._contour_heuristic_grid(parameter_name_1, parameter_name_2, sigma=sigma,
                                                initial_points=_initial_points, iterations=_iterations,
                                                area_scale_factor=_area_scale_factor)

    def _contour_heuristic_grid(self, parameter_name_1, parameter_name_2, sigma=1.0, initial_points=1,
                                iterations=5, area_scale_factor=1.5):
        initial_points = int(initial_points)
        iterations = int(iterations)

        if initial_points < 1:
            raise MinimizerScipyOptimize("initial_points must be a >= 1")
        if iterations < 0:
            raise MinimizerScipyOptimize("iterations must be a >= 0")

        _initial_points_per_axis = 1 + initial_points * 2
        _target_points_per_axis = 1 + initial_points * 2 ** (iterations + 1)
        _ids = (self._par_names.index(parameter_name_1), self._par_names.index(parameter_name_2))
        _minimum = np.asarray([self._par_val[_ids[0]], self._par_val[_ids[1]]])
        _err = np.asarray([self.parameter_errors[_ids[0]], self.parameter_errors[_ids[1]]])

        _x_values = np.linspace(start=-area_scale_factor * sigma * _err[0], stop=area_scale_factor * sigma * _err[0],
                                num=_target_points_per_axis, endpoint=True)
        _x_values += _minimum[0]
        _y_values = np.linspace(start=-area_scale_factor * sigma * _err[1], stop=area_scale_factor * sigma * _err[1],
                                num=_target_points_per_axis, endpoint=True)
        _y_values += _minimum[1]

        _grid = np.zeros((_target_points_per_axis, _target_points_per_axis)) - 1
        _x_step = int((_target_points_per_axis - 1) / (_initial_points_per_axis - 1))
        _y_step = int((_target_points_per_axis - 1) / (_initial_points_per_axis - 1))

        _min_coords = int((_target_points_per_axis - 1) / 2)
        _confirmed_coords = set()
        _unsure_coords = set()

        for _x in range(0, _target_points_per_axis, _x_step):
            for _y in range(0, _target_points_per_axis, _y_step):
                _local_constraints = [{'type': 'eq', 'fun': lambda x: x[_ids[0]] - _x_values[_x]},
                                      {'type': 'eq', 'fun': lambda x: x[_ids[1]] - _y_values[_y]}]
                _grid[_x, _y] = self._calc_fun_with_constraints(_local_constraints)

        _min_fun = min(self.function_value, _grid[_min_coords, _min_coords])
        _contour_fun = _min_fun + sigma ** 2

        _iterations = 0
        while _x_step > 0 and _y_step > 1:
            if _iterations % 2 == 0:
                _x_0 = int(_x_step / 2)
                _y_0 = int(_y_step / 2)
                _vector_1 = (int(_x_step / 2), int(_y_step / 2))
                _vector_2 = (int(_x_step / 2), -int(_y_step / 2))
            else:
                _x_0 = 0
                _y_0 = 0
                _vector_1 = (_x_step, 0)
                _vector_2 = (0, int(_y_step / 2))

            for _x in range(_x_0, _target_points_per_axis, _x_step):
                if _iterations % 2 == 1 and _x % (2 * _x_step) == 0:
                    _current_y_0 = _y_0 + int(_y_step / 2)
                else:
                    _current_y_0 = _y_0
                for _y in range(_current_y_0, _target_points_per_axis, _y_step):
                    _point_value = self._heuristic_point_evaluation(_contour_fun, _grid, _x, _y, _vector_1, _vector_2)
                    if _point_value == -1:
                        _local_constraints = [{'type': 'eq', 'fun': lambda x: x[_ids[0]] - _x_values[_x]},
                                              {'type': 'eq', 'fun': lambda x: x[_ids[1]] - _y_values[_y]}]
                        _grid[_x, _y] = self._calc_fun_with_constraints(_local_constraints)
                        _confirmed_coords.add((_x, _y))
                        if _iterations % 2 == 0:
                            _unsure_coords.add((_x - _x_step, _y))
                            _unsure_coords.add((_x, _y - _y_step))
                            _unsure_coords.add((_x + _x_step, _y))
                            _unsure_coords.add((_x, _y + _y_step))
                        else:
                            _unsure_coords.add((_x - _x_step, _y - int(_y_step / 2)))
                            _unsure_coords.add((_x - _x_step, _y + int(_y_step / 2)))
                            _unsure_coords.add((_x + _x_step, _y - int(_y_step / 2)))
                            _unsure_coords.add((_x + _x_step, _y + int(_y_step / 2)))
                    else:
                        _grid[_x, _y] = _point_value

            while _unsure_coords:
                _current_coords = _unsure_coords.pop()
                if (_current_coords[0] < 0 or _current_coords[0] >= _target_points_per_axis or
                        _current_coords[1] < 0 or _current_coords[1] >= _target_points_per_axis):
                    continue
                if _current_coords in _confirmed_coords:
                    continue
                _x = _current_coords[0]
                _y = _current_coords[1]
                _local_constraints = [{'type': 'eq', 'fun': lambda x: x[_ids[0]] - _x_values[_x]},
                                      {'type': 'eq', 'fun': lambda x: x[_ids[1]] - _y_values[_y]}]
                _current_fun = self._calc_fun_with_constraints(_local_constraints)
                _grid_fun = _grid[_current_coords[0], _current_coords[1]]
                if ((_current_fun > _contour_fun and _grid_fun < _contour_fun) or
                        (_current_fun < _contour_fun and _grid_fun > _contour_fun)):
                    if _iterations % 2 == 0:
                        _unsure_coords.add((_x - _x_step, _y))
                        _unsure_coords.add((_x, _y - _y_step))
                        _unsure_coords.add((_x + _x_step, _y))
                        _unsure_coords.add((_x, _y + _y_step))
                    else:
                        _unsure_coords.add((_x - _x_step, _y - int(_y_step / 2)))
                        _unsure_coords.add((_x - _x_step, _y + int(_y_step / 2)))
                        _unsure_coords.add((_x + _x_step, _y - int(_y_step / 2)))
                        _unsure_coords.add((_x + _x_step, _y + int(_y_step / 2)))
                _grid[_current_coords[0], _current_coords[1]] = _current_fun
                _confirmed_coords.add(_current_coords)

            if _iterations % 2 == 0:
                _x_step = int(_x_step / 2)
            else:
                _y_step = int(_y_step / 2)
            _iterations += 1

        _left_cutoff = 0
        _right_cutoff = _target_points_per_axis - 1
        _bottom_cutoff = 0
        _top_cutoff = _target_points_per_axis - 1
        _padding = int(3 / area_scale_factor * max(1, 2 ** (iterations - 4)))

        while _right_cutoff > 0 and np.min(_grid[_right_cutoff]) > _contour_fun:
            _right_cutoff -= 1
        _right_cutoff += _padding
        _right_cutoff = min(_right_cutoff, _target_points_per_axis - 1)

        while _left_cutoff < _right_cutoff and np.min(_grid[_left_cutoff]) > _contour_fun:
            _left_cutoff += 1
        _left_cutoff -= _padding
        _left_cutoff = max(_left_cutoff, 0)

        _grid = _grid[_left_cutoff:_right_cutoff]
        _grid = _grid.T

        while _top_cutoff > 0 and np.min(_grid[_top_cutoff]) > _contour_fun:
            _top_cutoff -= 1
        _top_cutoff += _padding
        _top_cutoff = min(_top_cutoff, _target_points_per_axis - 1)

        while _bottom_cutoff < _top_cutoff and np.min(_grid[_bottom_cutoff]) > _contour_fun:
            _bottom_cutoff += 1
        _bottom_cutoff -= _padding
        _bottom_cutoff = max(_bottom_cutoff, 0)

        _grid = _grid[_bottom_cutoff:_top_cutoff]
        _grid = _grid.T

        _x_values = _x_values[_left_cutoff:_right_cutoff]
        _y_values = _y_values[_bottom_cutoff:_top_cutoff]

        _grid = np.sqrt(_grid - _min_fun)
        self._func_wrapper_unpack_args(self._par_val)
        return ContourFactory.create_grid_contour(_x_values, _y_values, _grid, sigma)

    @staticmethod
    def _heuristic_point_evaluation(contour_fun, grid, x, y, vector_1, vector_2):
        _adjacent_points = MinimizerScipyOptimize._get_adjacent_grid_points(grid, x, y, vector_1, vector_2)
        if np.max(_adjacent_points) < contour_fun:
            return np.mean(_adjacent_points)
        if np.min(_adjacent_points) > contour_fun:
            return np.mean(_adjacent_points)
        return -1

    @staticmethod
    def _get_adjacent_grid_points(grid, x_0, y_0, vector_1, vector_2):
        _x_size = np.ma.size(grid, 0)
        _y_size = np.ma.size(grid, 1)
        _grid_points = []
        for i in range(4):
            _x = x_0
            _y = y_0
            if i == 0:
                _x -= vector_1[0]
                _y -= vector_1[1]
            elif i == 1:
                _x -= vector_2[0]
                _y -= vector_2[1]
            elif i == 2:
                _x += vector_1[0]
                _y += vector_1[1]
            elif i == 3:
                _x += vector_2[0]
                _y += vector_2[1]
            if _x >= 0 and _x < _x_size and _y >= 0 and _y < _y_size:
                _grid_points.append(grid[_x][_y])
        return np.asarray(_grid_points)

    def _contour_beacon(self, parameter_name_1, parameter_name_2, sigma=1.0, beacon_size=0.02):

        _contour_fun = self.function_value + sigma ** 2
        _contour_fun_upper_tolerance = self.function_value + (1.2 * sigma) ** 2
        _contour_fun_lower_tolerance = self.function_value + (0.8 * sigma) ** 2
        _ids = (self._par_names.index(parameter_name_1), self._par_names.index(parameter_name_2))
        _minimum = np.asarray([self._par_val[_ids[0]], self._par_val[_ids[1]]])
        _err = np.asarray([self._par_err[_ids[0]], self._par_err[_ids[1]]])

        _angles = []

        CONTOUR_ELLIPSE_POINTS = 21
        CONTOUR_STRETCHING = 4.0
        _unstretched_angles = np.linspace(-np.pi / 2, np.pi / 2, CONTOUR_ELLIPSE_POINTS, endpoint=True)
        _contour_search_ellipse = np.empty((2, CONTOUR_ELLIPSE_POINTS))
        _contour_search_ellipse[0] = sigma * beacon_size * np.sin(_unstretched_angles)
        _contour_search_ellipse[1] = sigma * CONTOUR_STRETCHING * beacon_size * np.cos(_unstretched_angles)
        _stretched_absolute_angles = np.abs(np.arctan(np.tan(_unstretched_angles) / CONTOUR_STRETCHING))
        _curvature_adjustion_factors = 1 + 0.025 * (10 - _stretched_absolute_angles * 180 / np.pi)
        _curvature_adjustion_factors = np.where(_curvature_adjustion_factors >= 0.25, _curvature_adjustion_factors,
                                                0.25)

        _termination_distance = (sigma * CONTOUR_STRETCHING * beacon_size) ** 2

        _meta_cost_function = lambda z: (_contour_fun - self._calc_fun_with_constraints(
            [{'type': 'eq', 'fun': lambda x: x[_ids[0]] - (_minimum[0] + _err[0] * z)},
             {'type': 'eq', 'fun': lambda x: x[_ids[1]] - _minimum[1]}]))

        _start_x = opt.brentq(_meta_cost_function, 0, 2 * sigma, maxiter=1000)
        _start_point = np.asarray([_start_x, 0.0])

        _phi = self._calculate_tangential_angle(_start_point, _ids)
        _coords = _start_point
        _curvature_adjustion = 1.0
        _last_backtrack = 0

        _loops = 0

        _contour_coords = [_start_point]

        while (True):
            _transformed_search_ellipse = self._rotate_clockwise(_contour_search_ellipse * _curvature_adjustion, _phi)
            _transformed_search_ellipse[0] += _coords[0]
            _transformed_search_ellipse[1] += _coords[1]
            _transformed_search_ellipse = _transformed_search_ellipse.T
            _ellipse_fun_values = np.empty(CONTOUR_ELLIPSE_POINTS)
            for i in range(CONTOUR_ELLIPSE_POINTS):
                _ellipse_coords = _transformed_search_ellipse[i]
                _transformed_coords = self._transform_coordinates(_minimum, _ellipse_coords, _err)
                _point_constraints = [{"type": "eq", "fun": lambda x: x[_ids[0]] - _transformed_coords[0]},
                                      {"type": "eq", "fun": lambda x: x[_ids[1]] - _transformed_coords[1]}]
                _ellipse_fun_values[i] = self._calc_fun_with_constraints(_point_constraints)
            _min_index = np.argmin(np.abs(_ellipse_fun_values - _contour_fun))
            _new_coords = _transformed_search_ellipse[_min_index]

            _curvature_adjustion *= _curvature_adjustion_factors[_min_index]
            if _curvature_adjustion > 1.0:
                _curvature_adjustion = 1.0

            if _stretched_absolute_angles[_min_index] > 0.349111:
                print("BACKTRACK")
                _contour_coords = _contour_coords[0:-1]
            else:
                _contour_coords.append(_new_coords)

            # _delta = _contour_coords[-1] - _contour_coords[-2]
            # _phi = np.arctan2(_delta[0], _delta[1])
            _coords = _contour_coords[-1]
            _phi = self._calculate_tangential_angle(_coords, _ids)

            if np.sum((_coords - _start_point) ** 2) < _termination_distance and _loops > 10:
                break

            if _loops < 200:
                _loops += 1
            else:
                break
        self._func_wrapper_unpack_args(self._par_val)
        return ContourFactory.create_xy_contour(self._transform_contour(_minimum, _contour_coords, _err), sigma)

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
        _rotated_xy_values[0] = np.cos(phi) * xy_values[0] + np.sin(phi) * xy_values[1]
        _rotated_xy_values[1] = -np.sin(phi) * xy_values[0] + np.cos(phi) * xy_values[1]
        return _rotated_xy_values

    def _calculate_tangential_angle(self, coords, ids):
        _meta_cost_function_gradient = lambda pars: self._calc_fun_with_constraints(
            [{'type': 'eq', 'fun': lambda x: x[ids[0]] - (self._par_val[ids[0]] + self._par_err[ids[0]] * pars[0])},
             {'type': 'eq', 'fun': lambda x: x[ids[1]] - (self._par_val[ids[1]] + self._par_err[ids[1]] * pars[1])}])
        _grad = nd.Gradient(_meta_cost_function_gradient)(coords)
        return np.arctan2(_grad[0], _grad[1]) + np.pi / 2

    def _calc_fun_with_constraints(self, additional_constraints, continuous_x0=False):
        _local_constraints = self._par_constraints + additional_constraints
        if continuous_x0:
            _x0 = self._x0
        else:
            _x0 = self._par_val
        _result = opt.minimize(self._func_wrapper_unpack_args,
                               _x0,
                               args=(),
                               method="slsqp",
                               jac=None,
                               bounds=self._par_bounds,
                               constraints=_local_constraints,
                               tol=self.tolerance,
                               callback=None,
                               options=dict(maxiter=6000, disp=False))
        self._x0 = _result.x
        return _result.fun

    def profile(self, parameter_name, bins=21, bound=2, subtract_min=False):
        if not self.did_fit:
            raise MinimizerScipyOptimizeException("Need to perform a fit before calling profile()!")
        _par_id = self._par_names.index(parameter_name)
        _par_err = self.parameter_errors[_par_id]
        _par_min = self._par_val[_par_id]
        _par = np.linspace(
            start=_par_min - bound * _par_err,
            stop=_par_min + bound * _par_err,
            num=bins, endpoint=True
        )
        _y_offset = self.function_value if subtract_min else 0

        _y = np.empty(bins)
        self._x0 = self._par_val
        for i in range(bins):
            _y[i] = self._calc_fun_with_constraints(
                [{"type": "eq", "fun": lambda x: x[_par_id] - _par[i]}], continuous_x0=True
            )
        self._func_wrapper_unpack_args(self._par_val)
        return np.asarray([_par, _y - _y_offset])

    def _func_wrapper(self, *parameter_values):
        '''call FCN, but ensure fixed parameters are passed with their fixed value'''
        # Note: this is needed in order to ensure that derivatives of `_func_wrapper`
        #       take parameter fixing into account
        assert (len(self.parameter_values) == len(self._par_fixed) == len(parameter_values))
        # replace parameter values
        parameter_values = [
            _fixed_val if _is_fixed else _call_val
            for _call_val, _fixed_val, _is_fixed in zip(parameter_values, self.parameter_values, self._par_fixed)
        ]
        _res = MinimizerBase._func_wrapper(self, *parameter_values)
        # some scipy methods handle 'nan' incorrectly -> return MAX_FLOAT instead
        if np.isnan(_res):
            return np.finfo(float).max
        return _res
