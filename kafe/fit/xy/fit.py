from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ...core import NexusFitter, Nexus
from .._base import FitException, FitBase, DataContainerBase, ModelParameterFormatter, CostFunctionBase
from .container import XYContainer
from .cost import XYCostFunction_Chi2, XYCostFunction_UserDefined
from .format import XYModelFunctionFormatter
from .model import XYParametricModel, XYModelFunction


__all__ = ["XYFit"]


CONFIG_PARAMETER_DEFAULT_VALUE = 1.0
CONFIG_FIT_MAX_ITERATIONS = 10
CONFIG_FIT_CONVERGENCE_LIMIT = 1e-5

class XYFitException(FitException):
    pass


class XYFit(FitBase):
    CONTAINER_TYPE = XYContainer
    MODEL_TYPE = XYParametricModel
    MODEL_FUNCTION_TYPE = XYModelFunction
    EXCEPTION_TYPE = XYFitException
    RESERVED_NODE_NAMES = {'y_data', 'y_model', 'cost',
                          'x_error', 'y_data_error', 'y_model_error', 'total_error',
                          'x_cov_mat', 'y_data_cov_mat', 'y_model_cov_mat', 'total_cov_mat',
                          'x_cor_mat', 'y_data_cor_mat', 'y_model_cor_mat', 'total_cor_mat',
                          'x_cov_mat_inverse', 'y_data_cov_mat_inverse', 'y_model_cov_mat_inverse', 'total_cor_mat_inverse'}

    def __init__(self, xy_data, model_function, cost_function=XYCostFunction_Chi2(axes_to_use='xy', errors_to_use='covariance'), minimizer="iminuit",minimizer_kwargs=None):
        """
        Construct a fit of a model to *xy* data.

        :param xy_data: the x and y measurement values
        :type xy_data: (2, N)-array of float
        :param model_function: the model function
        :type model_function: :py:class:`~kafe.fit.indexed.XYModelFunction` or unwrapped native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe.fit._base.CostFunctionBase`-derived or unwrapped native Python function
        """
        # set the data
        self.data = xy_data

        # set/construct the model function object
        if isinstance(model_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_function)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        # set and validate the cost function
        if isinstance(cost_function, CostFunctionBase):
            self._cost_function = cost_function
        else:
            self._cost_function = XYCostFunction_UserDefined(cost_function)
            #self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?

        # declare cache
        self._invalidate_total_error_cache()
        
        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._initialize_fitter(minimizer, minimizer_kwargs)
        # create the child ParametricModel object
        self._param_model = self._new_parametric_model(self.x, self._model_function.func, self.parameter_values)


    # -- private methods

    def _init_nexus(self):
        self._nexus = Nexus()
        self._nexus.new(y_data=self.y_data)
        self._nexus.new(x=self.x)

        # create a NexusNode for each parameter of the model function

        _nexus_new_dict = OrderedDict()
        _arg_defaults = self._model_function.argspec.defaults
        _n_arg_defaults = 0 if _arg_defaults is None else len(_arg_defaults)
        self._fit_param_names = []
        for _arg_pos, _arg_name in enumerate(self._model_function.argspec.args):
            # skip independent variable parameter
            if _arg_name == self._model_function.x_name:
                continue
            if _arg_pos >= (self._model_function.argcount - _n_arg_defaults):
                _default_value = _arg_defaults[_arg_pos - (self._model_function.argcount - _n_arg_defaults)]
            else:
                _default_value = CONFIG_PARAMETER_DEFAULT_VALUE
            _nexus_new_dict[_arg_name] = _default_value
            self._fit_param_names.append(_arg_name)

        self._nexus.new(**_nexus_new_dict)  # Create nexus Nodes for function parameters

        self._nexus.new_function(self._model_function.func, function_name=self._model_function.name, add_unknown_parameters=False)

        # add an alias 'model' for accessing the model values
        self._nexus.new_alias(**{'y_model': self._model_function.name})

        # bind other reserved nodes
        self._nexus.new_function(lambda: self.x_data_error, function_name='x_data_error')
        self._nexus.new_function(lambda: self.x_data_cov_mat, function_name='x_data_cov_mat')
        self._nexus.new_function(lambda: self.x_data_cov_mat_inverse, function_name='x_data_cov_mat_inverse')
        self._nexus.new_function(lambda: self.x_model_error, function_name='x_model_error')
        self._nexus.new_function(lambda: self.x_model_cov_mat, function_name='x_model_cov_mat')
        self._nexus.new_function(lambda: self.x_model_cov_mat_inverse, function_name='x_model_cov_mat_inverse')
        self._nexus.new_function(lambda: self.x_total_error, function_name='x_total_error')
        self._nexus.new_function(lambda: self.x_total_cov_mat, function_name='x_total_cov_mat')
        self._nexus.new_function(lambda: self.x_total_cov_mat_inverse, function_name='x_total_cov_mat_inverse')

        self._nexus.new_function(lambda: self.projected_xy_total_error, function_name='projected_xy_total_error')
        self._nexus.new_function(lambda: self.projected_xy_total_cov_mat, function_name='projected_xy_total_cov_mat')
        self._nexus.new_function(lambda: self.projected_xy_total_cov_mat_inverse, function_name='projected_xy_total_cov_mat_inverse')

        self._nexus.new_function(lambda: self.y_data_error, function_name='y_data_error')
        self._nexus.new_function(lambda: self.y_data_cov_mat, function_name='y_data_cov_mat')
        self._nexus.new_function(lambda: self.y_data_cov_mat_inverse, function_name='y_data_cov_mat_inverse')
        self._nexus.new_function(lambda: self.y_model_error, function_name='y_model_error')
        self._nexus.new_function(lambda: self.y_model_cov_mat, function_name='y_model_cov_mat')
        self._nexus.new_function(lambda: self.y_model_cov_mat_inverse, function_name='y_model_cov_mat_inverse')
        self._nexus.new_function(lambda: self.y_total_error, function_name='y_total_error')
        self._nexus.new_function(lambda: self.y_total_cov_mat, function_name='y_total_cov_mat')
        self._nexus.new_function(lambda: self.y_total_cov_mat_inverse, function_name='y_total_cov_mat_inverse')

        # the cost function (the function to be minimized)
        self._nexus.new_function(self._cost_function.func, function_name=self._cost_function.name, add_unknown_parameters=False)
        self._nexus.new_alias(**{'cost': self._cost_function.name})

    def _invalidate_total_error_cache(self):
        self.__cache_x_total_error = None
        self.__cache_x_total_cov_mat = None
        self.__cache_x_total_cov_mat_inverse = None
        self.__cache_projected_xy_total_error = None
        self.__cache_projected_xy_total_cov_mat = None
        self.__cache_projected_xy_total_cov_mat_inverse = None
        self.__cache_y_total_error = None
        self.__cache_y_total_cov_mat = None
        self.__cache_y_total_cov_mat_inverse = None
        self.__cache_y_error_band = None
        

    def _mark_errors_for_update(self):
        # TODO: implement a mass 'mark_for_update' routine in Nexus
        self._nexus.get_by_name('x_data_error').mark_for_update()
        self._nexus.get_by_name('x_data_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_data_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('x_model_error').mark_for_update()
        self._nexus.get_by_name('x_model_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_model_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('x_total_error').mark_for_update()
        self._nexus.get_by_name('x_total_cov_mat').mark_for_update()
        self._nexus.get_by_name('x_total_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_data_error').mark_for_update()
        self._nexus.get_by_name('y_data_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_data_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_model_error').mark_for_update()
        self._nexus.get_by_name('y_model_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_model_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_total_error').mark_for_update()
        self._nexus.get_by_name('y_total_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_total_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('projected_xy_total_error').mark_for_update()
        self._nexus.get_by_name('projected_xy_total_cov_mat').mark_for_update()
        self._nexus.get_by_name('projected_xy_total_cov_mat_inverse').mark_for_update()

    def _mark_errors_for_update_invalidate_total_error_cache(self):
        self._mark_errors_for_update()
        self._invalidate_total_error_cache()

    def _calculate_y_error_band(self):
        _xmin, _xmax = self._data_container.x_range
        _band_x = np.linspace(_xmin, _xmax, 100)  # TODO: config
        _f_deriv_by_params = self._param_model.eval_model_function_derivative_by_parameters(x=_band_x)
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[par_idx][x_idx]

        _f_deriv_by_params = _f_deriv_by_params.T
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[x_idx][par_idx]

        _band_y = np.zeros_like(_band_x)
        for _x_idx, _x_val in enumerate(_band_x):
            _p_res = _f_deriv_by_params[_x_idx]
            _band_y[_x_idx] = _p_res.dot(self.parameter_cov_mat).dot(_p_res)[0, 0]

        self.__cache_y_error_band = _band_y

    # -- public properties

    @property
    def x(self):
        """array of measurement *x* values"""
        return self._data_container.x

    @property
    def x_error(self):
        """array of pointwise *x* uncertainties"""
        return self._data_container.x_err

    @property
    def x_cov_mat(self):
        """the *x* covariance matrix"""
        return self._data_container.x_cov_mat

    @property
    def y_data(self):
        """array of measurement data *y* values"""
        return self._data_container.y

    @property
    def data(self):
        """(2, N)-array containing *x* and *y* measurement values"""
        return self._data_container.data

    @data.setter
    def data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise XYFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            _x_data = new_data[0]
            _y_data = new_data[1]
            self._data_container = self._new_data_container(_x_data, _y_data, dtype=float)

    @property
    def x_data_error(self):
        """array of pointwise *x* data uncertainties"""
        return self._data_container.x_err

    @property
    def y_data_error(self):
        """array of pointwise *y* data uncertainties"""
        return self._data_container.y_err

    @property
    def x_data_cov_mat(self):
        """the data *x* covariance matrix"""
        return self._data_container.x_cov_mat
    
    @property
    def y_data_cov_mat(self):
        """the data *y* covariance matrix"""
        return self._data_container.y_cov_mat

    @property
    def x_data_cov_mat_inverse(self):
        """inverse of the data *x* covariance matrix (or ``None`` if singular)"""
        return self._data_container.x_cov_mat_inverse

    @property
    def y_data_cov_mat_inverse(self):
        """inverse of the data *y* covariance matrix (or ``None`` if singular)"""
        return self._data_container.y_cov_mat_inverse

    @property
    def y_model(self):
        """array of *y* model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y

    @property
    def x_model_error(self):
        """array of pointwise model *x* uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.x_err

    @property
    def y_model_error(self):
        """array of pointwise model *y* uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y_err

    @property
    def x_model_cov_mat(self):
        """the model *x* covariance matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.x_cov_mat

    @property
    def y_model_cov_mat(self):
        """the model *y* covariance matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y_cov_mat

    @property
    def x_model_cov_mat_inverse(self):
        """inverse of the model *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.x_cov_mat_inverse

    @property
    def y_model_cov_mat_inverse(self):
        """inverse of the model *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y_cov_mat_inverse

    @property
    def x_total_error(self):
        """array of pointwise total *x* uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_x_total_error is None:
            _tmp = self.x_data_error**2
            _tmp += self.x_model_error**2
            self.__cache_x_total_error = np.sqrt(_tmp)
        return self.__cache_x_total_error

    @property
    def y_total_error(self):
        """array of pointwise total *y* uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_y_total_error is None:
            _tmp = self.y_data_error**2
            _tmp += self.y_model_error**2
            self.__cache_y_total_error = np.sqrt(_tmp)
        return self.__cache_y_total_error

    @property
    def projected_xy_total_error(self):
        """array of pointwise total *y* with the x uncertainties projected on top of them"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_error
        if self.__cache_projected_xy_total_error is None:
            _x_errors = self.x_total_error
            _precision = 0.01 * np.min(_x_errors)
            _derivatives = self._param_model.eval_model_function_derivative_by_x(dx=_precision)
            self.__cache_projected_xy_total_error = np.sqrt(self.y_total_error**2 + self.x_total_error**2 * _derivatives**2)
        return self.__cache_projected_xy_total_error

    @property
    def x_total_cov_mat(self):
        """the total *x* covariance matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_x_total_cov_mat is None:
            _tmp = self.x_data_cov_mat
            _tmp += self.x_model_cov_mat
            self.__cache_x_total_cov_mat = _tmp
        return self.__cache_x_total_cov_mat

    @property
    def y_total_cov_mat(self):
        """the total *y* covariance matrix"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_y_total_cov_mat is None:
            _tmp = self.y_data_cov_mat
            _tmp += self.y_model_cov_mat
            self.__cache_y_total_cov_mat = _tmp
        return self.__cache_y_total_cov_mat

    @property
    def projected_xy_total_cov_mat(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if np.count_nonzero(self._data_container.x_err) == 0:
            return self.y_total_cov_mat
        if self.__cache_projected_xy_total_cov_mat is None:
            _x_errors = self.x_total_error            
            _precision = 0.01 * np.min(_x_errors)
            _derivatives = self._param_model.eval_model_function_derivative_by_x(dx=_precision)
            _outer_product = np.outer(_derivatives, _derivatives)
            _projected_x_cov_mat = np.asarray(self.x_total_cov_mat) * _outer_product
            self.__cache_projected_xy_total_cov_mat = self.y_total_cov_mat + np.asmatrix(_projected_x_cov_mat)
        return self.__cache_projected_xy_total_cov_mat

    @property
    def x_total_cov_mat_inverse(self):
        """inverse of the total *x* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_x_total_cov_mat_inverse is None:
            _tmp = self.x_total_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_x_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_x_total_cov_mat_inverse

    @property
    def y_total_cov_mat_inverse(self):
        """inverse of the total *y* covariance matrix (or ``None`` if singular)"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_y_total_cov_mat_inverse is None:
            _tmp = self.y_total_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_y_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_y_total_cov_mat_inverse

    @property
    def projected_xy_total_cov_mat_inverse(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_projected_xy_total_cov_mat_inverse is None:
            _tmp = self.projected_xy_total_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_projected_xy_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_projected_xy_total_cov_mat_inverse

    @property
    def y_error_band(self):
        """one-dimensional array representing the uncertainty band around the model function"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_y_error_band is None:
            self._calculate_y_error_band()
        return self.__cache_y_error_band

    @property
    def x_range(self):
        """range of the *x* measurement data"""
        return self._data_container.x_range

    @property
    def y_range(self):
        """range of the *y* measurement data"""
        return self._data_container.y_range

    # -- public methods

    def add_simple_error(self, axis, err_val, correlation=0, relative=False):
        """
        Add a simple uncertainty source for axis to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :return: error id
        :rtype: int
        """
        # delegate to data container
        _ret = self._data_container.add_simple_error(axis, err_val, correlation=correlation, relative=relative)
        # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret

    def add_matrix_error(self, axis, err_matrix, matrix_type, err_val=None, relative=False):
        """
        Add a matrix uncertainty source for an axis to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param axis: ``'x'``/``0`` or ``'y'``/``1``
        :type axis: str or int
        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :return: error id
        :rtype: int
        """
        # delegate to data container
        _ret = self._data_container.add_matrix_error(axis, err_matrix, matrix_type, err_val=err_val, relative=relative)
        # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret

    def disable_error(self, err_id):
        """
        Temporarily disable an uncertainty source so that it doesn't count towards calculating the
        total uncertainty.

        :param err_id: error id
        :type err_id: int
        """
        # delegate to data container
        _ret = self._data_container.disable_error(err_id)   # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret

    def do_fit(self):
        if not self._data_container.has_x_errors:
            super(XYFit, self).do_fit()
        else:
            self._fitter.do_fit()
            _previous_cost_function_value = self.cost_function_value
            for i in range(CONFIG_FIT_MAX_ITERATIONS):
                self._mark_errors_for_update_invalidate_total_error_cache()
                self._fitter.do_fit()
                if np.abs(self.cost_function_value - _previous_cost_function_value) < CONFIG_FIT_CONVERGENCE_LIMIT:
                    break
                _previous_cost_function_value = self.cost_function_value
            # update parameter formatters
            for _fpf, _pv, _pe in zip(self._model_function.argument_formatters, self.parameter_values, self.parameter_errors):
                _fpf.value = _pv
                _fpf.error = _pe

    def eval_model_function(self, x=None, model_parameters=None):
        """
        Evaluate the model function.

        :param x: values of *x* at which to evaluate the model function (if ``None``, the data *x* values are used)
        :type x: iterable of float
        :param model_parameters: the model parameter values (if ``None``, the current values are used)
        :type model_parameters: iterable of float
        :return: model function values
        :rtype: :py:class:`numpy.ndarray`
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.eval_model_function(x=x, model_parameters=model_parameters)