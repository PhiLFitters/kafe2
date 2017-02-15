import inspect
import string
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ...core import NexusFitter, Nexus
from .._base import FitException, FitBase, DataContainerBase, ParameterFormatter, ModelFunctionFormatter
from .container import XYContainer
from .model import XYParametricModel

CONFIG_PARAMETER_DEFAULT_VALUE = 1.0

class ModelXYFunctionFormatter(ModelFunctionFormatter):
    def __init__(self, name, latex_name=None, x_name='x', latex_x_name=None,
                 arg_formatters=None, expression_string=None, latex_expression_string=None):
        self._x_name = x_name
        self._latex_x_name = latex_x_name
        if self._latex_x_name is None:
            self._latex_x_name = self._latexify_ascii(self._x_name)

        super(ModelXYFunctionFormatter, self).__init__(
            name, latex_name=latex_name, arg_formatters=arg_formatters,
            expression_string=expression_string,
            latex_expression_string=latex_expression_string
        )

    def _get_formatted_expression(self, format_as_latex=False):
        if format_as_latex and self._latex_expr_string is not None:
            _par_expr_string = self._latex_expr_string % tuple([_af.latex_name for _af in self._arg_formatters])
            _par_expr_string = string.replace(_par_expr_string, '{x}', self._latex_x_name)
        elif not format_as_latex and self._expr_string is not None:
            _par_expr_string = self._expr_string % tuple([_af.name for _af in self._arg_formatters])
            _par_expr_string = string.replace(_par_expr_string, '{x}', self._x_name)
        elif format_as_latex and self._latex_expr_string is None:
            _par_expr_string = self.DEFAULT_LATEX_EXPRESSION_STRING
        else:
            _par_expr_string = self.DEFAULT_EXPRESSION_STRING
        return _par_expr_string

    def get_formatted(self, with_par_values=True, n_significant_digits=2, format_as_latex=False, with_expression=False):
        _par_strings = self._get_formatted_args(with_par_values=with_par_values,
                                                n_significant_digits=n_significant_digits,
                                                format_as_latex=format_as_latex)
        _par_expr_string = ""
        if with_expression:
            _par_expr_string = self._get_formatted_expression(format_as_latex=format_as_latex)

        if format_as_latex:
            _out_string = r"%s\left(%s,%s\right)" % (self._latex_name, self._latex_x_name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
            _out_string = "$%s$" % (_out_string,)
        else:
            _out_string = "%s(%s, %s)" % (self._name, self._x_name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
        return _out_string


class XYFitException(FitException):
    pass


class XYFit(FitBase):
    CONTAINER_TYPE = XYContainer
    MODEL_TYPE = XYParametricModel
    EXCEPTION_TYPE = XYFitException
    X_VAR_NAME = 'x'
    RESERVED_NODE_NAMES = {'y_data', 'y_model', 'cost',
                          'x_error', 'y_data_error', 'y_model_error', 'total_error',
                          'x_cov_mat', 'y_data_cov_mat', 'y_model_cov_mat', 'total_cov_mat',
                          'x_cor_mat', 'y_data_cor_mat', 'y_model_cor_mat', 'total_cor_mat',
                          'x_cov_mat_inverse', 'y_data_cov_mat_inverse', 'y_model_cov_mat_inverse', 'total_cor_mat_inverse'}

    def __init__(self, xy_data, model_function, cost_function):
        # set the data
        self.data = xy_data

        # set and validate the model function
        self._model_func_handle = model_function
        self._validate_model_function_raise()

        # set and validate the cost function
        self._cost_function_handle = cost_function
        self._validate_cost_function_raise()

        # declare cache
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None
        self.__cache_y_error_band = None

        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._fitter = NexusFitter(nexus=self._nexus,
                                   parameters_to_fit=self._fit_param_names,
                                   parameter_to_minimize=self._cost_function_handle.__name__)


        self._fit_param_formatters = [ParameterFormatter(name=_pn, value=_pv, error=None)
                                      for _pn, _pv in self._fitter.fit_parameter_values.iteritems()]
        self._model_func_formatter = ModelXYFunctionFormatter(self._model_func_handle.__name__,
                                                              arg_formatters=self._fit_param_formatters,
                                                              x_name=self.X_VAR_NAME)

        # create the child ParametricModel objet
        self._param_model = self._new_parametric_model(self.x, self._model_func_handle, self.parameter_values)


    # -- private methods

    def _validate_model_function_raise(self):
        self._model_func_argspec = inspect.getargspec(self._model_func_handle)
        if self.X_VAR_NAME not in self._model_func_argspec.args:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' must have independent variable '%s' among its arguments!"
                % (self._model_func_handle, self.X_VAR_NAME))

        self._model_func_argcount = self._model_func_handle.func_code.co_argcount
        if self._model_func_argcount < 2:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' needs at least one parameter beside independent variable '%s'!"
                % (self._model_func_handle, self.X_VAR_NAME))

        super(XYFit, self)._validate_model_function_raise()

    def _init_nexus(self):
        self._nexus = Nexus()
        self._nexus.new(y_data=self.y_data)
        self._nexus.new(x=self.x)

        # create a NexusNode for each parameter of the model function

        _nexus_new_dict = OrderedDict()
        _arg_defaults = self._model_func_argspec.defaults
        _n_arg_defaults = 0 if _arg_defaults is None else len(_arg_defaults)
        self._fit_param_names = []
        for _arg_pos, _arg_name in enumerate(self._model_func_argspec.args):
            # skip independent variable parameter
            if _arg_name == self.X_VAR_NAME:
                continue
            if _arg_pos >= (self._model_func_argcount - _n_arg_defaults):
                _default_value = _arg_defaults[_arg_pos - (self._model_func_argcount - _n_arg_defaults)]
            else:
                _default_value = CONFIG_PARAMETER_DEFAULT_VALUE
            _nexus_new_dict[_arg_name] = _default_value
            self._fit_param_names.append(_arg_name)

        self._nexus.new(**_nexus_new_dict)  # Create nexus Nodes for function parameters

        self._nexus.new_function(self._model_func_handle, add_unknown_parameters=False)

        # add an alias 'model' for accessing the model values
        self._nexus.new_alias(**{'y_model': self._model_func_handle.__name__})

        # bind other reserved nodes
        self._nexus.new_function(lambda: self.y_data_error, function_name='y_data_error')
        self._nexus.new_function(lambda: self.y_data_cov_mat, function_name='y_data_cov_mat')
        self._nexus.new_function(lambda: self.y_data_cov_mat_inverse, function_name='y_data_cov_mat_inverse')
        self._nexus.new_function(lambda: self.y_model_error, function_name='y_model_error')
        self._nexus.new_function(lambda: self.y_model_cov_mat, function_name='y_model_cov_mat')
        self._nexus.new_function(lambda: self.y_model_cov_mat, function_name='y_model_cov_mat_inverse')
        self._nexus.new_function(lambda: self.y_total_error, function_name='y_total_error')
        self._nexus.new_function(lambda: self.y_total_cov_mat, function_name='y_total_cov_mat')
        self._nexus.new_function(lambda: self.y_total_cov_mat_inverse, function_name='y_total_cov_mat_inverse')

        # the cost function (the function to be minimized)
        self._nexus.new_function(self._cost_function_handle, add_unknown_parameters=False)
        self._nexus.new_alias(**{'cost': self._cost_function_handle.__name__})

    def _mark_errors_for_update_invalidate_total_error_cache(self):
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None
        self.__cache_y_error_band = None
        # TODO: implement a mass 'mark_for_update' routine in Nexus
        self._nexus.get_by_name('y_data_error').mark_for_update()
        self._nexus.get_by_name('y_data_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_data_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_model_error').mark_for_update()
        self._nexus.get_by_name('y_model_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_model_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('y_total_error').mark_for_update()
        self._nexus.get_by_name('y_total_cov_mat').mark_for_update()
        self._nexus.get_by_name('y_total_cov_mat_inverse').mark_for_update()

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
        return self._data_container.x

    @property
    def x_error(self):
        return self._data_container.x_err

    @property
    def x_cov_mat(self):
        return self._data_container.x_cov_mat

    @property
    def y_data(self):
        return self._data_container.y

    @property
    def data(self):
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
    def y_data_error(self):
        return self._data_container.y_err

    @property
    def y_data_cov_mat(self):
        return self._data_container.y_cov_mat

    @property
    def y_data_cov_mat_inverse(self):
        return self._data_container.y_cov_mat_inverse

    @property
    def y_model(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y

    @property
    def y_model_error(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y_err

    @property
    def y_model_cov_mat(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y_cov_mat

    @property
    def y_model_cov_mat_inverse(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.y_cov_mat_inverse

    @property
    def y_total_error(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_total_error is None:
            _tmp = self.y_data_error**2
            _tmp += self.y_model_error**2
            self.__cache_total_error = np.sqrt(_tmp)
        return self.__cache_total_error

    @property
    def y_total_cov_mat(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_total_cov_mat is None:
            _tmp = self.y_data_cov_mat
            _tmp += self.y_model_cov_mat
            self.__cache_total_cov_mat = _tmp
        return self.__cache_total_cov_mat

    @property
    def y_total_cov_mat_inverse(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_total_cov_mat_inverse is None:
            _tmp = self.y_total_cov_mat
            _tmp = _tmp.I
            self.__cache_total_cov_mat_inverse = _tmp
        return self.__cache_total_cov_mat_inverse

    @property
    def y_error_band(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        if self.__cache_y_error_band is None:
            self._calculate_y_error_band()
        return self.__cache_y_error_band

    @property
    def x_range(self):
        return self._data_container.x_range

    @property
    def y_range(self):
        return self._data_container.y_range

    @property
    def parameter_values(self):
        return self.parameter_name_value_dict.values()

    # NOTE: not supported by kafe.core.fitters
    #       maybe implement _there_, but not here!
    # @parameter_values.setter
    # def parameter_values(self, param_values):
    #     return self.parameter_name_value_dict.values()

    @property
    def parameter_name_value_dict(self):
        return self._fitter.fit_parameter_values

    @property
    def parameter_errors(self):
        return self._fitter.fit_parameter_errors

    @property
    def parameter_cov_mat(self):
        return self._fitter.fit_parameter_cov_mat

    @property
    def cost_function_value(self):
        return self._fitter.parameter_to_minimize_value

    # -- public methods

    def add_simple_error(self, axis, err_val, correlation=0, relative=False):
        # delegate to data container
        _ret = self._data_container.add_simple_error(axis, err_val, correlation=correlation, relative=relative)
        # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret

    def add_matrix_error(self, axis, err_matrix, matrix_type, err_val=None, relative=False):
        # delegate to data container
        _ret = self._data_container.add_matrix_error(axis, err_matrix, matrix_type, err_val=err_val, relative=relative)
        # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret

    def disable_error(self, err_id):
        # delegate to data container
        _ret = self._data_container.disable_error(err_id)   # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret

    # def do_fit(self):
    #     self._fitter.do_fit()

    def eval_model_function(self, x=None, model_parameters=None):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        self._param_model.x = self.x
        return self._param_model.eval_model_function(x=x, model_parameters=model_parameters)