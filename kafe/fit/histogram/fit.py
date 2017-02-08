import inspect
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ...core import NexusFitter, Nexus
from .._base import FitException, FitBase, DataContainerBase
from .container import HistContainer
from .model import HistParametricModel

CONFIG_PARAMETER_DEFAULT_VALUE = 1.0


class HistFitException(FitException):
    pass


class HistFit(FitBase):
    CONTAINER_TYPE = HistContainer
    MODEL_TYPE = HistParametricModel
    EXCEPTION_TYPE = HistFitException
    X_VAR_NAME = 'x'
    RESERVED_NODE_NAMES = {'data', 'model', 'model_density', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}

    def __init__(self, data, model_density_function, cost_function, model_density_function_antiderivative=None):
        # set the data
        self.data = data

        # set and validate the model function
        self._model_func_handle = model_density_function
        self._validate_model_function_raise()

        self._model_func_antider_handle = model_density_function_antiderivative
        self._validate_model_function_antiderivative_raise()

        # set and validate the cost function
        self._cost_function_handle = cost_function
        self._validate_cost_function_raise()

        # declare cache
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None

        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._fitter = NexusFitter(nexus=self._nexus,
                                   parameters_to_fit=self._fit_param_names,
                                   parameter_to_minimize=self._cost_function_handle.__name__)

        # create the child ParametricModel object
        self._param_model = self._new_parametric_model(
            self._data_container.size,
            self._data_container.bin_range,
            self._model_func_handle,
            self.parameter_values,
            self._data_container.bin_edges,
            model_density_func_antiderivative=self._model_func_antider_handle)


    # -- private methods

    def _validate_model_function_raise(self):
        self._model_func_argspec = inspect.getargspec(self._model_func_handle)
        if self.X_VAR_NAME not in self._model_func_argspec.args:
            raise self.__class__.EXCEPTION_TYPE(
                "Model density function '%r' must have independent variable '%s' among its arguments!"
                % (self._model_func_handle, self.X_VAR_NAME))

        self._model_func_argcount = self._model_func_handle.func_code.co_argcount
        if self._model_func_argcount < 2:
            raise self.__class__.EXCEPTION_TYPE(
                "Model density function '%r' needs at least one parameter beside independent variable '%s'!"
                % (self._model_func_handle, self.X_VAR_NAME))

        super(HistFit, self)._validate_model_function_raise()

    def _validate_model_function_antiderivative_raise(self):
        if self._model_func_antider_handle is None:
            self._model_func_antider_argspec = None
            self._model_func_antider_argcount = None
            return
        self._model_func_antider_argspec = inspect.getargspec(self._model_func_antider_handle)
        if self.X_VAR_NAME not in self._model_func_argspec.args:
            raise self.__class__.EXCEPTION_TYPE(
                "Model density function antiderivative '%r' must have independent variable '%s' among its arguments!"
                % (self._model_func_antider_handle, self.X_VAR_NAME))

        self._model_func_antider_argcount = self._model_func_antider_handle.func_code.co_argcount
        if self._model_func_antider_argcount < 2:
            raise self.__class__.EXCEPTION_TYPE(
                "Model density function antiderivative '%r' needs at least one parameter beside independent variable '%s'!"
                % (self._model_func_antider_handle, self.X_VAR_NAME))


    def _init_nexus(self):
        self._nexus = Nexus()
        self._nexus.new(data=self.data)  # Node containing indexed data is called 'data'

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

        #self._nexus.new_function(self._model_func_handle, add_unknown_parameters=False)

        # add an alias 'model' for accessing the model values
        #self._nexus.new_alias(**{'model': self._model_func_handle.__name__})
        #self._nexus.new_alias(**{'model_density': self._model_func_handle.__name__})

        # bind 'model' node
        self._nexus.new_function(lambda: self.model, function_name='model')
        # need to set dependencies manually
        for _fpn in self._fit_param_names:
            self._nexus.add_dependency(_fpn, 'model')
        # bind other reserved nodes
        self._nexus.new_function(lambda: self.data_error, function_name='data_error')
        self._nexus.new_function(lambda: self.data_cov_mat, function_name='data_cov_mat')
        self._nexus.new_function(lambda: self.data_cov_mat_inverse, function_name='data_cov_mat_inverse')
        self._nexus.new_function(lambda: self.model_error, function_name='model_error')
        self._nexus.new_function(lambda: self.model_cov_mat, function_name='model_cov_mat')
        self._nexus.new_function(lambda: self.model_cov_mat, function_name='model_cov_mat_inverse')
        self._nexus.new_function(lambda: self.total_error, function_name='total_error')
        self._nexus.new_function(lambda: self.total_cov_mat, function_name='total_cov_mat')
        self._nexus.new_function(lambda: self.total_cov_mat_inverse, function_name='total_cov_mat_inverse')

        # the cost function (the function to be minimized)
        self._nexus.new_function(self._cost_function_handle, add_unknown_parameters=False)
        self._nexus.new_alias(**{'cost': self._cost_function_handle.__name__})

    def _mark_errors_for_update_invalidate_total_error_cache(self):
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None
        # TODO: implement a mass 'mark_for_update' routine in Nexus
        self._nexus.get_by_name('model').mark_for_update()
        self._nexus.get_by_name('data_error').mark_for_update()
        self._nexus.get_by_name('data_cov_mat').mark_for_update()
        self._nexus.get_by_name('data_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('model_error').mark_for_update()
        self._nexus.get_by_name('model_cov_mat').mark_for_update()
        self._nexus.get_by_name('model_cov_mat_inverse').mark_for_update()
        self._nexus.get_by_name('total_error').mark_for_update()
        self._nexus.get_by_name('total_cov_mat').mark_for_update()
        self._nexus.get_by_name('total_cov_mat_inverse').mark_for_update()

    # -- public properties

    @property
    def data(self):
        return self._data_container.data

    @data.setter
    def data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise HistFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            raise HistFitException("Fitting a histogram requires a HistContainer!")

    @property
    def data_error(self):
        return self._data_container.err

    @property
    def data_cov_mat(self):
        return self._data_container.cov_mat

    @property
    def data_cov_mat_inverse(self):
        return self._data_container.cov_mat_inverse

    @property
    def model(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data * self._data_container.n_entries  # NOTE: model is just a density->scale up

    @property
    def model_error(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.err  # FIXME: how to handle scaling

    @property
    def model_cov_mat(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cov_mat

    @property
    def model_cov_mat_inverse(self):
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.cov_mat_inverse

    @property
    def total_error(self):
        if self.__cache_total_error is None:
            _tmp = self.data_error**2
            _tmp += self.model_error**2
            self.__cache_total_error = np.sqrt(_tmp)
        return self.__cache_total_error

    @property
    def total_cov_mat(self):
        if self.__cache_total_cov_mat is None:
            _tmp = self.data_cov_mat
            _tmp += self.model_cov_mat
            self.__cache_total_cov_mat = _tmp
        return self.__cache_total_cov_mat

    @property
    def total_cov_mat_inverse(self):
        if self.__cache_total_cov_mat_inverse is None:
            _tmp = self.total_cov_mat
            try:
                _tmp = _tmp.I
                self.__cache_total_cov_mat_inverse = _tmp
            except np.linalg.LinAlgError:
                pass
        return self.__cache_total_cov_mat_inverse

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
    def cost_function_value(self):
        return self._fitter.parameter_to_minimize_value

    # -- public methods

    def add_simple_error(self, err_val, correlation=0, relative=False):
        # delegate to data container
        _ret = self._data_container.add_simple_error(err_val, correlation=correlation, relative=relative)
        # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret


    def add_matrix_error(self, err_matrix, matrix_type, err_val=None, relative=False):
        # delegate to data container
        _ret = self._data_container.add_matrix_error(err_matrix, matrix_type, err_val=err_val, relative=relative)
        # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret

    def disable_error(self, err_id):
        # delegate to data container
        _ret = self._data_container.disable_error(err_id)   # mark nexus error parameters as stale
        self._mark_errors_for_update_invalidate_total_error_cache()
        return _ret

    def do_fit(self):
        self._fitter.do_fit()