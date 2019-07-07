from collections import OrderedDict
from copy import deepcopy

import numpy as np
import six
import sys
import textwrap

from ...tools import print_dict_as_table
from ...core import NexusFitter, Nexus
from ...config import kc
from .._base import FitException, FitBase, DataContainerBase, CostFunctionBase
from .container import UnbinnedContainer
from .cost import UnbinnedCostFunction_UserDefined, UnbinnedCostFunction_NegLogLikelihood
from .model import UnbinnedParametricModel, UnbinnedModelDensityFunction
from ..util import function_library

__all__ = ["UnbinnedFit"]


class UnbinnedFitException(FitException):
    pass


class UnbinnedFit(FitBase):
    CONTAINER_TYPE = UnbinnedContainer
    MODEL_TYPE = UnbinnedParametricModel
    MODEL_FUNCTION_TYPE = UnbinnedModelDensityFunction
    EXCEPTION_TYPE = UnbinnedFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'model_density', 'cost'}

    def __init__(self, data, model_density_function='gaussian', minimizer=None, minimizer_kwargs=None):
        self.data = data

        # set/construct the model function object
        if isinstance(model_density_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_density_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_density_function)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()
        """
        # set and validate the cost function
        if isinstance(cost_function, CostFunctionBase):
            self._cost_function = cost_function
        elif isinstance(cost_function, str):
            self._cost_function = STRING_TO_COST_FUNCTION[cost_function]()
        else:
            self._cost_function = HistCostFunction_UserDefined(cost_function)
            # self._validate_cost_function_raise()
            # TODO: validate user-defined cost function? how?
        _data_and_cost_compatible, _reason = self._cost_function.is_data_compatible(self.data)
        if not _data_and_cost_compatible:
            raise self.EXCEPTION_TYPE('Fit data and cost function are not compatible: %s' % _reason)
        """  # Todo: implement different cost functions
        self._cost_function = UnbinnedCostFunction_NegLogLikelihood()

        # declare cache
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None

        # initialize the Nexus
        self._init_nexus()

        # initialize the Fitter
        self._initialize_fitter(minimizer, minimizer_kwargs)

        # create the child ParametricModel object
        self._param_model = self._new_parametric_model(
            self._model_function,
            self.parameter_values)
        # FIXME: nicer way than len()?
        self._cost_function.ndf = self._data_container.size - len(self._param_model.parameters)
        self._fit_param_constraints = []
        self._loaded_result_dict = None

    # private methods

    def _init_nexus(self):
        self._nexus = Nexus()
        self._nexus.new(data=self.data)

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
                _default_value = kc('core', 'default_initial_parameter_value')
            _nexus_new_dict[_arg_name] = _default_value
            self._fit_param_names.append(_arg_name)

        self._nexus.new(**_nexus_new_dict)  # Create nexus Nodes for function parameters

        # self._nexus.new_function(self._model_func_handle, add_unknown_parameters=False)

        # add an alias 'model' for accessing the model values
        # self._nexus.new_alias(**{'model': self._model_func_handle.__name__})
        # self._nexus.new_alias(**{'model_density': self._model_func_handle.__name__})

        # bind 'model' node
        self._nexus.new_function(lambda: self.model, function_name='model')
        # need to set dependencies manually
        for _fpn in self._fit_param_names:
            self._nexus.add_dependency(_fpn, 'model')
        # bind other reserved nodes
        #self._nexus.new_function(lambda: self.data_error, function_name='data_error')
        #self._nexus.new_function(lambda: self.data_cov_mat, function_name='data_cov_mat')
        #self._nexus.new_function(lambda: self.data_cov_mat_inverse, function_name='data_cov_mat_inverse')
        #self._nexus.new_function(lambda: self.model_error, function_name='model_error')
        #self._nexus.new_function(lambda: self.model_cov_mat, function_name='model_cov_mat')
        #self._nexus.new_function(lambda: self.model_cov_mat, function_name='model_cov_mat_inverse')
        self._nexus.new_function(lambda: self.total_error, function_name='total_error')
        #self._nexus.new_function(lambda: self.total_cov_mat, function_name='total_cov_mat')
        #self._nexus.new_function(lambda: self.total_cov_mat_inverse, function_name='total_cov_mat_inverse')
        self._nexus.new_function(lambda: self.parameter_values, function_name='parameter_values')
        self._nexus.new_function(lambda: self.parameter_constraints, function_name='parameter_constraints')

        # the cost function (the function to be minimized)
        self._nexus.new_function(self._cost_function.func, function_name=self._cost_function.name,
                                 add_unknown_parameters=False)
        self._nexus.new_alias(**{'cost': self._cost_function.name})

        for _arg_name in self._fit_param_names:
            self._nexus.add_dependency(source=_arg_name, target="parameter_values")

    def _invalidate_total_error_cache(self):
        self.__cache_total_error = None
        self.__cache_total_cov_mat = None
        self.__cache_total_cov_mat_inverse = None

    @property
    def data(self):
        return self._data_container.data

    @data.setter
    def data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise UnbinnedFitException("Incompatible container type '%s' (expected '%s')"
                                      % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = self._new_data_container(new_data, dtype=float)

    @property
    def total_error(self):
        """array of pointwise total uncertainties"""
        if self.__cache_total_error is None:
            _tmp = self.data_error**2
            _tmp += self.model_error**2
            self.__cache_total_error = np.sqrt(_tmp)
        return self.__cache_total_error

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data  # * self._data_container.n_entries  # NOTE: model is just a density->scale up
