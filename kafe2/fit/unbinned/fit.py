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
from .model import UnbinnedModelPDF, UnbinnedParametricModel
from ..util import function_library

__all__ = ["UnbinnedFit"]


class UnbinnedFitException(FitException):
    pass


class UnbinnedFit(FitBase):
    CONTAINER_TYPE = UnbinnedContainer
    MODEL_TYPE = UnbinnedParametricModel
    MODEL_FUNCTION_TYPE = UnbinnedModelPDF
    EXCEPTION_TYPE = UnbinnedFitException
    RESERVED_NODE_NAMES = {'data', 'pdf', 'cost'}

    def __init__(self, data, model_density_function='gaussian',
                 cost_function=UnbinnedCostFunction_NegLogLikelihood(), minimizer=None, minimizer_kwargs=None):
        self.data = data

        # set/construct the model function object
        if isinstance(model_density_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_density_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_density_function)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        self._cost_function = cost_function
        # Todo: implement different cost functions and check if data and cost function is compatible

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

        # create regular nexus node
        self._nexus.new(x=self.data)

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

        # create pdf function node
        self._nexus.new_function(function_handle=self._model_function.func)
        # create an alias for pdf
        self._nexus.new_alias(**{'model': self._model_function.func.__name__})

        # need to set dependencies manually
        for _fpn in self._fit_param_names:
            self._nexus.add_dependency(_fpn, 'model')

        # the cost function (the function to be minimized)
        self._nexus.new_function(self._cost_function.func, function_name=self._cost_function.name,
                                 add_unknown_parameters=False)
        self._nexus.new_alias(**{'cost': self._cost_function.name})

        self._nexus.add_dependency('model', 'cost')

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

    def report(self, output_stream=sys.stdout, asymmetric_parameter_errors=False):
        super(UnbinnedFit, self).report(output_stream=output_stream,
                                        asymmetric_parameter_errors=asymmetric_parameter_errors)
