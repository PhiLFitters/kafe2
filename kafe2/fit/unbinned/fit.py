from collections import OrderedDict
from copy import deepcopy

import sys

from ...core import NexusFitter, Nexus
from ...config import kc
from .._base import FitException, FitBase, DataContainerBase
from .container import UnbinnedContainer
from .cost import UnbinnedCostFunction_UserDefined, UnbinnedCostFunction_NegLogLikelihood
from .model import UnbinnedModelPDF, UnbinnedParametricModel

__all__ = ["UnbinnedFit"]


class UnbinnedFitException(FitException):
    pass


class UnbinnedFit(FitBase):
    CONTAINER_TYPE = UnbinnedContainer
    MODEL_TYPE = UnbinnedParametricModel
    MODEL_FUNCTION_TYPE = UnbinnedModelPDF
    EXCEPTION_TYPE = UnbinnedFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'cost', 'parameter_values', 'parameter_constraints'}

    def __init__(self, data, model_density_function='gaussian',
                 cost_function=UnbinnedCostFunction_NegLogLikelihood(), minimizer=None, minimizer_kwargs=None):
        """
        Construct a fit to a model of *unbinned* data.

        :param data: the data points
        :param model_density_function: the model density
        :type model_density_function: :py:class:`~kafe2.fit.unbinned.UnbinnedModelPDF` or unwrapped native Python function
        :param cost_function: the cost function
        :param minimizer: the minimizer to use
        :param minimizer_kwargs:
        """
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
        self._nexus.new_function(lambda: self.data, 'x')

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
        # bind other reserved node names
        self._nexus.new_function(lambda: self.parameter_values, function_name='parameter_values')
        self._nexus.new_function(lambda: self.parameter_constraints, function_name='parameter_constraints')

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

    # private methods
    def _invalidate_total_error_cache(self):
        pass

    def _mark_errors_for_update(self):
        pass

    @property
    def data(self):
        """The current data of the fit object"""
        return self._data_container.data

    @data.setter
    def data(self, new_data):
        """
        Set new data for the fit

        :param new_data: new data
        """
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise UnbinnedFitException("Incompatible container type '%s' (expected '%s')"
                                       % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = self._new_data_container(new_data, dtype=float)
        if hasattr(self, '_nexus'):
            self._nexus.get_by_name('x').mark_for_update()
        if hasattr(self, '_cost_function'):
            self._cost_function.ndf = self._data_container.size - len(self._param_model.parameters)

    @property
    def data_range(self):
        """The minimum and maximum value of the data"""
        return self._data_container.data_range

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data

    @property
    def model_error(self):
        """array of pointwise model uncertainties"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.err

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
        self._param_model.parameters = self.poi_values  # this is lazy, so just do it
        self._param_model.x = self.data
        return self._param_model.eval_model_function(x=x, model_parameters=model_parameters)

    def report(self, output_stream=sys.stdout, asymmetric_parameter_errors=False):
        super(UnbinnedFit, self).report(output_stream=output_stream,
                                        asymmetric_parameter_errors=asymmetric_parameter_errors)
