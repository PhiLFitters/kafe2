from copy import deepcopy

import sys
import six

from ...core import Nexus
from ...core.fitters.nexus import Parameter, NexusError
from .._base import FitException, FitBase, DataContainerBase, ModelFunctionBase
from .container import UnbinnedContainer
from .cost import UnbinnedCostFunction_NegLogLikelihood
from .model import UnbinnedParametricModel
from .plot import UnbinnedPlotAdapter
from ..util import collect

__all__ = ['UnbinnedFit', 'UnbinnedFitException']


class UnbinnedFitException(FitException):
    pass


class UnbinnedFit(FitBase):
    CONTAINER_TYPE = UnbinnedContainer
    MODEL_TYPE = UnbinnedParametricModel
    MODEL_FUNCTION_TYPE = ModelFunctionBase
    PLOT_ADAPTER_TYPE = UnbinnedPlotAdapter
    EXCEPTION_TYPE = UnbinnedFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'cost', 'parameter_values', 'parameter_constraints'}

    def __init__(self,
                 data,
                 model_density_function='normal_distribution_pdf',
                 cost_function=UnbinnedCostFunction_NegLogLikelihood(),
                 minimizer=None,
                 minimizer_kwargs=None):
        """
        Construct a fit to a model of *unbinned* data.

        :param data: the data points
        :param model_density_function: the model density
        :type model_density_function: :py:class:`~kafe2.fit._base.ModelFunctionBase` or unwrapped native Python function
        :param cost_function: the cost function
        :param minimizer: the minimizer to use for fitting.
        :type minimizer: None, "iminuit", "tminuit", or "scipy".
        :param minimizer_kwargs: dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """
        # set/construct the model function object
        if isinstance(model_density_function, self.__class__.MODEL_FUNCTION_TYPE):
            self._model_function = model_density_function
        else:
            self._model_function = self.__class__.MODEL_FUNCTION_TYPE(model_density_function)

        # validate the model function for this fit
        self._validate_model_function_for_fit_raise()

        self._cost_function = cost_function
        # Todo: implement different cost functions and check if data and cost function is compatible
        # TODO: convert cost_function to a kafe2 cost function object if it is a string

        # initialize the Nexus
        self._init_nexus()

        # save minimizer, minimizer_kwargs for serialization
        self._minimizer = minimizer
        self._minimizer_kwargs = minimizer_kwargs

        # initialize the Fitter
        self._initialize_fitter()

        self._fit_param_constraints = []
        self._loaded_result_dict = None

        self.data = data

    # private methods

    def _init_nexus(self):

        self._nexus = Nexus()

        for _type in ('data', 'model'):
            # add data and model for axis
            self._add_property_to_nexus(_type)

        # add 'x' as an alias of 'data'
        self._nexus.add_alias('x', alias_for='data')

        # get names and default values of all parameters
        _nexus_new_dict = self._get_default_values(
            model_function=self._model_function,
            x_name=self._model_function.x_name
        )

        # -- fit parameters

        self._fit_param_names = []  # names of all fit parameters (including nuisance parameters)
        self._poi_names = []  # names of the parameters of interest (i.e. the model parameters)
        for _par_name, _par_value in six.iteritems(_nexus_new_dict):
            # create nexus node for function parameter
            self._nexus.add(Parameter(_par_value, name=_par_name))

            # keep track of fit parameter names
            self._fit_param_names.append(_par_name)
            self._poi_names.append(_par_name)

        # -- nuisance parameters
        self._nuisance_names = []  # names of all nuisance parameters accounting for correlated errors

        self._nexus.add_function(lambda: self.poi_values, func_name='poi_values')
        self._nexus.add_function(lambda: self.parameter_values, func_name='parameter_values')
        self._nexus.add_function(lambda: self.parameter_constraints, func_name='parameter_constraints')

        # add the original function name as an alias to 'model'
        try:
            self._nexus.add_alias(self._model_function.name, alias_for='model')
        except NexusError:
            pass  # allow 'model' as function name for model

        self._nexus.add_function(
            collect,
            func_name="nuisance_vector"
        )

        # -- initialize nuisance parameters

        # TODO: implement nuisance parameters for unbinned data (?)

        # the cost function (the function to be minimized)
        _cost_node = self._nexus.add_function(
            self._cost_function.func,
            func_name=self._cost_function.name,
        )

        _cost_alias = self._nexus.add_alias('cost', alias_for=self._cost_function.name)

        self._nexus.add_dependency('poi_values', depends_on=self._poi_names)
        self._nexus.add_dependency('parameter_values', depends_on=self._fit_param_names)

        self._nexus.add_dependency(
            'model',
            depends_on=(
                'poi_values'
            )
        )

    # -- private methods

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise UnbinnedFitException("Incompatible container type '%s' (expected '%s')"
                                       % (type(new_data), self.CONTAINER_TYPE))
        else:
            self._data_container = self._new_data_container(new_data, dtype=float)
        self._data_container._on_error_change_callback = self._on_error_change

        self._nexus.get('x').mark_for_update()
        # TODO: make 'Alias' nodes pass on 'mark_for_update'
        self._nexus.get('data').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = self._new_parametric_model(
            data=self.data,
            model_density_function=self._model_function,
            model_parameters=self.parameter_values
        )
        self._param_model._on_error_change_callbacks = [self._on_error_change]

    @FitBase.data.getter
    def data(self):
        """The current data of the fit object"""
        return self._data_container.data

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
        self._param_model.support = self.data
        return self._param_model.eval_model_function(support=x, model_parameters=model_parameters)

    def report(self, output_stream=sys.stdout, asymmetric_parameter_errors=False):
        super(UnbinnedFit, self).report(output_stream=output_stream,
                                        asymmetric_parameter_errors=asymmetric_parameter_errors)
