from copy import deepcopy

import sys

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
        super(UnbinnedFit, self).__init__(
            data=data, model_function=model_density_function, cost_function=cost_function,
            minimizer=minimizer, minimizer_kwargs=minimizer_kwargs)

    # private methods

    def _init_nexus(self):
        super(UnbinnedFit, self)._init_nexus()

        # add 'x' as an alias of 'data'
        self._nexus.add_alias('x', alias_for='data')

        self._nexus.add_dependency(
            'model',
            depends_on=(
                'parameter_values'
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
            self._data_container = UnbinnedContainer(new_data, dtype=float)
        self._data_container._on_error_change_callback = self._on_error_change

        self._nexus.get('x').mark_for_update()
        # TODO: make 'Alias' nodes pass on 'mark_for_update'
        self._nexus.get('data').mark_for_update()

    def _set_new_parametric_model(self):
        self._param_model = UnbinnedParametricModel(
            data=self.data,
            model_density_function=self._model_function,
            model_parameters=self.parameter_values
        )

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
    def goodness_of_fit(self):
        return None

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
        self._param_model.support = self.data
        return self._param_model.eval_model_function(support=x, model_parameters=model_parameters)

    def report(self, output_stream=sys.stdout, asymmetric_parameter_errors=False):
        super(UnbinnedFit, self).report(output_stream=output_stream,
                                        asymmetric_parameter_errors=asymmetric_parameter_errors)
