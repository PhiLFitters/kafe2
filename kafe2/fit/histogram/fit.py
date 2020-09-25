from copy import deepcopy

from .._base import FitException, FitBase, DataContainerBase
from .container import HistContainer
from .._base.cost import CostFunction_NegLogLikelihood
from .model import HistParametricModel, HistModelFunction
from .plot import HistPlotAdapter
from ..util import function_library, collect

__all__ = ['HistFit', 'HistFitException']


class HistFitException(FitException):
    pass


class HistFit(FitBase):
    CONTAINER_TYPE = HistContainer
    MODEL_TYPE = HistParametricModel
    MODEL_FUNCTION_TYPE = HistModelFunction
    PLOT_ADAPTER_TYPE = HistPlotAdapter
    EXCEPTION_TYPE = HistFitException
    RESERVED_NODE_NAMES = {'data', 'model', 'model_density', 'cost',
                          'data_error', 'model_error', 'total_error',
                          'data_cov_mat', 'model_cov_mat', 'total_cov_mat',
                          'data_cor_mat', 'model_cor_mat', 'total_cor_mat'}
    _BASIC_ERROR_NAMES = {'data_error', 'model_error', 'data_cov_mat', 'model_cov_mat'}

    def __init__(self,
                 data,
                 model_density_function=function_library.normal_distribution_pdf,
                 cost_function=CostFunction_NegLogLikelihood(
                    data_point_distribution='poisson'),
                 bin_evaluation="simpson",
                 minimizer=None,
                 minimizer_kwargs=None,
                 dynamic_error_algorithm="nonlinear"):
        """
        Construct a fit of a model to a histogram. If bin_evaluation is a Python function or
        of a numpy.vectorize object it is interpreted as the antiderivative of
        model_density_function. If bin_evaluation is equal to "rectangle", "midpoint", "trapezoid",
        or "simpson" the bin heights are evaluated according to the corresponding quadrature
        formula. If bin_evaluation is equal to "numerical" the bin heights are evaluated by
        numerically integrating model_density_function.

        :param data: a :py:class:`~kafe2.fit.hist.HistContainer` representing histogrammed data
        :type data: :py:class:`~kafe2.fit.hist.HistContainer`
        :param model_density_function: the model density function
        :type model_density_function: :py:class:`~kafe2.fit.hist.HistModelFunction` or unwrapped
            native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped
            native Python function
        :param bin_evaluation: how the model evaluates bin heights.
        :type bin_evaluation: str, callable, or numpy.vectorize
        :param minimizer: the minimizer to use for fitting.
        :type minimizer: None, "iminuit", "tminuit", or "scipy".
        :param minimizer_kwargs: dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """
        self._bin_evaluation = bin_evaluation
        super(HistFit, self).__init__(
            data=data, model_function=model_density_function, cost_function=cost_function,
            minimizer=minimizer, minimizer_kwargs=minimizer_kwargs,
            dynamic_error_algorithm=dynamic_error_algorithm)

    # -- private methods

    def _init_nexus(self):
        super(HistFit, self)._init_nexus()

        self._nexus.add_dependency(
            'model',
            depends_on=(
                'parameter_values'
            )
        )

    def _set_new_data(self, new_data):
        if isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise HistFitException("Incompatible container type '{}' (expected '{}')"
                                   .format(type(new_data), self.CONTAINER_TYPE))
        else:
            raise HistFitException("Fitting a histogram requires a HistContainer!")
        self._data_container._on_error_change_callback = self._on_error_change

        self._nexus.get('data').mark_for_update()

    def _set_new_parametric_model(self):
        # create the child ParametricModel object
        self._param_model = HistParametricModel(
            self._data_container.size, self._data_container.bin_range, self._model_function,
            self.parameter_values, self._data_container.bin_edges,
            bin_evaluation=self._bin_evaluation)

    # -- public properties

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.data * self._data_container.n_entries  # NOTE: model is just a density->scale up

    # FIXME: how to handle scaling of model_error

    # -- public methods

    ## add_error... methods inherited from FitBase ##

    def eval_model_function_density(self, x, model_parameters=None):
        """
        Evaluate the model function density.

        :param x: values of *x* at which to evaluate the model function density
        :type x: iterable of float
        :param model_parameters: the model parameter values (if ``None``, the current values are used)
        :type model_parameters: iterable of float
        :return: model function density values
        :rtype: :py:class:`numpy.ndarray`
        """
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        return self._param_model.eval_model_function_density(x=x, model_parameters=model_parameters)
