from copy import deepcopy

from .._base import DataContainerBase, FitBase
from .._base.cost import CostFunction_NegLogLikelihood
from ..util import (  # noqa: F401 (collect imported but not used)
    collect,
    function_library,
)
from .container import HistContainer
from .model import HistModelFunction, HistParametricModel
from .plot import HistPlotAdapter

__all__ = ["HistFit"]


class HistFit(FitBase):
    CONTAINER_TYPE = HistContainer
    MODEL_TYPE = HistParametricModel
    MODEL_FUNCTION_TYPE = HistModelFunction
    PLOT_ADAPTER_TYPE = HistPlotAdapter
    RESERVED_NODE_NAMES = {
        "data",
        "model",
        "model_density",
        "cost",
        "data_error",
        "model_error",
        "total_error",
        "data_cov_mat",
        "model_cov_mat",
        "total_cov_mat",
        "data_cor_mat",
        "model_cor_mat",
        "total_cor_mat",
    }
    _BASIC_ERROR_NAMES = {"data_error", "model_error", "data_cov_mat", "model_cov_mat"}

    def __init__(
        self,
        data,
        model_function=function_library.normal_distribution,
        cost_function=CostFunction_NegLogLikelihood(data_point_distribution="poisson"),
        bin_evaluation="simpson",
        density=True,
        minimizer=None,
        minimizer_kwargs=None,
        dynamic_error_algorithm="nonlinear",
    ):
        """
        Construct a fit of a model to a histogram. If bin_evaluation is a Python function or
        of a numpy.vectorize object it is interpreted as the antiderivative of
        model_density_function. If bin_evaluation is equal to "rectangle", "midpoint", "trapezoid",
        or "simpson" the bin heights are evaluated according to the corresponding quadrature
        formula. If bin_evaluation is equal to "numerical" the bin heights are evaluated by
        numerically integrating model_density_function.

        :param data: an encapsulated representation of the histogrammed data.
        :type data: :py:class:`~kafe2.fit.hist.HistContainer` or two-dimensional iterable of bin
            heights and bin edges as returned by `np.histogram`.
        :param model_function: the model (density) function
        :type model__function: :py:class:`~kafe2.fit.hist.HistModelFunction` or unwrapped
            native Python function
        :param cost_function: the cost function
        :type cost_function: :py:class:`~kafe2.fit._base.CostFunctionBase`-derived or unwrapped
            native Python function
        :param bin_evaluation: how the model evaluates bin heights.
        :type bin_evaluation: str, callable, or numpy.vectorize
        :param density: if True, scale model function to the number of data points.
        :type density: bool
        :param minimizer: the minimizer to use for fitting.
        :type minimizer: None, "iminuit", "tminuit", or "scipy".
        :param minimizer_kwargs: dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict
        """
        self._bin_evaluation = bin_evaluation
        self._density = density
        super(HistFit, self).__init__(
            data=data,
            model_function=model_function,
            cost_function=cost_function,
            minimizer=minimizer,
            minimizer_kwargs=minimizer_kwargs,
            dynamic_error_algorithm=dynamic_error_algorithm,
        )

    # -- private methods

    def _init_nexus(self):
        super(HistFit, self)._init_nexus()

        self._nexus.add_dependency("model", depends_on=("parameter_values"))

    def _set_new_data(self, new_data):
        try:
            _new_data_is_numpy_histogram = len(new_data) == 2 and len(new_data[0]) + 1 == len(new_data[1])
        except TypeError:
            _new_data_is_numpy_histogram = False
        if _new_data_is_numpy_histogram:
            self._data_container = self.CONTAINER_TYPE(
                n_bins=len(new_data[0]),
                bin_range=(new_data[1][0], new_data[1][-1]),
                bin_edges=new_data[1],
            )
            self._data_container.set_bins(new_data[0])
        elif isinstance(new_data, self.CONTAINER_TYPE):
            self._data_container = deepcopy(new_data)
        elif isinstance(new_data, DataContainerBase):
            raise TypeError("Incompatible container type '{}' (expected '{}')".format(type(new_data), self.CONTAINER_TYPE))
        else:
            raise TypeError("Fitting a histogram requires a kafe2 HistContainer or a NumPy histogram as data " f"but received {new_data}")
        self._data_container._on_error_change_callback = self._on_error_change

        self._nexus.get("data").mark_for_update()

    def _set_new_parametric_model(self):
        # create the child ParametricModel object
        self._param_model = HistParametricModel(
            self._data_container.size,
            self._data_container.bin_range,
            self._model_function,
            self.parameter_values,
            self._data_container.bin_edges,
            bin_evaluation=self._bin_evaluation,
            density=self._density,
        )

    # -- public properties

    @property
    def model(self):
        """array of model predictions for the data points"""
        self._param_model.parameters = self.parameter_values  # this is lazy, so just do it
        # if model is just a density, scale it to the number of data points
        if self._param_model.density:
            return self._param_model.data * self._data_container.n_entries
        else:
            return self._param_model.data

    @property
    def density(self):
        return self._param_model.density

    # FIXME: how to handle scaling of model_error

    # -- public methods

    # add_error... methods inherited from FitBase ##

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
