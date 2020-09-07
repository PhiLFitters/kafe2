import numpy as np
import six
from scipy import integrate
from inspect import getsource

from .._base import ParametricModelBaseMixin, ModelFunctionBase, ModelFunctionException
from .container import HistContainer, HistContainerException
from ..util import function_library


if six.PY2:
    from funcsigs import signature
else:
    from inspect import signature


__all__ = ["HistParametricModel", "HistModelFunction"]


class HistModelFunctionException(ModelFunctionException):
    pass


class HistModelFunction(ModelFunctionBase):
    EXCEPTION_TYPE = HistModelFunctionException

    def __init__(self, model_density_function=None):
        """
        Construct :py:class:`XYModelFunction` object (a wrapper for a native Python function):

        :param model_density_function: function handle
        """
        # TODO: default model function
        super(HistModelFunction, self).__init__(
            model_function=model_density_function, independent_argcount=1)


class HistParametricModelException(HistContainerException):
    pass


class HistParametricModel(ParametricModelBaseMixin, HistContainer):
    MODEL_FUNCTION_TYPE = HistModelFunction

    #TODO n_bins, bin_range, bin_edges contain redundant information, should the arguments for HistParametricModel be refactored?
    def __init__(self, n_bins, bin_range,
                 model_density_func=function_library.normal_distribution_pdf,
                 model_parameters=[1.0, 1.0], bin_edges=None, bin_evaluation="simpson"):
        super(HistParametricModel, self).__init__(
            model_density_func, model_parameters, n_bins, bin_range,
            bin_edges=bin_edges, fill_data=None, dtype=float)
        self._bin_evaluation = bin_evaluation
        if isinstance(self._bin_evaluation, str):
            self._bin_evaluation = self._bin_evaluation.lower()
            self._bin_evaluation_string = self._bin_evaluation
            if self._bin_evaluation in ("rectangle", "midpoint"):
                self._bin_evaluation_method = self._bin_evaluation_rectangle
            elif self._bin_evaluation == "trapezoid":
                self._bin_evaluation_method = self._bin_evaluation_trapezoid
            elif self._bin_evaluation == "simpson":
                self._bin_evaluation_method = self._bin_evaluation_simpson
            elif self._bin_evaluation == "numerical":
                self._bin_evaluation_method = self._bin_evaluation_numerical
            else:
                raise ValueError("Unknown bin evaluation method: %s" % self._bin_evaluation)
        else:
            if isinstance(self._bin_evaluation, np.vectorize):
                # special handling of numpy vectorized antiderivative functions
                _antiderivative_function_handle = self._bin_evaluation.pyfunc
            elif callable(self._bin_evaluation):
                # handle generic callables
                _antiderivative_function_handle = self._bin_evaluation
            else:
                # raise if not string and not callable
                raise ValueError(
                    "Cannot use {} as bin evaluation method: not a string and not callable!".format(
                        self._bin_evaluation))

            # Retrieving source code will fail if the function was generated through exec.
            # For kafe2go self._bin_evaluation_string will be replaced.
            try:
                self._bin_evaluation_string = getsource(_antiderivative_function_handle)
            except OSError:
                self._bin_evaluation_string = "OSError"
            except IOError:
                self._bin_evaluation_string = "IOError"
            _antider_parameters = list(signature(_antiderivative_function_handle).parameters)
            if isinstance(self._model_function_object, HistModelFunction):
                _model_func_parameters = list(self._model_function_object.signature.parameters)
            else:
                _model_func_parameters = list(signature(self._model_function_object).parameters)

            # require antiderivative and density to have the same arguments
            if _model_func_parameters != _antider_parameters:
                raise ValueError(
                    "Model density function and its antiderivative have different argument "
                    "signatures: (%r vs %r)"
                    % (_model_func_parameters, _antider_parameters))
            self._bin_evaluation_method = self._bin_evaluation_antiderivative

    # -- private methods

    def _recalculate(self):
        # don't use parent class setter for 'data' -> set directly
        self._data[1:-1] = self._bin_evaluation_method()
        self._pm_calculation_stale = False

    def _bin_evaluation_rectangle(self):
        _height_centers = self._model_function_object(self.bin_centers, *self._model_parameters)
        return self.bin_widths * _height_centers

    def _bin_evaluation_trapezoid(self):
        _height_edges = self._model_function_object(self._bin_edges, *self._model_parameters)
        return self.bin_widths / 2.0 * (_height_edges[:-1] + _height_edges[1:])

    def _bin_evaluation_simpson(self):
        _height_edges = self._model_function_object(self._bin_edges, *self._model_parameters)
        _height_centers = self._model_function_object(self.bin_centers, *self._model_parameters)
        return self.bin_widths / 6.0 * (
                _height_edges[:-1] + 4.0 * _height_centers + _height_edges[1:])

    def _bin_evaluation_numerical(self):
        _integrand_func = lambda x: self._model_function_object(x, *self._model_parameters)
        _int_val = np.zeros(self.size)
        for _i, (_a, _b) in enumerate(zip(self._bin_edges[:-1], self._bin_edges[1:])):
            _int_val[_i], _ = integrate.quad(_integrand_func, _a, _b)
        return _int_val

    def _bin_evaluation_antiderivative(self):
        _fval_antider_as = self._bin_evaluation(self._bin_edges[:-1], *self._model_parameters)
        _fval_antider_bs = self._bin_evaluation(self._bin_edges[1:], *self._model_parameters)
        assert len(_fval_antider_as) == len(_fval_antider_bs) == self.size
        return np.asarray(_fval_antider_bs) - np.asarray(_fval_antider_as)

    # -- public properties

    @property
    def data(self):
        """model predictions (one-dimensional :py:obj:`numpy.ndarray`)"""
        if self._pm_calculation_stale:
            self._recalculate()
        return super(HistParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise HistParametricModelException("Parametric model data cannot be set!")

    @property
    def bin_evaluation(self):
        """
        :return: how the model evaluates bin heights.
        :rtype str, callable, or numpy.vectorize
        """
        return self._bin_evaluation

    @property
    def bin_evaluation_string(self):
        """
        :return: string representation of how the model evaluates bin heights.
        :rtype str
        """
        return self._bin_evaluation_string

    # -- public methods

    def eval_model_function_density(self, x, model_parameters=None):
        """
        Evaluate the model function density.

        :param x: *x* values of the support points
        :type x: list of float
        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :return: value(s) of the model function for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        return self._model_function_object(x, *_pars)

    def fill(self, entries):
        raise HistParametricModelException("Parametric model of histogram cannot be filled!")
