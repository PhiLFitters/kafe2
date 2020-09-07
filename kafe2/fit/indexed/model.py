import numpy as np

from scipy.misc import derivative

from .._base import ParametricModelBaseMixin, ModelFunctionBase, ModelFunctionException
from .container import IndexedContainer, IndexedContainerException
from .format import IndexedModelFunctionFormatter


__all__ = ["IndexedParametricModel", "IndexedModelFunction"]


class IndexedModelFunctionException(ModelFunctionException):
    pass


class IndexedModelFunction(ModelFunctionBase):
    EXCEPTION_TYPE = IndexedModelFunctionException
    FORMATTER_TYPE = IndexedModelFunctionFormatter

    def __init__(self, model_function):
        """
        Construct :py:class:`IndexedModelFunction` object (a wrapper for a native Python function):

        :param model_function: function handle
        """
        # Indexed Fit Functions don't have independent arguments. They solely consist of parameters to be fitted.
        super(IndexedModelFunction, self).__init__(model_function=model_function, independent_argcount=0)


class IndexedParametricModelException(IndexedContainerException):
    pass


class IndexedParametricModel(ParametricModelBaseMixin, IndexedContainer):
    MODEL_FUNCTION_TYPE = IndexedModelFunction

    def __init__(self, model_func, model_parameters, shape_like=None):
        """
        Construct an :py:obj:`IndexedParametricModel` object:

        :param model_func: handle of Python function (the model function)
        :param model_parameters: iterable of parameter values with which the model function should be initialized
        :param shape_like: array with the same shape as the model
        """
        # print "IndexedParametricModel.__init__(model_func=%r, model_parameters=%r)" % (model_func, model_parameters)
        if shape_like is not None:
            _data = np.zeros_like(shape_like)
            try:
                _data[:] = model_func(*model_parameters)
            except ValueError:
                raise IndexedParametricModelException("Indexed Data and Function must have the same shape! "
                                                      "Got {} and {}".format(len(_data),
                                                                             len(model_func(*model_parameters))))
        else:
            _data = model_func(*model_parameters)
        super(IndexedParametricModel, self).__init__(model_func, model_parameters, _data)

    # -- private methods

    def _recalculate(self):
        # use parent class setter for 'data'
        IndexedContainer.data.fset(self, self.eval_model_function())
        self._pm_calculation_stale = False


    # -- public properties

    @property
    def data(self):
        """model predictions (one-dimensional :py:obj:`numpy.ndarray`)"""
        if self._pm_calculation_stale:
            self._recalculate()
        return super(IndexedParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise IndexedParametricModelException("Parametric model data cannot be set!")

    @property
    def data_range(self):
        """tuple containing the minimum and maximum of all model predictions"""
        _data = self.data
        return np.min(_data), np.max(_data)

    # -- public methods

    def eval_model_function(self, model_parameters=None):
        """
        Evaluate the model function.

        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :return: value(s) of the model function for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        return self._model_function_object(*_pars)

    def eval_model_function_derivative_by_parameters(self, model_parameters=None, par_dx=None):
        """
        Evaluate the derivative of the model function with respect to the model parameters.

        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :param par_dx: step size for numeric differentiation
        :type par_dx: float
        :return: value(s) of the model function derivative for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _pars = np.asarray(_pars)
        _par_dxs = par_dx if par_dx is not None else 1e-2 * (np.abs(_pars) + 1.0 / (1.0 + np.abs(_pars)))

        _ret = []
        for _par_idx, (_par_val, _par_dx) in enumerate(zip(_pars, _par_dxs)):
            def _chipped_func(par):
                _chipped_pars = _pars.copy()
                _chipped_pars[_par_idx] = par
                return self._model_function_object(*_chipped_pars)

            _der_val = derivative(_chipped_func, _par_val, dx=_par_dx)
            _ret.append(_der_val)
        return np.array(_ret)
