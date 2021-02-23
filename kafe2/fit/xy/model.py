try:
    import typing  # help IDEs with type-hinting inside docstrings
except ImportError:
    pass
import numpy  # help IDEs with type-hinting inside docstrings
import numpy as np

from scipy.misc import derivative

from .._base import ParametricModelBaseMixin
from .container import XYContainer, XYContainerException
from ..util import function_library


__all__ = ['XYParametricModel', 'XYParametricModelException']


class XYParametricModelException(XYContainerException):
    pass


class XYParametricModel(ParametricModelBaseMixin, XYContainer):
    # TODO why is model_function abbreviated as model_func?
    def __init__(self, x_data, model_func=function_library.linear_model,
                 model_parameters=(1.0, 1.0)):
        """Construct an :py:obj:`XYParametricModel` object:

        :param x_data: 1D array containing the *x* values supporting the model
        :type x_data: typing.Collection[float]
        :param model_func: Python function handle of the model function.
        :type model_func: typing.Callable
        :param model_parameters: 1D array containing the parameter values with which the model
            function should be initialized.
        :type model_parameters: typing.Collection[float]
        """
        _x_data_array = np.array(x_data)
        _y_data = model_func(_x_data_array, *model_parameters)
        self._pm_calculation_stale = False
        super(XYParametricModel, self).__init__(model_func, model_parameters,
                                                _x_data_array, _y_data)

    # -- private methods

    def _recalculate(self):
        # use parent class setter for 'y'
        XYContainer.y.fset(self, self.eval_model_function())
        self._pm_calculation_stale = False

    # -- public properties

    @property
    def data(self):
        """2D array with shape ``(2, N)`` containing the model predictions.

        :rtype: numpy.ndarray[numpy.ndarray[float]]
        """
        if self._pm_calculation_stale:
            self._recalculate()
        return super(XYParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise XYParametricModelException("Parametric model data cannot be set!")

    @property
    def x(self):
        """1D array containing the *x* support values.

        :rtype: numpy.ndarray[float]
        """
        return super(XYParametricModel, self).x

    @x.setter
    def x(self, new_x):
        # resetting 'x' -> must reset entire data array
        self._data = np.zeros((2, len(new_x)))
        self._data[0] = new_x
        self._pm_calculation_stale = True
        self._clear_total_error_cache()

    @property
    def y(self):
        """1D array containing the *y* values calculated from the *x* support values and the
        current parameters.

        :rtype: numpy.ndarray[float]
        """
        if self._pm_calculation_stale:
            self._recalculate()
        return super(XYParametricModel, self).y

    @y.setter
    def y(self, new_y):
        raise XYParametricModelException("Parametric model data cannot be set!")

    # -- public methods

    def eval_model_function(self, x=None, model_parameters=None):
        """Evaluate the model function.

        :param x: 1D array containing the *x* values of the support points. If :py:obj:`None`,
            the model *x* values are used.
        :type x: numpy.ndarray[float]
        :param model_parameters: 1D array containing the values of the model parameters. If
            :py:obj:`None`, the current values are used.
        :type model_parameters: typing.Collection[float] or None
        :return: Values of the model function for the given parameters.
        :rtype: numpy.ndarray[float]
        """
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        return self._model_function_object(_x, *_pars)

    def eval_model_function_derivative_by_parameters(self, x=None, model_parameters=None,
                                                     par_dx=None):
        """Evaluate the derivative of the model function with respect to the model parameters.

        :param x: 1D array with length ``N`` containing the *x* values of the support points. If
            :py:obj:`None`, the model *x* values are used.
        :type x: numpy.ndarray[float] or None
        :param model_parameters: 1D array with length ``pars`` containing the values of the model
            parameters. If :py:obj:`None`, the current values are used.
        :type model_parameters: typing.Collection[float] or None
        :param par_dx: 1D array with length ``pars`` containing the numeric differentiation step
            size for each parameter.
        :type par_dx: typing.Collection[float]
        :return: 2D array with shape ``(pars, N)`` containing the values of the model function
            derivatives with respect to the parameters.
        :rtype: numpy.ndarray[numpy.ndarray[float]]
        """
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _pars = np.asarray(_pars)
        _par_dxs = par_dx if par_dx is not None \
            else 1e-2 * (np.abs(_pars) + 1.0 / (1.0 + np.abs(_pars)))

        _ret = np.zeros((len(_pars), len(_x)))
        for _par_idx, (_par_val, _par_dx) in enumerate(zip(_pars, _par_dxs)):
            def _chipped_func(par):
                _chipped_pars = _pars.copy()
                _chipped_pars[_par_idx] = par
                return self._model_function_object(_x, *_chipped_pars)

            _der_val = np.array(derivative(_chipped_func, _par_val, dx=_par_dx))
            _ret[_par_idx] = _der_val
        return _ret

    def eval_model_function_derivative_by_x(self, x=None, model_parameters=None, dx=None):
        """Evaluate the derivative of the model function with respect to the independent variable.

        :param x: 1D array containing the *x* values of the support points. If :py:obj:`None`,
            the model *x* values are used.
        :type x: numpy.ndarray[float] or None
        :param model_parameters: 1D array containing the values of the model parameters.
            If :py:obj:`None`, the current values are used.
        :type model_parameters: typing.Collection[float] or None
        :param dx: Step size for numeric differentiation.
        :type dx: float or typing.Collection[float]

        :return: 1D array containing the values of the model function derivative for each parameter.
        :rtype: numpy.ndarray[float]
        """
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _dxs = dx if dx is not None else 1e-2 * (np.abs(_x) + 1.0/(1.0+np.abs(_x)))
        _dxs = np.where(_dxs == 0, 1.0, _dxs)  # Replace zeros with ones

        _low = self._model_function_object(_x - _dxs, *_pars)
        _high = self._model_function_object(_x + _dxs, *_pars)
        return 0.5 * (_high - _low) / _dxs
