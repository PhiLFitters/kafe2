import numpy as np

from scipy.misc import derivative

from .._base import ParametricModelBaseMixin
from .container import XYContainer, XYContainerException
from ..util import function_library


__all__ = ['XYParametricModel', 'XYParametricModelException']


class XYParametricModelException(XYContainerException):
    pass


class XYParametricModel(ParametricModelBaseMixin, XYContainer):
    #TODO why does model_function get abbreviated as model_func?
    def __init__(self, x_data, model_func=function_library.linear_model, model_parameters=[1.0, 1.0]):
        """
        Construct an :py:obj:`XYParametricModel` object:

        :param x_data: array containing the *x* values supporting the model
        :param model_func: handle of Python function (the model function)
        :param model_parameters: iterable of parameter values with which the model function should be initialized
        """
        # print "XYParametricModel.__init__(x_data=%r, model_func=%r, model_parameters=%r)" % (x_data, model_func, model_parameters)
        _x_data_array = np.array(x_data)
        _y_data = model_func(_x_data_array, *model_parameters)
        super(XYParametricModel, self).__init__(model_func, model_parameters, _x_data_array, _y_data)

    # -- private methods

    def _recalculate(self):
        # use parent class setter for 'y'
        XYContainer.y.fset(self, self.eval_model_function())
        self._pm_calculation_stale = False


    # -- public properties

    @property
    def data(self):
        """model predictions (one-dimensional :py:obj:`numpy.ndarray`)"""
        if self._pm_calculation_stale:
            self._recalculate()
        return super(XYParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise XYParametricModelException("Parametric model data cannot be set!")

    @property
    def x(self):
        """model *x* support values"""
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
        """model *y* values"""
        if self._pm_calculation_stale:
            self._recalculate()
        return super(XYParametricModel, self).y

    @y.setter
    def y(self, new_y):
        raise XYParametricModelException("Parametric model data cannot be set!")

    # -- public methods

    def eval_model_function(self, x=None, model_parameters=None):
        """
        Evaluate the model function.

        :param x: *x* values of the support points (if ``None``, the model *x* values are used)
        :type x: list or ``None``
        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :return: value(s) of the model function for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        return self._model_function_object(_x, *_pars)

    def eval_model_function_derivative_by_parameters(self, x=None, model_parameters=None, par_dx=None):
        """
        Evaluate the derivative of the model function with respect to the model parameters.

        :param x: *x* values of the support points (if ``None``, the model *x* values are used)
        :type x: list or ``None``
        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :param par_dx: step size for numeric differentiation
        :type par_dx: float
        :return: value(s) of the model function derivative for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _pars = np.asarray(_pars)
        _par_dxs = par_dx if par_dx is not None else 1e-2 * (np.abs(_pars) + 1.0/(1.0+np.abs(_pars)))

        _ret = []
        for _par_idx, (_par_val, _par_dx) in enumerate(zip(_pars, _par_dxs)):
            def _chipped_func(par):
                _chipped_pars = _pars.copy()
                _chipped_pars[_par_idx] = par
                return self._model_function_object(_x, *_chipped_pars)

            _der_val = derivative(_chipped_func, _par_val, dx=_par_dx)
            _ret.append(_der_val)
        return np.array(_ret)

    def eval_model_function_derivative_by_x(self, x=None, model_parameters=None, dx=None):
        """
        Evaluate the derivative of the model function with respect to the independent variable (*x*).

        :param x: *x* values of the support points (if ``None``, the model *x* values are used)
        :type x: list or ``None``
        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :param dx: step size for numeric differentiation
        :type dx: float
        :return: value(s) of the model function derivative
        :rtype: :py:obj:`numpy.ndarray`
        """
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _dxs = dx if dx is not None else 1e-2 * (np.abs(_x) + 1.0/(1.0+np.abs(_x)))
        try:
            iter(_dxs)
        except TypeError:
            _dxs = np.ones_like(_x)*_dxs

        _ret = []
        for _x_idx, (_x_val, _dx) in enumerate(zip(_x, _dxs)):
            def _chipped_func(x):
                return self._model_function_object(x, *_pars)

            _der_val = derivative(_chipped_func, _x_val, dx=_dx)
            _ret.append(_der_val)
        return np.array(_ret)
