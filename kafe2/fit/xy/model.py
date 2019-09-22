import inspect
import numpy as np

from scipy.misc import derivative

from .._base import ParametricModelBaseMixin, ModelFunctionBase, ModelFunctionException, ModelParameterFormatter
from .container import XYContainer, XYContainerException
from .format import XYModelFunctionFormatter
from ..util import function_library



__all__ = ["XYParametricModel", "XYModelFunction"]


class XYModelFunctionException(ModelFunctionException):
    pass

class XYModelFunction(ModelFunctionBase):
    EXCEPTION_TYPE = XYModelFunctionException
    FORMATTER_TYPE = XYModelFunctionFormatter

    def __init__(self, model_function=function_library.linear_model):
        """
        Construct :py:class:`XYModelFunction` object (a wrapper for a native Python function):

        :param model_function: function handle
        """
        self._x_name = 'x'
        super(XYModelFunction, self).__init__(model_function=model_function)

    def _validate_model_function_raise(self):
        # require 'xy' model function agruments to include 'x'
        if self.x_name not in self.signature.parameters:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' must have independent variable '%s' among its arguments!"
                % (self.func, self.x_name))

        # require 'xy' model functions to have at least two arguments
        if self.argcount < 2:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' needs at least one parameter beside independent variable '%s'!"
                % (self.func, self.x_name))

        # evaluate general model function requirements
        super(XYModelFunction, self)._validate_model_function_raise()

    def _get_parameter_formatters(self):
        _start_at_arg = 1
        return [ModelParameterFormatter(name=_pn, value=_pv, error=None)
                for _pn, _pv in zip(list(self.signature.parameters)[_start_at_arg:], self.argvals[_start_at_arg:])]

    def _assign_function_formatter(self):
        self._formatter = self.__class__.FORMATTER_TYPE(self.name,
                                                        arg_formatters=self._get_parameter_formatters(),
                                                        x_name=self.x_name)

    @property
    def x_name(self):
        """the name of the independent variable"""
        return self._x_name


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
        self._xy_data = np.zeros((2, len(new_x)))
        self._xy_data[0] = new_x
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
