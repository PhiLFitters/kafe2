import inspect
import numpy as np

from scipy.misc import derivative

from .._base import ParametricModelBaseMixin, ModelFunctionBase, ModelFunctionException, ModelParameterFormatter
from .container import XYContainer, XYContainerException
from .format import XYModelFunctionFormatter


__all__ = ["XYParametricModel", "XYModelFunction"]


class XYModelFunctionException(ModelFunctionException):
    pass

class XYModelFunction(ModelFunctionBase):
    EXCEPTION_TYPE = XYModelFunctionException
    FORMATTER_TYPE = XYModelFunctionFormatter

    def __init__(self, model_function, data_indices):
        """
        Construct :py:class:`XYModelFunction` object (a wrapper for a native Python function):

        :param model_function: function handle
        """
        self._x_name = 'x'
        self._data_indices = data_indices
        self._model_arg_indices = []
        _args = []
        #TODO implement
        _varargs = None
        _keywords = None
        _defaults = None
        for _model_function in model_function:
            _model_arg_indices = []
            _argspec = inspect.getargspec(_model_function)
            for _arg_name in _argspec[0]:
                if _arg_name is self._x_name:
                    continue
                if _arg_name not in _args:
                    _args.append(_arg_name)
                _model_arg_indices.append(_args.index(_arg_name))
            self._model_arg_indices.append(_model_arg_indices)
                
        _args.insert(0, self._x_name)
        self._model_function_argspec = inspect.ArgSpec(_args, _varargs, _keywords, _defaults)
        self._model_count = len(model_function)
        self._model_function_handle = model_function
        self._model_function_argcount = len(_args)
        self._validate_model_function_raise()
        self._assign_parameter_formatters()
        self._assign_function_formatter()

        #super(XYModelFunction, self).__init__(model_function=model_function)

    def _validate_model_function_raise(self):
        # require 'xy' model function agruments to include 'x'
        if self.x_name not in self.argspec.args:
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

    def _assign_parameter_formatters(self):
        _start_at_arg = 1
        self._arg_formatters = [ModelParameterFormatter(name=_pn, value=_pv, error=None)
                                for _pn, _pv in zip(self.argspec.args[_start_at_arg:], self.argvals[_start_at_arg:])]

    def _assign_function_formatter(self):
        self._formatter = self.__class__.FORMATTER_TYPE(self.name,
                                                        arg_formatters=self._arg_formatters,
                                                        x_name=self.x_name)

    @property
    def x_name(self):
        """the name of the independent variable"""
        return self._x_name

    def _eval(self, x, *args, **kwargs):
        _y = np.empty(x.size)
        _arg_lists = []
        _x_indices = kwargs.get("x_indices")
        _x_indices = _x_indices if _x_indices is not None else self._data_indices
        for i in range(self._model_count):
            _x = x[_x_indices[i]:_x_indices[i + 1]]
            _arg_list = self._construct_arg_list(args, i)
            _y[self._data_indices[i]:self._data_indices[i + 1]] = self._model_function_handle[i](_x, *_arg_list)
        return _y

    @property
    def func(self):
        """The model function handle"""
        return self._eval

    @property
    def name(self):
        """The model function name (a valid Python identifier)"""
        return self._eval.__name__

    @property
    def model_function_count(self):
        """The number of model functions"""
        return len(self._model_function_handle)
    
    @property
    def model_function_list(self):
        """The list of model functions"""
        return self._model_function_handle
    
    def _construct_arg_list(self, args, model_index):
        _arg_list = []
        for _arg_index in self._model_arg_indices[model_index]:
            _arg_list.append(args[_arg_index])
        return _arg_list

    def eval_underlying_model_function(self, x, args, model_index):
        #TODO documentation
        return self.model_function_list[model_index](x, *self._construct_arg_list(args, model_index))
    
    @property
    def data_indices(self):
        """The indices by which the data is spliced and distributed to the individual models"""
        return self._data_indices

class XYParametricModelException(XYContainerException):
    pass


class XYParametricModel(ParametricModelBaseMixin, XYContainer):
    def __init__(self, x_data, model_func, model_parameters):
        """
        Construct an :py:obj:`XYParametricModel` object:

        :param x_data: array containing the *x* values supporting the model
        :param model_func: handle of Python function (the model function)
        :param model_parameters: iterable of parameter values with which the model function should be initialized
        """
        #TODO update documentation
        # print "XYParametricModel.__init__(x_data=%r, model_func=%r, model_parameters=%r)" % (x_data, model_func, model_parameters)
        _y_data = model_func.func(x_data, *model_parameters)
        super(XYParametricModel, self).__init__(model_func, model_parameters, x_data, _y_data)
        self._data_indices = model_func.data_indices

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

    def eval_model_function(self, x=None, model_parameters=None, model_index=None):
        """
        Evaluate the model function.

        :param x: *x* values of the support points (if ``None``, the model *x* values are used)
        :type x: list or ``None``
        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :return: value(s) of the model function for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        #TODO update documentation
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        if model_index is None:
            return self._model_function_handle.func(_x, *_pars)
        else:
            return self._model_function_handle.eval_underlying_model_function(_x, _pars, model_index)

    def eval_model_function_derivative_by_parameters(self, x=None, x_indices=None, model_parameters=None, par_dx=None):
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
        #TODO update documentation
        if x is not None and x_indices is None:
            raise XYParametricModelException('When x is specified x_indices also has to be specified!')
        
        _x = x if x is not None else self.x
        _x_indices = x_indices if x_indices is not None else self._data_indices
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _pars = np.asarray(_pars)
        _par_dxs = par_dx if par_dx is not None else 1e-2 * (np.abs(_pars) + 1.0/(1.0+np.abs(_pars)))

        try:
            iter(_par_dxs)
            if len(_pars) != len(_par_dxs):
                raise XYParametricModelException('When providing an iterable of par_dx values it must have the same length as model_parameters!')
        except TypeError:
            _par_dxs = np.ones_like(_pars)*_par_dxs

        _derivatives = []
        for _par in _pars:
            _derivatives.append([])
        for _i, (_model_function, _par_indices) in enumerate(zip(self._model_function_handle.model_function_list, self._model_function_handle._model_arg_indices)):
            _x_splice = _x[_x_indices[_i]:_x_indices[_i + 1]]
            _par_sublist = []
            for _par_index in _par_indices:
                _par_sublist.append(_pars[_par_index])
            _par_sublist = np.array(_par_sublist)
            _skipped_pars = 0
            for _j, (_par_val, _par_dx) in enumerate(zip(_pars, _par_dxs)):
                #if a model function does not have a parameter, 
                #the derivative for that parameter is 0
                if _j not in _par_indices:
                    _derivatives[_j].append(np.zeros_like(_x_splice))
                    _skipped_pars += 1
                    continue
                def _chipped_func(par):
                    _chipped_pars = _par_sublist.copy()
                    _chipped_pars[_j - _skipped_pars] = par
                    return _model_function(_x_splice, *_chipped_pars)
                _derivatives[_j].append(derivative(_chipped_func, _par_val, dx=_par_dx))
        
        _flattened_derivatives = []
        for _derivative in _derivatives:
            _flattened_derivatives.append(np.append(*_derivative))
        return np.array(_flattened_derivatives)
        #_ret = []
        #for _par_idx, (_par_val, _par_dx) in enumerate(zip(_pars, _par_dxs)):
        #    def _chipped_func(par):
        #        _chipped_pars = _pars.copy()
        #        _chipped_pars[_par_idx] = par
        #        return self._model_function_handle.func(_x, *_chipped_pars)
        #
        #    _der_val = derivative(_chipped_func, _par_val, dx=_par_dx)
        #    _ret.append(_der_val)
        #return np.array(_ret)

    def eval_model_function_derivative_by_x(self, x=None, x_indices=None, model_parameters=None, dx=None):
        """
        Evaluate the derivative of the model function with respect to the independent variable (*x*).

        :param x: *x* values of the support points (if ``None``, the model *x* values are used)
        :type x: list or ``None``
        :param x_indices: indices for slicing *x* values for distributing them to the model functions.
            Has to be specified if x is specified.
        :type x: list or ``None``
        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :param dx: step size for numeric differentiation
        :type dx: float
        :return: value(s) of the model function derivative
        :rtype: :py:obj:`numpy.ndarray`
        """
        if x is not None and x_indices is None:
            raise XYParametricModelException('When x is specified x_indices also has to be specified!')
        
        _x = x if x is not None else self.x
        _x_indices = x_indices if x_indices is not None else self._data_indices
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _dxs = dx if dx is not None else 1e-2 * (np.abs(_x) + 1.0/(1.0+np.abs(_x)))
        try:
            iter(_dxs)
            if len(_x) != len(_dxs):
                raise XYParametricModelException('When providing an iterable of dx values it must have the same length as x!')
        except TypeError:
            _dxs = np.ones_like(_x)*_dxs

        _derivatives = []
        for _i, (_model_function, _par_indices) in enumerate(zip(self._model_function_handle.model_function_list, self._model_function_handle._model_arg_indices)):
            _par_sublist = []
            for _par_index in _par_indices:
                _par_sublist.append(_pars[_par_index])
            
            def _chipped_func(x):
                return _model_function(x, *_par_sublist)
            
            for _j in range(_x_indices[_i], _x_indices[_i + 1]):
                _derivatives.append(derivative(_chipped_func, _x[_j], dx=_dxs[_j]))
                            
        return np.array(_derivatives)
    