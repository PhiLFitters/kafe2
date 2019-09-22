import inspect
import numpy as np
import six

from scipy.misc import derivative

from .._base import ParametricModelBaseMixin, ModelFunctionBase, ModelFunctionException, ModelParameterFormatter
from .container import XYMultiContainer, XYMultiContainerException
from .format import XYMultiModelFunctionFormatter
from ..xy.model import XYModelFunction
from collections import OrderedDict


if six.PY2:
    from funcsigs import signature, Signature, Parameter
else:
    from inspect import signature, Signature, Parameter


__all__ = ["XYMultiParametricModel", "XYMultiModelFunction"]


class XYMultiModelFunctionException(ModelFunctionException):
    pass

class XYMultiModelFunction(ModelFunctionBase):
    EXCEPTION_TYPE = XYMultiModelFunctionException
    FORMATTER_TYPE = XYMultiModelFunctionFormatter

    def __init__(self, model_function_list, data_indices):
        """
        Construct :py:class:`XYMultiModelFunction` object (a wrapper for an iterable of native Python function):

        :param model_function: function handles
        :type model_function: native Python function or iterable thereof
        :param data_indices: the indices at which iterables of *x* values should be split between the functions, 
               length must be length of model_funtion + 1
        :type iterable of int
        """
        #TODO default model function
        try: 
            iter(model_function_list)
        except:
            model_function_list = [model_function_list]
        if len(model_function_list) != len(data_indices) - 1:
            raise XYMultiModelFunctionException(
                "Received %s models but %s datasets, must be equal!"
                % (len(model_function_list), len(data_indices) - 1)
            )
        self._x_name = 'x'
        self._data_indices = data_indices
        self._singular_model_functions = [XYModelFunction(_model_function) 
                                          for _model_function in model_function_list]
        super(XYMultiModelFunction, self).__init__(model_function=self._eval)

    #in this case also assigns self._model_arg_indices
    def _assign_model_function_signature_and_argcount(self):

        #stores the location of singular model function arguments in the
        #combined multi model function argument list
        self._model_arg_indices = []

        #TODO implement
        _varargs = None
        _keywords = None

        _args_with_defaults = OrderedDict()  # the combined args
        for _model_function in self._singular_model_functions:
            _args = list(_model_function.signature.parameters)[1:]  # index 0 reserved for x
            _defaults = [_par.default for _par in _model_function.signature.parameters.values()][1:]  # index 0 reserved for x

            # add singular model function arg names and defaults to the combined lists
            for _arg_name, _default_value in zip(_args, _defaults):

                # replace dummy default values
                if _arg_name not in _args_with_defaults or _args_with_defaults[_arg_name] == Parameter.empty:
                    _args_with_defaults[_arg_name] = _default_value

                # raise on conflicting defaults
                elif _default_value != Parameter.empty and _args_with_defaults[_arg_name] != _default_value:
                    raise XYMultiModelFunctionException(
                        "Model functions have conflicting defaults for parameter %s: %s <-> %s" % 
                        (_arg_name, _args_with_defaults[_arg_name], _default_value))

            # save the locations of singular args in the combined list
            _keys = list(_args_with_defaults)
            self._model_arg_indices.append([_keys.index(_arg_name) for _arg_name in _args])

        _combined_args = list(_args_with_defaults)
        _combined_defaults = []
        for _arg_name in _combined_args:
            _default = _args_with_defaults.get(_arg_name)
            _combined_defaults.append(_default if _default != Parameter.empty else 1.0)

        _combined_args.insert(0, self._x_name)
        _combined_defaults.insert(0, Parameter.empty)

        self._model_function_signature = Signature(
            parameters=[
                Parameter(
                    name=_arg,
                    default=_default,
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                )
                for _arg, _default in zip(_combined_args, _combined_defaults)
            ]
        )

        self._model_function_argcount = len(_combined_args)
        self._model_function_parcount = self._model_function_argcount - 1

    def _validate_model_function_raise(self):
        for _model_function in self.singular_model_functions:
            _model_function._validate_model_function_raise()

    def _get_parameter_formatters(self):
        _start_at_arg = 1
        return [ModelParameterFormatter(name=_pn, value=_pv, error=None)
                for _pn, _pv in zip(list(self.signature.parameters)[_start_at_arg:], self.argvals[_start_at_arg:])]

    def _assign_function_formatter(self):
        _singular_formatters=[_model_function.formatter 
            for _model_function in self._singular_model_functions]
        _parameter_formatters = self._get_parameter_formatters()
        for _i in range(self.model_function_count):
            _singular_formatters[_i].arg_formatters = self._construct_arg_list(
                _parameter_formatters, _i)
        self._formatter=self.__class__.FORMATTER_TYPE(
            singular_formatters=_singular_formatters,
            arg_formatters=_parameter_formatters
        )

    def _construct_arg_list(self, args, model_index):
        _arg_list = []
        for _arg_index in self._model_arg_indices[model_index]:
            _arg_list.append(args[_arg_index])
        return _arg_list

    def _eval(self, x, *args, **kwargs):
        _y = np.empty(x.size)
        _arg_lists = []
        _x_indices = kwargs.get("x_indices")
        _x_indices = _x_indices if _x_indices is not None else self.data_indices
        for _i, _model_function in enumerate(self._singular_model_functions):
            _x = x[_x_indices[_i]:_x_indices[_i + 1]]
            _arg_list = self._construct_arg_list(args, _i)
            _y[self.data_indices[_i]:self.data_indices[_i + 1]] = _model_function(_x, *_arg_list)
        return _y

    @property
    def x_name(self):
        """the name of the independent variable"""
        return self._x_name

    @property
    def model_function_count(self):
        """The number of model functions"""
        return len(self._singular_model_functions)
    
    @property
    def singular_model_functions(self):
        """The list of singular model functions"""
        return self._singular_model_functions
    
    @property
    def data_indices(self):
        """The indices by which the data is spliced and distributed to the individual models"""
        return self._data_indices

    @data_indices.setter
    def data_indices(self, new_data_indices):
        self._data_indices = new_data_indices

    def eval_underlying_model_function(self, x, args, model_index):
        """
        evaluate one of the underlying model functions
        :param x: the *x* values passed on the model function
        :type x: float or :py:obj:`numpy.ndarray` of float
        :param args: an iterable of all parameters used between all model functions
        :type args: iterable of float
        :param model_index: the index of the model function to be evaluated
        :type model_index: int
        :return: y-values of the specified model function at the given *x* values
        :rtype: float of :py:obj:`numpy.ndarray` of float
        """
        return self.singular_model_functions[model_index](x, *self._construct_arg_list(args, model_index))

    def get_argument_formatters(self, model_index):
        """
        return the argument formatters for a single underlying model function
        :param model_index: the index of the model function whose parameter formatters to return
        :type model_index: int
        :return: the list of argument formatters corresponding to the specified model function
        :rtype: :py:obj:`Formatter`
        """
        return self._construct_arg_list(self.argument_formatters, model_index)

    def assign_model_function_expression(self, expression_format_string, model_index):
        """Assign a plain-text-formatted expression string to the model function."""
        self._formatter._singular_formatters[model_index].expression_format_string = expression_format_string

    def assign_model_function_latex_expression(self, latex_expression_format_string, model_index):
        """Assign a LaTeX-formatted expression string to the model function."""
        self._formatter._singular_formatters[model_index].latex_expression_format_string = latex_expression_format_string
        
class XYMultiParametricModelException(XYMultiContainerException):
    pass


class XYMultiParametricModel(ParametricModelBaseMixin, XYMultiContainer):
    def __init__(self, x_data, model_func, model_parameters):
        """
        Construct an :py:obj:`XYMultiParametricModel` object:

        :param x_data: array containing the *x* values supporting the models
        :type x_data: :py:obj:`numpy.ndarray`
        :param model_func: the model functions for the multi model encapsulated in a kafe2 model function object
        :type model_func: :py:obj:`XYMultiModelFunction`
        :param model_parameters: iterable of parameter values with which the model functions should be initialized
        :type model_parameters: iterable of float
        """
        # print "XYMultiParametricModel.__init__(x_data=%r, model_func=%r, model_parameters=%r)" % (x_data, model_func, model_parameters)
        _x_data_array = np.asarray(x_data)
        _xy_data = np.empty([2, _x_data_array.size])
        _xy_data[0] = _x_data_array
        _xy_data[1] = model_func(_x_data_array, *model_parameters)
        super(XYMultiParametricModel, self).__init__(model_func, model_parameters, _xy_data)

    # -- private methods

    def _recalculate(self):
        # use parent class setter for 'y'
        XYMultiContainer.y.fset(self, self.eval_model_function())
        self._pm_calculation_stale = False


    # -- public properties

    @property
    def data(self):
        """model predictions (one-dimensional :py:obj:`numpy.ndarray`)"""
        if self._pm_calculation_stale:
            self._recalculate()
        return super(XYMultiParametricModel, self).data

    @data.setter
    def data(self, new_data):
        raise XYMultiParametricModelException("Parametric model data cannot be set!")

    @property
    def x(self):
        """model *x* support values"""
        return super(XYMultiParametricModel, self).x

    @x.setter
    def x(self, new_x):
        # resetting 'x' -> must reset entire data array
        if len(new_x) != len(self.x):
            raise XYMultiParametricModelException("When setting a new x for an XYMultiParametricModel, the length must stay the same!",
                                                "To change the length, simultaneously provide a new set of data indices via set_x!")
        self._xy_data = np.zeros((2, len(new_x)))
        self._xy_data[0] = new_x
        self._pm_calculation_stale = True
        self._clear_total_error_cache()

    def set_x(self, new_x, new_data_indices):
        if len(self._data_indices) != len(new_data_indices):
            raise XYMultiParametricModelException("When assigning new data indices the length cannot change!")
        self.data_indices = new_data_indices
        self._xy_data = np.zeros((2, len(new_x)))
        self._xy_data[0] = new_x
        self._pm_calculation_stale = True
        self._clear_total_error_cache()

    @property
    def y(self):
        """model *y* values"""
        if self._pm_calculation_stale:
            self._recalculate()
        return super(XYMultiParametricModel, self).y

    @y.setter
    def y(self, new_y):
        raise XYMultiParametricModelException("Parametric model data cannot be set!")
    
    @property
    def data_indices(self):
        return self._model_function_object.data_indices

    @data_indices.setter
    def data_indices(self, new_data_indices):
        self._model_function_object.data_indices = new_data_indices
    
    # -- public methods

    def eval_model_function(self, x=None, model_parameters=None, model_index=None):
        """
        Evaluate all model functions or just one of them.

        :param x: *x* values of the support points (if ``None``, the model *x* values are used)
        :type x: list or ``None``
        :param model_parameters: values of the model parameters used between all model functions
                                 (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :param model_index: the index of the model function to be evaluated (if ``None``, the *x* 
                            values are distributed according to `data_indices`)
        :type model_index: int
        :return: value(s) of the model function for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _x = x if x is not None else self.x
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        if model_index is None:
            return self._model_function_object(_x, *_pars)
        else:
            return self._model_function_object.eval_underlying_model_function(_x, _pars, model_index)

    def eval_model_function_derivative_by_parameters(self, x=None, x_indices=None, model_parameters=None, par_dx=None):
        """
        Evaluate the derivative of the model function with respect to the model parameters.

        :param x: *x* values of the support points (if ``None``, the model *x* values are used)
        :type x: list or ``None``
        :param x_indices: the indices at which the *x* values should be split when distributing
                          them to the model functions (if ``None``, `data_indices` is used)
        :type x_indices: iterable of int
        :param model_parameters: values of the model parameters between all model functions
                                 (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :param par_dx: step size for numeric differentiation for the parameters
        :type par_dx: float or `numpy.ndarray` of float
        :return: value(s) of the model function derivative for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _x = x if x is not None else self.x
        _x_indices = x_indices if x_indices is not None else self.data_indices
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _pars = np.asarray(_pars)
        _par_dxs = par_dx if par_dx is not None else 1e-2 * (np.abs(_pars) + 1.0/(1.0+np.abs(_pars)))

        try:
            iter(_par_dxs)
            if len(_pars) != len(_par_dxs):
                raise XYMultiParametricModelException('When providing an iterable of par_dx values it must have the same length as model_parameters!')
        except TypeError:
            _par_dxs = np.ones_like(_pars)*_par_dxs

        _derivatives = []
        for _par in _pars:
            _derivatives.append([])
        for _i, (_model_function, _par_indices) in enumerate(zip(
                self._model_function_object.singular_model_functions, 
                self._model_function_object._model_arg_indices)):
            _x_splice = _x[_x_indices[_i]:_x_indices[_i + 1]]
            _par_sublist = []
            for _par_index in _par_indices:
                _par_sublist.append(_pars[_par_index])
            _par_sublist = np.array(_par_sublist)
            for _j, (_par_val, _par_dx) in enumerate(zip(_pars, _par_dxs)):
                #if a model function does not have a parameter, 
                #the derivative for that parameter is 0
                if _j not in _par_indices:
                    _derivatives[_j].append(np.zeros_like(_x_splice))
                else:
                    _par_sublist_index = _par_indices.index(_j)
                    def _chipped_func(par):
                        _chipped_pars = _par_sublist.copy()
                        _chipped_pars[_par_sublist_index] = par
                        return _model_function(_x_splice, *_chipped_pars)
                    _derivatives[_j].append(derivative(_chipped_func, _par_val, dx=_par_dx))

        _flattened_derivatives = []
        for _derivative in _derivatives:
            _flattened_derivatives.append(np.append(np.array([]), _derivative))
        return np.array(_flattened_derivatives)

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
            raise XYMultiParametricModelException('When x is specified x_indices also has to be specified!')
        
        _x = x if x is not None else self.x
        _x_indices = x_indices if x_indices is not None else self.data_indices
        _pars = model_parameters if model_parameters is not None else self._model_parameters
        _dxs = dx if dx is not None else 1e-2 * (np.abs(_x) + 1.0/(1.0+np.abs(_x)))
        try:
            iter(_dxs)
            if len(_x) != len(_dxs):
                raise XYMultiParametricModelException('When providing an iterable of dx values it must have the same length as x!')
        except TypeError:
            _dxs = np.ones_like(_x)*_dxs

        _derivatives = []
        for _i, (_model_function, _par_indices) in enumerate(zip(
                self._model_function_object.singular_model_functions, 
                self._model_function_object._model_arg_indices)):
            _par_sublist = []
            for _par_index in _par_indices:
                _par_sublist.append(_pars[_par_index])
            
            def _chipped_func(x):
                return _model_function(x, *_par_sublist)

            for _j in range(_x_indices[_i], _x_indices[_i + 1]):
                _derivatives.append(derivative(_chipped_func, _x[_j], dx=_dxs[_j]))
                            
        return np.array(_derivatives)
    