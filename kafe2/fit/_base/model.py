import abc
import inspect
from collections import OrderedDict

import numpy as np
import six

try:
    import sympy as sp
except ModuleNotFoundError:
    sp = None

from ...config import kc
from ..io.file import FileIOMixin
from ..util import function_library
from .format import ModelFunctionFormatter, ParameterFormatter, latexify_ascii

if six.PY2:
    from funcsigs import Parameter, Signature, signature
else:
    from inspect import Parameter, Signature, signature


__all__ = ["ParametricModelBaseMixin", "ModelFunctionBase"]


@six.add_metaclass(abc.ABCMeta)
class ModelFunctionBase(FileIOMixin, object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    model functions.

    In order to be used as a model function, a native Python function must be wrapped
    by an object whose class derives from this base class.
    There is a dedicated :py:class:`ModelFunction` specialization for each type of
    data container.

    This class provides the basic functionality used by all :py:class:`ModelFunction` objects.
    These use introspection (:py:mod:`inspect`) for determining the parameter structure of the
    model function and to ensure the function can be used as a model function (validation).

    """

    FORMATTER_TYPE = ModelFunctionFormatter

    def __init__(self, model_function=function_library.linear_model, independent_argcount=1):
        """
        Construct :py:class:`ModelFunction` object (a wrapper for a native Python function):

        :param model_function: function handle
        :param independent_argcount: The amount of independent variables for this model. The first n variables of the
                                      model function will be treated as independent variables and will not be fitted.
        :type independent_argcount: int
        """
        _custom_defaults = OrderedDict()
        self._name = None

        # determine library function from string specification
        if isinstance(model_function, str):
            self._model_function_handle = function_library.STRING_TO_FUNCTION.get(model_function, None)
            if self._model_function_handle is None and "->" in model_function:
                if sp is None:
                    raise ModuleNotFoundError("SymPy is not installed so it cannot be used to define model functions.")
                _symbol_string, _function_string = model_function.split("->")
                _latex_name = None
                if ":" in _symbol_string:
                    self._name, _symbol_string = _symbol_string.split(":")
                else:
                    self._name = "model"
                _latex_name = latexify_ascii(self._name)
                _symbols = list(sp.symbols(_symbol_string))
                for _i, _symbol_i in enumerate(_symbols):
                    if _i < independent_argcount:
                        continue
                    _symbol_i_str = str(_symbol_i)
                    if "=" in _symbol_i_str:
                        _symbol_i_str, _values_str = _symbol_i_str.split("=")
                        _symbol_i = sp.symbols(_symbol_i_str)
                        _custom_defaults[_symbol_i_str] = float(_values_str)
                        _symbols[_i] = _symbol_i
                _symbolic_function = sp.sympify(_function_string)
                self._model_function_handle = sp.lambdify(_symbols, _symbolic_function)
                _latex_string = sp.latex(_symbolic_function)
                _latex_string = _latex_string.replace(r"{", r"{{")
                _latex_string = _latex_string.replace(r"}", r"}}")
                for _symbol in _symbols:
                    _latex_string = _latex_string.replace(r"{{%s}}" % _symbol, r"{%s}" % _symbol)

                self._model_function_handle.latex_name = _latex_name
                self._model_function_handle.latex_expression_format_string = _latex_string
            if not self._model_function_handle:
                raise ValueError("Unknown model function: %s" % model_function)
            self._callable = self._model_function_handle

        # special handling of numpy vectorized functions
        elif isinstance(model_function, np.vectorize):
            self._callable = model_function
            self._model_function_handle = model_function.pyfunc

        # handle generic callables
        elif callable(model_function):
            self._model_function_handle = model_function if model_function else self._get_default()
            self._callable = self._model_function_handle

        # raise if not callable
        else:
            raise TypeError("Cannot use {} as model function: object not callable!".format(model_function))

        if self._name is None:
            self._name = self._model_function_handle.__name__

        assert int(independent_argcount) >= 0, "The number of independent parameters must be greater than 0"
        self._independent_argcount = int(independent_argcount)
        self._assign_model_function_signature_and_argcount(_custom_defaults)
        self._validate_model_function_raise()
        self._assign_function_formatter()
        self._source_code = None
        super(ModelFunctionBase, self).__init__()

    @classmethod
    def _get_base_class(cls):
        return ModelFunctionBase

    @classmethod
    def _get_object_type_name(cls):
        return "model_function"

    @classmethod
    def _get_default(cls):
        return function_library.linear_model

    def _assign_model_function_signature_and_argcount(self, custom_defaults={}):
        self._model_function_signature = signature(self._model_function_handle)
        self._model_function_argcount = self._model_function_handle.__code__.co_argcount
        # remove the amount of independent variables from the parameter count
        self._model_function_parcount = self._model_function_argcount - self._independent_argcount
        if custom_defaults:
            self.defaults = [custom_defaults.get(_p_name, _p_val) for _p_name, _p_val in self.defaults_dict.items()]

    def _validate_model_function_raise(self):
        # evaluate general model function requirements
        for _par in self.signature.parameters.values():
            if _par.kind == _par.VAR_POSITIONAL:
                raise ValueError(
                    "Model function '{}' with variable number of positional "
                    "arguments (*{}) is not supported".format(
                        self._model_function_handle.__name__,
                        _par.name,
                    )
                )
            if _par.kind == _par.VAR_KEYWORD:
                raise ValueError(
                    "Model function '{}' with variable number of keyword "
                    "arguments (**{}) is not supported".format(
                        self._model_function_handle.__name__,
                        _par.name,
                    )
                )
        # require at least one parameter to fit
        if self._model_function_parcount < 1:
            raise ValueError(
                "Model function {0!r} needs at least one parameter besides the "  # noqa: F523 (unused argument in format string)
                "first {0!s} independent variable(s)!".format(self._model_function_handle, self._independent_argcount)
            )

    def _get_argument_formatters(self):
        return [ParameterFormatter(_arg_name) for _arg_name in self.signature.parameters.keys()]

    def _assign_function_formatter(self):
        self._formatter = self.__class__.FORMATTER_TYPE(self.name, arg_formatters=self._get_argument_formatters())
        try:
            _latex_name = self._model_function_handle.latex_name
            if _latex_name is not None:
                self._formatter.latex_name = _latex_name
        except AttributeError:
            pass
        try:
            self._formatter.expression_format_string = self._model_function_handle.expression_format_string
        except AttributeError:
            pass
        try:
            self._formatter.latex_expression_format_string = self._model_function_handle.latex_expression_format_string
        except AttributeError:
            pass

    def __call__(self, *args, **kwargs):
        return self._callable(*args, **kwargs)

    @property
    def name(self):
        """The model function name (a valid Python identifier)"""
        return self._name

    @property
    def func(self):
        """The underlying model function handle"""
        return self._model_function_handle

    @property
    def signature(self):
        """The model function argument specification, as returned by :py:meth:`inspect.signature`"""
        return self._model_function_signature

    @property
    def argcount(self):
        """The number of arguments the model function accepts.
        (including any independent variables which are not parameters)"""
        return self._model_function_argcount

    @property
    def parcount(self):
        """The number of fitting parameters in the model function."""
        return self._model_function_parcount

    @property
    def x_name(self):
        """The name of the independent variable. ``None`` for 0 independent variables."""
        _pars = list(self.signature.parameters.keys())
        return _pars[0 : self._independent_argcount]

    @property
    def parameter_names(self):
        """The names of the parameters."""
        return list(self.signature.parameters.keys())[self._independent_argcount :]

    @property
    def formatter(self):
        """The :py:obj:`ModelFunctionFormatter`-derived object for this function"""
        return self._formatter

    @property
    def defaults(self):
        """The default values for model function parameters as a list"""
        return list(self.defaults_dict.values())

    @defaults.setter
    def defaults(self, new_defaults):
        if self.parcount != len(new_defaults):  # first arg is independent variable, but not a parameter
            raise ValueError("Expected %s parameter defaults (1 per parameter), but received %s" % (self.parcount, len(new_defaults)))

        # pad defaults with empties for 'x'
        new_defaults = [Parameter.empty] * (self.argcount - self.parcount) + new_defaults

        # set new signature
        self._model_function_signature = Signature(
            parameters=[
                Parameter(
                    name=_par.name,
                    kind=_par.kind,
                    default=_par_default,
                )
                for _par, _par_default in zip(self.signature.parameters.values(), new_defaults)
            ]
        )

    @property
    def defaults_dict(self):
        """The default values for model function parameters as a dict"""
        _defaults_dict = OrderedDict()
        _x_name = self.x_name  # list of strings

        for _par in self.signature.parameters.values():
            # skip independent variable parameter
            if _x_name is not None and _par.name in _x_name:
                continue
            if _par.default == _par.empty:
                _defaults_dict[_par.name] = kc("core", "default_initial_parameter_value")
            else:
                _defaults_dict[_par.name] = _par.default

        return _defaults_dict

    @property
    def source_code(self):
        if self._source_code is None:
            return inspect.getsource(self.func)
        return self._source_code


class ParametricModelBaseMixin(object):
    """
    A "mixin" class for representing a parametric model.
    Inheriting from this class in addition to a data container class
    additionally stores a Python function handle referring to the
    model function. The argument structure of this function must
    be compatible with the data container type and it must return
    a numpy array of the same shape as the
    :py:meth:`~kafe2.fit._base.DataContainerBase.data` property of
    the data container.

    This mixin class introduces an additional :py:func:`parameters` property for
    the object, which can be used to obtain and set the values of the parameter

    Derived classes should inherit from :py:class:`ParametricModelBaseMixin` and the
    relevant data container (in that order).
    """

    MODEL_FUNCTION_TYPE = ModelFunctionBase

    def __init__(self, model_func, model_parameters, *args, **kwargs):
        """
        Mixin constructor: sets and initialized the model function.

        :param model_func: handle of Python function (the model function)
        :param model_parameters: iterable of parameter values with which the model function should be initialized
        """
        if isinstance(model_func, self.MODEL_FUNCTION_TYPE):
            self._model_function_object = model_func
        else:
            self._model_function_object = self.MODEL_FUNCTION_TYPE(model_func)
        self.parameters = model_parameters
        super(ParametricModelBaseMixin, self).__init__(*args, **kwargs)
        self.label = "$%s$" % self._model_function_object.formatter.latex_name

    @classmethod
    def _get_base_class(cls):
        return ParametricModelBaseMixin

    @classmethod
    def _get_object_type_name(cls):
        return "parametric_model"

    @property
    def ndf(self):
        return self.size - self._model_function_object.parcount

    @property
    def parameters(self):
        """Model parameter values"""
        return self._model_parameters

    @parameters.setter
    def parameters(self, parameters):
        """Setter for parameter values"""
        self._model_parameters = parameters

        # flag: recalculate the model values next time they are requested
        self._pm_calculation_stale = True
        self._clear_total_error_cache()  # declared in the container class
