import numpy as np

from types import FunctionType

from .._base import ParametricModelBaseMixin, ModelFunctionBase, ModelFunctionException, ModelParameterFormatter
from .container import UnbinnedContainer, UnbinnedContainerException
from ..xy.format import XYModelFunctionFormatter
from ..util import function_library


class UnbinnedModelPDFException(ModelFunctionException):
    pass


class UnbinnedModelPDF(ModelFunctionBase):
    EXCEPTION_TYPE = UnbinnedModelPDFException
    FORMATTER_TYPE = XYModelFunctionFormatter  # TODO: check for more duplicates and use the xy Formatter where possible

    def __init__(self, model_density_function=None):
        self._x_name = 'x'
        if isinstance(model_density_function, FunctionType):
            _pdf = model_density_function
        elif model_density_function.lower() == "gaussian":
            _pdf = function_library.normal_distribution_pdf
        else:
            raise UnbinnedModelPDFException("Unknown value '%s' for 'model_density_function':"
                                            "It must be a function or one of ('gaussian')!")
        super(UnbinnedModelPDF, self).__init__(model_function=_pdf)

    def _validate_model_function_raise(self):
        # require pdf model function arguments to include 'x'
        if self.x_name not in self.signature.parameters:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' must have independent variable '%s' among its arguments!"
                % (self.func, self.x_name))

        # require 'hist' model functions to have more than two arguments
        if self.argcount < 2:
            raise self.__class__.EXCEPTION_TYPE(
                "Model function '%r' needs at least one parameter beside independent variable '%s'!"
                % (self.func, self.x_name))

        # evaluate general model function requirements
        super(UnbinnedModelPDF, self)._validate_model_function_raise()

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


class UnbinnedParametricModelException(UnbinnedContainerException):
    pass


class UnbinnedParametricModel(ParametricModelBaseMixin, UnbinnedContainer):
    def __init__(self, data, model_density_function=function_library.normal_distribution_pdf,
                 model_parameters=[1.0, 1.0]):

        self.support = np.array(data)

        super(UnbinnedParametricModel, self).__init__(
            # this gets passed to ParametricModelBaseMixin.__init__
            model_func=model_density_function,
            model_parameters=model_parameters,
            # this gets passed to UnbinnedContainer.__init__
            data=model_density_function(
                self.support, *model_parameters)
        )

    # -- private methods

    def _recalculate(self):
        # use parent class setter for 'data'
        UnbinnedContainer.data.fset(self, self.eval_model_function())
        self._pm_calculation_stale = False

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, model_support):
        self._support = model_support
        self._pm_calculation_stale = True

    @property
    def data(self):
        if self._pm_calculation_stale:
            self._recalculate()
        return super(UnbinnedParametricModel, self).data

    @data.setter
    def data(self):
        raise UnbinnedParametricModelException("Parametric model data cannot be set!")

    def eval_model_function(self, support=None, model_parameters=None):
        """
        Evaluate the model function.

        :param support: *x* values of the support points (if ``None``, the model *support* values are used)
        :type support: list or ``None``
        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :return: value(s) of the model function for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _x = support if support is not None else self.support
        _pars = model_parameters if model_parameters is not None else self.parameters
        return self._model_function_object(_x, *_pars)
