import abc
import inspect
import numpy as np
import re
import string


class FitException(Exception):
    pass


class FitBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    CONTAINER_TYPE = None
    MODEL_TYPE = None
    EXCEPTION_TYPE = FitException
    RESERVED_NODE_NAMES = None

    # -- private methods

    def _new_data_container(self, *args, **kwargs):
        return self.__class__.CONTAINER_TYPE(*args, **kwargs)

    def _new_parametric_model(self, *args, **kwargs):
        return self.__class__.MODEL_TYPE(*args, **kwargs)

    def _validate_model_function_for_fit_raise(self):
        # disallow using reserved keywords as model function arguments
        if not self.RESERVED_NODE_NAMES.isdisjoint(set(self._model_function.argspec.args)):
            _invalid_args = self.RESERVED_NODE_NAMES.intersection(set(self._model_function.argspec.args))
            raise self.__class__.EXCEPTION_TYPE(
                "The following names are reserved and cannot be used as model function arguments: %r"
                % (_invalid_args,))

    @staticmethod
    def _latexify_ascii(ascii_string):
        _lpn = string.replace(ascii_string, '_', r"\_")
        return r"{\tt %s}" % (_lpn,)

    # -- public properties

    # @abc.abstractproperty
    # def data(self): pass
    #
    # @abc.abstractproperty
    # def model(self): pass
    #
    # @abc.abstractproperty
    # def data_error(self): pass
    #
    # @abc.abstractproperty
    # def data_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def data_cov_mat_inverse(self): pass
    #
    # @abc.abstractproperty
    # def model_error(self): pass
    #
    # @abc.abstractproperty
    # def model_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def model_cov_mat_inverse(self): pass
    #
    # @abc.abstractproperty
    # def total_error(self): pass
    #
    # @abc.abstractproperty
    # def total_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def total_cov_mat_inverse(self): pass

    @abc.abstractproperty
    def parameter_values(self): pass

    @abc.abstractproperty
    def parameter_errors(self): pass

    @abc.abstractproperty
    def parameter_cov_mat(self): pass

    @abc.abstractproperty
    def parameter_name_value_dict(self): pass

    @abc.abstractproperty
    def cost_function_value(self): pass

    @property
    def data_size(self):
        return self._data_container.size

    @property
    def has_model_errors(self):
        return self._param_model.has_errors

    @property
    def has_data_errors(self):
        return self._data_container.has_errors

    @property
    def has_errors(self):
        return True if self.has_data_errors or self.has_model_errors else False

    # -- public methods

    @abc.abstractmethod
    def add_simple_error(self): pass

    @abc.abstractmethod
    def add_matrix_error(self): pass

    # @abc.abstractmethod
    # def do_fit(self): pass

    def get_model_function_string(self, with_par_names=True, with_par_values=False, format_as_latex=False):
        _func_name_string = self._model_function.name
        if format_as_latex:
            _filter = self._latexify_ascii
            _args_string_circumfix = r"\left(%s\right)"
        else:
            _filter = lambda x: x
            _args_string_circumfix = "(%s)"

        if with_par_values and with_par_values:
            _individual_arg_strings = ["%s=%g" % (_filter(_pn), _pv) for _pn, _pv in self.parameter_name_value_dict.iteritems()]
            _args_string = _args_string_circumfix % (", ".join(_individual_arg_strings))
        elif with_par_names:
            _individual_arg_strings = ["%s" % (_filter(_pn),) for _pn in self._fit_param_names]
            _args_string = _args_string_circumfix % (", ".join(_individual_arg_strings))
        elif with_par_values:
            _individual_arg_strings = ["%g" % (_pv,) for _pv in self.parameter_name_value_dict.values()]
            _args_string = _args_string_circumfix % (", ".join(_individual_arg_strings))
        else:
            _args_string = ""
        return "%s%s" % (_filter(_func_name_string), _args_string)

    def do_fit(self):
        self._fitter.do_fit()
        # update parameter formatters
        for _fpf, _pv, _pe in zip(self._fit_param_formatters, self.parameter_values, self.parameter_errors):
            _fpf.value = _pv
            _fpf.error = _pe

    def assign_model_function_expression(self, expression_format_string):
        self._model_func_formatter.expression_format_string = expression_format_string

    def assign_model_function_latex_expression(self, latex_expression_format_string):
        self._model_func_formatter.latex_expression_format_string = latex_expression_format_string

    def assign_parameter_latex_names(self, **par_latex_names_dict):
        for _pf in self._fit_param_formatters:
            _pln = par_latex_names_dict.get(_pf.name, None)
            if _pln is not None:
                _pf.latex_name = _pln