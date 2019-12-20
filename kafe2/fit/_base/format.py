import re
import string

import numpy as np
from kafe2.fit.io.file import FileIOMixin


__all__ = ["ModelParameterFormatter", "ModelFunctionFormatter", "CostFunctionFormatter", "FormatterException"]


# -- formatters for model parameters and model functions

class FormatterException(Exception):
    pass


class ModelParameterFormatter(FileIOMixin, object):
    """
    :py:obj:`Formatter` class for model parameter objects.

    These objects store the relevant information for constructing
    plain-text and/or LaTeX string representations of model function parameters.

    For this, :py:obj:`ModelParameterFormatter` objects store the parameter name, formatted as a plain-text/LaTeX
    string, its value (a ``float``) and its error (a ``float`` for symmetric, a tuple of ``floats`` for
    asymmetric errors).

    The formatted string is obtained by calling the :py:meth:`~ModelParameterFormatter.get_formatted` method.
    """
    def __init__(self, name, value=None, error=None, asymmetric_error=None, latex_name=None):
        """

        Construct a :py:obj:`Formatter` for a model function:

        :param name:
        :param latex_name: a LaTeX-formatted string indicating the function name
        :param arg_formatters: list of :py:obj:`ModelParameterFormatter`-derived objects,
                               formatters for function arguments
        :param expression_string:  a plain-text-formatted string indicating the function expression
        :param latex_expression_string:  a LaTeX-formatted string indicating the function expression

        :param name: a plain-text-formatted string indicating the parameter name
        :param value: the parameter value (``float``)
        :param error: the parameter error: ``float`` (tuple of 2 ``floats``) for symmetric (asymmetric) error
        :param latex_name: a LaTeX-formatted string indicating the parameter name
        """
        self.value = value

        self.error = error
        self.asymmetric_error = asymmetric_error

        self._name = name
        self.latex_name = latex_name  # latex_name setter requires self._name to be set beforehand

        self._fixed = False
        super(ModelParameterFormatter, self).__init__()

    @classmethod
    def _get_base_class(cls):
        return ModelParameterFormatter

    @classmethod
    def _get_object_type_name(cls):
        return 'model_parameter_formatter'


    @staticmethod
    def _latexify_ascii(ascii_string):
        _lpn = ascii_string.replace('_', r"\_")
        return r"{\tt %s}" % (_lpn,)

    @property
    def name(self):
        """a plain-text-formatted string indicating the parameter name"""
        return self._name

    @property
    def latex_name(self):
        """a LaTeX-formatted string indicating the parameter name"""
        return self._latex_name

    @latex_name.setter
    def latex_name(self, new_latex_name):
        """Set the LaTeX-formatted string indicating the parameter name

        :param new_latex_name: LaTeX-formatted string
        :type new_latex_name: str
        """
        # TODO: validate
        if new_latex_name is None:
            self._latex_name = self._latexify_ascii(self.name)
        elif new_latex_name.startswith('{') and new_latex_name.endswith('}'):
            self._latex_name = new_latex_name
        else:
            self._latex_name = '{' + new_latex_name + '}'

    @property
    def value(self):
        """the parameter value"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def error(self):
        """the parameter error (``float``/tuple of 2 ``floats``)"""
        return self._error

    @error.setter
    def error(self, err_spec):
        self._error = err_spec

    @property
    def error_rel(self):
        """the relative parameter error (``float``/tuple of 2 ``floats``)"""
        if self._error is None:
            return None
        else:
            return self._error/self._value

    @property
    def asymmetric_error(self):
        return self._asymmetric_error

    @asymmetric_error.setter
    def asymmetric_error(self, err_spec):
        self._asymmetric_error = err_spec

    @property
    def error_up(self):
        """the "up" error (only for asymmetric errors)"""
        if self.asymmetric_error is None:
            return None
        return self.asymmetric_error[1]

    @property
    def error_down(self):
        """the "down" error (only for asymmetric errors)"""
        if self.asymmetric_error is None:
            return None
        return self.asymmetric_error[0]

    @property
    def fixed(self):
        """if the parameter has been fixed by the user."""
        return self._fixed

    @fixed.setter
    def fixed(self, fixed):
        self._fixed = fixed

    def get_formatted(self, with_name=False, with_value=True, with_errors=True, n_significant_digits=2,
                      round_value_to_error=True, asymmetric_error=False, format_as_latex=False):
        """
        Get a formatted string representing this model parameter.

        :param with_name:  if ``True``, output will include the parameter name
        :type with_name: bool
        :param with_value: if ``True``, output will include the parameter value
        :type with_value: bool
        :param with_errors: if ``True``, output will include the parameter error/errors
        :type with_errors: bool
        :param n_significant_digits: number of significant digits for rounding
        :type n_significant_digits: int
        :param round_value_to_error: if ``True``, the parameter value will be rounded to the same precision as the error
        :type round_value_to_error: bool
        :param asymmetric_error: if ``True``, use two different errors for up/down directions
        :type asymmetric_error: bool
        :param format_as_latex: if ``True``, the returned string will be formatted using LaTeX syntax
        :type format_as_latex: bool
        :return: the string representation of the parameter
        :rtype: str
        """
        _display_string = ""
        if with_name:
            if format_as_latex:
                _display_string += "$%s$" % (self.latex_name,)
            else:
                _display_string += "%s" % (self.name,)

        if with_value:
            if with_name:
                _display_string += " = "

            # fallback to rounding to 10^(-1) if value is zero
            _log_abs_value = -1
            if self._value:
                _log_abs_value = np.log(np.abs(self._value))

            if not with_errors or (not asymmetric_error and self.error is None) or \
                    (asymmetric_error and self.asymmetric_error is None):
                _sig = int(-np.floor(_log_abs_value / np.log(10))) + n_significant_digits - 1
                _display_val = round(self._value, _sig)
                if format_as_latex:
                    _display_string += "$%g$" % (_display_val,)
                else:
                    _display_string += "%g" % (_display_val,)
            elif self.fixed:
                if format_as_latex:
                    _display_string += r"$%g$ (fixed)" % self._value
                else:
                    _display_string += "%g (fixed)" % self._value
            else:
                if asymmetric_error:
                    _min_err = min(abs(self.error_up), abs(self.error_down))
                else:
                    _min_err = self.error
                # fallback to rounding to 10^(-1) if error is zero
                if not _min_err or np.isnan(_min_err):
                    _min_err = 1e-1
                if round_value_to_error:
                    _sig = int(-np.floor(np.log(_min_err)/np.log(10))) + n_significant_digits - 1
                else:
                    _sig = int(-np.floor(_log_abs_value / np.log(10))) + n_significant_digits - 1

                _display_val = round(self._value, _sig)
                if asymmetric_error:
                    _display_err_dn = round(self.error_down, _sig)
                    _display_err_up = round(self.error_up, _sig)
                    if format_as_latex:
                        _display_string += "${%g}^{%+g}_{%g}$" % (_display_val, _display_err_up, _display_err_dn)
                    else:
                        _display_string += "%g + %g (up) - %g (down)" % (_display_val, _display_err_up,
                                                                         abs(_display_err_dn))
                else:
                    _display_err = round(self.error, _sig)
                    if format_as_latex:
                        _display_string += r"$%g \pm %g$" % (_display_val, _display_err)
                    else:
                        _display_string += "%g +/- %g" % (_display_val, _display_err)

            # replace scientific notation with power of ten (LaTeX only)
            if format_as_latex:
                _display_string = re.sub(r'(-?\d*\.?\d+?)0*e\+?(-?[0-9]*[1-9]?)',
                                         r'\1\\times10^{\2}', _display_string)
        return _display_string


class ModelFunctionFormatter(FileIOMixin, object):
    """
    Base class for model function :py:obj:`Formatter` objects. Requires further specialization for
    each type of model function.

    Objects derived from :py:class:`ModelFunctionFormatter` store information relevant for constructing
    plain-text and/or LaTeX string representations of model functions.

    For this, :py:obj:`ModelFunctionFormatter` objects store the function name, formatted as a plain-text/LaTeX
    string, as well as a list of references to :py:obj:`ModelParameterFormatter` objects which contain information
    on how to format the model function arguments.

    Optionally, plain-text/LaTeX expression strings can be provided. These are strings representing the model
    function expression (i.e. mathematical formula).

    The formatted string is obtained by calling the :py:meth:`~ModelFunctionFormatter.get_formatted` method.
    """
    DEFAULT_EXPRESSION_STRING = "<not_specified>"
    DEFAULT_LATEX_EXPRESSION_STRING = r"\langle{\it not\,\,specified}\rangle"

    def __init__(self, name, latex_name=None, arg_formatters=None, expression_string=None, latex_expression_string=None):
        """
        Construct a :py:obj:`Formatter` for a model function:

        :param name: a plain-text-formatted string indicating the function name
        :param latex_name: a LaTeX-formatted string indicating the function name
        :param arg_formatters: list of :py:obj:`ModelParameterFormatter`-derived objects,
                               formatters for function arguments
        :param expression_string:  a plain-text-formatted string indicating the function expression
        :param latex_expression_string:  a LaTeX-formatted string indicating the function expression
        """
        #TODO should name be allowed to be None?
        self._name = name
        self._arg_formatters = arg_formatters
        self.expression_format_string = expression_string
        self.latex_expression_format_string = latex_expression_string

        self._latex_name = latex_name
        if self._latex_name is None:
            self._latex_name = self._latexify_ascii(self._name)

        self._description = None
        super(ModelFunctionFormatter, self).__init__()

    @classmethod
    def _get_base_class(cls):
        return ModelFunctionFormatter

    @classmethod
    def _get_object_type_name(cls):
        return 'model_function_formatter'

    @staticmethod
    def _latexify_ascii(ascii_string):
        _lpn = ascii_string.replace('_', r"\_")
        return r"{\tt %s}" % (_lpn,)

    def _get_format_kwargs(self, format_as_latex=False):
        if format_as_latex:
            _par_name_string_dict = {_af.name: _af.latex_name for _af in self._arg_formatters}
        else:
            _par_name_string_dict = {_af.name: _af.name for _af in self._arg_formatters}
        return _par_name_string_dict

    def _get_formatted_name(self, format_as_latex=False):
        if format_as_latex:
            return self._latex_name
        else:
            return self._name

    def _get_formatted_args(self, with_par_values=True, n_significant_digits=2, format_as_latex=False):
        if format_as_latex:
            _par_name_strings = [_af.latex_name for _af in self._arg_formatters]
        else:
            _par_name_strings = [_af.name for _af in self._arg_formatters]

        if with_par_values:
            _par_val_strings = []
            for _af in self._arg_formatters:
                _par_val_strings.append(_af.get_formatted(with_name=False,
                                                          with_value=True,
                                                          with_errors=False,
                                                          n_significant_digits=n_significant_digits,
                                                          round_value_to_error=False,
                                                          format_as_latex=format_as_latex))
            return ["%s=%s" % (_pn, _pv) for _pn, _pv in zip(_par_name_strings, _par_val_strings)]
        return ["%s" % (_pn,) for _pn in _par_name_strings]

    def _get_formatted_expression(self, format_as_latex=False):
        # TODO: nicer way then always use two curly braces for latex expression?
        # Currently when creating a latex expression to curly braces are needed. E.g. \\frac{{1}}{{x}} for 1/x
        # If only one curly brace is used kafe will lookup the associated param
        _kwargs = self._get_format_kwargs(format_as_latex=format_as_latex)
        if format_as_latex and self._latex_expr_string is not None:
            _par_expr_string = self._latex_expr_string.format(*[_af.latex_name for _af in self._arg_formatters], **_kwargs)
        elif not format_as_latex and self._expr_string is not None:
            _par_expr_string = self._expr_string.format(*[_af.name for _af in self._arg_formatters], **_kwargs)
        elif format_as_latex and self._latex_expr_string is None:
            _par_expr_string = self.DEFAULT_LATEX_EXPRESSION_STRING
        else:
            _par_expr_string = self.DEFAULT_EXPRESSION_STRING
        return _par_expr_string

    @property
    def expression_format_string(self):
        """a plain-text-formatted expression for the function"""
        return self._expr_string

    @expression_format_string.setter
    def expression_format_string(self, expression_format_string):
        self._expr_string = expression_format_string
        try:
            self._get_formatted_expression(format_as_latex=False)
        except:
            _af_names = [_af.name for _af in self._arg_formatters] if self._arg_formatters else None
            raise FormatterException("Expression string %s does not match argument structure %s"
                                     % (expression_format_string, _af_names))

    @property
    def latex_expression_format_string(self):
        """a LaTeX-formatted expression for the function"""
        return self._latex_expr_string

    @latex_expression_format_string.setter
    def latex_expression_format_string(self, latex_expression_format_string):
        self._latex_expr_string = latex_expression_format_string
        try:
            self._get_formatted_expression(format_as_latex=True)
        except:
            _af_latex_names = [_af.latex_name for _af in self._arg_formatters] if self._arg_formatters else None
            raise FormatterException("LaTeX expression string %s does not match argument structure %s"
                                     % (latex_expression_format_string, _af_latex_names))

    @property
    def name(self):
        """a plain-text-formatted string indicating the parameter name"""
        return self._name

    @name.setter
    def name(self, new_name):
        # TODO: validate
        self._name = new_name

    @property
    def latex_name(self):
        """a LaTeX-formatted string indicating the function name"""
        return self._latex_name

    @latex_name.setter
    def latex_name(self, new_latex_name):
        # TODO: validate
        self._latex_name = new_latex_name

    @property
    def description(self):
        """a short plain-text description of the function"""
        return self._description or "<no description provided>"

    @description.setter
    def description(self, new_description):
        # TODO: validate
        self._description = new_description

    @property
    def arg_formatters(self):
        """
        the list of :py:obj:`ModelParameterFormatter`-derived objects used for formatting
        model function arguments
        """
        return self._arg_formatters

    @arg_formatters.setter
    def arg_formatters(self, arg_formatters):
        self._arg_formatters = arg_formatters

    def get_formatted(self, with_par_values=True, n_significant_digits=2, format_as_latex=False, with_expression=False):
        """
        Get a formatted string representing this model function.

        :param with_par_values: if ``True``, output will include the value of each function parameter
                                (e.g. ``f(a=1, b=2, ...)``)
        :param n_significant_digits: number of significant digits for rounding
        :param format_as_latex: if ``True``, the returned string will be formatted using LaTeX syntax
        :param with_expression: if ``True``, the returned string will include the expression assigned to the function
        :return: string
        """
        # FIXME: default should actually *not* show the parameter values
        _par_strings = self._get_formatted_args(with_par_values=with_par_values,
                                                n_significant_digits=n_significant_digits,
                                                format_as_latex=format_as_latex)
        _par_expr_string = ""
        if with_expression:
            _par_expr_string = self._get_formatted_expression(format_as_latex=format_as_latex)

        if format_as_latex:
            _out_string = r"%s\left(%s\right)" % (self._latex_name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
            _out_string = "$%s$" % (_out_string,)
        else:
            _out_string = "%s(%s)" % (self._name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
        return _out_string


class CostFunctionFormatter(ModelFunctionFormatter):

    def get_formatted(self, value=None, n_degrees_of_freedom=None,
                      with_name=True,
                      with_value_per_ndf=True, format_as_latex=False):
        """
        Get a formatted string representing this cost function.

        :param value: value of the cost function (if not ``None``, the returned string will include this)
        :type value: float
        :param n_degrees_of_freedom: number of degrees of freedom (if not ``None``, the returned string will include this)
        :type n_degrees_of_freedom: int
        :param with_name: if ``True``, the returned string will include the cost function name
        :param with_value_per_ndf: if ``True``, the returned string will include the value-ndf ratio as a decimal value
        :param format_as_latex: if ``True``, the returned string will be formatted using LaTeX syntax
        :return: string
        """

        _name_string = "%s" % (self._latex_name)
        _value_string = ""
        if value is not None:
            _value_string = "%.4g" % (value,)
            if n_degrees_of_freedom is not None:
                if format_as_latex:
                    _name_string = r"%s / {\rm ndf}" % (self._latex_name)
                else:
                    _name_string = "%s / ndf" % (self._latex_name)
                _value_string = "%s / %d" % (_value_string, n_degrees_of_freedom)
                if with_value_per_ndf:
                    _value_string = "%s = %.4g" % (_value_string, float(value)/n_degrees_of_freedom)

        if with_name:
            _out_string = "%s = %s" % (_name_string, _value_string)
        else:
            _out_string = "%s" % (_value_string,)

        if format_as_latex:
            _out_string = "$%s$" % (_out_string ,)

        return _out_string
