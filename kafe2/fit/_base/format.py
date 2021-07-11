import copy
import re

import numpy as np

from ..io.file import FileIOMixin

__all__ = ["FormatterException", "ScalarFormatter", "ParameterFormatter", "FunctionFormatter",
           "ModelFunctionFormatter", "CostFunctionFormatter", "latexify_ascii"]


# -- formatters for model parameters and model functions
def latexify_ascii(ascii_string):
    """Create a true type latex string of an standard ascii string.

    :param ascii_string: The string to be converted
    :type ascii_string: str
    :rtype: str
    """
    _lpn = ascii_string.replace('_', r"\_")
    return r"{\tt %s}" % _lpn


# Naming convention: Arguments describe all arguments of a function, parameters only the fitted
# parameters excluding the independent variable(s)


class FormatterException(Exception):
    pass


class ScalarFormatter(object):
    def __init__(self, sigma, n_significant_digits=2):
        """Format a scalar to a specified precision, according to the uncertainty.

        :param float sigma: The uncertainty of the parameter.
        :param int n_significant_digits: Number of significant digits.
        """
        self._sigma = sigma
        self._n_significant_digits = n_significant_digits
        _sig = int(-np.floor(np.log10(self._sigma))) + self._n_significant_digits - 1
        # inner rounding needed for errors like 0.9999999 -> 1.0 (shift in decimal place)
        self._sig = int(-np.floor(np.log10(np.around(self._sigma, _sig)))) + self._n_significant_digits - 1

    def __call__(self, x):
        """Format the input to the precision given by the uncertainty.

        :param float x: The value to format.
        :rtype: str
        """
        # needed e.g. when rounding values like 9.999999 -> 10.0 (shift in decimal place)
        _rounded_x = abs(np.around(x, self._sig))
        # fallback to rounding to 10^(-1) if value is zero
        _log_abs_x = -1
        if _rounded_x:
            _log_abs_x = np.log10(np.abs(_rounded_x))
        _val_sig = int(self._sig - int(-np.floor(_log_abs_x)) + self._n_significant_digits - 1)

        if _val_sig < 0:
            raise FormatterException("Value significance is smaller than 0. Did you try to format a value which is "
                                     "less precise than the uncertainty?")

        _template = "%#.{significance}g".format(significance=_val_sig)
        return _template % x


class ParameterFormatter(FileIOMixin, object):
    """Formatter class for model parameter objects.

    These objects store the relevant information for constructing plain-text and/or LaTeX string
    representations of model function parameters.

    For this, the original argument name, the name for representation, formatted as a
    plain-text/LaTeX string, its value and its
    uncertainty is stored.

    The formatted string is obtained by calling the :py:meth:`~.get_formatted` method.
    """

    def __init__(self, arg_name, value=None, error=None, asymmetric_error=None, name=None,
                 latex_name=None):
        """Construct a Parameter Formatter.

        :param str arg_name: A plain string indicating the parameter's signature inside the
            function call.
        :param value: The parameter value.
        :type value: float or None
        :param error: The symmetric parameter error.
        :type error: float or None
        :param asymmetric_error: The asymmetric parameter errors.
        :type asymmetric_error: tuple[float, float] or None
        :param name: A plain-text-formatted string indicating the parameter name.
        :type name: str or None
        :param latex_name: A LaTeX-formatted string indicating the parameter name.
        :type latex_name: str or None
        :rtype: ParameterFormatter
        """
        super(ParameterFormatter, self).__init__()
        self._arg_name = arg_name
        self.value = value
        self.error = error
        self.asymmetric_error = asymmetric_error

        self.name = name
        self.latex_name = latex_name  # latex_name setter requires self._name to be set beforehand

        self._fixed = False

    @classmethod
    def _get_base_class(cls):
        return ParameterFormatter

    @classmethod
    def _get_object_type_name(cls):
        return 'parameter_formatter'

    @property
    def arg_name(self):
        """Name of the function argument this formatter represents.

        :rtype: str
        """
        return self._arg_name

    @property
    def name(self):
        """The plain-text-formatted string indicating the parameter name.

        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            self._name = self.arg_name
        else:
            self._name = name

    @property
    def latex_name(self):
        """The LaTeX-formatted string indicating the parameter name.

        :rtype: str
        """
        return self._latex_name

    @latex_name.setter
    def latex_name(self, new_latex_name):
        # TODO: validate
        if new_latex_name is None:
            self._latex_name = latexify_ascii(self.name)
        elif new_latex_name.startswith('{') and new_latex_name.endswith('}'):
            self._latex_name = new_latex_name
        else:
            self._latex_name = '{' + new_latex_name + '}'

    @property
    def value(self):
        """The parameter value.

        :rtype: float or None"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def error(self):
        """The symmetric parameter error.

        :rtype: float or None"""
        return self._error

    @error.setter
    def error(self, err_spec):
        self._error = err_spec

    @property
    def error_rel(self):
        """The relative parameter error.

        :rtype: float or None"""
        if self._error is None:
            return None
        return self._error / self._value

    @property
    def asymmetric_error(self):
        """Tuple containing the asymmetric parameter errors.

        :rtype: tuple[float, float] or None"""
        return self._asymmetric_error

    @asymmetric_error.setter
    def asymmetric_error(self, err_spec):
        self._asymmetric_error = err_spec

    @property
    def error_up(self):
        """The upper uncertainty (only for asymmetric errors).

        :rtype: float or None
        """
        if self.asymmetric_error is None:
            return None
        return self.asymmetric_error[1]

    @property
    def error_down(self):
        """The lower uncertainty (only for asymmetric errors).

        :rtype: float or None
        """
        if self.asymmetric_error is None:
            return None
        return self.asymmetric_error[0]

    @property
    def fixed(self):
        """If the parameter has been fixed by the user. :py:obj:`True` when it's fixed,
        :py:obj:`False` when not.

        :rtype: bool
        """
        return self._fixed

    @fixed.setter
    def fixed(self, fixed):
        self._fixed = fixed

    def get_formatted(self, with_name=False, with_value=True, with_errors=True,
                      n_significant_digits=2, round_value_to_error=True, asymmetric_error=False,
                      format_as_latex=False):
        """Get a formatted string representing this model parameter.

        :param bool with_name: If :py:obj:`True`, the output will include the parameter name.
        :param bool with_value: If :py:obj:`True`, the output will include the parameter value.
        :param bool with_errors: If :py:obj:`True`, the output will include the parameter error(s).
        :param int n_significant_digits: Number of significant digits for rounding.
        :param bool round_value_to_error: If :py:obj:`True`, the parameter value will be rounded to
            the same precision as the uncertainty.
        :param bool asymmetric_error: If :py:obj:`True`, the asymmetric parameter uncertainties are
            used.
        :param bool format_as_latex: If :py:obj:`True`, the returned string will be formatted using
            LaTeX syntax.
        :return: The string representation of the parameter.
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

            if self.fixed:
                if format_as_latex:
                    _display_string += r"$%g$ (fixed)" % self._value
                else:
                    _display_string += "%g (fixed)" % self._value
            elif not with_errors or (not asymmetric_error and self.error in (None, 0)) or \
                    (asymmetric_error and (
                            self.asymmetric_error is None or np.all(self.asymmetric_error == 0))):
                if format_as_latex:
                    _display_string += "$%g$" % self.value
                else:
                    _display_string += "%g" % self.value
            else:
                if asymmetric_error:
                    _min_err = min(abs(self.error_up), abs(self.error_down))
                else:
                    _min_err = self.error
                # fallback to rounding to 10^(-1) if error is zero
                if not _min_err or np.isnan(_min_err):
                    _min_err = 1e-1

                # calculate decimal precision if rounding:
                if round_value_to_error:
                    val_formatter = ScalarFormatter(_min_err, n_significant_digits=n_significant_digits)
                    _val = val_formatter(self.value)
                    _err = "%#.{n}g".format(n=n_significant_digits) % self.error
                    if asymmetric_error:  # needed for different powers of 10 in asymmetric errs
                        if abs(self.error_down) <= abs(self.error_up):
                            _err_u = val_formatter(abs(self.error_up))
                            _err_d = "%#.{n}g".format(n=n_significant_digits) % abs(self.error_down)
                        else:
                            _err_d = val_formatter(abs(self.error_down))
                            _err_u = "%#.{n}g".format(n=n_significant_digits) % abs(self.error_up)
                # default cases if no rounding
                else:
                    _val = "%.g" % self.value
                    _err = "%.g" % self.error
                    if asymmetric_error:
                        _err_u = "%.g" % self.error_up
                        _err_d = "%.g" % self.error_down

                if asymmetric_error:
                    if format_as_latex:
                        _display_string += "${%s}^{+%s}_{-%s}$" % (_val, _err_u, _err_d)
                    else:
                        _display_string += "%s + %s (up) - %s (down)" % (_val, _err_u, _err_d)
                else:
                    if format_as_latex:
                        _display_string += "$%s \\pm %s$" % (_val, _err)
                    else:
                        _display_string += "%s +/- %s" % (_val, _err)

            # replace scientific notation with power of ten (LaTeX only)
            if format_as_latex:
                _display_string = re.sub(r'(-?\d*\.?\d+?)0*e\+?(-?[0-9]*[1-9]?)',
                                         r'\1\\times10^{\2}', _display_string)
        return _display_string


class FunctionFormatter(FileIOMixin, object):
    """Base class for function formatter objects. Requires further specialization for each type of
    model function.
    Objects derived from this class store information relevant for constructing plain-text and/or
    LaTeX string representations of functions.

    For this, the function name, formatted as a plain-text/LaTeX string, as well as a list of
    references to :py:obj:`ParameterFormatter` objects which contain information on how to format
    the model function arguments is stored.

    Optionally, plain-text/LaTeX expression strings can be provided. These are strings representing
    the model function expression (i.e. mathematical formula).

    The formatted string is obtained by calling the :py:meth:`~get_formatted` method.
    """
    DEFAULT_EXPRESSION_STRING = None
    DEFAULT_LATEX_EXPRESSION_STRING = None

    def __init__(self, name, latex_name=None, arg_formatters=None, expression_string=None,
                 latex_expression_string=None):
        """Construct a formatter for a model function:

        :param name: A plain-text-formatted string indicating the function name.
        :type name: str
        :param latex_name: A LaTeX-formatted string indicating the function name.
        :type latex_name: str
        :param arg_formatters: List of :py:obj:`ParameterFormatter`-derived objects, formatters for
            function arguments.
        :type arg_formatters: list[kafe2.fit._base.ParameterFormatter]
        :param expression_string: A plain-text-formatted string indicating the function expression.
        :type expression_string: str
        :param latex_expression_string: A LaTeX-formatted string indicating the function expression.
        :type latex_expression_string: str
        """
        # TODO should name be allowed to be None?
        self._name = name
        self._arg_formatters = arg_formatters
        self.expression_format_string = expression_string
        self.latex_expression_format_string = latex_expression_string

        self.latex_name = latex_name

        self._description = None
        super(FunctionFormatter, self).__init__()

    @classmethod
    def _get_base_class(cls):
        return FunctionFormatter

    @classmethod
    def _get_object_type_name(cls):
        return 'function_formatter'

    def _get_format_kwargs(self, format_as_latex=False):
        """Create a dictionary containing argument name and format pairs.

        :param format_as_latex: If the format string is a latex formatted string.
        :return: Dictionary containing argument name and format pairs.
        :rtype: dict[str, str]
        """
        if format_as_latex:
            _par_name_string_dict = {_af.arg_name: _af.latex_name for _af in self.arg_formatters}
        else:
            _par_name_string_dict = {_af.arg_name: _af.name for _af in self.arg_formatters}
        return _par_name_string_dict

    def _get_formatted_name(self, format_as_latex=False):
        """Get the formatted function name.

        :param bool format_as_latex: If the format string is a latex formatted string.
        :return: The formatted function name.
        :rtype: str
        """
        if format_as_latex:
            return self._latex_name
        return self._name

    def _get_formatted_pars(self, with_par_values=True, n_significant_digits=2,
                            format_as_latex=False):
        """Get a list of the formatted parameters including their values. This can be turned off.

        :param bool with_par_values: If the strings should contain the parameter values.
        :param int n_significant_digits: The number of significant digits when using the parameter
            values.
        :param bool format_as_latex: If the string should be formatted as a latex string.
        :return: List of formatted parameter strings.
        :rtype: list[str]
        """
        if format_as_latex:
            _par_name_strings = [_pf.latex_name for _pf in self.par_formatters]
        else:
            _par_name_strings = [_pf.name for _pf in self.par_formatters]

        if with_par_values:
            _par_val_strings = []
            for _pf in self.par_formatters:
                _par_val_strings.append(_pf.get_formatted(with_name=False,
                                                          with_value=True,
                                                          with_errors=False,
                                                          n_significant_digits=n_significant_digits,
                                                          round_value_to_error=False,
                                                          format_as_latex=format_as_latex))
            return ["%s=%s" % (_pn, _pv) for _pn, _pv in zip(_par_name_strings, _par_val_strings)]
        return ["%s" % (_pn,) for _pn in _par_name_strings]

    def _get_formatted_expression(self, format_as_latex=False):
        """Get the formatted function expression.

        :param format_as_latex: If the string should be formatted as a latex string.
        :rtype: str
        """
        _kwargs = self._get_format_kwargs(format_as_latex=format_as_latex)
        if format_as_latex and self._latex_expr_string is not None:
            # use all arguments to format the expression string
            _par_expr_string = self._latex_expr_string.format(
                *[_af.latex_name for _af in self.arg_formatters], **_kwargs)
        elif not format_as_latex and self._expr_string is not None:
            _par_expr_string = self._expr_string.format(
                *[_af.name for _af in self.arg_formatters], **_kwargs)
        elif format_as_latex and self._latex_expr_string is None:
            _par_expr_string = self.DEFAULT_LATEX_EXPRESSION_STRING
        else:
            _par_expr_string = self.DEFAULT_EXPRESSION_STRING
        return _par_expr_string

    @property
    def expression_format_string(self):
        """A plain-text-formatted expression for the function.
        This function will replace all function parameters with their corresponding strings.
        For example the string "{a}*{x}+{b}" will turn into "A*x + B" when the name of the parameter
        a was set to "A", and the name of b is set to "B".

        :rtype: str
        """
        return self._expr_string

    @expression_format_string.setter
    def expression_format_string(self, expression_format_string):
        self._expr_string = expression_format_string
        try:
            self._get_formatted_expression(format_as_latex=False)
        except LookupError:  # fetch key and index errors. Other Errors have other causes.
            _af_arg_names = [_af.arg_name for _af in self.arg_formatters] if self.arg_formatters \
                else None
            raise FormatterException("Expression string %s does not match argument structure %s"
                                     % (expression_format_string, _af_arg_names))

    @property
    def latex_expression_format_string(self):
        r"""A LaTeX-formatted expression for the function.
        This function will replace all function parameters with their corresponding latex string.
        For example the string ``"{a}{x}+{b}"`` will turn into ``"A_0 x + B"`` when the latex name
        of the parameter ``a`` was set to ``"A_0"``, and the latex name of ``b`` is set to ``"B"``.

        .. note:
            Due to the way python handles string formatting, please always use two curly braces in
            standard LaTeX expressions. E.g. ``"\\frac{{{a}*{b}}}{{{x}-{c}}}"``.
            When not using a raw string, please double all backslashes as well! When using a raw
            string the above translates to ``r"\frac{{{a}*{b}}}{{{x}-{c}}}"``.

        :rtype: str
        """
        return self._latex_expr_string

    @latex_expression_format_string.setter
    def latex_expression_format_string(self, latex_expression_format_string):
        self._latex_expr_string = latex_expression_format_string
        try:
            self._get_formatted_expression(format_as_latex=True)
        except LookupError:  # fetch key and index errors. Other Errors have other causes.
            _af_arg_names = [_af.arg_name for _af in self.arg_formatters] if self.arg_formatters \
                else None
            raise FormatterException(
                "LaTeX expression string %s does not match argument structure %s" % (
                    latex_expression_format_string, _af_arg_names)
            )

    @property
    def name(self):
        """A plain-text-formatted string indicating the function name.

        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name):
        # TODO: validate
        self._name = name

    @property
    def latex_name(self):
        """A LaTeX-formatted string indicating the function name.

        :rtype: str
        """
        return self._latex_name

    @latex_name.setter
    def latex_name(self, latex_name):
        if latex_name is None:
            self._latex_name = latexify_ascii(self.name)
        # TODO: validate
        else:
            self._latex_name = latex_name

    @property
    def description(self):
        """A short plain-text description of the function.

        :rtype: str
        """
        return self._description or "<no description provided>"

    @description.setter
    def description(self, description):
        # TODO: validate
        self._description = description

    @property
    def arg_formatters(self):
        """The list of :py:obj:`ParameterFormatter`-derived objects used for formatting all model
        function arguments.

        :rtype: list[ParameterFormatter]
        """
        return self._arg_formatters

    @property
    def par_formatters(self):
        """List of :py:obj:`ParameterFormatter`-derived objects used for formatting the fit parameters,
        excluding the independent parameter(s).

        :rtype: list[ParameterFormatter]
        """
        return self._arg_formatters  # same as all arguments for base, overwrite when necessary

    def get_formatted(self, with_par_values=False, n_significant_digits=2, format_as_latex=False,
                      with_expression=False):
        """Get a formatted string representing this model function.

        :param bool with_par_values: If :py:obj:`True`, output will include the value of each function parameter
                                     (e.g. ``f(a=1, b=2, ...)``).
        :param int n_significant_digits: Number of significant digits for rounding.
        :param bool format_as_latex: If :py:obj:`True`, the returned string will be formatted using LaTeX syntax.
        :param bool with_expression: If :py:obj:`True`, the returned string will include the expression assigned to the
                                     function.
        """
        _par_strings = self._get_formatted_pars(with_par_values=with_par_values,
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


class CostFunctionFormatter(FunctionFormatter):
    """A Formatter class for Cost Functions.
    """

    def __init__(
            self, name, name_saturated=None, latex_name=None, latex_name_saturated=None,
            arg_formatters=None, expression_string=None, latex_expression_string=None):
        super(CostFunctionFormatter, self).__init__(
            name=name, latex_name=latex_name, arg_formatters=arg_formatters,
            expression_string=expression_string, latex_expression_string=latex_expression_string)
        self.name_saturated = name_saturated
        self.latex_name_saturated = latex_name_saturated

    @property
    def name_saturated(self):
        """A plain-text-formatted string indicating the saturated function name.

        :rtype: str
        """
        return self._name_saturated if self._name_saturated is not None else self._name

    @name_saturated.setter
    def name_saturated(self, name_saturated):
        # TODO: validate
        self._name_saturated = name_saturated

    @property
    def latex_name_saturated(self):
        """A LaTeX-formatted string indicating the saturated function name.

        :rtype: str
        """
        return self._latex_name_saturated if self._latex_name_saturated is not None \
            else self.latex_name

    @latex_name_saturated.setter
    def latex_name_saturated(self, latex_name_saturated):
        self._latex_name_saturated = latex_name_saturated  # TODO validate

    def get_formatted(
            self, value=None, n_degrees_of_freedom=None, with_name=True, saturated=False,
            with_value_per_ndf=True, format_as_latex=False):
        """Get a formatted string representing this cost function.

        :param value: Value of the cost function (if not :py:obj:`None`, the returned string will
            include this).
        :type value: float or None
        :param n_degrees_of_freedom: Number of degrees of freedom (if not :py:obj:`None`, the
            returned string will include this).
        :type n_degrees_of_freedom: int or None
        :param bool with_name: If :py:obj:`True`, the returned string will include the cost function
            name
        :param bool saturated: If :py:obj:`True`, the cost function name for the saturated
            Likelihood will be used (no effect for chi2).
        :param bool with_value_per_ndf: If :py:obj:`True`, the returned string will include the
            value-ndf ratio as a decimal value
        :param bool format_as_latex: If :py:obj:`True`, the returned string will be formatted using
            LaTeX syntax
        :rtype: str
        """

        if format_as_latex:
            _name = self.latex_name_saturated if saturated else self.latex_name
        else:
            _name = self.name_saturated if saturated else self.name
        _name_string = "%s" % _name
        _value_string = ""
        if value is not None:
            _value_string = "%.4g" % value
            if n_degrees_of_freedom is not None:
                if format_as_latex:
                    _name_string = r"%s / {\rm ndf}" % _name
                else:
                    _name_string = "%s / ndf" % _name
                _value_string = "%s / %d" % (_value_string, n_degrees_of_freedom)
                if with_value_per_ndf and n_degrees_of_freedom > 0:
                    _value_string = "%s = %.4g" % (_value_string,
                                                   float(value) / n_degrees_of_freedom)

        if with_name:
            _out_string = "%s = %s" % (_name_string, _value_string)
        else:
            _out_string = "%s" % _value_string

        if format_as_latex:
            _out_string = "$%s$" % _out_string

        return _out_string


class ModelFunctionFormatter(FunctionFormatter):
    """A formatter class for model functions.

    This object stores the function name, formatted as a plain-text/LaTeX string, as well as a list
    of references to :py:obj:`ParameterFormatter` objects which contain information on how to format
    the model function arguments.
    Additionally formatting information about the independent variable is stored.

    Optionally, plain-text/LaTeX expression strings can be provided. These are strings representing
    the model function expression (i.e. mathematical formula).

    The formatted string is obtained by calling the :py:meth:`~.get_formatted` method.
    """

    @classmethod
    def _get_object_type_name(cls):
        return 'model_function_formatter'

    @property
    def par_formatters(self):
        # copy so we don't modify the original formatters
        formatters = copy.copy(self.arg_formatters)
        formatters.pop(0)  # first formatter is independent var, delete it
        return formatters

    def get_formatted(self, with_par_values=False, n_significant_digits=2, format_as_latex=False,
                      with_expression=False):
        """Create a formatted string representing this model function.

        :param bool with_par_values: If :py:obj:`True`, output will include the value of each
            function parameter (e.g. ``f(a=1, b=2, ...)``).
        :param int n_significant_digits: number of significant digits for rounding
        :param bool format_as_latex: If :py:obj:`True`, the returned string will be formatted using
            LaTeX syntax.
        :param bool with_expression: If :py:obj:`True`, the returned string will include the
            expression assigned to the function.
        :returns: The formatted string representing this model function.
        :rtype: str
        """
        _par_strings = self._get_formatted_pars(with_par_values=with_par_values,
                                                n_significant_digits=n_significant_digits,
                                                format_as_latex=format_as_latex)
        _par_expr_string = ""
        if with_expression:
            _par_expr_string = self._get_formatted_expression(format_as_latex=format_as_latex)

        x_formatter = self.arg_formatters[0]
        if format_as_latex:
            _out_string = r"%s\left(%s;%s\right)" % (
            self._latex_name, x_formatter.latex_name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
            _out_string = "$%s$" % (_out_string,)
        else:
            _out_string = "%s(%s; %s)" % (self._name, x_formatter.name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
        return _out_string
