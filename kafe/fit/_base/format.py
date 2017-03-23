import re
import string

import numpy as np


# -- formatters for model parameters and model functions

class FormatterException(Exception):
    pass


class ModelParameterFormatter(object):
    def __init__(self, name, value, error=None, latex_name=None):
        self.value = value

        self._error_is_asymmetric = False
        self.error = error

        self._name = name
        self._latex_name = latex_name
        if self._latex_name is None:
            self._latex_name = self._latexify_ascii(self._name)

    @staticmethod
    def _latexify_ascii(ascii_string):
        _lpn = string.replace(ascii_string, '_', r"\_")
        return r"{\tt %s}" % (_lpn,)

    @property
    def name(self):
        return self._name

    @property
    def latex_name(self):
        return self._latex_name

    @latex_name.setter
    def latex_name(self, new_latex_name):
        # TODO: validate
        self._latex_name = new_latex_name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, err_spec):
        _err_spec = err_spec
        if _err_spec is None:
            self._error = None
            self._error_is_asymmetric = False
            return

        try:
            iter(_err_spec)
        except TypeError:
            _err_spec = (_err_spec,)

        if len(_err_spec) == 2:
            self._error = tuple(_err_spec)  # asymmetric error
            self._error_is_asymmetric = True
        elif len(_err_spec) == 1:
            self._error = (_err_spec[0], _err_spec[0])  # asymmetric error
            self._error_is_asymmetric = False
        else:
            raise FormatterException("Error specification not understood: %r" % (err_spec,))

    @property
    def error_rel(self):
        if self._error is None:
            return None
        if self._error_is_asymmetric:
            return (self._error[0]/self._value, self._error[1]/self._value)
        else:
            return self._error[0]/self._value

    @property
    def error_up(self):
        if self._error is None:
            return None
        return self._error[1]

    @property
    def error_down(self):
        if self._error is None:
            return None
        return self._error[0]


    def get_formatted(self, with_name=False, with_value=True, with_errors=True,
                      n_significant_digits=2, round_value_to_error=True, format_as_latex=False):
        _display_string = ""
        if with_name:
            if format_as_latex:
                _display_string += "$%s$" % (self.latex_name,)
            else:
                _display_string += "%s" % (self.name,)

        if with_value:
            _display_string += " = "
            if not with_errors or self._error is None:
                _sig = int(-np.floor(np.log(self._value) / np.log(10))) + n_significant_digits - 1
                _display_val = round(self._value, _sig)
                if format_as_latex:
                    _display_string += "$%g$" % (_display_val,)
                else:
                    _display_string += "%g" % (_display_val,)
            else:
                _min_err = np.min(map(abs, self._error))
                if round_value_to_error:
                    _sig = int(-np.floor(np.log(_min_err)/np.log(10))) + n_significant_digits - 1
                else:
                    _sig = int(-np.floor(np.log(self._value) / np.log(10))) + n_significant_digits - 1

                _display_val = round(self._value, _sig)
                if self._error_is_asymmetric:
                    _display_err_dn = round(self.error_down, _sig)
                    _display_err_up = round(self.error_up, _sig)
                    if format_as_latex:
                        _display_string += "%g^{%g}_{%g}" % (_display_val, _display_err_up, _display_err_dn)
                    else:
                        _display_string += "%g + %g (up) - %g (down)" % (_display_val, _display_err_up, _display_err_dn)
                else:
                    _display_err = round(self._error[0], _sig)
                    if format_as_latex:
                        _display_string += r"$%g \pm %g$" % (_display_val, _display_err)
                    else:
                        _display_string += "%g +/- %g" % (_display_val, _display_err)

            # replace scientific notation with power of ten (LaTeX only)
            if format_as_latex:
                _display_string = re.sub(r'(-?\d*\.?\d+?)0*e\+?(-?[0-9]*[1-9]?)',
                                         r'\1\\times10^{\2}', _display_string)
        return _display_string


class ModelFunctionFormatter(object):
    DEFAULT_EXPRESSION_STRING = "<not_specified>"
    DEFAULT_LATEX_EXPRESSION_STRING = r"\langle{\it not\,\,specified}\rangle"
    def __init__(self, name, latex_name=None, arg_formatters=None, expression_string=None, latex_expression_string=None):
        self._name = name
        self._arg_formatters = arg_formatters
        self.expression_format_string = expression_string
        self.latex_expression_format_string = latex_expression_string

        self._latex_name = latex_name
        if self._latex_name is None:
            self._latex_name = self._latexify_ascii(self._name)

    @staticmethod
    def _latexify_ascii(ascii_string):
        _lpn = string.replace(ascii_string, '_', r"\_")
        return r"{\tt %s}" % (_lpn,)

    def _get_format_kwargs(self, format_as_latex=False):
        return dict()

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
        return self._expr_string

    @expression_format_string.setter
    def expression_format_string(self, expression_format_string):
        self._expr_string = expression_format_string
        try:
            self._get_formatted_expression(format_as_latex=False)
        except:
            raise FormatterException("Expression string does not match argument structure: %s"
                                     % (expression_format_string,))

    @property
    def latex_expression_format_string(self):
        return self._latex_expr_string

    @latex_expression_format_string.setter
    def latex_expression_format_string(self, latex_expression_format_string):
        self._latex_expr_string = latex_expression_format_string
        try:
            self._get_formatted_expression(format_as_latex=True)
        except:
            raise FormatterException("LaTeX expression string does not match argument structure: %s"
                                     % (latex_expression_format_string,))

    def get_formatted(self, with_par_values=True, n_significant_digits=2, format_as_latex=False, with_expression=False):
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

