from .._base import ModelFunctionFormatter


__all__ = ["XYModelFunctionFormatter"]


class XYModelFunctionFormatter(ModelFunctionFormatter):
    def __init__(self, name, latex_name=None, x_name='x', latex_x_name=None,
                 arg_formatters=None, expression_string=None, latex_expression_string=None):
        """
        Construct a :py:obj:`Formatter` for a model function for *xy* data:

        :param name: a plain-text-formatted string indicating the function name
        :param latex_name: a LaTeX-formatted string indicating the function name
        :param x_name: a plain-text-formatted string representing the independent variable
        :param latex_x_name: a LaTeX-formatted string representing the independent variable
        :param arg_formatters: list of :py:obj:`ModelParameterFormatter`-derived objects,
                               formatters for function arguments
        :param expression_string:  a plain-text-formatted string indicating the function expression
        :param latex_expression_string:  a LaTeX-formatted string indicating the function expression
        """
        self._x_name = x_name
        self._latex_x_name = latex_x_name
        if self._latex_x_name is None:
            self._latex_x_name = self._latexify_ascii(self._x_name)

        super(XYModelFunctionFormatter, self).__init__(
            name, latex_name=latex_name, arg_formatters=arg_formatters,
            expression_string=expression_string,
            latex_expression_string=latex_expression_string
        )

    def _get_format_kwargs(self, format_as_latex=False):
        _dct = super(XYModelFunctionFormatter, self)._get_format_kwargs(format_as_latex=format_as_latex)
        if format_as_latex:
            _dct.update(x=self._latex_x_name)
        else:
            _dct.update(x=self._x_name)
        return _dct

    def get_formatted(self, with_par_values=True, n_significant_digits=2, format_as_latex=False, with_expression=False):
        """
        Get a formatted string representing this model function.

        :param with_par_values: if ``True``, output will include the value of each function parameter
                                (e.g. ``f_i(a=1, b=2, ...)``)
        :param n_significant_digits: number of significant digits for rounding
        :type n_significant_digits: int
        :param format_as_latex: if ``True``, the returned string will be formatted using LaTeX syntax
        :param with_expression: if ``True``, the returned string will include the expression assigned to the function
        :return: string
        """
        _par_strings = self._get_formatted_args(with_par_values=with_par_values,
                                                n_significant_digits=n_significant_digits,
                                                format_as_latex=format_as_latex)
        _par_expr_string = ""
        if with_expression:
            _par_expr_string = self._get_formatted_expression(format_as_latex=format_as_latex)

        if format_as_latex:
            _out_string = r"%s\left(%s;%s\right)" % (self._latex_name, self._latex_x_name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
            _out_string = "$%s$" % (_out_string,)
        else:
            _out_string = "%s(%s; %s)" % (self._name, self._x_name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
        return _out_string
