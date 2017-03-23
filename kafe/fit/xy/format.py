from .._base import ModelFunctionFormatter


class XYModelFunctionFormatter(ModelFunctionFormatter):
    def __init__(self, name, latex_name=None, x_name='x', latex_x_name=None,
                 arg_formatters=None, expression_string=None, latex_expression_string=None):
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
        if format_as_latex:
            return dict(x=self._latex_x_name)
        else:
            return dict(x=self._x_name)

    def get_formatted(self, with_par_values=True, n_significant_digits=2, format_as_latex=False, with_expression=False):
        _par_strings = self._get_formatted_args(with_par_values=with_par_values,
                                                n_significant_digits=n_significant_digits,
                                                format_as_latex=format_as_latex)
        _par_expr_string = ""
        if with_expression:
            if format_as_latex:
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
