from .._base import ModelFunctionFormatter


__all__ = ["IndexedModelFunctionFormatter"]


class IndexedModelFunctionFormatter(ModelFunctionFormatter):
    def __init__(self, name, latex_name=None, index_name='i', latex_index_name='i',
                 arg_formatters=None, expression_string=None, latex_expression_string=None):
        """
        Construct a :py:obj:`Formatter` for a model function for *indexed* data:

        :param name: a plain-text-formatted string indicating the function name
        :param latex_name: a LaTeX-formatted string indicating the function name
        :param index_name: a plain-text-formatted string representing the index
        :param latex_index_name: a LaTeX-formatted string representing the index
        :param arg_formatters: list of :py:obj:`ParameterFormatter`-derived objects,
                               formatters for function arguments
        :param expression_string:  a plain-text-formatted string indicating the function expression
        :param latex_expression_string:  a LaTeX-formatted string indicating the function expression
        """
        super(IndexedModelFunctionFormatter, self).__init__(
            name, latex_name=latex_name,
            x_name=index_name, latex_x_name=latex_index_name,
            arg_formatters=arg_formatters,
            expression_string=expression_string,
            latex_expression_string=latex_expression_string
        )

    def _get_format_kwargs(self, format_as_latex=False):
        _dct = super(IndexedModelFunctionFormatter, self)._get_format_kwargs(format_as_latex=format_as_latex)
        if format_as_latex:
            _dct.update(x=self._latex_x_name)
        else:
            _dct.update(x=self._x_name)
        return _dct

    @property
    def index_name(self):
        """The parameter name of the index.

        :rtype: str
        """
        return self.x_name

    @index_name.setter
    def index_name(self, index_name):
        self.x_name = index_name

    @property
    def latex_index_name(self):
        """The LaTeX parameter name of the index.

        :rtype: str"""
        return self.latex_x_name

    @latex_index_name.setter
    def latex_index_name(self, latex_index_name):
        self.latex_x_name = latex_index_name

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
            _out_string = r"%s_{%s}\left(%s\right)" % (self._latex_name, self.latex_index_name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
            _out_string = "$%s$" % (_out_string,)
        else:
            _out_string = "%s_%s(%s)" % (self._name, self.index_name, ", ".join(_par_strings))
            if _par_expr_string:
                _out_string += " = " + _par_expr_string
        return _out_string
