from .._base import ModelFunctionFormatter


__all__ = ["XYMultiModelFunctionFormatter"]


class XYMultiModelFunctionFormatter(ModelFunctionFormatter):
    def __init__(self, singular_formatters, arg_formatters):
        """
        Construct a :py:obj:`Formatter` for a multi model function for *xy* data:
        
        :param singular_formatters: the list of singular model function formatters to be contained in this
            multi model function formatter, one per model function
        :type singular_formatters: list of :py:obj:`XYModelFunctionFormatter`
        :param arg_formatters: the combined list of argument formatters to be used by all singular 
            model function formatters
        :type arg_formatters: list of :py:obj:`ModelParameterFormatter`
        """
        self._singular_formatters = singular_formatters
        self._arg_formatters = arg_formatters
        
        #super(XYMultiModelFunctionFormatter, self).__init__()

    def get_formatted(self, model_index, with_par_values=True, n_significant_digits=2, format_as_latex=False, with_expression=False):
        """
        Get a formatted string representing this model function.

        :param model_index: determines the index of the model function to be formatted
        :type model_index: int
        :param with_par_values: if ``True``, output will include the value of each function parameter
                                (e.g. ``f_i(a=1, b=2, ...)``)
        :param n_significant_digits: number of significant digits for rounding
        :type n_significant_digits: int
        :param format_as_latex: if ``True``, the returned string will be formatted using LaTeX syntax
        :param with_expression: if ``True``, the returned string will include the expression assigned to the function
        :return: string
        """
        return self._singular_formatters[model_index].get_formatted(
            with_par_values=with_par_values,
            n_significant_digits=n_significant_digits,
            format_as_latex=format_as_latex,
            with_expression=with_expression
        )
