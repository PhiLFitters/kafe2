class MultiPlot(object):
    """Class for making plots from multi fits"""

    def __init__(self, fit_objects, separate_plots=True):
        """
        Parent constructor for multi plots

        :param fit_objects: the fit objects for which plots should be created
        :type fit_objects: specified by subclass
        :param separate_plots: if ``True``, will create separate plots for each model
                               within each fit object, if ``False`` will create one plot
                               for each fit object
        :type separate_plots: bool
        """
        try:
            iter(fit_objects)
        except TypeError:
            fit_objects = [fit_objects]
        if separate_plots:
            for _fit_object in fit_objects:
                self._underlying_plots = [_underlying_fit.generate_plot()
                                          for _underlying_fit in _fit_object.underlying_fits]
        else:
            raise NotImplementedError()

    @property
    def underlying_plots(self):
        return self._underlying_plots

    # -- public methods

    def plot(self):
        for _underlying_plot in self._underlying_plots:
            _underlying_plot.plot()

    def show_fit_info_box(self, asymmetric_parameter_errors=False, format_as_latex=True):
        """Render text information about each plot on the figure.

        :param format_as_latex: if ``True``, the infobox text will be formatted as a LaTeX string
        :type format_as_latex: bool
        :param asymmetric_parameter_errors: if ``True``, use two different parameter errors for up/down directions
        :type asymmetric_parameter_errors: bool
        """
        for _underlying_plot in self._underlying_plots:
            _underlying_plot.show_fit_info_box(
                asymmetric_parameter_errors=asymmetric_parameter_errors, format_as_latex=format_as_latex)
