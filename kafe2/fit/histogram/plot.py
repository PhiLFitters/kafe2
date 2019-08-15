import numpy as np

from .._base import PlotContainerBase, PlotContainerException, PlotFigureBase
from .._aux import step_fill_between
from . import HistFit

__all__ = ["HistPlot", "HistPlotContainer"]

class HistPlotContainerException(PlotContainerException):
    pass

class HistPlotContainer(PlotContainerBase):
    FIT_TYPE = HistFit

    def __init__(self, hist_fit_object, n_plot_points_model_density=100):
        """
        Construct an :py:obj:`HistPlotContainer` for a :py:obj:`~kafe2.fit.histogram.HistFit` object:

        :param fit_object: an :py:obj:`~kafe2.fit.histogram.HistFit` object
        :param n_plot_points_model_density: number of plot points to use for plotting the model density
        """
        super(HistPlotContainer, self).__init__(fit_object=hist_fit_object)
        self._n_plot_points_model_density = n_plot_points_model_density

    # -- private methods

    @property
    def data_x(self):
        """data x values"""
        return self._fitter._data_container.bin_centers

    @property
    def data_y(self):
        """data y values"""
        return self._fitter.data

    @property
    def data_xerr(self):
        """x error bars for data (actually used to represent the bins)"""
        return self._fitter._data_container.bin_widths*0.5

    @property
    def data_yerr(self):
        """y error bars for data: total data uncertainty"""
        return self._fitter.data_error

    @property
    def model_x(self):
        """model prediction x values"""
        return self.data_x

    @property
    def model_y(self):
        """model prediction y values"""
        return self._fitter.model

    @property
    def model_xerr(self):
        """x error bars for model (actually used to represent the bins)"""
        return self._fitter._param_model.bin_widths*0.5

    @property
    def model_yerr(self):
        """y error bars for model: ``None`` for :py:obj:`HistPlotContainer`"""
        return None #self._fitter.model_error

    @property
    def model_density_x(self):
        """x support points for model density plot"""
        _xmin, _xmax = self.x_range
        return np.linspace(_xmin, _xmax, self._n_plot_points_model_density)

    @property
    def model_density_y(self):
        """value of model density at the support points"""
        _hist_cont = self._fitter._data_container
        _mean_bin_size = float(_hist_cont.high - _hist_cont.low)/_hist_cont.size
        _factor = _hist_cont.n_entries * _mean_bin_size
        return _factor * self._fitter.eval_model_function_density(x=self.model_density_x)

    @property
    def x_range(self):
        """x plot range (the histogram bin range)"""
        return self._fitter._data_container.bin_range

    @property
    def y_range(self):
        """y plot range: ``None`` for :py:obj:`IndexedPlotContainer`"""
        return None  # no fixed range

    # public methods

    def plot_data(self, target_axis, **kwargs):
        """
        Plot the measurement data to a specified ``matplotlib`` ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` method ``errorbar``
        :return: plot handle(s)
        """
        _yerr = np.sqrt(
            self.data_yerr ** 2 + self._fitter._cost_function.get_uncertainty_gaussian_approximation(self.data_y) ** 2
        )
        return target_axis.errorbar(self.data_x,
                                    self.data_y,
                                    xerr=self.data_xerr,
                                    yerr=_yerr,
                                    **kwargs)

    def plot_model(self, target_axis, **kwargs):
        """
        Plot the model predictions to a specified matplotlib ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` method ``bar``
        :return: plot handle(s)
        """
        #_pad = kwargs.pop('bar_width_pad')
        _sf = kwargs.pop('bar_width_scale_factor')
        return target_axis.bar(
                             left=self.model_x,
                             align='center',
                             height=self.model_y,
                             width=self.model_xerr*2.0 * _sf,
                             bottom=None,
                             **kwargs
                             )

    def plot_model_density(self, target_axis, **kwargs):
        """
        Plot the model density to a specified ``matplotlib`` ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` method ``plot``
        :return: plot handle(s)
        """
        # TODO: how to handle/display "error" on the model density?
        return target_axis.plot(self.model_density_x,
                                self.model_density_y,
                                **kwargs)


class HistPlot(PlotFigureBase):

    PLOT_CONTAINER_TYPE = HistPlotContainer
    PLOT_STYLE_CONFIG_DATA_TYPE = 'histogram'

    PLOT_SUBPLOT_TYPES = PlotFigureBase.PLOT_SUBPLOT_TYPES.copy()  # don't change original class variable
    PLOT_SUBPLOT_TYPES['model_density'] = dict(
        plot_container_method='plot_model_density',
    )

    def __init__(self, fit_objects):
        super(HistPlot, self).__init__(fit_objects=fit_objects)
