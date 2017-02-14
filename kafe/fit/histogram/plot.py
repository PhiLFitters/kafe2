import numpy as np

from .._base import PlotContainerBase, PlotFigureBase
from .._aux import step_fill_between
from . import HistFit


class HistPlotContainer(PlotContainerBase):
    FIT_TYPE = HistFit

    def __init__(self, hist_fit_object):
        super(HistPlotContainer, self).__init__(fit_object=hist_fit_object)

    # -- private methods

    @property
    def data_x(self):
        return self._fitter._data_container.bin_centers

    @property
    def data_y(self):
        return self._fitter.data

    @property
    def data_xerr(self):
        return self._fitter._data_container.bin_widths*0.5

    @property
    def data_yerr(self):
        return self._fitter.data_error

    @property
    def model_x(self):
        return self.data_x

    @property
    def model_y(self):
        return self._fitter.model

    @property
    def model_xerr(self):
        return self._fitter._param_model.bin_widths*0.5

    @property
    def model_yerr(self):
        return None #self._fitter.model_error

    @property
    def x_range(self):
        return self._fitter._data_container.bin_range

    @property
    def y_range(self):
        return None  # no fixed range

    # public methods

    def plot_data(self, target_axis, **kwargs):
        return target_axis.errorbar(self.data_x,
                                    self.data_y,
                                    xerr=self.data_xerr,
                                    yerr=self.data_yerr,
                                    **kwargs)

    def plot_model(self, target_axis, **kwargs):
        _pad = kwargs.pop('bar_width_pad')
        return target_axis.bar(
                             left=self.model_x - self.model_xerr + _pad/2.,
                             height=self.model_y,
                             width=self.model_xerr*2.0 - _pad,
                             bottom=None,
                             **kwargs
                             )


class HistPlot(PlotFigureBase):

    PLOT_CONTAINER_TYPE = HistPlotContainer

    PLOT_TYPE_DEFAULT_CONFIGS = PlotFigureBase.PLOT_TYPE_DEFAULT_CONFIGS.copy()  # don't change original class variable
    PLOT_TYPE_DEFAULT_CONFIGS['model'] = dict(
        plot_container_method='plot_model',
        plot_container_method_static_kwargs=dict(
            bar_width_pad = 0.05,
            alpha=0.5,
            linestyle='-',
            label='model %(subplot_id)s',
            edgecolor='none',
            linewidth=0,
            zorder=-100
        ),
        plot_container_method_kwargs_cycler_args=tuple((
            dict(
                facecolor=('#f59a96', '#a6cee3', '#b0dd8b', '#fdbe6f', '#cbb1d2', '#b39c9a'),
            ),))
    )

    def __init__(self, fit_objects):
        super(HistPlot, self).__init__(fit_objects=fit_objects)
