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
    def plot_data_x(self):
        return np.arange(self._fitter.data_size)

    @property
    def plot_data_y(self):
        return self._fitter.data

    @property
    def plot_data_xerr(self):
        return None

    @property
    def plot_data_yerr(self):
        return self._fitter.data_error

    @property
    def plot_model_x(self):
        return self.plot_data_x

    @property
    def plot_model_y(self):
        return self._fitter.model

    @property
    def plot_model_xerr(self):
        return 0.5

    @property
    def plot_model_yerr(self):
        return None #self._fitter.model_error

    @property
    def plot_range_x(self):
        return (-0.5, self._fitter.data_size-0.5)

    @property
    def plot_range_y(self):
        return None  # no fixed range

    # public methods

    def plot_data(self, target_axis, **kwargs):
        if self._fitter.has_data_errors:
            return target_axis.errorbar(self.plot_data_x,
                                 self.plot_data_y,
                                 xerr=self.plot_data_xerr,
                                 yerr=self.plot_data_yerr,
                                 **kwargs)
        else:
            return target_axis.plot(self.plot_data_x,
                             self.plot_data_y,
                             **kwargs)

    def plot_model(self, target_axis, **kwargs):
        return step_fill_between(target_axis,
                          self.plot_model_x,
                          self.plot_model_y,
                          xerr=self.plot_model_xerr,
                          yerr=self.plot_model_yerr,
                          draw_central_value=True,
                          **kwargs
                          )


class HistPlot(PlotFigureBase):

    PLOT_CONTAINER_TYPE = HistPlotContainer

    def __init__(self, fit_objects):
        super(HistPlot, self).__init__(fit_objects=fit_objects)
