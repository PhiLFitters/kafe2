import numpy as np

from .._base import PlotContainerBase, PlotFigureBase
from .._aux import step_fill_between
from . import IndexedFit


class IndexedPlotContainer(PlotContainerBase):
    FIT_TYPE = IndexedFit

    def __init__(self, indexed_fit_object):
        super(IndexedPlotContainer, self).__init__(fit_object=indexed_fit_object)

    @property
    def data_x(self):
        return np.arange(self._fitter.data_size)

    @property
    def data_y(self):
        return self._fitter.data

    @property
    def data_xerr(self):
        return None

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
        return 0.5

    @property
    def model_yerr(self):
        return None #self._fitter.model_error

    @property
    def x_range(self):
        return (-0.5, self._fitter.data_size-0.5)

    @property
    def y_range(self):
        return None  # no fixed range

    # public methods

    def plot_data(self, target_axis, **kwargs):
        if self._fitter.has_errors:
            return target_axis.errorbar(self.data_x,
                                        self.data_y,
                                        xerr=self.data_xerr,
                                        yerr=self.data_yerr,
                                        **kwargs)
        else:
            return target_axis.plot(self.data_x,
                                    self.data_y,
                                    **kwargs)

    def plot_model(self, target_axis, **kwargs):
        return step_fill_between(target_axis,
                                 self.model_x,
                                 self.model_y,
                                 xerr=self.model_xerr,
                                 yerr=self.model_yerr,
                                 draw_central_value=True,
                                 **kwargs
                                 )


class IndexedPlot(PlotFigureBase):

    PLOT_CONTAINER_TYPE = IndexedPlotContainer

    def __init__(self, fit_objects):
        super(IndexedPlot, self).__init__(fit_objects=fit_objects)
