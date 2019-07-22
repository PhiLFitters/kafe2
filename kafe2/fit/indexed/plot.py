import numpy as np

from .._base import PlotContainerBase, PlotContainerException, PlotFigureBase
from .._aux import step_fill_between
from . import IndexedFit

__all__ = ["IndexedPlot", "IndexedPlotContainer"]

class IndexedPlotContainerException(PlotContainerException):
    pass

class IndexedPlotContainer(PlotContainerBase):
    FIT_TYPE = IndexedFit

    def __init__(self, indexed_fit_object):
        """
        Construct an :py:obj:`IndexedPlotContainer` for a :py:obj:`~kafe2.fit.indexed.IndexedFit` object:

        :param fit_object: an :py:obj:`~kafe2.fit.indexed.IndexedFit` object
        """
        super(IndexedPlotContainer, self).__init__(fit_object=indexed_fit_object)

    @property
    def data_x(self):
        """data x values"""
        return np.arange(self._fitter.data_size)

    @property
    def data_y(self):
        """data y values"""
        return self._fitter.data

    @property
    def data_xerr(self):
        """x error bars for data: ``None`` for :py:obj:`IndexedPlotContainer`"""
        return None

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
        """x error bars for model (actually used to represent the horizontal step length)"""
        return 0.5

    @property
    def model_yerr(self):
        """y error bars for model: ``None`` for :py:obj:`IndexedPlotContainer`"""
        return None #self._fitter.model_error

    @property
    def x_range(self):
        """x plot range: (-0.5, N-0.5) for :py:obj:`IndexedPlotContainer`"""
        return (-0.5, self._fitter.data_size-0.5)

    @property
    def y_range(self):
        """y plot range: ``None`` for :py:obj:`IndexedPlotContainer`"""
        return None  # no fixed range

    # public methods

    def plot_data(self, target_axis, **kwargs):
        """
        Plot the measurement data to a specified ``matplotlib`` ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` methods ``errorbar`` or ``plot``
        :return: plot handle(s)
        """
        if self._fitter.has_errors:
            _yerr = np.sqrt(
                self.data_yerr ** 2
                + self._fitter._cost_function.get_uncertainty_gaussian_approximation(self.data_y) ** 2
            )
            return target_axis.errorbar(self.data_x,
                                        self.data_y,
                                        xerr=self.data_xerr,
                                        yerr=_yerr,
                                        **kwargs)
        else:
            _yerr = self._fitter._cost_function.get_uncertainty_gaussian_approximation(self.data_y)
            if np.all(_yerr == 0):
                return target_axis.plot(self.data_x,
                                        self.data_y,
                                        **kwargs)
            else:
                return target_axis.errorbar(self.data_x,
                                            self.data_y,
                                            yerr=_yerr,
                                            **kwargs)

    def plot_model(self, target_axis, **kwargs):
        """
        Plot the model predictions to a specified matplotlib ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the :py:func:`~kafe2._aux.step_fill_between` method
        :return: plot handle(s)
        """
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
    PLOT_STYLE_CONFIG_DATA_TYPE = 'indexed'

    def __init__(self, fit_objects):
        super(IndexedPlot, self).__init__(fit_objects=fit_objects)
