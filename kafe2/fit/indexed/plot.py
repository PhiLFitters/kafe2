import numpy as np

from .._aux import step_fill_between
from .._base import PlotAdapterBase

__all__ = ["IndexedPlotAdapter"]


class IndexedPlotAdapter(PlotAdapterBase):
    PLOT_STYLE_CONFIG_DATA_TYPE = "indexed"

    PLOT_SUBPLOT_TYPES = dict(
        PlotAdapterBase.PLOT_SUBPLOT_TYPES,
    )

    def __init__(self, indexed_fit_object, from_container=False):
        """
        Construct an :py:obj:`IndexedPlotContainer` for a :py:obj:`~kafe2.fit.indexed.IndexedFit` object:

        :param fit_object: an :py:obj:`~kafe2.fit.indexed.IndexedFit` object
        :param from_container: Whether indexed_fit_object was created ad-hoc from just a data
            container.
        :type from_container: bool
        """
        super(IndexedPlotAdapter, self).__init__(fit_object=indexed_fit_object, from_container=from_container)
        self.x_range = (-0.5, self._fit.data_size - 0.5)

    @property
    def data_x(self):
        """data x values"""
        return np.arange(self._fit.data_size)

    @property
    def data_y(self):
        """data y values"""
        return self._fit.data

    @property
    def data_xerr(self):
        """x error bars for data: ``None`` for :py:obj:`IndexedPlotContainer`"""
        return None

    @property
    def data_yerr(self):
        """y error bars for data: total uncertainty"""
        return self._fit.total_error

    @property
    def model_x(self):
        """model prediction x values"""
        return self.data_x

    @property
    def model_y(self):
        """model prediction y values"""
        return self._fit.model

    @property
    def model_xerr(self):
        """x error bars for model (actually used to represent the horizontal step length)"""
        return 0.5

    @property
    def model_yerr(self):
        """y error bars for model: ``None`` for :py:obj:`IndexedPlotContainer`"""
        return None  # self.fit.model_error

    # public methods

    def plot_data(self, target_axes, **kwargs):
        """
        Plot the measurement data to a specified ``matplotlib`` ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` methods ``errorbar`` or ``plot``
        :return: plot handle(s)
        """
        if self._fit.has_errors:
            _yerr = np.sqrt(self.data_yerr**2 + self._fit._cost_function.get_uncertainty_gaussian_approximation(self.data_y) ** 2)
            return target_axes.errorbar(self.data_x, self.data_y, xerr=self.data_xerr, yerr=_yerr, **kwargs)
        _yerr = self._fit._cost_function.get_uncertainty_gaussian_approximation(self.data_y)
        if np.all(_yerr == 0):
            return target_axes.plot(self.data_x, self.data_y, **kwargs)
        return target_axes.errorbar(self.data_x, self.data_y, yerr=_yerr, **kwargs)

    def plot_model(self, target_axes, **kwargs):
        """
        Plot the model predictions to a specified matplotlib ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the :py:func:`~kafe2._aux.step_fill_between` method
        :return: plot handle(s)
        """
        return step_fill_between(
            target_axes,
            self.model_x,
            self.model_y,
            xerr=self.model_xerr,
            yerr=self.model_yerr,
            draw_central_value=True,
            continuous=False,
            **kwargs,
        )
