import numpy as np
from matplotlib.collections import LineCollection
import warnings

from .._base import PlotAdapterBase, PlotAdapterException


__all__ = ["UnbinnedPlotAdapter"]


class UnbinnedPlotAdapterException(PlotAdapterException):
    pass


class UnbinnedPlotAdapter(PlotAdapterBase):

    PLOT_STYLE_CONFIG_DATA_TYPE = 'unbinned'

    PLOT_SUBPLOT_TYPES = dict(
        PlotAdapterBase.PLOT_SUBPLOT_TYPES,
        model_line=dict(
            plot_adapter_method='plot_model_line',
            target_axes='main'
        )
    )
    del PLOT_SUBPLOT_TYPES['model']  # don't show "model" points

    def __init__(self, unbinned_fit_object):
        """
        Construct an :py:obj:`UnbinnedPlotAdapter` for a :py:obj:`~kafe2.fit.unbinned.UnbinnedFit` object:
        :param unbinned_fit_object: an :py:obj:`~kafe2.fit.unbinned.UnbinnedFit` object
        """
        super(UnbinnedPlotAdapter, self).__init__(fit_object=unbinned_fit_object)
        self.n_plot_points = 100 if len(self.data_x) < 100 else len(self.data_x)
        self.x_range = self._compute_plot_range_x()
        self.y_range = self._compute_plot_range_y()

    # -- private methods

    def _compute_plot_range_x(self, pad_coeff=1.1, additional_pad=None):
        if additional_pad is None:
            additional_pad = (0, 0)
        _xmin, _xmax = self._fit.data_range
        _w = _xmax - _xmin
        return (0.5 * (_xmin + _xmax - _w * pad_coeff) - additional_pad[0],
                0.5 * (_xmin + _xmax + _w * pad_coeff) + additional_pad[1])

    def _compute_plot_range_y(self, pad_coeff=1.1, additional_pad=None):
        if additional_pad is None:
            additional_pad = (0, 0)
        model = self.model_y
        _ymax = np.nanmax(model[model != np.inf])
        # no negative densities possible, ymin has to be 0, for data density to show
        _ymin = 0
        _w = _ymax - _ymin
        return _ymin, 0.5 * (_ymin + _ymax + _w * pad_coeff) + additional_pad[1]

    @property
    def data_x(self):
        """
        The 'x' coordinates of the data (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        return self._fit.data

    @property
    def data_y(self):
        """
        The 'y' coordinates of the data (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        raise UnbinnedPlotAdapterException("There's no y-data in the unbinned container")

    @property
    def data_xerr(self):
        """
        The magnitude of the data 'x' error bars (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        return self._fit.data.err

    @property
    def data_yerr(self):
        raise UnbinnedPlotAdapterException("There's no y-data in the unbinned container, hence no y-error")

    @property
    def model_x(self):
        """x support values for model function"""
        return self.data_x

    @property
    def model_y(self):
        """
        The 'y' coordinates of the model (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        return self._fit.model
        #raise UnbinnedPlotAdapterException("There's no y-model in the unbinned container")

    @property
    def model_line_x(self):
        """x support values for model function"""
        _xmin, _xmax = self.x_range
        return np.linspace(_xmin, _xmax, self.n_plot_points)

    @property
    def model_line_y(self):
        """
        The 'y' coordinates of the model (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        return self._fit.eval_model_function(x=self.model_line_x)

    @property
    def model_xerr(self):
        """
        The magnitude of the model 'x' error bars (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        # Fixme: No static value
        return 0.5

    @property
    def model_yerr(self):
        """
        The magnitude of the model 'y' error bars (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        return None

    @property
    def x_range(self):
        return self._x_range

    @x_range.setter
    def x_range(self, x_range):
        if x_range is None:  # needed for model_line_x
            warnings.warn("x_range can't be None, using defaults")
            x_range = self._compute_plot_range_x()
        self._x_range = x_range

    @property
    def y_range(self):
        return self._y_range

    @y_range.setter
    def y_range(self, y_range):
        if y_range is None:  # needed for plot_data
            warnings.warn("y_range can't be None, using defaults")
            y_range = self._compute_plot_range_y()
        self._y_range = y_range

    # public methods
    def plot_data(self, target_axes, height=None, **kwargs):
        """
        Method called by the main plot routine to plot the data points to a specified matplotlib ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param height: The height of the lines which represent the density
        :type height: float
        :return: plot handle(s)
        """
        kwargs.pop('marker', None)  # pop marker keyword, as LineCollection doesn't support it

        if height is None:
            height = self.y_range[1]/10  # set height to 1/10th of the max height of the model

        data = self.data_x
        xy_pairs = np.column_stack([np.repeat(data, 2), np.tile([0, height], len(data))])
        lines = xy_pairs.reshape([len(data), 2, 2])
        line_segments = LineCollection(lines, **kwargs)
        return target_axes.add_collection(line_segments)

    def plot_model(self, target_axes, **kwargs):
        """
        Method called by the main plot routine to plot the model to a specified matplotlib ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the :py:func:`~kafe2._aux.step_fill_between` method
        :return: plot handle(s)
        """
        return target_axes.plot(self.model_x, self.model_y, **kwargs)

    def plot_ratio(self, target_axes, **kwargs):
        """
        Plot the data/model ratio to a specified ``matplotlib`` ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` methods ``errorbar`` or ``plot``
        :return: plot handle(s)
        """
        raise NotImplementedError("Data/model ratio cannot be plotted for unbinned fits.")

    def plot_model_line(self, target_axes, **kwargs):
        """
        Method called by the main plot routine to plot the model to a specified matplotlib ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the :py:func:`~kafe2._aux.step_fill_between` method
        :return: plot handle(s)
        """
        return target_axes.plot(self.model_line_x, self.model_line_y, **kwargs)
