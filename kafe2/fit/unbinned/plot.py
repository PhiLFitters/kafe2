import numpy as np
from matplotlib.collections import LineCollection
from collections import OrderedDict

from ...config import kc
from .._base import PlotContainerBase, PlotContainerException, PlotFigureBase, kc_plot_style
from .._aux import step_fill_between
from .fit import UnbinnedFit


__all__ = ["UnbinnedPlot", "UnbinnedPlotContainer"]


class UnbinnedPlotContainerException(PlotContainerException):
    pass


class UnbinnedPlotContainer(PlotContainerBase):
    FIT_TYPE = UnbinnedFit

    def __init__(self, unbinned_fit_object, n_plot_points_model=100):
        """
        Construc an :py:obj:`UnbinnedPlotContainer` for a :py:obj:`~kafe2.fit.unbinned.UnbinnedFit` object:
        :param unbinned_fit_object: an :py:obj:`~kafe2.fit.unbinned.UnbinnedFit` object
        :param n_plot_points_model: Number of data points for plotting the model
        :type n_plot_points_model: int
        """
        super(UnbinnedPlotContainer, self).__init__(fit_object=unbinned_fit_object)
        self._n_plot_points_model = n_plot_points_model
        self._plot_range_x = None
        self._plot_range_y = None

    # -- private methods

    def _compute_plot_range_x(self, pad_coeff=1.1, additional_pad=None):
        if additional_pad is None:
            additional_pad = (0, 0)
        _xmin, _xmax = self._fitter.data_range
        _w = _xmax - _xmin
        self._plot_range_x = (
            0.5 * (_xmin + _xmax - _w * pad_coeff) - additional_pad[0],
            0.5 * (_xmin + _xmax + _w * pad_coeff) + additional_pad[1]
        )

    def _compute_plot_range_y(self, pad_coeff=1.1, additional_pad=None):
        if additional_pad is None:
            additional_pad = (0, 0)
        model = self.model_y
        _ymax = np.amax(model)
        # no negative densities possible, ymin has to be 0, fo data density to show
        _ymin = 0
        _w = _ymax - _ymin
        self._plot_range_y = (_ymin, 0.5 * (_ymin + _ymax + _w * pad_coeff) + additional_pad[1])

    @property
    def data_x(self):
        """
        The 'x' coordinates of the data (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        return self._fitter.data

    @property
    def data_y(self):
        """
        The 'y' coordinates of the data (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        raise UnbinnedPlotContainerException("There's no y-data in the unbinned container")

    @property
    def data_xerr(self):
        """
        The magnitude of the data 'x' error bars (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        return self._fitter.data.err

    @property
    def data_yerr(self):
        raise UnbinnedPlotContainerException("There's no y-data in the unbinned container, hence no y-error")

    @property
    def model_x(self):
        """x support values for model function"""
        _xmin, _xmax = self.x_range
        return np.linspace(_xmin, _xmax, self._n_plot_points_model)

    @property
    def model_y(self):
        """
        The 'y' coordinates of the model (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        return self._fitter.eval_model_function(x=self.model_x)

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
        """x plot range"""
        if self._plot_range_x is None:
            self._compute_plot_range_x()
        return self._plot_range_x

    @property
    def y_range(self):
        """y plot range"""
        if self._plot_range_y is None:
            self._compute_plot_range_y()
        return self._plot_range_y

    # public methods
    def plot_data(self, target_axis, **kwargs):
        pass

    def plot_data_density(self, target_axis, height=0.05, **kwargs):
        """
        Method called by the main plot routine to plot the data points to a specified matplotlib ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :return: plot handle(s)
        """
        data = self.data_x
        xy_pairs = np.column_stack([np.repeat(data, 2), np.tile([0, height], len(data))])
        lines = xy_pairs.reshape([len(data), 2, 2])
        line_segments = LineCollection(lines, **kwargs)
        return target_axis.add_collection(line_segments)

    def plot_model(self, target_axis, **kwargs):
        """
        Method called by the main plot routine to plot the model to a specified matplotlib ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :return: plot handle(s)
        """
        """
        Plot the model predictions to a specified matplotlib ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the :py:func:`~kafe2._aux.step_fill_between` method
        :return: plot handle(s)
        """
        return target_axis.plot(self.model_x, self.model_y, **kwargs)


class UnbinnedPlot(PlotFigureBase):

    PLOT_CONTAINER_TYPE = UnbinnedPlotContainer
    PLOT_STYLE_CONFIG_DATA_TYPE = 'unbinned'
    PLOT_SUBPLOT_TYPES = OrderedDict()
    PLOT_SUBPLOT_TYPES.update(
        data_density=dict(
            plot_container_method='plot_data_density',
        ),
        model=dict(
            plot_container_method='plot_model',
        ),
    )

    def __init__(self, fit_objects):
        super(UnbinnedPlot, self).__init__(fit_objects=fit_objects)
