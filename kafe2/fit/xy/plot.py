import numpy  # help IDEs with type-hinting inside docstrings

from .._base import PlotAdapterBase, PlotAdapterException
from .._aux import add_pad_to_range
np = numpy


__all__ = ["XYPlotAdapter"]


class XYPlotAdapterException(PlotAdapterException):
    pass


class XYPlotAdapter(PlotAdapterBase):

    PLOT_STYLE_CONFIG_DATA_TYPE = 'xy'
    PLOT_SUBPLOT_TYPES = dict(
        PlotAdapterBase.PLOT_SUBPLOT_TYPES,
        model_line=dict(
            plot_adapter_method='plot_model_line',
            target_axes='main'
        ),
        model_error_band=dict(
            plot_adapter_method='plot_model_error_band',
            target_axes='main'
        ),
        ratio_error_band=dict(
            plot_style_as='model_error_band',
            plot_adapter_method='plot_ratio_error_band',
            target_axes='ratio'
        ),
    )

    AVAILABLE_X_SCALES = ('linear', 'log')

    def __init__(self, xy_fit_object):
        """Construct an :py:obj:`XYPlotContainer` for a :py:obj:`~.XYFit` object:

        :param kafe2.XYFit xy_fit_object: The :py:obj:`~.XYFit` object handled by this plot
            adapter.
        """
        self._fit = xy_fit_object  # needed for type hinting to work correctly
        super(XYPlotAdapter, self).__init__(fit_object=xy_fit_object)
        self.n_plot_points = 100 if len(self.data_x) < 25 else 4*len(self.data_x)
        self.x_range = add_pad_to_range(self._fit.x_range, scale=self.x_scale)

    # -- public properties

    @property
    def data_x(self):
        return self._fit.x_data

    @property
    def data_y(self):
        return self._fit.y_data

    @property
    def data_xerr(self):
        return self._fit.x_total_error

    @property
    def data_yerr(self):
        return self._fit.y_total_error

    @property
    def model_x(self):
        return self._fit.x_model

    @property
    def model_y(self):
        return self._fit.y_model

    @property
    def model_xerr(self):
        return self._fit.x_model_error

    @property
    def model_yerr(self):
        return self._fit.y_model_error

    @PlotAdapterBase.x_scale.setter
    def x_scale(self, scale):
        update_xrange = self.x_range == add_pad_to_range(self._fit.x_range, scale=self.x_scale)
        PlotAdapterBase.x_scale.fset(self, scale)  # use parent setter
        if update_xrange:
            self.x_range = add_pad_to_range(self._fit.x_range, scale=self.x_scale)

    @property
    def model_line_x(self):
        """*x* support values for model function. Adapts spacing to :py:obj:`.x_scale`.

        :rtype: numpy.ndarray[float]
        """
        _xmin, _xmax = self.x_range
        if self.x_scale == 'linear':
            return np.linspace(_xmin, _xmax, self.n_plot_points)
        if self.x_scale == 'log':
            try:
                return np.geomspace(_xmin, _xmax, self.n_plot_points)
            except ValueError:
                raise XYPlotAdapterException("Support point calculation failed. "
                                             "The plot range can't include 0 when using log scale.")
        raise XYPlotAdapterException("x_range has to be one of {}. Found {} instead.".format(
            self.AVAILABLE_X_SCALES, self.x_scale))

    @property
    def model_line_y(self):
        """*y* values of the model function at the support points :py:obj:`.model_line_x`.

        :rtype: numpy.ndarray[float]
        """
        return self._fit.eval_model_function(x=self.model_line_x)

    @property
    def y_error_band(self):
        """1D array representing the uncertainty band around the model function at the support
        points :py:obj:`.model_line_x`.

        :rtype: numpy.ndarray[float]
        """
        return self._fit.error_band(self.model_line_x)

    # public methods

    def plot_data(self, target_axes, error_contributions=('data',), **kwargs):
        """Plot the measurement data to a specified :py:obj:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:obj:`matplotlib` axes used for plotting.
        :param error_contributions: Which error contributions to include when plotting the data.
            Can either be ``data``, ``'model'`` or both.
        :type error_contributions: str or Tuple[str]
        :param dict kwargs: Keyword arguments accepted by :py:obj:`matplotlib.pyplot.errorbar`.
        :return: plot handle(s)
        """

        _yerr = self._get_total_error(error_contributions)

        return target_axes.errorbar(self.data_x,
                                    self.data_y,
                                    xerr=self.data_xerr,
                                    yerr=_yerr,
                                    **kwargs)

    def plot_model(self, target_axes, error_contributions=('model',), **kwargs):
        """Plot the model data to a specified :py:obj:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:obj:`matplotlib` axes used for plotting.
        :param error_contributions: Which error contributions to include when plotting the model.
        :type error_contributions: str or Tuple[str]
            Can either be ``data``, ``'model'`` or both.
        :param dict kwargs: Keyword arguments accepted by :py:obj:`matplotlib.pyplot.errorbar`.
        :return: plot handle(s)
        """

        _yerr = self._get_total_error(error_contributions)

        return target_axes.errorbar(self.model_x,
                                    self.model_y,
                                    xerr=self.data_xerr,
                                    yerr=_yerr,
                                    **kwargs)

    def plot_model_line(self, target_axes, **kwargs):
        """Plot the model function to a specified :py:obj:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:obj:`matplotlib` axes used for plotting.
        :param dict kwargs: Keyword arguments accepted by :py:obj:`matplotlib.pyplot.plot`.
        :return: plot handle(s)
        """
        # TODO: how to handle 'data' errors and 'model' errors?
        return target_axes.plot(self.model_line_x,
                                self.model_line_y,
                                **kwargs)

    def plot_model_error_band(self, target_axes, **kwargs):
        """Plot an error band around the model model function.

        :param matplotlib.axes.Axes target_axes: The :py:obj:`matplotlib` axes used for plotting.
        :param dict kwargs: Keyword arguments accepted by :py:obj:`matplotlib.pyplot.fill_between`.
        :return: plot handle(s)
        """
        if self._fit.did_fit and (self._fit.has_errors or not self._fit._cost_function.needs_errors):
            _band_y = self.y_error_band
            _y = self.model_line_y
            return target_axes.fill_between(
                self.model_line_x,
                _y - _band_y, _y + _band_y,
                **kwargs)
        return None  # don't plot error band if fitter input data has no errors...

    def plot_ratio(self, target_axes, error_contributions=('data',), **kwargs):
        """Plot the data/model ratio to a specified :py:obj:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:obj:`matplotlib` axes used for plotting.
        :param error_contributions: Which error contributions to include when plotting the data.
            Can either be ``data``, ``'model'`` or both.
        :type error_contributions: str or Tuple[str]
        :param dict kwargs: Keyword arguments accepted by :py:obj:`matplotlib.pyplot.errorbar`.
        :return: plot handle(s)        :return: error id
        """

        _yerr = self._get_total_error(error_contributions)
        if _yerr is not None:
            _yerr /= self.model_y

        # TODO: how to handle case when x and y error/model differ?
        return target_axes.errorbar(
            self.data_x,
            self.data_y / self.model_y,
            xerr=self.data_xerr,
            yerr=_yerr,
            **kwargs
        )

    def plot_ratio_error_band(self, target_axes, **kwargs):
        """Plot model error band around the data/model ratio to specified
        :py:obj:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:obj:`matplotlib` axes used for plotting.
        :param dict kwargs: Keyword arguments accepted by :py:obj:`matplotlib.pyplot.fill_between`.
        :return: plot handle(s)
        """
        if self._fit.did_fit and (self._fit.has_errors or not self._fit._cost_function.needs_errors):
            _band_y = self.y_error_band
            _y = self.model_line_y
            return target_axes.fill_between(
                self.model_line_x,
                1 - _band_y/_y, 1 + _band_y/_y,
                **kwargs)
        return None  # don't plot error band if fitter input data has no errors...

    def update_plot_kwargs(self, plot_type, plot_kwargs):
        # update ratio kwargs as well, when corresponding plot_types are updated
        # can be overwritten by the user by explicitly setting the ratio kwargs last
        if plot_type == 'data':
            super(XYPlotAdapter, self).update_plot_kwargs(plot_type='ratio', plot_kwargs=plot_kwargs)
        elif plot_type == 'model_error_band':
            super(XYPlotAdapter, self).update_plot_kwargs(plot_type='ratio_error_band', plot_kwargs=plot_kwargs)
        super(XYPlotAdapter, self).update_plot_kwargs(plot_type=plot_type, plot_kwargs=plot_kwargs)
