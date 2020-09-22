import numpy as np

from ...config import kc
from .._base import PlotAdapterBase, PlotAdapterException, Plot, kc_plot_style
from .._aux import step_fill_between


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
    del PLOT_SUBPLOT_TYPES['model']  # don't plot model xy points

    def __init__(self, xy_fit_object):
        """
        Construct an :py:obj:`XYPlotContainer` for a :py:obj:`~kafe2.fit.xy.XYFit` object:

        :param xy_fit_object: an :py:obj:`~kafe2.fit.xy.XYFit` object
        :type xy_fit_object: :py:class:`~kafe2.fit.xy.XYFit`
        """
        super(XYPlotAdapter, self).__init__(fit_object=xy_fit_object)
        self.n_plot_points = 100 if len(self.data_x) < 50 else 2*len(self.data_x)
        self.x_range = self._compute_plot_range_x()

    # -- private methods

    def _compute_plot_range_x(self, pad_coeff=1.1, additional_pad=None):
        if additional_pad is None:
            additional_pad = (0, 0)
        _xmin, _xmax = self._fit.x_range
        _w = _xmax - _xmin
        return (0.5 * (_xmin + _xmax - _w * pad_coeff) - additional_pad[0],
                0.5 * (_xmin + _xmax + _w * pad_coeff) + additional_pad[1])

    # -- public properties

    @property
    def data_x(self):
        """data x values"""
        return self._fit.x_data

    @property
    def data_y(self):
        """data y values"""
        return self._fit.y_data

    @property
    def data_xerr(self):
        """x error bars for data: total x uncertainty"""
        return self._fit.x_total_error

    @property
    def data_yerr(self):
        """y error bars for data: total y uncertainty"""
        return self._fit.y_total_error

    @property
    def model_x(self):
        """model x values"""
        return self._fit.x_model

    @property
    def model_y(self):
        """model y values"""
        return self._fit.y_model

    @property
    def model_xerr(self):
        """x error bars for model: ``None`` for :py:obj:`IndexedPlotContainer`"""
        return self._fit.x_model_error

    @property
    def model_yerr(self):
        """y error bars for model: total model uncertainty"""
        return self._fit.y_model_error

    @property
    def model_line_x(self):
        """x support values for model function"""
        _xmin, _xmax = self.x_range
        return np.linspace(_xmin, _xmax, self.n_plot_points)

    @property
    def model_line_y(self):
        """y values at support points for model function"""
        return self._fit.eval_model_function(x=self.model_line_x)

    @property
    def y_error_band(self):
        """one-dimensional array representing the uncertainty band around the model function"""
        if not self._fit.did_fit:
            raise XYPlotAdapterException('Cannot calculate an error band without first performing a fit.')
        _band_x = self.model_line_x
        if self._fit.parameter_cov_mat is None:
            return np.zeros_like(_band_x)

        _f_deriv_by_params = self._fit.eval_model_function_derivative_by_parameters(
            x=_band_x,
            model_parameters=self._fit.parameter_values)
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[par_idx][x_idx]

        _f_deriv_by_params = _f_deriv_by_params.T
        # here: df/dp[par_idx]|x=x[x_idx] = _f_deriv_by_params[x_idx][par_idx]

        _band_y = np.zeros_like(_band_x)
        _n_poi = len(self._fit.parameter_values)
        for _x_idx, _x_val in enumerate(_band_x):
            _p_res = _f_deriv_by_params[_x_idx]
            _band_y[_x_idx] = _p_res.dot(self._fit.parameter_cov_mat[:_n_poi, :_n_poi]).dot(_p_res)

        return np.sqrt(_band_y)

    # public methods

    def plot_data(self, target_axes, error_contributions=('data',), **kwargs):
        """
        Plot the measurement data to a specified ``matplotlib`` ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` methods ``errorbar`` or ``plot``
        :return: plot handle(s)
        """

        _yerr = self._get_total_error(error_contributions)

        return target_axes.errorbar(self.data_x,
                                    self.data_y,
                                    xerr=self.data_xerr,
                                    yerr=_yerr,
                                    **kwargs)

    def plot_model(self, target_axes, error_contributions=('model',), **kwargs):
        """
        Plot the measurement data to a specified ``matplotlib`` ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` methods ``errorbar`` or ``plot``
        :return: plot handle(s)
        """

        _yerr = self._get_total_error(error_contributions)

        return target_axes.errorbar(self.model_x,
                                    self.model_y,
                                    xerr=self.data_xerr,
                                    yerr=_yerr,
                                    **kwargs)

    def plot_model_line(self, target_axes, **kwargs):
        """
        Plot the model function to a specified matplotlib ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` ``plot`` method
        :return: plot handle(s)
        """
        # TODO: how to handle 'data' errors and 'model' errors?
        return target_axes.plot(self.model_line_x,
                                self.model_line_y,
                                **kwargs)

    def plot_model_error_band(self, target_axes, **kwargs):
        """
        Plot an error band around the model model function.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` ``fill_between`` method
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
        """
        Plot the data/model ratio to a specified ``matplotlib`` ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` methods ``errorbar`` or ``plot``
        :return: plot handle(s)
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
        """
        Plot model error band around the data/model ratio to a specified ``matplotlib`` ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :param kwargs: keyword arguments accepted by the ``matplotlib`` methods ``errorbar`` or ``plot``
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
