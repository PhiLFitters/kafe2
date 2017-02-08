import numpy as np

from .._base import FitPlotBase
from .._aux import step_fill_between


class XYFitPlot(FitPlotBase):

    SUBPLOT_CONFIGS_DEFAULT = FitPlotBase.SUBPLOT_CONFIGS_DEFAULT
    SUBPLOT_CONFIGS_DEFAULT['model_error_band'] = dict(
        alpha=0.5,
        linestyle='-',
        label='model error',
        edgecolor='none',
        linewidth=2
    )

    N_PLOT_POINTS = 100

    def __init__(self, parent_fit):
        super(XYFitPlot, self).__init__(parent_fit=parent_fit)

        _sz = len(self._fitter.data)
        self._axes.set_xlim(-0.5, _sz-0.5)

        self._plot_range_x = None

        self.__plot_dicts = self.SUBPLOT_CONFIGS_DEFAULT.copy()

    # -- private methods

    def _compute_plot_range_x(self, pad_coeff=1.1, additional_pad=None):
        if additional_pad is None:
            additional_pad = (0, 0)
        _xmin, _xmax = self._fitter.x_range
        _w = _xmax - _xmin
        self._plot_range_x = (
            0.5 * (_xmin + _xmax - _w * pad_coeff) - additional_pad[0],
            0.5 * (_xmin + _xmax + _w * pad_coeff) + additional_pad[1]
        )

    @property
    def plot_data_x(self):
        return self._fitter.x

    @property
    def plot_data_y(self):
        return self._fitter.y_data

    @property
    def plot_data_xerr(self):
        return self._fitter.x_error

    @property
    def plot_data_yerr(self):
        return self._fitter.y_data_error

    @property
    def plot_model_x(self):
        _xmin, _xmax = self.plot_range_x
        return np.linspace(_xmin, _xmax, self.N_PLOT_POINTS)

    @property
    def plot_model_y(self):
        return self._fitter.eval_model_function(x=self.plot_model_x)

    @property
    def plot_model_xerr(self):
        return None if np.allclose(self._fitter.x_error, 0) else self._fitter.x_error,

    @property
    def plot_model_yerr(self):
        return None if np.allclose(self._fitter.y_data_error, 0) else self._fitter.y_data_error

    @property
    def plot_range_x(self):
        if self._plot_range_x is None:
            self._compute_plot_range_x()
        return self._plot_range_x

    @property
    def plot_range_y(self):
        return None

    # def _plot_data(self, target_axis):
    #     _x = self._fitter.x
    #     _y = self._fitter.y_data
    #     if self._fitter.has_errors:
    #         target_axis.errorbar(_x, _y,
    #                              yerr=None if np.allclose(self._fitter.y_data_error, 0) else self._fitter.y_data_error,
    #                              xerr=None if np.allclose(self._fitter.x_error, 0) else self._fitter.x_error,
    #                              **self.__plot_dicts['data'])
    #     else:
    #         target_axis.plot(self._fitter.data,
    #                          **self.__plot_dicts['data'])

    # def _plot_model(self, target_axis):
    #     _xmin, _xmax = self.plot_x_range
    #     _x = np.linspace(_xmin, _xmax, 100)
    #     _y = self._fitter.eval_model_function(x=_x)
    #     target_axis.plot(_x, _y,
    #                      **self.__plot_dicts['model'])

    def _plot_model_error_band(self, target_axis):
        _xmin, _xmax = self.plot_range_x
        _x = np.linspace(_xmin, _xmax, 100)
        _y = self._fitter._param_model.eval_model_function(x=_x)
        _band_y = self._fitter.y_error_band
        target_axis.fill_between(_x, _y-_band_y, _y+_band_y,
                                 **self.__plot_dicts['model_error_band'])

    def _plot_model(self, target_axis, **kwargs):
        super(XYFitPlot, self)._plot_model(target_axis=target_axis, **kwargs)
        self._plot_model_error_band(target_axis=target_axis)

    # -- public properties

    # -- public methods