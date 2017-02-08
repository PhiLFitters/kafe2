import abc
import numpy as np
from kafe.config import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


class FitPlotBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    SUBPLOT_CONFIGS_DEFAULT = dict(
        data=dict(
            linestyle='',
            marker='o',
            label='data'
        ),
        model=dict(
            linestyle='-',
            marker='',
            label='model',
            linewidth=2
        ),
        model_error_band=dict(
            alpha=0.5,
            linestyle='-',
            label='model error',
            edgecolor='none',
            linewidth=2
        )
    )

    def __init__(self, parent_fit):
        self._fig = plt.figure()  # defaults from matplotlibrc
        self._gs = gs.GridSpec(nrows=1, ncols=1)
        self._axes = plt.subplot(self._gs[0, 0])
        self._fitter = parent_fit

        self._plot_range_x = None
        self._plot_range_y = None

        self._subplot_kwarg_dicts = self.__class__.SUBPLOT_CONFIGS_DEFAULT.copy()

        self._store_artists = dict()

    # -- private methods

    # TODO: turn into properties

    @abc.abstractmethod
    def _get_plot_data_x(self): pass

    @abc.abstractmethod
    def _get_plot_data_y(self): pass

    @abc.abstractmethod
    def _get_plot_data_xerr(self): pass

    @abc.abstractmethod
    def _get_plot_data_yerr(self): pass

    @abc.abstractmethod
    def _get_plot_model_x(self): pass

    @abc.abstractmethod
    def _get_plot_model_y(self): pass

    @abc.abstractmethod
    def _get_plot_model_xerr(self): pass

    @abc.abstractmethod
    def _get_plot_model_yerr(self): pass

    @abc.abstractmethod
    def _get_plot_range_x(self): pass

    @abc.abstractmethod
    def _get_plot_range_y(self): pass


    def _plot_data(self, target_axis, **kwargs):
        if self._fitter.has_data_errors:
            self._store_artists['data'] = target_axis.errorbar(self._get_plot_data_x(),
                                 self._get_plot_data_y(),
                                 xerr=self._get_plot_data_xerr(),
                                 yerr=self._get_plot_data_yerr(),
                                 **self._subplot_kwarg_dicts['data'])
        else:
            self._store_artists['data'] = target_axis.plot(self._get_plot_data_x(),
                             self._get_plot_data_y(),
                             **self._subplot_kwarg_dicts['data'])

    def _plot_model(self, target_axis, **kwargs):
        if self._fitter.has_model_errors:
            self._store_artists['model'] = target_axis.errorbar(self._get_plot_model_x(),
                                 self._get_plot_model_y(),
                                 xerr=self._get_plot_model_xerr(),
                                 yerr=self._get_plot_model_yerr(),
                                 **self._subplot_kwarg_dicts['model'])
        else:
            self._store_artists['model'] = target_axis.plot(self._get_plot_model_x(),
                             self._get_plot_model_y(),
                             **self._subplot_kwarg_dicts['model'])


    def _render_parameter_info_box(self, target_axis, **kwargs):
        for _pi, (_pn, _pv) in enumerate(reversed(self._fitter.parameter_name_value_dict.items())):
            target_axis.text(.2, .1+.05*_pi, "%s = %g" % (_pn, _pv), transform=target_axis.transAxes)
        target_axis.text(.1, .1 + .05 * (_pi+1.), r"Model parameters", transform=target_axis.transAxes, fontdict={'weight': 'bold'})

    def _render_legend(self, target_axis, **kwargs):
        target_axis.legend()

    # -- public properties

    @property
    def figure(self):
        return self._fig

    # -- public methods

    def plot(self):
        # TODO: hooks?
        self._plot_data(self._axes)
        self._plot_model(self._axes)

        _xlim = self._get_plot_range_x()
        if _xlim is not None:
            self._axes.set_xlim(_xlim[0], _xlim[1])
        _ylim = self._get_plot_range_y()
        if _ylim is not None:
            self._axes.set_ylim(_ylim[0], _ylim[1])

        self._render_parameter_info_box(self._axes)
        self._render_legend(self._axes)

