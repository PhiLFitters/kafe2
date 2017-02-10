import abc
import numpy as np

from collections import OrderedDict
from copy import copy

from ...config import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs



class CyclerException(Exception):
    pass

class Cycler(object):
    # TODO: handle_mismatching_lengths in ['repeat', 'repeat_last', 'reflect']
    def __init__(self, *args):
        self._props = []
        self._modulo = 1

        # read in properties and check
        _pv_sizes = []
        _processed_names = set()
        for _i, _content in enumerate(args):
            _prop_size_i = None
            _prop_dict_i = dict()
            for _prop_name, _prop_vals in _content.iteritems():
                # for the time being: 'refuse' any mismatching property value lengths
                if _prop_size_i is None:
                    _prop_size_i = len(_prop_vals)
                else:
                    if len(_prop_vals) != _prop_size_i:
                        raise CyclerException("Cannot cycle properties with mismatching value sequence lengths!")
                if _prop_name in _processed_names:
                    raise CyclerException("Cycle already contains a property named '%s'!" % (_prop_name,))
                _prop_dict_i[_prop_name] = tuple(_prop_vals)
                _processed_names.add(_prop_name)
            _pv_sizes.append(_prop_size_i)
            self._modulo *= _prop_size_i
            self._props.append(_prop_dict_i)
        self._dim = len(self._props)

        self._prop_val_sizes = np.array(_pv_sizes, dtype=int)
        self._counter_divisors = np.ones_like(self._prop_val_sizes, dtype=int)
        for i in range(1, self._dim):
            self._counter_divisors[i] = self._counter_divisors[i-1] * self._prop_val_sizes[i-1]
        self._cycle_counter = 0

    # public properties

    @property
    def modulo(self):
        return self._modulo

    # public methods

    def get_next(self):
        _prop_positions = [(self._cycle_counter//self._counter_divisors[i])%self._prop_val_sizes[i] for i in xrange(self._dim)]
        _ps = {}
        for _i, _content in enumerate(self._props):
            for (_name, _values) in _content.iteritems():
                _ps[_name] = _values[_prop_positions[_i]]
        self._cycle_counter += 1
        return _ps

    def reset(self):
        # TODO: test
        self._cycle_counter = 0

    def combine(self, other_cycler):
        # TODO: test
        # TODO: more control over combination
        _args = self._props + other_cycler._props
        return Cycler(*_args)

    def subset_cycler(self, properties):
        # TODO: test
        _args = []
        for _i, _content in enumerate(self._props):
            _tmp_dict = {}
            for (_name, _values) in _content.iteritems():
                if _name in properties:
                    _tmp_dict[_name] = _values
            _args.append(_tmp_dict)
        return Cycler(*_args)

# TODO: get from config
DEFAULT_PROPERTY_CYCLER_ARGS = dict(
    data=tuple(
        (
            dict(
                color=('#2079b4', '#36a12e', '#e41f21', '#ff8001', '#6d409c', '#b15928'),
            ),
            dict(
                marker=('o', '^', 's'),
            ),
        )
    ),
    model=tuple(
        (
            dict(
                color=('#a6cee3', '#b0dd8b', '#f59a96', '#fdbe6f', '#cbb1d2', '#faf899'),
            ),
            dict(
                linestyle=('-', '--', '-.'),
            ),
        )
    ),
    model_error_band=tuple(
        (
            dict(
                facecolor=('#a6cee3', '#b0dd8b', '#f59a96', '#fdbe6f', '#cbb1d2', '#faf899'),
            ),
        )
    ),
)


class FitPlotBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    SUBPLOT_CONFIGS_DEFAULT = dict(
        data=dict(
            linestyle='',
            marker='o',
            label='data',
            zorder=10
        ),
        model=dict(
            linestyle='-',
            marker='',
            label='model',
            linewidth=2,
            zorder=-10
        ),
        model_error_band=dict(
            alpha=0.5,
            linestyle='-',
            label='model error',
            edgecolor='none',
            linewidth=2,
            zorder=-100
        )
    )
    SUBPLOT_PROPERTY_CYCLER_ARGS_DEFAULT = DEFAULT_PROPERTY_CYCLER_ARGS

    def __init__(self, parent_fit):
        self._fig = plt.figure()  # defaults from matplotlibrc
        self._gs = gs.GridSpec(nrows=1, ncols=1)
        self._axes = plt.subplot(self._gs[0, 0])
        self._fitter = parent_fit

        self._plot_range_x = None
        self._plot_range_y = None

        # default kwargs (static) for different subplots ('data', 'model', 'model_error_band', ...)
        self._subplot_kwarg_dicts = self.__class__.SUBPLOT_CONFIGS_DEFAULT.copy()
        # default kwarg cyclers for different subplots (these override original static properties)
        self._subplot_prop_cyclers = dict()
        for _subplot_name, _subplot_cycler_args in self.__class__.SUBPLOT_PROPERTY_CYCLER_ARGS_DEFAULT.iteritems():
            self._subplot_prop_cyclers[_subplot_name] = Cycler(*_subplot_cycler_args)

        self._store_artists = dict()

    # -- private methods

    def _get_next_subplot_kwargs(self, subplot_name):
        _kwargs = self._subplot_kwarg_dicts.get(subplot_name, dict())
        if subplot_name in self._subplot_prop_cyclers:
            _cycler = self._subplot_prop_cyclers[subplot_name]
            _cycler_kwargs = _cycler.get_next()
            _kwargs.update(_cycler_kwargs)
        return _kwargs

    # -- properties

    @abc.abstractproperty
    def plot_data_x(self): pass

    @abc.abstractproperty
    def plot_data_y(self): pass

    @abc.abstractproperty
    def plot_data_xerr(self): pass

    @abc.abstractproperty
    def plot_data_yerr(self): pass

    @abc.abstractproperty
    def plot_model_x(self): pass

    @abc.abstractproperty
    def plot_model_y(self): pass

    @abc.abstractproperty
    def plot_model_xerr(self): pass

    @abc.abstractproperty
    def plot_model_yerr(self): pass

    @abc.abstractproperty
    def plot_range_x(self): pass

    @abc.abstractproperty
    def plot_range_y(self): pass


    def _plot_data(self, target_axis, **kwargs):
        if self._fitter.has_data_errors:
            self._store_artists['data'] = target_axis.errorbar(self.plot_data_x,
                                 self.plot_data_y,
                                 xerr=self.plot_data_xerr,
                                 yerr=self.plot_data_yerr,
                                 **self._get_next_subplot_kwargs('data'))
        else:
            self._store_artists['data'] = target_axis.plot(self.plot_data_x,
                             self.plot_data_y,
                             **self._get_next_subplot_kwargs('data'))

    def _plot_model(self, target_axis, **kwargs):
        if self._fitter.has_model_errors:
            self._store_artists['model'] = target_axis.errorbar(self.plot_model_x,
                                 self.plot_model_y,
                                 xerr=self.plot_model_xerr,
                                 yerr=self.plot_model_yerr,
                                 **self._get_next_subplot_kwargs('model'))
        else:
            self._store_artists['model'] = target_axis.plot(self.plot_model_x,
                             self.plot_model_y,
                             **self._get_next_subplot_kwargs('model'))


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

        _xlim = self.plot_range_x
        if _xlim is not None:
            self._axes.set_xlim(_xlim[0], _xlim[1])
        _ylim = self.plot_range_y
        if _ylim is not None:
            self._axes.set_ylim(_ylim[0], _ylim[1])

        self._render_parameter_info_box(self._axes)
        self._render_legend(self._axes)
