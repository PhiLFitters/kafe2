import abc
import numpy as np

from ...config import matplotlib as mpl
from fit import FitBase

from collections import OrderedDict
from copy import copy
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

    def get(self, cycle_position):
        _prop_positions = [(cycle_position//self._counter_divisors[i])%self._prop_val_sizes[i] for i in xrange(self._dim)]
        _ps = {}
        for _i, _content in enumerate(self._props):
            for (_name, _values) in _content.iteritems():
                _ps[_name] = _values[_prop_positions[_i]]
        return _ps

    def get_next(self):
        _ps = self.get(self._cycle_counter)
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

# class FitPlotException(Exception):
#     pass

class PlotContainerException(Exception):
    pass

class PlotContainerBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    FIT_TYPE = None

    def __init__(self, fit_object):
        if not isinstance(fit_object, self.__class__.FIT_TYPE):
            raise PlotContainerException("PlotContainer of type '%s' is incompatible with Fit of type '%s'"
                                         % self.__class__, self.__class__.FIT_TYPE)
        self._fitter = fit_object

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


    @abc.abstractmethod
    def plot_data(self, target_axis, **kwargs): pass

    @abc.abstractmethod
    def plot_model(self, target_axis, **kwargs): pass

# class PlotException(object):
#     pass
#
# class PlotBase(object):
#     pass


# -- must come last!

class PlotFigureException(Exception):
    pass

class PlotFigureBase(object):

    __metaclass__ = abc.ABCMeta  # TODO: check if needed

    PLOT_CONTAINER_TYPE = None

    PLOT_TYPE_DEFAULT_CONFIGS = OrderedDict()
    PLOT_TYPE_DEFAULT_CONFIGS.update(
        data=dict(
            plot_container_method='plot_data',
            plot_container_method_static_kwargs=dict(
                linestyle='',
                marker='o',
                label='data %(subplot_id)s',
                #zorder=10
            ),
            plot_container_method_kwargs_cycler_args=tuple((
                dict(
                    color=('#2079b4', '#36a12e', '#e41f21', '#ff8001', '#6d409c', '#b15928'),
                ),
                dict(
                    marker=('o', '^', 's'),
                ))
            ),
        )
    )
    PLOT_TYPE_DEFAULT_CONFIGS.update(
        model=dict(
            plot_container_method='plot_model',
            plot_container_method_static_kwargs=dict(
                linestyle='-',
                marker='',
                label='model %(subplot_id)s',
                linewidth=2,
                #zorder=-10
            ),
            plot_container_method_kwargs_cycler_args=tuple((
                dict(
                    color=('#a6cee3', '#b0dd8b', '#f59a96', '#fdbe6f', '#cbb1d2', '#b39c9a'),
                ),
                dict(
                    linestyle=('-', '--', '-.'),
                ))
            )
        ),
    )


    def __init__(self, fit_objects):
        self._fig = plt.figure()  # defaults from matplotlibrc
        self._gs = gs.GridSpec(nrows=1, ncols=1)
        self._axes = plt.subplot(self._gs[0, 0])

        self._plot_data_containers = []
        self._artist_store = []
        try:
            iter(fit_objects)
        except TypeError:
            fit_objects = (fit_objects,)

        for _fit in fit_objects:
            _pdc = self.__class__.PLOT_CONTAINER_TYPE(_fit)
            self._plot_data_containers.append(_pdc)
            self._artist_store.append(dict())

        self._plot_range_x = None
        self._plot_range_y = None

        # store defined plot types for conveniient access
        self._defined_plot_types = self.PLOT_TYPE_DEFAULT_CONFIGS.keys()

        # fill meta-information structures for all plot_types
        self._subplot_static_kwarg_dicts = dict()
        self._subplot_container_plot_method_name = dict()
        self._subplot_prop_cyclers = dict()
        for _pt in self._defined_plot_types:
            self._subplot_container_plot_method_name[_pt] = self.PLOT_TYPE_DEFAULT_CONFIGS[_pt]['plot_container_method']
            self._subplot_static_kwarg_dicts[_pt] = self.PLOT_TYPE_DEFAULT_CONFIGS[_pt]['plot_container_method_static_kwargs']
            self._subplot_prop_cyclers[_pt] = Cycler(*self.PLOT_TYPE_DEFAULT_CONFIGS[_pt]['plot_container_method_kwargs_cycler_args'])

    # -- private methods

    def _get_interpolated_label(self, subplot_id, plot_type):
        _kwargs = self._subplot_static_kwarg_dicts[subplot_id][plot_type]
        _label_raw =_kwargs.pop('label')
        return _label_raw % _kwargs

    def _get_subplot_kwargs(self, subplot_id, plot_type):
        # get static kwargs
        _kwargs = self._subplot_static_kwarg_dicts.get(plot_type, dict()).copy()
        # get kwargs from property cyclers
        if plot_type in self._subplot_prop_cyclers:
            _cycler = self._subplot_prop_cyclers[plot_type]
            _cycler_kwargs = _cycler.get(subplot_id)
            _kwargs.update(_cycler_kwargs)

        # apply interpolation to legend labels
        _label_raw = _kwargs.pop('label')
        # TODO: think of better way to handle this (and make for flexible)
        _kwargs['label'] = _label_raw % dict(subplot_id=subplot_id, plot_type=plot_type)

        # calculate zorder if not explicitly given
        _n_defined_plot_types = len(self._defined_plot_types)
        if 'zorder' not in _kwargs:
            _kwargs['zorder'] = subplot_id * _n_defined_plot_types + self._defined_plot_types.index(plot_type)


        return _kwargs

    def _get_plot_handle_for_plot_type(self, plot_type, plot_data_container):
        _plot_method_name = self.PLOT_TYPE_DEFAULT_CONFIGS[plot_type]['plot_container_method']

        try:
            _plot_method_handle = getattr(plot_data_container, _plot_method_name)
        except AttributeError:
            raise PlotFigureException("Cannot handle plot of type '%s': cannot find corresponding "
                                      "plot method '%s' in %s!"
                                      % (plot_type, _plot_method_name, self.PLOT_CONTAINER_TYPE))
        return _plot_method_handle

    def _call_plot_method_for_plot_type(self, subplot_id, plot_type, target_axis):
        _pdc = self._plot_data_containers[subplot_id]
        _plot_method_handle = self._get_plot_handle_for_plot_type(plot_type, _pdc)
        _artist = _plot_method_handle(target_axis, **self._get_subplot_kwargs(subplot_id, plot_type))
        # TODO: warn if plot function does not return artist (?)
        # store the artist returned by the plot method
        self._artist_store[subplot_id][plot_type] = _artist

    def _plot_all_subplots_all_plot_types(self):
        for _spid, _ in enumerate(self._plot_data_containers):
            for _pt in self._defined_plot_types:
                self._call_plot_method_for_plot_type(_spid, _pt, target_axis=self._axes)

    def _render_parameter_info_box(self, target_axis, **kwargs):
        if 'transform' not in kwargs:
            kwargs['transform'] = target_axis.transAxes
        if 'zorder' not in kwargs:
            kwargs['zorder'] = 999

        _y_inc_offset = 0.
        for _pdc in reversed(self._plot_data_containers):
            for _pi, (_pn, _pv) in enumerate(reversed(_pdc._fitter.parameter_name_value_dict.items())):
                target_axis.text(.2, .1+.05*_y_inc_offset, "%s = %g" % (_pn, _pv), **kwargs)
                _y_inc_offset += 1
            target_axis.text(.1, .1 + .05 * (_y_inc_offset), r"Model parameters", fontdict={'weight': 'bold'}, **kwargs)
            _y_inc_offset += 1

    def _render_legend(self, target_axis, **kwargs):
        _hs_unsorted, _ls_unsorted = target_axis.get_legend_handles_labels()
        _hs_sorted, _ls_sorted = [], []

        # sort legend entries by drawing order
        for _subplot_artist_map in self._artist_store:
            for _pt in self._defined_plot_types:
                _artist = _subplot_artist_map.get(_pt , None)
                if _artist is None:
                    continue
                try:
                    _artist_index = _hs_unsorted.index(_artist)
                except ValueError:
                    _artist_index = _hs_unsorted.index(_artist[0])
                _hs_sorted.append(_hs_unsorted[_artist_index])
                _ls_sorted.append(_ls_unsorted[_artist_index])

        _zorder = kwargs.pop('zorder', 999)
        _bbox_to_anchor = kwargs.pop('bbox_to_anchor', (0., -0.120, 1., .1))
        _mode = kwargs.pop('mode', "expand")
        _borderaxespad = kwargs.pop('borderaxespad', 0.)
        _ncol = kwargs.pop('ncol', len(self._defined_plot_types))

        target_axis.legend(_hs_sorted, _ls_sorted,
                           #bbox_to_anchor=_bbox_to_anchor,
                           #mode=_mode,
                           #borderaxespad=_borderaxespad,
                           #ncol=_ncol,
                           **kwargs).set_zorder(_zorder)


    def _get_total_data_range_x(self):
        _min, _max = None, None
        for _pdc in self._plot_data_containers:
            _lim = _pdc.plot_range_x
            if _lim is None:
                continue
            if _min is None or _lim[0] < _min:
                _min = _lim[0]
            if _max is None or _lim[1] > _max:
                _max = _lim[1]
        return _min, _max

    def _get_total_data_range_y(self):
        _min, _max = None, None
        for _pdc in self._plot_data_containers:
            _lim = _pdc.plot_range_y
            if _lim is None:
                continue
            if _min is None or _lim[0] < _min:
                _min = _lim[0]
            if _max is None or _lim[1] > _max:
                _max = _lim[1]
        return _min, _max

    def _set_plot_range_to_total_data_range(self):
        _xlim = self._get_total_data_range_x()
        if None not in _xlim:
            self._axes.set_xlim(_xlim[0], _xlim[1])
        _ylim = self._get_total_data_range_y()
        if None not in _ylim:
            self._axes.set_ylim(_ylim[0], _ylim[1])

    # -- public properties

    @property
    def figure(self):
        return self._fig

    # -- public methods

    def plot(self):
        # TODO: hooks?
        self._plot_all_subplots_all_plot_types()
        self._set_plot_range_to_total_data_range()
        self._render_parameter_info_box(self._axes)
        self._render_legend(self._axes)
