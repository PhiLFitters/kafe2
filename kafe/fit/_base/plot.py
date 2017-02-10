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

    PLOT_CONTAINER_METHODS_BY_PLOT_TYPE = dict(data='plot_data',
                                               model='plot_model')

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
    )

    # don't take more keys from the default than is necessary
    SUBPLOT_PROPERTY_CYCLER_ARGS_DEFAULT = dict(
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
                    color=('#a6cee3', '#b0dd8b', '#f59a96', '#fdbe6f', '#cbb1d2', '#b39c9a'),
                ),
                dict(
                    linestyle=('-', '--', '-.'),
                ),
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

        # default kwargs (static) for different subplots ('data', 'model', 'model_error_band', ...)
        self._subplot_kwarg_dicts = self.__class__.SUBPLOT_CONFIGS_DEFAULT.copy()
        # default kwarg cyclers for different subplots (these override original static properties)
        self._subplot_prop_cyclers = dict()
        for _subplot_name, _subplot_cycler_args in self.__class__.SUBPLOT_PROPERTY_CYCLER_ARGS_DEFAULT.iteritems():
            self._subplot_prop_cyclers[_subplot_name] = Cycler(*_subplot_cycler_args)

        self._defined_plot_types = self._subplot_kwarg_dicts.keys()


    # -- private methods

    def _get_next_subplot_kwargs(self, subplot_name):
        _kwargs = self._subplot_kwarg_dicts.get(subplot_name, dict())
        if subplot_name in self._subplot_prop_cyclers:
            _cycler = self._subplot_prop_cyclers[subplot_name]
            _cycler_kwargs = _cycler.get_next()
            _kwargs.update(_cycler_kwargs)
        return _kwargs

    def _get_plot_handle_for_plot_type(self, plot_type, plot_data_container):
        _plot_method_name = self.PLOT_CONTAINER_METHODS_BY_PLOT_TYPE.get(plot_type, None)
        if _plot_method_name is None:
            raise PlotFigureException("Cannot handle plot of type '%s': no entry in class dictionary "
                                      "for corresponding plot method in %s..."
                                      % (plot_type, self.PLOT_CONTAINER_TYPE))
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
        self._artist_store[subplot_id][plot_type] = _plot_method_handle(target_axis, **self._get_next_subplot_kwargs(plot_type))

    def _plot_all_subplots_all_plot_types(self):
        for _spid, _ in enumerate(self._plot_data_containers):
            for _pt in self._defined_plot_types:
                self._call_plot_method_for_plot_type(_spid, _pt, target_axis=self._axes)

    def _render_parameter_info_box(self, target_axis, **kwargs):
        _y_inc_offset = 0.
        for _pdc in reversed(self._plot_data_containers):
            for _pi, (_pn, _pv) in enumerate(reversed(_pdc._fitter.parameter_name_value_dict.items())):
                target_axis.text(.2, .1+.05*_y_inc_offset, "%s = %g" % (_pn, _pv), transform=target_axis.transAxes)
                _y_inc_offset += 1
            target_axis.text(.1, .1 + .05 * (_y_inc_offset), r"Model parameters", transform=target_axis.transAxes, fontdict={'weight': 'bold'})
            _y_inc_offset += 1

    def _render_legend(self, target_axis):
        target_axis.legend()

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
