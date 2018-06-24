import abc
import numpy as np
import six

from ...config import matplotlib as mpl
from ...config import kc, ConfigError
from .fit import FitBase

from collections import OrderedDict
from copy import copy
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs


__all__ = ["PlotContainerBase", "PlotFigureBase", "MultiPlotBase", "PlotContainerException", "PlotFigureException",
           "kc_plot_style"]


def kc_plot_style(data_type, subplot_key, property_key):
    try:
        # try to find plot style-related configuration entry
        return kc('fit', 'plot', 'style', data_type, subplot_key, property_key)
    except ConfigError:
        # if not available, do lookup for the default data type
        return kc('fit', 'plot', 'style', 'default', subplot_key, property_key)


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
            for _prop_name, _prop_vals in six.iteritems(_content):
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
        _prop_positions = [(cycle_position//self._counter_divisors[i])%self._prop_val_sizes[i] for i in six.moves.range(self._dim)]
        _ps = {}
        for _i, _content in enumerate(self._props):
            for (_name, _values) in six.iteritems(_content):
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
            for (_name, _values) in six.iteritems(_content):
                if _name in properties:
                    _tmp_dict[_name] = _values
            _args.append(_tmp_dict)
        return Cycler(*_args)


class PlotContainerException(Exception):
    pass


class PlotContainerBase(object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    types of plotters.

    A :py:obj:`PlotContainer` object can be constructed for a :py:obj:`Fit` object of the
    corresponding type.
    Its main purpose is to provide an interface for accessing data stored in the
    :py:obj:`Fit` object, for the purposes of plotting.
    Most importantly, it provides methods to call the relevant ``matplotlib`` methods
    for plotting the data, model (and other information, depending on the fit type),
    and constructs the arrays required by these routines in a meaningful way.

    Classes derived from :py:obj:`PlotContainer` must at the very least contain
    properties for constructing the ``x`` and ``y`` point arrays for both the
    data and the fitted model, as well as methods calling the ``matplotlib`` routines
    doing the actual plotting.
    """
    __metaclass__ = abc.ABCMeta

    FIT_TYPE = None

    def __init__(self, fit_object, model_index=0):
        """
        Construct a :py:obj:`PlotContainer` for a :py:obj:`Fit` object:

        :param fit_object: an object derived from :py:obj:`~kafe.fit._base.FitBase`
        """
        #TODO: update documentation
        if not isinstance(fit_object, self.__class__.FIT_TYPE):
            raise PlotContainerException("PlotContainer of type '%s' is incompatible with Fit of type '%s'"
                                         % (self.__class__, fit_object.__class__))
        if fit_object.model_count <= model_index:
            raise PlotContainerException("Received %s as model index but fit object only has %s models"
                                         % (fit_object.model_count, model_index))
        self._fitter = fit_object
        self._model_index = model_index

    # -- properties

    @abc.abstractproperty
    def data_x(self):
        """
        The 'x' coordinates of the data (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def data_y(self):
        """
        The 'y' coordinates of the data (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def data_xerr(self):
        """
        The magnitude of the data 'x' error bars (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def data_yerr(self):
        """
        The magnitude of the data 'y' error bars (used by :py:meth:`~plot_data`).

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def model_x(self):
        """
        The 'x' coordinates of the model (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def model_y(self):
        """
        The 'y' coordinates of the model (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def model_xerr(self):
        """
        The magnitude of the model 'x' error bars (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def model_yerr(self):
        """
        The magnitude of the model 'y' error bars (used by :py:meth:`~plot_model`).

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def x_range(self):
        """
        The 'x' axis plot range.

        :return: iterable
        """
        pass

    @abc.abstractproperty
    def y_range(self):
        """
        The 'y' axis plot range.

        :return: iterable
        """
        pass


    @abc.abstractmethod
    def plot_data(self, target_axis, **kwargs):
        """
        Method called by the main plot routine to plot the data points to a specified matplotlib ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :return: plot handle(s)
        """
        pass

    @abc.abstractmethod
    def plot_model(self, target_axis, **kwargs):
        """
        Method called by the main plot routine to plot the model to a specified matplotlib ``Axes`` object.

        :param target_axis: ``matplotlib`` ``Axes`` object
        :return: plot handle(s)
        """
        pass

    def get_formatted_model_function(self, **kwargs):
        return self._fitter._model_function.formatter.get_formatted(**kwargs)

    @property
    def model_function_argument_formatters(self):
        return self._fitter._model_function.argument_formatters

# -- must come last!

class PlotFigureException(Exception):
    pass

class PlotFigureBase(object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    types of plotters.

    A :py:obj:`PlotFigure` object corresponds to a single ``matplotlib`` figure and
    can contain plots coming from different :py:obj:`FitBase`-derived objects simultaneously.

    It controls the overall figure layout and is responsible for axes, subplot and legend management.
    """
    #TODO update documentation
    __metaclass__ = abc.ABCMeta  # TODO: check if needed

    PLOT_CONTAINER_TYPE = None
    PLOT_STYLE_CONFIG_DATA_TYPE = 'default'

    PLOT_SUBPLOT_TYPES = OrderedDict()
    PLOT_SUBPLOT_TYPES.update(
        data=dict(
            plot_container_method='plot_data',
        ),
        model=dict(
            plot_container_method='plot_model',
        ),
    )

    IS_MULTI_PLOT = False

    def __init__(self, fit_objects, model_indices=0):
        try:
            iter(model_indices)
        except:
            model_indices = [model_indices]
        self._model_indices = model_indices
        self._fig = plt.figure()  # defaults from matplotlibrc
        # self._figsize = (self._fig.get_figwidth()*self._fig.dpi, self._fig.get_figheight()*self._fig.dpi)
        self._outer_gs = gs.GridSpec(nrows=1,
                                     ncols=3,
                                     left=0.075,
                                     bottom=0.1,
                                     right=0.925,
                                     top=0.9,
                                     wspace=None,
                                     hspace=None,
                                     width_ratios=[6, 2, 4],
                                     height_ratios=None)

        # TODO: use these later to maintain absolute margin width on resize
        self._absolute_outer_sizes = dict(left=self._outer_gs.left*self._fig.get_figwidth()*self._fig.dpi,
                                          bottom=self._outer_gs.bottom*self._fig.get_figheight()*self._fig.dpi,
                                          right=self._outer_gs.right*self._fig.get_figwidth()*self._fig.dpi,
                                          top=self._outer_gs.top*self._fig.get_figheight()*self._fig.dpi)

        self._plot_axes_gs = gs.GridSpecFromSubplotSpec(
            nrows=1,
            ncols=1,
            wspace=None,
            hspace=None,
            width_ratios=None,
            #height_ratios=[8, 1],
            subplot_spec=self._outer_gs[0, 0]
        )
        self._main_plot_axes = plt.subplot(self._plot_axes_gs[0, 0])

        # NOTE: not working because mpl_connect overwrites existing handlers for the same event x_x
        #_cid = self._fig.canvas.mpl_connect('resize_event', self._on_resize)

        self._plot_data_containers = []
        self._artist_store = []
        try:
            iter(fit_objects)
        except TypeError:
            fit_objects = (fit_objects,)

        for _i, _fit in enumerate(fit_objects):
            if self.__class__.IS_MULTI_PLOT:
                _pdc = self.__class__.PLOT_CONTAINER_TYPE(_fit, model_index=self._model_indices[_i])
            else:
                _pdc = self.__class__.PLOT_CONTAINER_TYPE(_fit)                
            self._plot_data_containers.append(_pdc)
            self._artist_store.append(dict())

        self._plot_range_x = None
        self._plot_range_y = None

        # store defined plot types for conveniient access
        self._defined_plot_types = self.PLOT_SUBPLOT_TYPES.keys()

        # fill meta-information structures for all plot_types
        self._subplot_static_kwarg_dicts = dict()
        self._subplot_container_plot_method_name = dict()
        self._subplot_prop_cyclers = dict()
        for _pt in self._defined_plot_types:
            self._subplot_container_plot_method_name[_pt] = self.PLOT_SUBPLOT_TYPES[_pt]['plot_container_method']
            self._subplot_static_kwarg_dicts[_pt] = kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, _pt, 'plot_kwargs')
            self._subplot_prop_cyclers[_pt] = Cycler(*kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, _pt, 'property_cycler'))

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
        _kwargs['label'] = _label_raw % dict(subplot_id=subplot_id,
                                             plot_type=plot_type)

        # calculate zorder if not explicitly given
        _n_defined_plot_types = len(self._defined_plot_types)
        if 'zorder' not in _kwargs:
            _kwargs['zorder'] = subplot_id * _n_defined_plot_types + self._defined_plot_types.index(plot_type)


        return _kwargs

    def _get_plot_handle_for_plot_type(self, plot_type, plot_data_container):
        _plot_method_name = self.PLOT_SUBPLOT_TYPES[plot_type]['plot_container_method']

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
        _artist = _plot_method_handle(target_axis, **self._get_subplot_kwargs(self._model_indices[subplot_id], plot_type))
        # TODO: warn if plot function does not return artist (?)
        # store the artist returned by the plot method
        self._artist_store[subplot_id][plot_type] = _artist

    def _plot_all_subplots_all_plot_types(self):
        for _spid, _ in enumerate(self._plot_data_containers):
            for _pt in self._defined_plot_types:
                self._call_plot_method_for_plot_type(_spid, _pt, target_axis=self._main_plot_axes)

    def _render_parameter_info_box(self, target_figure, format_as_latex, **kwargs):
        if 'transform' not in kwargs:
            kwargs['transform'] = target_figure.transFigure
        if 'zorder' not in kwargs:
            kwargs['zorder'] = 999

        _fig_bs, _fig_ts, _fig_ls, _fig_rs = self._outer_gs.get_grid_positions(self._fig)

        _n_text_lines = len(self._plot_data_containers) * 2
        _n_text_lines += np.sum([len(_pdc._fitter._model_function.argument_formatters) for _pdc in self._plot_data_containers])
        # FIXME: access to "_pdc._fitter._model_function" not public!

        _y_inc_size = min(.05, (_fig_ts[0] - _fig_ls[0])/_n_text_lines)
        _y_inc_offset = _fig_ts[0] - 0.05
        _y_inc_counter = 0
        for _id, _pdc in enumerate(self._plot_data_containers):
            _y = _y_inc_offset - _y_inc_size*_y_inc_counter
            target_figure.text(_fig_ls[2], _y, r"Model %d" % (self._model_indices[_id],), fontdict={'weight': 'bold'}, **kwargs)
            _y_inc_counter += 1

            _y = _y_inc_offset - _y_inc_size * _y_inc_counter
            _formatted_string = _pdc.get_formatted_model_function(
                with_par_values=False, n_significant_digits=2, format_as_latex=format_as_latex, with_expression=True)
            target_figure.text(_fig_ls[2] + .025, _y, _formatted_string, **kwargs)
            _y_inc_counter += 1

            for _pi, _pf in enumerate(_pdc.model_function_argument_formatters):
                _y = _y_inc_offset - _y_inc_size * _y_inc_counter
                _formatted_string = _pf.get_formatted(with_name=True, with_value=True, with_errors=True, format_as_latex=format_as_latex)
                target_figure.text(_fig_ls[2]+.05, _y, _formatted_string, **kwargs)
                _y_inc_counter += 1

            if not self.__class__.IS_MULTI_PLOT:
                # print info about cost function per degree of freedom
                _y = _y_inc_offset - _y_inc_size * _y_inc_counter
                _pf = _pdc._fitter._cost_function._formatter
                _formatted_string = _pf.get_formatted(value=_pdc._fitter.cost_function_value,
                                                n_degrees_of_freedom=_pdc._fitter._cost_function.ndf, # TODO: public interface
                                                with_value_per_ndf=True,
                                                format_as_latex=format_as_latex)
                target_figure.text(_fig_ls[2] + .05, _y, _formatted_string, **kwargs)
                _y_inc_counter += 1
        if self.__class__.IS_MULTI_PLOT:
            _y_inc_counter += 1
            _pdc = self._plot_data_containers[0]
            _y = _y_inc_offset - _y_inc_size * _y_inc_counter
            _pf = _pdc._fitter._cost_function._formatter
            _formatted_string = _pf.get_formatted(value=_pdc._fitter.cost_function_value,
                                            n_degrees_of_freedom=_pdc._fitter._cost_function.ndf, # TODO: public interface
                                            with_value_per_ndf=True,
                                            format_as_latex=format_as_latex)
            target_figure.text(_fig_ls[2], _y, _formatted_string, **kwargs)
            _y_inc_counter += 1



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
        _bbox_to_anchor = kwargs.pop('bbox_to_anchor', None)
        if _bbox_to_anchor is None:
            # _ls, _rs = self._outer_gs.get_grid_positions(self._fig)[2:]
            # _l = _ls[1]
            # _b = self._outer_gs.bottom
            # _w = self._outer_gs.right - _l
            # _h = self._outer_gs.top - _b
            # _bbox_to_anchor = (_l, _b, _w, _h)
            _bbox_to_anchor = (1.05, 0.0, 0.67, 1.0)  # axes coordinates FIXME: no hardcoding!
        _mode = kwargs.pop('mode', "expand")
        _borderaxespad = kwargs.pop('borderaxespad', 0.1)
        _ncol = kwargs.pop('ncol', 1)

        target_axis.legend(_hs_sorted, _ls_sorted,
                           bbox_to_anchor=_bbox_to_anchor,
                           mode=_mode,
                           borderaxespad=_borderaxespad,
                           ncol=_ncol,
                           **kwargs).set_zorder(_zorder)


    def _get_total_data_range_x(self):
        _min, _max = None, None
        for _pdc in self._plot_data_containers:
            _lim = _pdc.x_range
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
            _lim = _pdc.y_range
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
            self._main_plot_axes.set_xlim(_xlim[0], _xlim[1])
        _ylim = self._get_total_data_range_y()
        if None not in _ylim:
            self._main_plot_axes.set_ylim(_ylim[0], _ylim[1])

    # -- public properties

    @property
    def figure(self):
        """The ``matplotlib`` figure managed by this object."""
        return self._fig

    # -- public methods

    def plot(self):
        """Plot data, model (and other subplots) for all child :py:obj:`Fit` objects, and show legend."""
        # TODO: hooks?
        # TODO: more fine-grained control over what is plotted
        self._plot_all_subplots_all_plot_types()
        self._set_plot_range_to_total_data_range()
        self._render_legend(self._main_plot_axes)
        # set axis labels
        self._main_plot_axes.set_xlabel(kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, 'axis_labels', 'x'))
        self._main_plot_axes.set_ylabel(kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, 'axis_labels', 'y'))

    def show_fit_info_box(self, format_as_latex=False):
        """Render text information about each plot on the figure.

        :param format_as_latex: if ``True``, the infobox text will be formatted as a LaTeX string
        :type format_as_latex: bool
        """
        self._render_parameter_info_box(self._fig, format_as_latex=format_as_latex)


class MultiPlotBase(object):
    """Abstract class for making plots from multi fits"""
    SINGULAR_PLOT_TYPE = None
    
    def __init__(self, fit_objects, separate_plots=True):
        """
        Parent constructor for multi plots
        
        :param fit_objects: the fit objects for which plots should be created
        :type fit_objects: specified by subclass
        :param separate_plots: if true, will create separate plots for each model
                               within each fit object, if false will create one plot
                               for each fir object
        :type separate_plots: bool
        """
        self._underlying_plots = []
        try:
            iter(fit_objects)
        except:
            fit_objects=[fit_objects]
        if separate_plots:
            for _fit_object in fit_objects:
                for _i in range(_fit_object.model_count):
                    self._underlying_plots.append(self.__class__.SINGULAR_PLOT_TYPE(_fit_object, model_indices=_i))
        else:
            for _fit_object in fit_objects:
                _fit_object_list = [] 
                _model_indices = []
                for _i in range(_fit_object.model_count):
                    _fit_object_list.append(_fit_object)
                    _model_indices.append(_i)
                self._underlying_plots.append(self.__class__.SINGULAR_PLOT_TYPE(_fit_object_list, model_indices=_model_indices))

    def get_figure(self, plot_index):
        """return the figure with the specified index"""
        return self._underlying_plots[plot_index].figure
    
    def plot(self):
        """Plot data, model (and other subplots) for all child :py:obj:`Fit` objects, and show legend."""
        for _plot in self._underlying_plots:
            _plot.plot()
    
    def show_fit_info_box(self, format_as_latex=False):
        """Render text information about each plot on the figure.

        :param format_as_latex: if ``True``, the infobox text will be formatted as a LaTeX string
        :type format_as_latex: bool
        """
        for _plot in self._underlying_plots:
            _plot.show_fit_info_box(format_as_latex=format_as_latex)
        