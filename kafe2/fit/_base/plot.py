import abc
import numpy as np
import six
import textwrap
import warnings

from ...config import matplotlib as mpl
from ...config import kc, ConfigError

from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from matplotlib.legend_handler import HandlerBase

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

class DummyLegendHandler(HandlerBase):
    """Dummy legend handler (nothing is drawn)"""
    def legend_artist(self, *args, **kwargs):
        return None

class PlotContainerException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
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

    FIT_TYPE = None

    def __init__(self, fit_object, model_index=0):
        """
        Construct a :py:obj:`PlotContainer` for a :py:obj:`Fit` object:

        :param fit_object: an object derived from :py:obj:`~kafe2.fit._base.FitBase`
        """
        # TODO: update documentation
        if not isinstance(fit_object, self.__class__.FIT_TYPE):
            raise PlotContainerException("PlotContainer of type '%s' is incompatible with Fit of type '%s'"
                                         % (self.__class__, fit_object.__class__))
        if fit_object.model_count <= model_index:
            raise PlotContainerException("Received %s as model index but fit object only has %s models"
                                         % (fit_object.model_count, model_index))
        self._fitter = fit_object
        self._model_index = model_index

    def _get_total_error(self, error_contributions):
        _total_err = np.zeros_like(self.data_y)
        for _ec in error_contributions:
            _ec = _ec.lower()
            if _ec not in ('data', 'model'):
                raise ValueError(
                    "Unknown error contribution specification '{}': "
                    "expecting 'data' or 'model'".format(_ec))
            _total_err += getattr(self, _ec + '_yerr') ** 2
            _total_err += self._fit._cost_function.get_uncertainty_gaussian_approximation(
                getattr(self, _ec + '_y')) ** 2

        _total_err = np.sqrt(_total_err)

        if np.all(_total_err==0):
            return None
        else:
            return _total_err

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
    def plot_data(self, target_axes, **kwargs):
        """
        Method called by the main plot routine to plot the data points to a specified matplotlib ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :return: plot handle(s)
        """
        pass

    @abc.abstractmethod
    def plot_model(self, target_axes, **kwargs):
        """
        Method called by the main plot routine to plot the model to a specified matplotlib ``Axes`` object.

        :param target_axes: ``matplotlib`` ``Axes`` object
        :return: plot handle(s)
        """
        pass

    #Overridden by multi plot containers
    def get_formatted_model_function(self, **kwargs):
        """return model function string"""
        return self._fitter._model_function.formatter.get_formatted(**kwargs)

    #Overridden by multi plot containers
    @property
    def model_function_argument_formatters(self):
        """return model function argument formatters"""
        return self._fitter._model_function.argument_formatters

# -- must come last!


class PlotFigureException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)  # TODO: check if needed
class PlotFigureBase(object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    types of plotters.

    A :py:obj:`PlotFigure` object corresponds to a single ``matplotlib`` figure and
    can contain plots coming from different :py:obj:`FitBase`-derived objects simultaneously.

    It controls the overall figure layout and is responsible for axes, subplot and legend management.
    """
    # TODO update documentation

    PLOT_CONTAINER_TYPE = None
    PLOT_STYLE_CONFIG_DATA_TYPE = 'default'

    PLOT_SUBPLOT_TYPES = OrderedDict()
    PLOT_SUBPLOT_TYPES.update(
        data=dict(
            plot_container_method='plot_data',
            target_axes='main',
        ),
        model=dict(
            plot_container_method='plot_model',
            target_axes='main',
        ),
    )

    FIT_INFO_STRING_FORMAT = textwrap.dedent("""\
        {model_function}
        {parameters}
            $\\hookrightarrow${fit_quality}
    """)

    def __init__(self, fit_objects, model_indices=None):

        # set the managed fit objects
        try:
            iter(fit_objects)
        except TypeError:
            fit_objects = (fit_objects,)
        self._fits = fit_objects

        # model indices (relevant for Multiplots)
        if model_indices is None:
            try:
                model_indices = range(len(fit_objects))
            except TypeError:
                model_indices = [0]
        else:
            try:
                iter(model_indices)
            except:
                model_indices = [model_indices]
        self._model_indices = model_indices

        # figure layout (TODO: no hardcoding)
        self._outer_gs = gs.GridSpec(nrows=1,
                                     ncols=2,
                                     left=0.075,
                                     bottom=0.1,
                                     right=0.925,
                                     top=0.9,
                                     wspace=None,
                                     hspace=None,
                                     width_ratios=[1, 1],
                                     height_ratios=None)

        # default axis labels
        self._plot_label_x = kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, 'axis_labels', 'x')
        self._plot_label_y = kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, 'axis_labels', 'y')

        # store defined plot types for convenient access
        self._defined_plot_types = self.PLOT_SUBPLOT_TYPES.keys()

        # fill meta-information structures for all plot_types
        self._subplot_static_kwarg_dicts = dict()
        self._subplot_container_plot_method_name = dict()
        self._subplot_prop_cyclers = dict()
        for _pt in self._defined_plot_types:
            _plot_style_as = self.PLOT_SUBPLOT_TYPES[_pt].get('plot_style_as', _pt)

            self._subplot_container_plot_method_name[_pt] = self.PLOT_SUBPLOT_TYPES[_pt]['plot_container_method']
            self._subplot_static_kwarg_dicts[_pt] = kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, _plot_style_as, 'plot_kwargs')
            self._subplot_prop_cyclers[_pt] = Cycler(*kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, _plot_style_as, 'property_cycler'))

        # owned objects
        self._fig = None
        self._plot_containers = None

    # -- private methods

    def _get_axes(self, axes_key):
        try:
            return self._axes[axes_key]
        except KeyError:
            raise KeyError("No axes found for name '{}'!".format(axes_key))

    def _create_figure_axes(self, axes_keys, height_ratios=None):

        if height_ratios:
            assert len(axes_keys) == len(height_ratios)

        # plot axes layout
        _plot_axes_gs = gs.GridSpecFromSubplotSpec(
            nrows=len(axes_keys),
            ncols=1,
            wspace=None,
            hspace=0.06,
            width_ratios=None,
            height_ratios=height_ratios,
            subplot_spec=self._outer_gs[0, 0]
        )

        # create figure (overwrite the old one)
        self._fig = plt.figure()

        # create named axes
        self._axes = {
            _k : plt.subplot(_plot_axes_gs[_i])
            for _i, _k in enumerate(axes_keys)
        }

        # hide x tick labels in all but the lowest Axes
        for _key in axes_keys[:-1]:
            self._axes[_key].set_xticklabels([])
            self._axes[_key].set_xlabel(None)

        self._axes[axes_keys[-1]].set_xlabel(self.x_label)
        self._axes[axes_keys[0]].set_ylabel(self.y_label)

    def _get_plot_containers(self):
        '''retrieve plot containers, creating them if needed'''

        if self._plot_containers is None:
            self._plot_containers = []
            for _i, _fit in enumerate(self._fits):
                _pdc = self.__class__.PLOT_CONTAINER_TYPE(_fit)

                self._plot_containers.append(_pdc)

        return self._plot_containers

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
            _kwargs['zorder'] = subplot_id * _n_defined_plot_types + list(self._defined_plot_types).index(plot_type)

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

    def _call_plot_method_for_plot_type(self, plot_container, plot_type, target_axes):
        _plot_method_handle = self._get_plot_handle_for_plot_type(
            plot_type, plot_container)

        return _plot_method_handle(
            target_axes,
            **self._get_subplot_kwargs(
                self._model_indices[plot_container._model_index],
                plot_type
            )
        )

    def _plot_and_get_results(self):
        _plot_containers = self._get_plot_containers()

        _plots = {}
        for _i, _pdc in enumerate(_plot_containers):

            _pdc._model_index = _i
            for _pt in self._defined_plot_types:
                _axes_key = self.PLOT_SUBPLOT_TYPES[_pt]['target_axes']

                # skip plot elements meant for an inexistent axes
                if _axes_key not in self._axes:
                    continue

                _axes_plot_dicts = _plots.setdefault(_axes_key, {})

                _axes_plots = _axes_plot_dicts.setdefault('plots', [])

                _artist = self._call_plot_method_for_plot_type(
                    _pdc,
                    _pt,
                    target_axes=self._get_axes(_axes_key)
                )

                _axes_plots.append({
                    'type' : _pt,
                    'fit_index' : _i,
                    'container' : _pdc,
                    'artist' : _artist,
                })

                if _pdc.x_range is not None:
                    _xlim = _axes_plot_dicts.setdefault(
                        'x_range', _pdc.x_range)
                    _axes_plot_dicts['x_range'] = (
                        min(_xlim[0], _pdc.x_range[0]),
                        max(_xlim[1], _pdc.x_range[1])
                    )

                if _pdc.y_range is not None:
                    _ylim = _axes_plot_dicts.setdefault(
                        'y_range', _pdc.y_range)
                    _axes_plot_dicts['y_range'] = (
                        min(_ylim[0], _pdc.y_range[0]),
                        max(_ylim[1], _pdc.y_range[1])
                    )

        return _plots

    @classmethod
    def _get_fit_info(cls, plot_data_container, format_as_latex, asymmetric_parameter_errors):

        plot_data_container._fitter._update_parameter_formatters(
            update_asymmetric_errors=asymmetric_parameter_errors
        )

        _cost_func = plot_data_container._fitter._cost_function  # TODO: public interface

        return cls.FIT_INFO_STRING_FORMAT.format(
            model_function=plot_data_container.get_formatted_model_function(
                with_par_values=False,
                n_significant_digits=2,
                format_as_latex=format_as_latex,
                with_expression=True
            ),
            parameters='\n'.join([
                '    ' + _pf.get_formatted(
                    with_name=True,
                    with_value=True,
                    with_errors=True,
                    asymmetric_error=asymmetric_parameter_errors,
                    format_as_latex=format_as_latex
                )
                for _pf in plot_data_container.model_function_argument_formatters
            ]),
            fit_quality=_cost_func._formatter.get_formatted(
                value=plot_data_container._fitter.cost_function_value,
                n_degrees_of_freedom=_cost_func.ndf,
                with_value_per_ndf=True,
                format_as_latex=format_as_latex
            ),
        )

    def _render_legend(self, plot_results, axes_keys, with_fit_info=True, with_asymmetric_parameter_errors=False, **kwargs):
        '''render the legend for axes `axes_keys`'''
        for _axes_key in axes_keys:
            _axes = self._get_axes(_axes_key)

            _hs_unsorted, _ls_unsorted = _axes.get_legend_handles_labels()
            _hs_sorted, _ls_sorted = [], []

            _axes_plots = plot_results[_axes_key]['plots']

            if with_fit_info:
                #for _i_plot, _plot_dict in zip(reversed(range(len(_axes_plots))), _axes_plots):
                for _plot_dict in _axes_plots:

                    try:
                        try:
                            _artist_index = _hs_unsorted.index(_plot_dict['artist'][0])
                        except (ValueError, TypeError):
                            _artist_index = _hs_unsorted.index(_plot_dict['artist'])
                    except (KeyError, ValueError):
                        # artist not available or not plottable -> skip
                        continue

                    _hs_sorted.append(_hs_unsorted[_artist_index])
                    _ls_sorted.append(_ls_unsorted[_artist_index])

                    if _plot_dict['type'] != 'model_line':
                        continue

                    if with_fit_info:
                        _fit_info = self._get_fit_info(
                            _plot_dict['container'],
                            format_as_latex=True,
                            asymmetric_parameter_errors=with_asymmetric_parameter_errors
                        )

                        _hs_sorted.append('_nokey_')
                        _ls_sorted.append(_fit_info)

            _zorder = kwargs.pop('zorder', 999)
            _bbox_to_anchor = kwargs.pop('bbox_to_anchor', None)
            if _bbox_to_anchor is None:
                _bbox_to_anchor = (1.05, 0.0, 0.67, 1.0)  # axes coordinates FIXME: no hardcoding!

            _mode = kwargs.pop('mode', "expand")
            _borderaxespad = kwargs.pop('borderaxespad', 0.1)
            _ncol = kwargs.pop('ncol', 1)

            kwargs['loc'] = 'upper left'

            _axes.legend(_hs_sorted, _ls_sorted,
                         bbox_to_anchor=_bbox_to_anchor,
                         mode=_mode,
                         borderaxespad=_borderaxespad,
                         ncol=_ncol,
                         handler_map={'_nokey_': DummyLegendHandler()},
                         **kwargs).set_zorder(_zorder)

    def _adjust_plot_ranges(self, plot_results):
        '''set the x and y ranges across (all axes) to the total data range reported by the plot containers'''
        for _axes_name, _axes_dict in six.iteritems(plot_results):
            _ax = self._get_axes(_axes_name)

            _xlim = _axes_dict.get('x_range', None)
            if _xlim:
                _ax.set_xlim(_xlim)

            _ylim = _axes_dict.get('y_range', None)
            if _ylim:
                _ax.set_ylim(_ylim)

    # -- public properties

    @property
    def figure(self):
        """The ``matplotlib`` figure managed by this object."""
        return self._fig

    @property
    def axes(self):
        """A dictionary mapping names to ``matplotlib`` `Axes` objects
        contained in this figure."""
        return self._axes

    @property
    def x_label(self):
        """The current x-label."""
        return self._plot_label_x

    @x_label.setter
    def x_label(self, x_label):
        """Set the x-label for the plot.

        :param x_label: x-label to set
        :type x_label: str
        """
        if x_label is not None:
            self._plot_label_x = x_label

    @property
    def y_label(self):
        """The current y-label"""
        return self._plot_label_y

    @y_label.setter
    def y_label(self, y_label):
        """Set the y-label for the plot.

        :param y_label: y-label to set
        :type y_label: str
        """
        if y_label is not None:
            self._plot_label_y = y_label

    # -- public methods

    def plot(self,
             with_legend=True,
             with_fit_info=True,
             with_asymmetric_parameter_errors=False):
        """Plot data, model (and other subplots) for all child :py:obj:`Fit` objects."""

        _axes_keys = ('main',)
        _height_ratios = None

        self._create_figure_axes(
            _axes_keys,
            height_ratios=_height_ratios
        )

        _plot_results = self._plot_and_get_results()

        if with_legend:
            self._render_legend(
                plot_results=_plot_results,
                axes_keys=('main',),
                with_fit_info=with_fit_info,
                with_asymmetric_parameter_errors=with_asymmetric_parameter_errors
            )

        self._adjust_plot_ranges(_plot_results)


        return _plot_results

    def show_fit_info_box(self, asymmetric_parameter_errors=False, format_as_latex=True):
        """[DEPRECATED] Render text information about each plot on the figure.

        :param format_as_latex: if ``True``, the infobox text will be formatted as a LaTeX string
        :type format_as_latex: bool
        :param asymmetric_parameter_errors: if ``True``, use two different parameter errors for up/down directions
        :type asymmetric_parameter_errors: bool
        """

        # API is deprecated
        warnings.warn(
            "Method `show_fit_info_box` of `{}` object is deprecated. "
            "This call will have no effect. Pass 'with_fit_info=True' to `plot` "
            "method to show fit results as part of the legend instead.".format(self.__class__.__name__), UserWarning)

class MultiPlotBase(object):
    """Abstract class for making plots from multi fits"""
    SINGULAR_PLOT_TYPE = None

    def __init__(self, fit_objects, separate_plots=True):
        """
        Parent constructor for multi plots

        :param fit_objects: the fit objects for which plots should be created
        :type fit_objects: specified by subclass
        :param separate_plots: if ``True``, will create separate plots for each model
                               within each fit object, if ``False`` will create one plot
                               for each fit object
        :type separate_plots: bool
        """
        self._underlying_plots = []
        self._axis_labels = []
        try:
            iter(fit_objects)
        except:
            fit_objects = [fit_objects]
        if separate_plots:
            for _fit_object in fit_objects:
                for _i in range(_fit_object.model_count):
                    self._underlying_plots.append(self.__class__.SINGULAR_PLOT_TYPE(_fit_object, model_indices=_i))
                    # set size of labels array, if label is None it should be set to the default by SINGULAR_PLOT_TYPE
                    self._axis_labels.append([None, None])
        else:
            for _fit_object in fit_objects:
                _fit_object_list = []
                _model_indices = []
                for _i in range(_fit_object.model_count):
                    _fit_object_list.append(_fit_object)
                    _model_indices.append(_i)
                self._underlying_plots.append(self.__class__.SINGULAR_PLOT_TYPE(_fit_object_list, model_indices=_model_indices))

    @property
    def axis_labels(self):
        """the axis-labels of the plot"""
        return self._axis_labels

    @axis_labels.setter
    def axis_labels(self, labels):
        """sets the axis labels of the plot

        :param labels: list of axis labels
        :type labels: list
        """
        if len(labels) != len(self._axis_labels) or len(labels[0]) != len(self._axis_labels[0]):
            raise PlotContainerException("The dimensions of labels must fit the dimension of the data")
        self._axis_labels = labels

    def get_figure(self, plot_index):
        """return the figure with the specified index"""
        return self._underlying_plots[plot_index].figure

    def plot(self,
             with_legend=True,
             with_fit_info=True,
             with_asymmetric_parameter_errors=False):
        """Plot data, model (and other subplots) for all child :py:obj:`Fit` objects, and show legend."""
        for _i, _plot in enumerate(self._underlying_plots):
            _plot.x_label = self._axis_labels[_i][0]
            _plot.y_label = self._axis_labels[_i][1]
            _plot.plot(
                with_legend=with_legend,
                with_fit_info=with_fit_info,
                with_asymmetric_parameter_errors=with_asymmetric_parameter_errors
            )


    def get_fit_info_strings(self, format_as_latex=False, asymmetric_parameter_errors=False):
        return [
            _plot.get_fit_info_strings(
                format_as_latex=format_as_latex,
                asymmetric_parameter_errors=asymmetric_parameter_errors
            )
            for _plot in self._underlying_plots
        ]

    def show_fit_info_box(self, format_as_latex=True):
        """[DEPRECATED] Render text information about each plot on the figure.

        :param format_as_latex: if ``True``, the infobox text will be formatted as a LaTeX string
        :type format_as_latex: bool
        """
        for _plot in self._underlying_plots:
            _plot.show_fit_info_box(format_as_latex=format_as_latex)
