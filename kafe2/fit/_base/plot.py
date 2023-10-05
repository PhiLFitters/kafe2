import abc
import itertools
import os
import textwrap
import warnings
from collections import OrderedDict
from collections.abc import Iterable

import matplotlib as mpl
import numpy  # help IDEs with type-hinting inside docstrings
import numpy as np
import six
from matplotlib import gridspec as gs
from matplotlib import pyplot as plt
from matplotlib import rc_context, rcParams
from matplotlib.legend_handler import HandlerBase

from ...config import ConfigError, kafe2_rc, kc
from ..multi.fit import MultiFit
from ..util.wrapper import _fit_history
from .container import DataContainerBase
from .format import ParameterFormatter

try:
    import typing  # help IDEs with type-hinting inside docstrings  # noqa: F401 (unused import)
except ImportError:
    pass

__all__ = ["PlotAdapterBase", "Plot", "kc_plot_style"]


def kc_plot_style(data_type, subplot_key, property_key):
    try:
        # try to find plot style-related configuration entry
        return kc("fit", "plot", "style", data_type, subplot_key, property_key)
    except ConfigError:
        # if not available, do lookup for the default data type
        return kc("fit", "plot", "style", "default", subplot_key, property_key)


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
                        raise ValueError("Cannot cycle properties with mismatching value " "sequence lengths!")
                if _prop_name in _processed_names:
                    raise ValueError("Cycle already contains a property named '%s'!" % (_prop_name,))
                _prop_dict_i[_prop_name] = tuple(_prop_vals)
                _processed_names.add(_prop_name)
            _pv_sizes.append(_prop_size_i)
            self._modulo *= _prop_size_i
            self._props.append(_prop_dict_i)
        self._dim = len(self._props)

        self._prop_val_sizes = np.array(_pv_sizes, dtype=int)
        self._counter_divisors = np.ones_like(self._prop_val_sizes, dtype=int)
        for i in range(1, self._dim):
            self._counter_divisors[i] = self._counter_divisors[i - 1] * self._prop_val_sizes[i - 1]
        self._cycle_counter = 0

    # public properties

    @property
    def modulo(self):
        return self._modulo

    # public methods

    def get(self, cycle_position):
        _prop_positions = [(cycle_position // self._counter_divisors[i]) % self._prop_val_sizes[i] for i in six.moves.range(self._dim)]
        _ps = {}
        for _i, _content in enumerate(self._props):
            for _name, _values in six.iteritems(_content):
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
            for _name, _values in six.iteritems(_content):
                if _name in properties:
                    _tmp_dict[_name] = _values
            _args.append(_tmp_dict)
        return Cycler(*_args)


class DummyLegendHandler(HandlerBase):
    """Dummy legend handler (nothing is drawn)"""

    def legend_artist(self, *args, **kwargs):
        return None


@six.add_metaclass(abc.ABCMeta)
class PlotAdapterBase:
    """This is a purely abstract class implementing the minimal interface required by all
    types of plot adapters.

    A :py:obj:`PlotAdapter` object can be constructed for a :py:obj:`Fit` object of the
    corresponding type.
    Its main purpose is to provide an interface for accessing data stored in the
    :py:obj:`Fit` object, for the purposes of plotting.
    Most importantly, it provides methods to call the relevant :py:mod:`matplotlib` methods
    for plotting the data, model (and other information, depending on the fit type),
    and constructs the arrays required by these routines in a meaningful way.

    Classes derived from :py:class:`PlotAdapterBase` must at the very least contain
    properties for constructing the *x* and *y* point arrays for both the
    data and the fitted model, as well as methods calling the :py:mod:`matplotlib` routines
    doing the actual plotting.
    """

    PLOT_STYLE_CONFIG_DATA_TYPE = "default"

    PLOT_SUBPLOT_TYPES = OrderedDict(
        data=dict(
            plot_adapter_method="plot_data",
            target_axes="main",
            container_valid=True,
        ),
        model=dict(
            plot_adapter_method="plot_model",
            target_axes="main",
        ),
        ratio=dict(
            plot_style_as="data",
            plot_adapter_method="plot_ratio",
            target_axes="ratio",
        ),
        residual=dict(
            plot_style_as="data",
            plot_adapter_method="plot_residual",
            target_axes="residual",
        ),
    )

    AVAILABLE_X_SCALES = ("linear",)
    AVAILABLE_Y_SCALES = ("linear", "log")

    def __init__(self, fit_object, from_container=False):
        """Construct a :py:obj:`PlotAdapter` for a :py:obj:`Fit` object:

        :param fit_object: An object derived from :py:obj:`~.FitBase`
        :type fit_object: kafe2.fit._base.FitBase
        :param from_container: Whether fit_object was created ad-hoc from just a data container.
        :type from_container: bool
        """
        self._fit = fit_object
        self._from_container = from_container
        self._x_range = None
        self._y_range = None
        self._x_scale = "linear"
        self._y_scale = "linear"
        _axes = ("x", "y")
        _axis_labels = list()
        for i, axis in enumerate(_axes):
            label = self._fit.data_container.axis_labels[i]  # load from data_container
            if label is None and axis == "x" and type(fit_object).__name__ == "XYFit":
                _latex_name = fit_object.model_function.formatter.arg_formatters[0].latex_name
                label = f"${_latex_name}$"
            if label is None:  # fallback to default
                label = kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, "axis_labels", axis)
            if label == "__del__":  # set axis label to None for special string __del__
                label = None
            _axis_labels.append(label)
        self._axis_labels = _axis_labels
        _x_ticks = fit_object._get_default_x_ticks()
        self._ticks = [_x_ticks, []]

        # specification of subplots for which this adapter provided plot routines
        self._subplots = None
        self._get_subplots()

        # set labels if present and according subplots are available
        self._set_plot_labels()

    def _get_subplots(self):
        """Create dictionary containing all subplot specifications.

        :return: An ordered dictionary containing all information about the available plot methods.
        :rtype: OrderedDict
        """
        if not self._subplots:
            # create subplot dict
            self._subplots = OrderedDict()
            for _pt, _pt_spec in six.iteritems(self.PLOT_SUBPLOT_TYPES):
                try:
                    # replace plot adapter method string with real method
                    self._subplots[_pt] = dict(
                        _pt_spec,
                        plot_adapter_method=getattr(self, _pt_spec["plot_adapter_method"]),
                    )
                except KeyError as _e:
                    raise ValueError(
                        "Invalid subplot configuration: missing "
                        "key `plot_adapter_method` for subplot type '{}' "
                        "in PLOT_SUBPLOT_TYPES in {}!".format(_pt, self.__class__)
                    ) from _e
                except AttributeError as _e:
                    raise TypeError(
                        "Cannot handle plot of type '{}': "
                        "cannot find corresponding plot method "
                        "'{}' in {}!".format(_pt, _pt_spec["plot_adapter_method"], self.__class__)
                    ) from _e
                self._subplots[_pt].setdefault("plot_method_keywords", {})

        return self._subplots

    def _set_plot_labels(self):
        """Obtain the data and model labels from data and model container and set them
        accordingly.
        """
        if self._fit.data_container.label is not None:
            try:
                self.update_plot_kwargs("data", dict(label=self._fit.data_container.label))
            except ValueError:
                pass  # no data present
        if self._fit.model_label is not None:
            for plot_model_name in ("model", "model_line"):
                try:
                    self.update_plot_kwargs(plot_model_name, dict(label=self._fit.model_label))
                except ValueError:
                    pass  # no model plot function available
            _model_error_name = (
                kc("fit", "plot", "error_label") % dict(model_label=self._fit.model_label) if self._fit.model_label != "__del__" else "__del__"
            )
            try:
                self.update_plot_kwargs("model_error_band", dict(label=_model_error_name))
            except ValueError:
                pass  # no error band available

    def _get_subplot_kwargs(self, plot_index, plot_type):
        """Resolve the keyword arguments passed to the plot method."""
        _subplots = self._get_subplots()

        try:
            _subplot = _subplots[plot_type]
        except KeyError as exc:
            raise ValueError(f"Unknown subplot: {plot_type} . Available: {list(_subplots.keys())}") from exc
        _explicit_kwargs = _subplot.get("plot_method_keywords", {})

        # get static kwargs
        _plot_style_as = _subplots[plot_type].get("plot_style_as", plot_type)

        # retrieve default plot keywords from style config
        _kwargs = dict(kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, _plot_style_as, "plot_kwargs"))

        # initialize property cycler from style config and commit keywords
        _prop_cycler_args = kc_plot_style(self.PLOT_STYLE_CONFIG_DATA_TYPE, _plot_style_as, "property_cycler")
        _prop_cycler = Cycler(*_prop_cycler_args)
        _kwargs.update(**_prop_cycler.get(plot_index))

        # override keywords with one explicitly set via the API
        _kwargs.update(**_explicit_kwargs)

        # remove keywords set to the special value '__del__'
        _kwargs = {_k: _v for _k, _v in six.iteritems(_kwargs) if _v != "__del__"}

        # apply interpolation to legend labels
        _label = _kwargs.pop("label", None)
        if _label:
            _kwargs["label"] = _label % dict(subplot_id=plot_index, plot_type=plot_type)

        # calculate zorder if not explicitly given
        _n_defined_plot_types = len(_subplots)
        if "zorder" not in _kwargs:
            _kwargs["zorder"] = plot_index * _n_defined_plot_types + list(_subplots).index(plot_type)

        _container_valid = _subplots[plot_type].get("container_valid", None)
        if _container_valid is not None:
            _kwargs["container_valid"] = _container_valid

        return _kwargs

    def _get_total_error(self, error_contributions):
        _total_err = np.zeros_like(self.data_y)
        for _ec in error_contributions:
            _ec = _ec.lower()
            if _ec not in ("data", "model"):
                raise ValueError("Unknown error contribution specification '{}': " "expecting 'data' or 'model'".format(_ec))
            _total_err += getattr(self, _ec + "_yerr") ** 2
            _total_err += self._fit._cost_function.get_uncertainty_gaussian_approximation(getattr(self, _ec + "_y")) ** 2

        _total_err = np.sqrt(_total_err)

        if np.all(_total_err == 0):
            return None
        return _total_err

    # -- public API

    def call_plot_method(self, plot_type, target_axes, **kwargs):
        """Call the registered plot method for `plot_type`.

        :param plot_type: key identifying a registered plot type for this :py:obj:`PlotAdapter`
        :type plot_type: str
        :param target_axes: axes to plot to
        :type target_axes: `matplotlib.Axes` object
        :param kwargs: keyword arguments to pass to the plot method
        :type kwargs: dict
        :return: return value of the plot method
        """
        _subplots = self._get_subplots()

        if plot_type not in _subplots:
            raise ValueError("Cannot call plot method: unknown plot type '{}'! " "Expecting one of: {!r}".format(plot_type, list(_subplots)))

        _callable = _subplots[plot_type].get("plot_adapter_method", None)

        if _callable is None:
            raise ValueError("Cannot call plot method: missing key `plot_adapter_method` " "in configuration for plot type '{}'!".format(plot_type))
        if not callable(_callable):
            raise ValueError(
                "Cannot call plot method: registered `plot_adapter_method` "
                "with type {!r} in configuration for plot type '{}' "
                "is not callable!".format(type(_callable), plot_type)
            )

        if kwargs.pop("hide", False):
            return None

        if not kwargs.pop("container_valid", False) and self._from_container:
            return

        return _callable(target_axes=target_axes, **kwargs)

    def update_plot_kwargs(self, plot_type, plot_kwargs):
        """Update the value of keyword arguments `plot_kwargs` to be passed
        to the plot method for for `plot_type`.

        If a keyword argument should be removed, the value of the keyword
        in `plot_kwargs` can be set to the special value ``'__del__'``.
        To indicate that the default value should be used, the special
        value ``'__default__'`` can be set as a value.

        :param plot_type: key identifying a registered plot type for this :py:obj:`PlotAdapter`
        :type plot_type: str
        :param plot_kwargs: dictionary containing keywords arguments to override
        :type plot_kwargs: dict
        """
        _subplots = self._get_subplots()

        if plot_type not in _subplots:
            raise ValueError(
                "Cannot set custom plot keyword arguments "
                "for plot type '{}': no plot with this type defined "
                "for this adapter!".format(plot_type)
            )

        # remove keys set to '__default__': the defaults will be substituted in
        # then _get_subplot_kwargs() is called. The '__del__' special value is
        # handled by _get_subplot_kwargs()
        _keys_to_delete = [_k for _k, _v in six.iteritems(plot_kwargs) if _v == "__default__"]

        _config_dict = _subplots[plot_type].setdefault("plot_method_keywords", {})
        _config_dict.update(plot_kwargs)

        for _key in _keys_to_delete:
            try:
                del _config_dict[_key]
            except KeyError:
                pass  # ok, key not present

    # -- properties

    @property
    @abc.abstractmethod
    def data_x(self):
        """The *x* coordinates of the data (used by :py:meth:`~plot_data`).

        :rtype: numpy.ndarray
        """
        pass

    @property
    @abc.abstractmethod
    def data_y(self):
        """The *y* coordinates of the data (used by :py:meth:`~plot_data`).

        :rtype: numpy.ndarray
        """
        pass

    @property
    @abc.abstractmethod
    def data_xerr(self):
        """The magnitude of the data *x* error bars (used by :py:meth:`~plot_data`).

        :rtype: numpy.ndarray
        """
        pass

    @property
    @abc.abstractmethod
    def data_yerr(self):
        """The magnitude of the data *y* error bars (used by :py:meth:`~plot_data`).

        :rtype: numpy.ndarray
        """
        pass

    @property
    @abc.abstractmethod
    def model_x(self):
        """The *x* coordinates of the model (used by :py:meth:`~plot_model`).

        :rtype: numpy.ndarray
        """
        pass

    @property
    @abc.abstractmethod
    def model_y(self):
        """The *y* coordinates of the model (used by :py:meth:`~plot_model`).

        :rtype: numpy.ndarray
        """
        pass

    @property
    @abc.abstractmethod
    def model_xerr(self):
        """The magnitude of the model *x* error bars (used by :py:meth:`~plot_model`).

        :rtype: numpy.ndarray
        """
        pass

    @property
    @abc.abstractmethod
    def model_yerr(self):
        """The magnitude of the model *y* error bars (used by :py:meth:`~plot_model`).

        :rtype: numpy.ndarray
        """
        pass

    @property
    def x_range(self):
        """The *x* axis plot range.

        :rtype: tuple[float, float]
        """
        return self._x_range

    @x_range.setter
    def x_range(self, x_range):
        self._x_range = tuple(x_range)

    @property
    def y_range(self):
        """The *y* axis plot range.

        :rtype: tuple[float, float]
        """
        return self._y_range

    @y_range.setter
    def y_range(self, y_range):
        self._y_range = tuple(y_range)

    @property
    def x_scale(self):
        """The *x* axis scale. Available scales are given in :py:const:`~.AVAILABLE_X_SCALES`

        :rtype: str
        """
        return self._x_scale

    @x_scale.setter
    def x_scale(self, scale):
        if scale not in self.AVAILABLE_X_SCALES:
            raise TypeError("x_scale {} is not supported for this type of fit, " "use one of {}".format(scale, self.AVAILABLE_X_SCALES))
        self._x_scale = scale

    @property
    def y_scale(self):
        """The *y* axis scale. Available scales are given in :py:const:`~.AVAILABLE_Y_SCALES`

        :rtype: str
        """
        return self._y_scale

    @y_scale.setter
    def y_scale(self, scale):
        if scale not in self.AVAILABLE_Y_SCALES:
            raise TypeError("y_scale {} is not supported for this type of fit, " "use one of {}".format(scale, self.AVAILABLE_Y_SCALES))
        self._y_scale = scale

    @property
    def x_label(self):
        """The *x* axis label of the fit handled by this plot adapter. If ``'__del__'`` is used,
        the label will be set to :py:obj:`None`.

        :rtype: str
        """
        return self._axis_labels[0]

    @x_label.setter
    def x_label(self, label):
        if label == "__del__":
            label = None
        self._axis_labels[0] = label

    @property
    def y_label(self):
        """The *x* axis label of the fit handled by this plot adapter. If ``'__del__'`` is used,
        the label will be set to :py:obj:`None`.

        :rtype: str
        """
        return self._axis_labels[1]

    @y_label.setter
    def y_label(self, label):
        if label == "__del__":
            label = None
        self._axis_labels[1] = label

    @property
    def x_ticks(self):
        return self._ticks[0]

    @x_ticks.setter
    def x_ticks(self, ticks):
        self._ticks[0] = ticks

    @property
    def y_ticks(self):
        return self._ticks[1]

    @y_ticks.setter
    def y_ticks(self, ticks):
        self._ticks[1] = ticks

    @property
    def from_container(self):
        """Whether the contained fit object was created ad-hoc from just a data container."""
        return self._from_container

    @abc.abstractmethod
    def plot_data(self, target_axes, **kwargs):
        """Method called by the main plot routine to plot the data points to a specified
        :py:class:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:mod:`matplotlib` target axes.
        :return: plot handle(s)
        """
        pass

    @abc.abstractmethod
    def plot_model(self, target_axes, **kwargs):
        """Method called by the main plot routine to plot the model to a specified
        :py:class:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:mod:`matplotlib` target axes.
        :return: plot handle(s)
        """
        pass

    def plot_ratio(self, target_axes, error_contributions=("data",), **kwargs):
        """Plot the data/model ratio to a specified :py:obj:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:obj:`matplotlib` axes used for plotting.
        :param error_contributions: Which error contributions to include when plotting the data.
            Can either be ``data``, ``'model'`` or both.
        :type error_contributions: str or Tuple[str]
        :param dict kwargs: Keyword arguments accepted by :py:obj:`matplotlib.pyplot.errorbar`.
        :return: plot handle(s)
        """

        _yerr = self._get_total_error(error_contributions)
        if _yerr is not None:
            _yerr /= self.model_y

        # TODO: how to handle case when x and y error/model differ?
        return target_axes.errorbar(self.data_x, self.data_y / self.model_y, xerr=self.data_xerr, yerr=_yerr, **kwargs)

    def plot_residual(self, target_axes, error_contributions=("data",), **kwargs):
        """Plot the residuals to a :py:obj:`matplotlib.axes.Axes` object.

        :param matplotlib.axes.Axes target_axes: The :py:obj:`matplotlib` axes used for plotting.
        :param error_contributions: Which error contributions to include when plotting the data.
            Can either be ``data``, ``'model'`` or both.
        :type error_contributions: str or Tuple[str]
        :param dict kwargs: Keyword arguments accepted by :py:obj:`matplotlib.pyplot.errorbar`.
        :return: plot handle(s)
        """
        # TODO: how to handle case when x and y error/model differ?
        return target_axes.errorbar(
            self.data_x,
            self.data_y - self.model_y,
            xerr=self.data_xerr,
            yerr=self._get_total_error(error_contributions),
            **kwargs,
        )

    # Overridden by multi plot adapters
    def get_formatted_model_function(self, **kwargs):
        """return model function string"""
        return self._fit.model_function.formatter.get_formatted(**kwargs)

    # Overridden by multi plot adapters
    @property
    def model_function_parameter_formatters(self):
        """The model function parameter formatters, excluding the independent variable."""
        return self._fit.model_function.formatter.par_formatters


# -- must come last!


class Plot:
    """
    This is a class implementing the creation of Fits from one of more subclasses of
    :py:obj`PlotAdapterBase`. Consequently a :py:obj:`Plot` object manages one or several
    ``matplotlib`` figures. It controls the overall figure layout and is responsible for axes,
    subplot and legend management.
    """

    FIT_INFO_STRING_FORMAT_CHI2 = textwrap.dedent(
        """\
        {model_function}
        {parameters}
            $\\hookrightarrow${fit_quality}
            $\\hookrightarrow \\chi^2 \\, \\mathrm{{probability =}}${chi2_probability}
    """
    )
    FIT_INFO_STRING_FORMAT_SATURATED = textwrap.dedent(
        """\
        {model_function}
        {parameters}
            $\\hookrightarrow${fit_quality}
    """
    )
    FIT_INFO_STRING_FORMAT_NOT_SATURATED = textwrap.dedent(
        """\
        {model_function}
        {parameters}
            $\\hookrightarrow${cost}
            $\\hookrightarrow${fit_quality}
    """
    )

    def __init__(self, fit_objects, separate_figures=False):
        """
        :param fit_objects: which kafe2 fits to use for the plot. A positive integer is interpreted
            as the fit with the given index that has been performed (with wrappers) since the
            program started. A negative integer *-n* is interpreted as the last *n* fits. kafe2 fit
            objects are used directly. If a sequence of fit objects is passed they can be displayed
            as part of the same plot.
        :type fit_objects: int or :py:class:`~kafe2.fit._base.FitBase`
            or Sequence[:py:class:`~kafe2.fit._base.FitBase`]
        :param separate_figures: whether the fits should be displayed in separate figures.
        :type separate_figures: bool
        """
        from kafe2.fit.tools.fit_wrapper import Fit

        # set the managed fit objects
        if isinstance(fit_objects, MultiFit):
            self._multifit = fit_objects
            fit_objects = fit_objects.fits
        else:
            self._multifit = None
        if isinstance(fit_objects, int):
            if fit_objects >= 0:
                fit_objects = [_fit_history[fit_objects]["fit"]]
            else:
                fit_objects = [_f["fit"] for _f in _fit_history[fit_objects:]]
        try:
            iter(fit_objects)
        except TypeError:
            fit_objects = (fit_objects,)
        self._from_container = tuple(isinstance(_fo, DataContainerBase) for _fo in fit_objects)
        self._fits = tuple(Fit(_fo) if _fc else _fo for _fc, _fo in zip(self._from_container, fit_objects))

        self._separate_figs = separate_figures

        # owned objects
        self._figure_dicts = []
        self._plot_adapters = None
        self._current_results = None

    # -- private methods

    def _get_axes(self, axes_key):
        try:
            return self._current_axes[axes_key]
        except KeyError:
            raise KeyError("No axes found for name '{}'!".format(axes_key))

    def _create_figure_axes(self, axes_keys, height_ratios=None, width_ratios=None, figsize=None):
        if height_ratios:
            assert len(axes_keys) == len(height_ratios)

        # plot axes layout
        _plot_axes_gs = gs.GridSpec(
            nrows=len(axes_keys),
            ncols=2,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
        )

        # create figure
        self._current_figure = plt.figure(figsize=figsize)
        self._figure_dicts.append(dict(figure=self._current_figure))
        # 'tight_layout' has a bug in matplotlib < 2
        if not mpl.__version__.startswith("1"):
            self._current_figure.set_tight_layout(dict(h_pad=0.1))

        # create named axes
        self._current_axes = self._figure_dicts[-1]["axes"] = {
            _k: self._current_figure.add_subplot(_plot_axes_gs[_i, 0]) for _i, _k in enumerate(axes_keys)
        }
        # create a fake axes for the legend
        self._current_axes["__legendfakeaxes__"] = self._current_figure.add_subplot(_plot_axes_gs[:, 1])
        self._current_axes["__legendfakeaxes__"].set_visible(False)

        self._current_results = None  # populated on 'plot()'

    def _get_plot_adapters(self, plot_indices=None):
        """retrieve plot adapters, creating them if needed"""

        plot_indices = plot_indices or range(len(self._fits))

        if self._plot_adapters is None:
            self._plot_adapters = []
            for _fit, _from_container in zip(self._fits, self._from_container):
                self._plot_adapters.append(_fit.PLOT_ADAPTER_TYPE(_fit, _from_container))
        if not self._separate_figs:
            _x_range_min = np.min([_pa.x_range[0] for _pa in self._plot_adapters])
            _x_range_max = np.max([_pa.x_range[1] for _pa in self._plot_adapters])
            for _plot_adapter in self._plot_adapters:
                _plot_adapter.x_range = (_x_range_min, _x_range_max)

        return [self._plot_adapters[_idx] for _idx in plot_indices]

    def _plot_and_get_results(self, plot_indices=None):
        plot_indices = plot_indices or range(len(self._fits))
        _plot_adapters = self._get_plot_adapters(plot_indices)
        if self._multifit is None:
            for _plot_adapter in _plot_adapters:
                if not _plot_adapter._fit.did_fit and not _plot_adapter.from_container:
                    warnings.warn("No fit has been performed for {}. Did you forget to run fit.do_fit()?".format(_plot_adapter._fit))
        elif not self._multifit.did_fit:
            warnings.warn("No fit has been performed for {}. Did you forget to run fit.do_fit()?".format(self._multifit))

        _plots = {}
        for _i_pdc, _pdc in zip(plot_indices, _plot_adapters):
            if not _pdc.PLOT_SUBPLOT_TYPES:
                continue

            for _i_pt, (_pt, _pt_spec) in enumerate(six.iteritems(_pdc.PLOT_SUBPLOT_TYPES)):
                _axes_key = _pt_spec["target_axes"]

                # skip plot elements meant for an inexistent axes
                if _axes_key not in self._current_axes:
                    continue

                _axes_plot_dicts = _plots.setdefault(_axes_key, {})

                _axes_plots = _axes_plot_dicts.setdefault("plots", [])

                _plot_kwargs = _pdc._get_subplot_kwargs(_i_pdc, _pt)
                if "zorder" not in _plot_kwargs:
                    _plot_kwargs["zorder"] = 0
                _plot_kwargs["zorder"] = _plot_kwargs["zorder"] - 10 * _i_pdc

                _artist = _pdc.call_plot_method(_pt, target_axes=self._get_axes(_axes_key), **_plot_kwargs)

                _axes_plots.append(
                    {
                        "type": _pt,
                        "fit_index": _i_pdc,
                        "adapter": _pdc,
                        "artist": _artist,
                    }
                )

                if _pdc.x_range is not None:
                    _xlim = _axes_plot_dicts.setdefault("x_range", _pdc.x_range)
                    _axes_plot_dicts["x_range"] = (
                        min(_xlim[0], _pdc.x_range[0]),
                        max(_xlim[1], _pdc.x_range[1]),
                    )

                if _pdc.y_range is not None and _axes_key != "ratio":  # y_range of ratio can be adjusted by plot kwargs
                    _ylim = _axes_plot_dicts.setdefault("y_range", _pdc.y_range)
                    _axes_plot_dicts["y_range"] = (
                        min(_ylim[0], _pdc.y_range[0]),
                        max(_ylim[1], _pdc.y_range[1]),
                    )

        return _plots

    def _get_fit_info(self, plot_adapter, format_as_latex, asymmetric_parameter_errors):
        if self._multifit is None:
            plot_adapter._fit._update_parameter_formatters(update_asymmetric_errors=asymmetric_parameter_errors)
        else:
            if asymmetric_parameter_errors:
                self._multifit.asymmetric_parameter_errors
            self._multifit._update_parameter_formatters(update_asymmetric_errors=asymmetric_parameter_errors)

        _cost_func = plot_adapter._fit._cost_function  # TODO: public interface

        _ndf = plot_adapter._fit.ndf
        _cost_function_value = plot_adapter._fit.cost_function_value
        _gof_value = plot_adapter._fit.goodness_of_fit
        _info_format_dict = dict(
            model_function=plot_adapter.get_formatted_model_function(
                with_par_values=False,
                n_significant_digits=2,
                format_as_latex=format_as_latex,
                with_expression=True,
            ),
            parameters="\n".join(
                [
                    "    "
                    + _pf.get_formatted(
                        with_name=True,
                        with_value=True,
                        with_errors=plot_adapter._fit.errors_valid,
                        asymmetric_error=asymmetric_parameter_errors,
                        format_as_latex=format_as_latex,
                    )
                    for _pf in plot_adapter.model_function_parameter_formatters
                ]
            ),
        )
        # _gof_value is None for UnbinnedFit and some user-defined cost functions.
        if plot_adapter._fit.errors_valid and _gof_value is not None:
            _info_format_dict["fit_quality"] = _cost_func.formatter.get_formatted(
                value=_gof_value,
                with_name=True,
                saturated=True,
                n_degrees_of_freedom=_ndf,
                with_value_per_ndf=True,
                format_as_latex=format_as_latex,
            )
            if plot_adapter._fit._cost_function.is_chi2:
                _info_format_string = self.FIT_INFO_STRING_FORMAT_CHI2
                _chi2_pf = ParameterFormatter("chi2", plot_adapter._fit.chi2_probability)
                _info_format_dict["chi2_probability"] = _chi2_pf.get_formatted(n_significant_digits=3, format_as_latex=format_as_latex)
            elif plot_adapter._fit._cost_function.saturated:
                _info_format_string = self.FIT_INFO_STRING_FORMAT_SATURATED
            else:
                _info_format_string = self.FIT_INFO_STRING_FORMAT_NOT_SATURATED
                _info_format_dict["cost"] = _cost_func.formatter.get_formatted(
                    value=_cost_function_value, with_name=True, format_as_latex=format_as_latex
                )
        else:
            _info_format_string = self.FIT_INFO_STRING_FORMAT_SATURATED
            _info_format_dict["fit_quality"] = _cost_func.formatter.get_formatted(
                value=_cost_function_value, with_name=True, format_as_latex=format_as_latex
            )

        _info_text = _info_format_string.format(**_info_format_dict)

        if self._multifit is not None:
            _multi_ndf = self._multifit.ndf
            _multi_cost_function = self._multifit._cost_function
            _multi_cost_function_value = self._multifit.cost_function_value
            _multi_gof = self._multifit.goodness_of_fit

            _multi_info_dict = dict()
            if _multi_cost_function.is_chi2:
                _template = "    $\\hookrightarrow$ global {fit_quality}\n"
                _chi2_pf = ParameterFormatter("chi2", self._multifit.chi2_probability)
                _multi_info_dict["chi2_probability"] = _chi2_pf.get_formatted(n_significant_digits=3, format_as_latex=format_as_latex)
                _template += "    $\\hookrightarrow$ global $\\chi^2 \\, \\mathrm{{probability}} " "= ${chi2_probability}"
            elif _multi_cost_function.saturated:
                _template = "    $\\hookrightarrow$ global cost / ndf = {fit_quality}\n"
            else:
                _template = "    $\\hookrightarrow$ global cost = {cost}\n"
                _multi_info_dict["cost"] = _multi_cost_function.formatter.get_formatted(
                    value=_multi_cost_function_value,
                    with_name=False,
                    format_as_latex=format_as_latex,
                )
                if _multi_gof is not None:
                    _template += "    $\\hookrightarrow$ global GoF / ndf = {fit_quality}\n"

            if _multi_gof is not None:
                _multi_info_dict["fit_quality"] = _multi_cost_function.formatter.get_formatted(
                    value=_multi_gof,
                    with_name=_multi_cost_function.is_chi2,
                    n_degrees_of_freedom=_multi_ndf,
                    with_value_per_ndf=True,
                    format_as_latex=format_as_latex,
                )

            _info_text += _template.format(**_multi_info_dict)
        return _info_text

    def _render_legend(self, plot_results, axes_keys, fit_info=True, asymmetric_parameter_errors=False, **kwargs):
        """render the legend for axes `axes_keys`"""
        for _axes_key in axes_keys:
            _axes = self._get_axes(_axes_key)

            _hs_unsorted, _ls_unsorted = _axes.get_legend_handles_labels()
            _hs_sorted, _ls_sorted = [], []

            _axes_plots = plot_results[_axes_key]["plots"]

            # -- go through each plot in order and generate the legend entry

            _prev_fit_index = None
            _fit_info = {}
            if isinstance(fit_info, bool):
                fit_info = [fit_info] * len(self._fits)
            for _i_plot, _plot_dict in enumerate(_axes_plots):
                # check if artist available for this plot
                try:
                    # if multiple artists were stored for a plot,
                    # only show the first  in the legend
                    try:
                        _artist_index = _hs_unsorted.index(_plot_dict["artist"][0])
                    except (ValueError, TypeError):
                        _artist_index = _hs_unsorted.index(_plot_dict["artist"])

                except (KeyError, ValueError):
                    # artist not available or not plottable -> skip
                    continue

                # append handle and label to the legend
                _hs_sorted.append(_hs_unsorted[_artist_index])
                _ls_sorted.append(_ls_unsorted[_artist_index])

                # if requested, compute info of fit associated to this artist
                _fit_index = _plot_dict["fit_index"]
                _plot_adapter = _plot_dict["adapter"]
                if fit_info[_fit_index] and not _plot_adapter.from_container:
                    if _fit_index not in _fit_info:
                        # compute fit info string (if not computed yet)
                        _fit_info[_fit_index] = dict(
                            text=self._get_fit_info(
                                _plot_adapter,
                                format_as_latex=True,
                                asymmetric_parameter_errors=asymmetric_parameter_errors,
                            )
                        )

                    # update the legend position at which to insert the text
                    _fit_info[_fit_index].update(
                        # put fit info directly after the last visible legend
                        # entry that corresponds to this fit
                        pos=len(_hs_sorted)
                    )

            # insert fit infos at the right positions
            for _i, _fi_dict in _fit_info.items():
                _hs_sorted.insert(_fi_dict["pos"] + _i, "_nokey_")
                _ls_sorted.insert(_fi_dict["pos"] + _i, _fi_dict["text"])

            # -- legend layout

            _zorder = kwargs.pop("zorder", 999)
            _bbox_to_anchor = kwargs.pop("bbox_to_anchor", None)
            if _bbox_to_anchor is None:
                _bbox_to_anchor = (1.05, 0.0, 0.67, 1.0)  # axes coordinates FIXME: no hardcoding!

            _mode = kwargs.pop("mode", "expand")
            _borderaxespad = kwargs.pop("borderaxespad", 0.1)
            _ncol = kwargs.pop("ncol", 1)

            kwargs["loc"] = "upper left"

            # Note: legend must be attached to *figure*, not *axes*,
            # otherwise 'tight_layout' will consider it part of the axes
            # and produce undesirable layouts

            _leg = _axes.get_figure().legend(
                _hs_sorted,
                _ls_sorted,
                mode=_mode,
                borderaxespad=_borderaxespad,
                ncol=_ncol,
                fontsize=rcParams["font.size"],
                handler_map={"_nokey_": DummyLegendHandler()},
                **kwargs,
            )
            _leg.set_zorder(_zorder)

            # manually change bbox from figure to axes
            _leg._bbox_to_anchor = self._get_axes("__legendfakeaxes__").bbox

    def _adjust_plot_ranges(self, plot_results):
        """set the x and y ranges (all axes) to the total data range reported by the plot adapters"""
        for _axes_name, _axes_dict in six.iteritems(plot_results):
            _ax = self._get_axes(_axes_name)

            _xlim = _axes_dict.get("x_range", None)
            if _xlim:
                _ax.set_xlim(_xlim)

            _ylim = _axes_dict.get("y_range", None)
            if _ylim:
                _ax.set_ylim(_ylim)

    def _set_axis_labels(self, plot_results, axes_keys):
        """Set the x and y axis labels from given plot adapter labels"""
        for _axes_name, _axes_dict in six.iteritems(plot_results):
            _ax = self._get_axes(_axes_name)

            # collect different sets of axis labels
            _seen_x_labels = list()
            _seen_y_labels = list()
            for _plot in _axes_dict["plots"]:
                _x_label = _plot["adapter"].x_label
                _y_label = _plot["adapter"].y_label
                if _x_label not in _seen_x_labels:
                    _seen_x_labels.append(_x_label)
                if _y_label not in _seen_y_labels:
                    _seen_y_labels.append(_y_label)

            # use concatenation of labels as axis label, filter to skip None labels
            _ax.set_xlabel(", ".join(filter(None, [label for label in _seen_x_labels])))
            _ax.set_ylabel(", ".join(filter(None, [label for label in _seen_y_labels])))

            _adapter = _axes_dict["plots"][0]["adapter"]
            _x_ticks = _adapter.x_ticks
            _y_ticks = _adapter.y_ticks

            if len(_x_ticks) > 0:
                if isinstance(_x_ticks[0], str):
                    _ax.set_xticks(np.arange(len(_x_ticks)), labels=_x_ticks)
                else:
                    _ax.set_xticks(_x_ticks)
                # Use default formatter for custom ticks, log formatter doesn't seem to work:
                _ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

            if len(_y_ticks) > 0:
                _ax.set_yticks(_y_ticks)
                # Use default formatter for custom ticks, log formatter doesn't seem to work:
                _ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

        # hide x tick labels in all but the lowest axes
        for _key in axes_keys[:-1]:
            self._current_axes[_key].set_xlabel(None)
            for _label in self._current_axes[_key].get_xticklabels():
                _label.set_visible(False)

    # -- public properties

    @property
    def figures(self):
        """The ``matplotlib`` figures managed by this object."""
        return [_d["figure"] for _d in self._figure_dicts]

    @property
    def axes(self):
        """A list of dictionaries (one per figure) mapping names to
        ``matplotlib`` `Axes` objects contained in this figure."""
        return [_d["axes"] for _d in self._figure_dicts]

    @property
    def x_range(self):
        """The plotting x-range for each fit handled by this :py:obj:`~kafe2.Plot` object.
        :param: List of tuples containing the x_ranges for each fit.
        :type: list[tuple[float, float]] or tuple[float, float]
        """
        return [_adapter.x_range for _adapter in self._get_plot_adapters()]

    @x_range.setter
    def x_range(self, x_range):
        _adapters = self._get_plot_adapters()
        if np.ndim(x_range) == 1:
            if len(x_range) != 2:
                raise ValueError("x_range must contain two elements. A lower and an upper limit. " "Got {} elements".format(len(x_range)))
            x_range = itertools.repeat(x_range, len(_adapters))
        elif len(_adapters) != len(x_range):
            raise ValueError("Amount of x_ranges and fits does not match. Got {} x_ranges and have " "{} fits".format(len(x_range), len(_adapters)))
        for i, _range in enumerate(x_range):
            _adapters[i].x_range = _range

    @property
    def y_range(self):
        """The plotting y-range for each fit handled by this :py:obj:`~kafe2.Plot` object.
        :param: List of tuples containing the y_ranges for each fit.
        :type: list[tuple[float, float]] or tuple[float, float]
        """
        return [_adapter.y_range for _adapter in self._get_plot_adapters()]

    @y_range.setter
    def y_range(self, y_range):
        _adapters = self._get_plot_adapters()
        if np.ndim(y_range) == 1:
            if len(y_range) != 2:
                raise ValueError("y_range must contain two elements. A lower and an upper limit. " "Got {} elements".format(len(y_range)))
            y_range = itertools.repeat(y_range, len(_adapters))
        elif len(_adapters) != len(y_range):
            raise ValueError("Amount of y_ranges and fits does not match. Got {} y_ranges and " "have {} fits".format(len(y_range), len(_adapters)))
        for i, _range in enumerate(y_range):
            _adapters[i].y_range = _range

    def _repeat_str(self, var):
        """Repeat a string with the count of the given plot adapters. Used by setters."""
        _adapters = self._get_plot_adapters()
        if isinstance(var, str):  # check if string, if not probably list
            var = itertools.repeat(var, len(_adapters))
        elif len(_adapters) != len(var):
            raise ValueError("Length of input and fits does not match. Got {} inputs and have {} " "fits".format(len(var), len(_adapters)))
        return var

    def _repeat_ticks(self, ticks):
        try:
            iter(ticks)
        except TypeError:
            raise TypeError(f"Ticks have to be iterable but received {ticks}")
        try:
            iter(ticks[0])
            _ticks_is_1d_iter = isinstance(ticks[0], str)
        except TypeError:
            _ticks_is_1d_iter = True
        if _ticks_is_1d_iter:
            ticks = itertools.repeat(ticks, len(self._get_plot_adapters()))
        return ticks

    @property
    def x_scale(self):
        """The x-scale for each fit used for creating the support values when plotting and axis
        scaling.
        :type: list[str] or str
        """
        return [_adapter.x_scale for _adapter in self._get_plot_adapters()]

    @x_scale.setter
    def x_scale(self, x_scale):
        x_scale = self._repeat_str(x_scale)
        for adapter, _scale in zip(self._get_plot_adapters(), x_scale):
            adapter.x_scale = _scale

    @property
    def y_scale(self):
        """The y-scale for each fit used when plotting.
        :type: list[str] or str
        """
        return [_adapter.y_scale for _adapter in self._get_plot_adapters()]

    @y_scale.setter
    def y_scale(self, y_scale):
        y_scale = self._repeat_str(y_scale)
        for adapter, _scale in zip(self._get_plot_adapters(), y_scale):
            adapter.y_scale = _scale

    @property
    def x_label(self):
        """The x-label(s) of the plot. If multiple fits are handled by this plot this is a list of
        strings. Multiple labels will be separated by a comma in the final plot while skipping
        duplicates.
        If a label is :py:obJ:`None` or ``'__del__'`` it will be removed.
        :type: str or list[str]"""
        return [_adapter.x_label for _adapter in self._get_plot_adapters()]

    @x_label.setter
    def x_label(self, label):
        label = self._repeat_str(label)
        for _l, adapter in zip(label, self._get_plot_adapters()):
            adapter.x_label = _l

    @property
    def y_label(self):
        """The y-label(s) of the plot. If multiple fits are handled by this plot this is a list of
        strings. Multiple labels will be separated by a comma in the final plot while skipping
        duplicates.
        If a label is :py:obJ:`None` or ``'__del__'`` it will be removed.
        :type: str or list[str]"""
        return [_adapter.y_label for _adapter in self._get_plot_adapters()]

    @y_label.setter
    def y_label(self, label):
        label = self._repeat_str(label)
        for _l, adapter in zip(label, self._get_plot_adapters()):
            adapter.y_label = _l

    @property
    def x_ticks(self):
        return [_adapter.x_ticks for _adapter in self._get_plot_adapters()]

    @x_ticks.setter
    def x_ticks(self, ticks):
        for _t, adapter in zip(self._repeat_ticks(ticks), self._get_plot_adapters()):
            adapter.x_ticks = _t

    @property
    def y_ticks(self):
        return [_adapter.y_ticks for _adapter in self._get_plot_adapters()]

    @y_ticks.setter
    def y_ticks(self, ticks):
        for _t, adapter in zip(self._repeat_ticks(ticks), self._get_plot_adapters()):
            adapter.y_ticks = _t

    # public methods

    @staticmethod
    def show(*args, **kwargs):
        """Convenience wrapper for matplotlib.pyplot.show()"""
        plt.show(*args, **kwargs)

    def plot(
        self,
        legend=True,
        fit_info=True,
        asymmetric_parameter_errors=False,
        ratio=False,
        ratio_range=None,
        ratio_height_share=0.25,
        residual=False,
        residual_range=None,
        residual_height_share=0.25,
        plot_width_share=0.5,
        font_scale=1.0,
        figsize=None,
    ):
        """
        Plot data, model (and other subplots) for all child :py:obj:`Fit` objects.

        :param legend: if ``True``, a legend is rendered
        :param fit_info: If :py:obj:`True`, fit results will be shown in the legend. This can also
            be a list of booleans, corresponding to the fits handled by this
            :py:obj:`~.Plot`-object.
        :type legend: bool or typing.Collection[bool]
        :param asymmetric_parameter_errors: if ``True``, parameter errors in fit results will be asymmetric
        :param ratio: if ``True``, a secondary plot containing data/model ratios is shown below the main plot
        :param ratio_range: the *y* range to set in the secondary plot
        :type ratio_range: tuple of 2 floats
        :param ratio_height_share: share of the total height to be taken up by the secondary plot
        :type ratio_height_share: float
        :param plot_width_share: share of the total width to be taken up by the plot(s)
        :type plot_width_share: float
        :param font_scale: multiply font size by this amount.
        :type font_scale: float
        :param figsize: the (*width*, *height*) of the figure (in inches) or ``None`` to use default
        :type figsize: tuple of 2 floats

        :return: dictionary containing information about the plotted objects
        :rtype: dict
        """
        if ratio and residual:
            raise NotImplementedError("Cannot plot ratio and residual at the same time.")

        with rc_context(kafe2_rc):
            rcParams["font.size"] *= font_scale
            _axes_keys = ("main",)
            _height_ratios = [1.0]
            _width_ratios = (plot_width_share, 1.0 - plot_width_share)

            if ratio:
                _axes_keys += ("ratio",)
                _height_ratios[0] -= ratio_height_share
                _height_ratios.append(ratio_height_share)
            elif residual:
                _axes_keys += ("residual",)
                _height_ratios[0] -= residual_height_share
                _height_ratios.append(residual_height_share)

            _all_plot_results = []
            for i in range(len(self._fits) if self._separate_figs else 1):
                self._create_figure_axes(
                    _axes_keys,
                    width_ratios=_width_ratios,
                    height_ratios=_height_ratios,
                    figsize=figsize,
                )

                _plot_results = self._plot_and_get_results(plot_indices=(i,) if self._separate_figs else None)

                # set axis scales for the main plot, x-axis is shared with ratio plot
                self._get_axes("main").set_xscale(self.x_scale[i])
                self._get_axes("main").set_yscale(self.y_scale[i])

                if legend:
                    self._render_legend(
                        plot_results=_plot_results,
                        axes_keys=("main",),
                        fit_info=fit_info,
                        asymmetric_parameter_errors=asymmetric_parameter_errors,
                    )

                self._adjust_plot_ranges(_plot_results)
                self._set_axis_labels(_plot_results, axes_keys=_axes_keys)
                try:
                    self._current_figure.align_ylabels()
                except AttributeError:
                    # matplotlib < 2.0.0
                    pass

                if ratio:
                    _axis = self._current_axes["ratio"]
                    _ratio_label = kc("fit", "plot", "ratio_label")
                    _axis.set_ylabel(_ratio_label)
                    if ratio_range is None:
                        _plot_adapters = self._get_plot_adapters()[i : i + 1] if self._separate_figs else self._get_plot_adapters()
                        _max_abs_deviation = 0
                        for _plot_adapter in _plot_adapters:
                            _max_abs_deviation = max(
                                _max_abs_deviation,
                                np.max(
                                    (np.abs(_plot_adapter.data_yerr) + np.abs(_plot_adapter.data_y - _plot_adapter.model_y))
                                    / np.abs(_plot_adapter.model_y)
                                ),
                            )
                        # Small gap between highest error bar and plot border:
                        _low = 1 - _max_abs_deviation * 1.05
                        _high = 1 + _max_abs_deviation * 1.05
                        _axis.set_ylim((_low, _high))
                    else:
                        _axis.set_ylim(ratio_range)
                if residual:
                    _axis = self._current_axes["residual"]
                    _residual_label = kc("fit", "plot", "residual_label")
                    _axis.set_ylabel(_residual_label)
                    if residual_range is None:
                        _plot_adapters = self._get_plot_adapters()[i : i + 1] if self._separate_figs else self._get_plot_adapters()
                        _max_abs_deviation = 0
                        for _plot_adapter in _plot_adapters:
                            _max_abs_deviation = max(
                                _max_abs_deviation,
                                np.max(np.abs(_plot_adapter.data_yerr) + np.abs(_plot_adapter.data_y - _plot_adapter.model_y)),
                            )
                        # Small gap between highest error bar and plot border:
                        _low = -_max_abs_deviation * 1.05
                        _high = _max_abs_deviation * 1.05
                        _axis.set_ylim((_low, _high))
                    else:
                        _axis.set_ylim(residual_range)

                _all_plot_results.append(_plot_results)

            self._current_results = _all_plot_results

            return _all_plot_results

    def get_keywords(self, plot_type):
        """Retrieve keyword arguments for plots with type `plot_type` as they would be used when calling `plot`.

        This is an advanced function. An understanding of how plotting with
        `matplotlib` and the `PlotAdapter` classes in *kafe2* work is recommended.

        The `plot_type` must be one of the plot types registered in the
        `PlotAdapter` (e.g. ``'data'``, ``'model_line'`` etc.).

        :param plot_type: keyword identifying the plot type for which to set a custom keyword argument
        :type plot_type: str

        :return: list of dictionaries (one per fit instance) containing plot keywords and their values
        :rtype: list of dict
        """

        _adapters = self._get_plot_adapters()

        _keywords_list = []
        for _i_pdc, _pdc in enumerate(_adapters):
            _keywords_list.append(_pdc._get_subplot_kwargs(_i_pdc, plot_type))

        return _keywords_list

    def set_keywords(self, plot_type, keyword_spec):
        """Set values for keyword arguments used for plots with type `plot_type`.

        This is an advanced function. An understanding of how plotting with
        `matplotlib` and the `PlotAdapter` classes in *kafe2* work is recommended.

        The `plot_type` must be one of the plot types registered in the
        `PlotAdapter` (e.g. ``'data'``, ``'model_line'`` etc.).

        The `keyword_spec` contains dictionaries whose contents will be passed as
        keyword arguments to the plot adapter method responsible for plotting the
        `plot_type`. If `keyword` spec contains a key for which a default value is
        configured, it will be overridden.

        Passing the following special values for a keyword will have the following effects:

        * ``'__del__'``: the value will be removed from the keyword arguments. This includes
          default values, meaning that the plot call will be made **without** the keyword
          argument even if a default value for it exists.
        * ``'__default__'``: the customized value will be replaced by the default value.

        .. note::

            No keyword/value validation is done: everything is passed to the underlying plot
            methods as specified. Incorrect or incompatible keywords may be ignored or lead to errors.

        As an example, to override the labels shown in the legend entries for the `data`

        .. code:: python

            p = Plot([fit_1, fit_2])
            p.customize('data', [dict(label='My Data Label'), dict(label='Another Data Label')])

        To set keywords for a single `fit`, pass values as ``(index, value)``, where `index` is
        the index of the `fit` object:

        .. code:: python

            p = Plot([fit_1, fit_2])
            p.customize('data', [(1, dict(label='Another Data Label'))])

        :param plot_type: keyword identifying the plot type for which to set a custom keyword argument
        :type plot_type: str
        :param keyword_spec: specification of dictionaries containing the keyword arguments to use for fit.
            Can be either a list of dictionaries with a length corresponding to the number of `fit` objects
            managed by this `Plot` instance, or a list of tuples of the form ``(index, dict)``, where
            ``index``  denotes the index of the `fit` object for which the dictionary `dict` should be used.
        :type keyword_spec: list of values or list of 2-tuples like ``(index, value)``

        :return: this `Plot` instance
        :rtype: `Plot`
        """

        _adapters = self._get_plot_adapters()

        _has_tuples = None
        for _spec in keyword_spec:
            if isinstance(_spec, tuple):
                # raise if both tuples and non-tuples provided
                if _has_tuples is None:
                    _has_tuples = True
                elif not _has_tuples:
                    raise ValueError("Cannot set custom plot keyword arguments: " "provided `values` contain a mix of tuples and non-tuples!")

                # validate tuple
                if not len(_spec) == 2:
                    raise ValueError(
                        "Cannot set custom plot keyword arguments: " "tuple {!r} has length {} (expected " "2) ".format(_spec, len(_spec))
                    )

                # validate index
                try:
                    _adapters[_spec[0]]
                except IndexError:
                    raise ValueError(
                        "Cannot set custom plot keyword arguments: "
                        "invalid index {!r} encountered (object manages {} fit "
                        "objects)!".format(_spec[0], len(_adapters))
                    )
            else:
                if _has_tuples is None:
                    _has_tuples = False
                elif _has_tuples:
                    raise ValueError("Cannot set custom plot keyword arguments: " "provided `values` contain a mix of tuples and non-tuples!")

        if not _has_tuples:
            # must provide same amount of values as fits
            if len(_adapters) != len(keyword_spec):
                raise ValueError(
                    "Cannot set custom plot keyword argument: "
                    "{} values provided but this object manages {} "
                    "fit objects!".format(len(keyword_spec), len(_adapters))
                )

            # convert plain list to tuple
            keyword_spec = [(_index, _dict) for _index, _dict in enumerate(keyword_spec)]
        if _has_tuples:
            # can provide less tuples than fits but not more
            if len(_adapters) < len(keyword_spec):
                raise ValueError(
                    "Cannot set custom plot keyword argument: "
                    "{} values provided but this object manages {} "
                    "fit objects!".format(len(keyword_spec), len(_adapters))
                )

        for _index, _dict in keyword_spec:
            _adapters[_index].update_plot_kwargs(plot_type, _dict)

        return self  # allow chaining calls

    def customize(self, plot_type, keyword, values):
        """Set values for keyword arguments used for plots with type `plot_type`.

        This is a convenience wrapper around `set_keywords`.

        The `keyword` will be passed to the plot adapter method responsible for
        plotting the `plot_type` as a keyword argument with a value taken
        from `values`. If a default value for `keyword` is configured, it is
        overridden.

        The `values` can be specified in two ways:

        #. as a list with a length corresponding to the number of `fit` objects managed
           by this `Plot` instance. The special value ``'__skip__'`` can be used to skip
           `fit` objects.
        #. as a list of tuples of the form ``(index, value)``, where `index` denotes the
           index of the `fit` object for which the `value` should be used.

        Passing the following special values for a keyword will have the following effects:

        * ``'__del__'``: the value will be removed from the keyword arguments. This includes
          default values, meaning that the plot call will be made **without** the keyword
          argument even if a default value for it exists.
        * ``'__default__'``: the customized value will be replaced by the default value.
        * ``'__skip__'``: the keywords for this `fit` will not be changed.

        .. note::

            No keyword/value validation is done: everything is passed to the underlying plot
            methods as specified. Incorrect or incompatible keywords may be ignored or lead to errors.

        As an example, to override the labels shown in the legend entries for the `data`

        .. code:: python

            p = Plot([fit_1, fit_2])
            p.customize('data', 'label', ['My Data Label', 'Another Data Label'])

        To set keywords for a single `fit`, pass values as ``(index, value)``, where `index` is
        the index of the `fit` object:

        .. code:: python

            p = Plot([fit_1, fit_2])
            p.customize('data', 'label', [(1, 'Another Data Label')])

        :param plot_type: keyword identifying the plot type for which to set a custom keyword argument
        :type plot_type: str
        :param keyword: the keyword argument. The corresponding value in `values` will be passed
            to the plot adapter method using this keyword argument
        :type keyword: str
        :param values: values that the keyword argument should take for each fit. Can be a list of values
            with a length corresponding to the number of `fit` objects managed by this `Plot` instance,
            or a list of tuples of the form ``(index, value)``
        :type values: list of values or list of 2-tuples like ``(index, value)``

        :return: this `Plot` instance
        :rtype: `Plot`
        """
        if values is None or isinstance(values, (int, float, str)):
            values = (values,) * len(self._fits)

        _has_tuples = None
        for _val in values:
            if isinstance(_val, tuple):
                # raise if both tuples and non-tuples provided
                if _has_tuples is None:
                    _has_tuples = True
                elif not _has_tuples:
                    raise ValueError("Cannot set custom plot keyword arguments: " "provided `values` contain a mix of tuples and non-tuples!")

                # validate tuple
                if not len(_val) == 2:
                    raise ValueError("Cannot set custom plot keyword arguments: " "tuple {!r} has length {} (expected " "2) ".format(_val, len(_val)))
            else:
                if _has_tuples is None:
                    _has_tuples = False
                elif _has_tuples:
                    raise ValueError("Cannot set custom plot keyword arguments: " "provided `values` contain a mix of tuples and non-tuples!")

        if not _has_tuples:
            _dicts = [{keyword: _value} if _value != "__skip__" else {} for _value in values]
        else:
            _dicts = [(_index, {keyword: _value}) for _index, _value in values if _value != "__skip__"]

        return self.set_keywords(plot_type, _dicts)

    def save(self, fname=None, figures="all", *args, **kwargs):
        """
        Saves the plot figures to files.
        args and kwargs are passed on to matplotlib.Figure.savefig() .

        :param fname: Output file name(s), defaults to fit.png or fit_0.png, fit_1.png, ...
        :type fname: None or str or iterable of str.
        :param figures: Which figures to save.
        :type figures: 'all' or int or iterable of int
        """
        if figures == "all":
            _figure_indices = range(len(self._figure_dicts))
        elif isinstance(figures, int):
            _figure_indices = [figures]
        _file_name_list = fname
        if fname is None:
            if len(self._figure_dicts) == 1:
                _file_name_list = ["fit.png"]
            else:
                _file_name_list = [f"fit_{_i}.png" for _i in _figure_indices]
        elif isinstance(fname, str):
            if len(_figure_indices) == 1:
                _file_name_list = [fname]
            else:
                name, extension = os.path.splitext(fname)
                _file_name_list = [f"{name}_{_i}{extension}" for _i in _figure_indices]
        elif isinstance(fname, Iterable):
            _file_name_list = fname
        else:
            raise ValueError("fname must be either None, a string, or an iterable, but received type " f"{type(fname)}")
        if len(_file_name_list) != len(_figure_indices):
            raise ValueError(f"Received {len(_file_name_list)} file names for {len(_figure_indices)} figures." "Must be the same!")
        for _figure_index, _file_name in zip(_figure_indices, _file_name_list):
            self._figure_dicts[_figure_index]["figure"].savefig(_file_name, *args, **kwargs)
