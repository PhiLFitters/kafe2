import numpy as np
import six
import sys
import matplotlib as mpl

from ...config import kafe2_rc
from ...core.confidence import ConfidenceLevel
from .._base import FitBase
from .._base.format import ScalarFormatter as ScalarBaseFormatter
from matplotlib import pyplot as plt, rcParams
from matplotlib import gridspec as gs
from matplotlib import ticker as plticker
from matplotlib.axes import Axes
from matplotlib import rc_context


__all__ = ["ContoursProfiler"]

_python_version = sys.version_info[0]
_float_template = ".2g" if _python_version == 2 else "#.2g"


def _linear_range_transform(range_, factor, asymmetry=0.0):
    """Return a range that is `factor` larger than `range_`.

    The amount by which the extension is done is the direction of
    each boundary is controlled by a float `asymmetry`, which should
    be between -1 and 1. An `asymmetry` of 0 will extend the range
    symmetrically in the direction of both boundaries. An `asymmetry`
    of -1 (+1) will extend the range in the direction of the first
    (second) boundary, leaving the other boundary unchanged."""

    assert len(range_) == 2
    assert factor > 0.0
    assert -1.0 <= asymmetry <= 1.0

    _m = 0.5 * (range_[1] + range_[0])
    _d = 0.5 * (range_[1] - range_[0])
    _asymm_factor = (1.0 + asymmetry)

    return _m - factor*(2.0 - _asymm_factor) * _d, _m + factor*_asymm_factor*_d


def _maybe_set_tight_layout(figure):
    """Enable 'tight_layout' for a figure if the matplotlib version is high enough."""
    # older versions of matplotlib cause problems with 'tight_layout'
    if not mpl.__version__.startswith('1'):
        figure.set_tight_layout(True)


class SigmaLocator(plticker.Locator):
    """
    Create ticks at evenly spaced offsets from a central value.
    The offsets are integer multiples of a fixed value ('sigma')
    """
    def __init__(self, central_value, sigma):
        self._cval = central_value
        self._sigma = sigma

    def __call__(self):
        """Return the locations of the ticks"""
        _vmin, _vmax = self.axis.get_view_interval()
        return self.tick_values(_vmin, _vmax)

    def tick_values(self, vmin, vmax):
        _n_sigma_dn = int((vmin - self._cval)/self._sigma)
        _n_sigma_up = int((vmax - self._cval)/self._sigma)
        return self.raise_if_exceeds(
            np.arange(_n_sigma_dn, _n_sigma_up+1, 1) * self._sigma + self._cval)


class SigmaFormatter(plticker.Formatter):
    """
    Set the tick labels to indicate the distance to the
    central value, in intervals
    """
    def __init__(self, central_value, sigma):
        """set the tick labels to correspond to sigma"""
        self._cval = central_value
        self._sigma = sigma

    def __call__(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""
        # _vmin, _vmax = self.axis.get_data_interval()
        _numeric_label = (x - self._cval)/self._sigma
        _str_label = r"%.2g$\sigma$" % (_numeric_label,)
        return _str_label

    def format_data_short(self, value):
        """Short version of format string for tick"""
        return '{:g}'.format(value)


class ScalarFormatter(plticker.Formatter):
    """Format the tick labels to a specified precision.
    """
    def __init__(self, sigma, n_significant_digits=2):
        """Format the tick label to a specified precision, according to the uncertainty.

        :param float sigma: The uncertainty of the parameter.
        :param int n_significant_digits: Number of significant digits.
        """
        self.formatter = ScalarBaseFormatter(sigma=sigma, n_significant_digits=n_significant_digits)

    def __call__(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""
        return self.formatter(x)


class ContoursProfilerException(Exception):
    pass


class ContoursProfiler(object):
    """
    Object dedicated to *profiling* the cost function in one or two parameters.

    Once a fit has converged and the cost function is at its minimum, the technique of *profiling*
    can be applied to examine the behavior of the cost function around its minimum. Profiling
    involves repeating the cost function minimization with the value of one particular parameter
    being fixed to a series of points in the vicinity of the minimum.
    The function resulting from mapping the parameter values to the cost function value thus
    obtained, is known as a *profile* in that parameter.

    In two dimensions, two parameters can be profiled simultaneously. The resulting profiles are
    functions of two variables and can be visualized in the plane by drawing the 3D *contours*
    along which this function remains constant.

    This object offers a means of calculating both profiles and contours
    """

    _DEFAULT_PLOT_PROFILE_KWARGS = dict(marker='', linewidth=2)
    _DEFAULT_PLOT_PARABOLA_KWARGS = dict(marker='', linewidth=2, linestyle='--')
    _DEFAULT_PLOT_ERROR_SPAN_ON_PROFILE_KWARGS = dict(color='gray', alpha=0.5)
    _DEFAULT_PLOT_MINIMUM_KWARGS = dict(markersize=12, marker='*', linewidth=1.5, linestyle='',
                                        capsize=4, color='green', ecolor='green', elinewidth=1.5)
    _DEFAULT_PLOT_MINIMUM_HVLINES_KWARGS = dict(linewidth=1.0, linestyle='--', color='green', marker='')
    _DEFAULT_PLOT_FILL_CONTOUR_KWARGS = dict(alpha=0.3, linewidth=2)

    def __init__(self, fit_object,
                 profile_points=100, profile_subtract_min=True, profile_bound=2.45,
                 contour_points=100, contour_sigma_values=(1.0, 2.0), contour_smoothing_sigma=0.0,
                 contour_method_kwargs=None):
        """
        Construct a :py:obj:`~kafe2.fit._base.profile.ContoursProfiler` object:

        :param fit_object: fit object for which to calculate profiles/contours
        :type fit_object: :py:obj:`~kafe2.fit._base.FitBase`-derived object
        :param profile_points: number of points at which to sample each profile
        :type profile_points: int
        :param profile_subtract_min: if ``True``, the value of the cost function at the minimum is subtracted from
                                     the profiles, so that only the difference to the minimum is plotted
        :type profile_subtract_min: bool
        :param profile_bound: sample the profile at most this far from the minimum (in sigma)
        :type profile_bound: float
        :param contour_points: number of points at which to sample each contour
        :type contour_points: int
        :param contour_sigma_values: evaluate and show contours for these confidences (in sigma)
        :type contour_sigma_values: iterable of float
        :param contour_smoothing_sigma: apply a smoothing Gaussian filter with this sigma parameter to each contour
                                        (default is ``0.0``, meaning no smoothing)
        :type contour_smoothing_sigma: float
        """
        if not isinstance(fit_object, FitBase):
            raise ContoursProfilerException("Object %r is not a fit object!" % (fit_object,))

        _contour_confidence_levels = [ConfidenceLevel(n_dimensions=2, sigma=_sigma)
                                      for _sigma in contour_sigma_values]

        self._fit = fit_object
        self._profile_kwargs = dict(points=profile_points, subtract_min=profile_subtract_min, bound=profile_bound)
        self._contour_kwargs = dict(points=contour_points,
                                    confidence_levels=_contour_confidence_levels,
                                    smoothing_sigma=contour_smoothing_sigma,
                                    method_kwargs=contour_method_kwargs)

        self._cost_function_formatted_name = "${}$".format(self._fit._cost_function.formatter.latex_name)
        # FIXME MultiFit does not have an internal _model_function field
        self._parameters_formatted_names = ["${}$".format(pf.latex_name)
                                            for pf in self._fit._get_model_function_parameter_formatters()]

        self._figures = []

    def _make_figure_gs(self, nrows=1, ncols=1):
        _fig = plt.figure(figsize=(8, 8))  # defaults from matplotlibrc

        if mpl.__version__.startswith('1'):
            # old API does not support passing 'figure'
            _gs = gs.GridSpec(nrows=nrows, ncols=ncols)
        else:
            _gs = gs.GridSpec(nrows=nrows, ncols=ncols, figure=_fig)

        self._figures.append(_fig)  # store figure

        return _fig, _gs

    @staticmethod
    def _plot_profile_xy(target_axes, x, y, label):
        _kwargs = ContoursProfiler._DEFAULT_PLOT_PROFILE_KWARGS.copy()
        return target_axes.plot(x, y, label=label, **_kwargs)

    @staticmethod
    def _plot_parabolic_cost(target_axes, x, quad_coeff, x_offset, y_offset, label):
        _kwargs = ContoursProfiler._DEFAULT_PLOT_PARABOLA_KWARGS.copy()
        _y = quad_coeff * (x - x_offset) ** 2 + y_offset
        return target_axes.plot(x, _y, label=label, **_kwargs)

    @staticmethod
    def _plot_error_span_on_profile(target_axes, xmin, xmax, label):
        _kwargs = ContoursProfiler._DEFAULT_PLOT_ERROR_SPAN_ON_PROFILE_KWARGS.copy()

        _err_span_artist = target_axes.axvspan(xmin, xmax, label=label, **_kwargs)

        return _err_span_artist

    @staticmethod
    def _plot_minimum(target_axes, x, y, xerr, yerr, label):
        _kwargs = ContoursProfiler._DEFAULT_PLOT_MINIMUM_KWARGS.copy()

        _min_pt_artists = target_axes.errorbar(x, y, xerr=xerr, yerr=yerr, label=label, **_kwargs)

        # make error bar caps have the same line width as the error bars
        _cap_artists = _min_pt_artists[1]
        for _cap in _cap_artists:
            _cap.set_markeredgewidth(_kwargs.get('elinewidth', 1.5))

        _kwargs = ContoursProfiler._DEFAULT_PLOT_MINIMUM_HVLINES_KWARGS.copy()

        _min_lines_artists = (target_axes.axvline(x, **_kwargs),
                              target_axes.axhline(y, **_kwargs))

        return _min_pt_artists, _min_lines_artists

    @staticmethod
    def _plot_contour_xy(target_axes, contour, label, contour_color):
        _kwargs = ContoursProfiler._DEFAULT_PLOT_FILL_CONTOUR_KWARGS.copy()
        if contour.xy_points is not None:
            return [target_axes.fill(contour.xy_points[0], contour.xy_points[1], label=label, **_kwargs)]
        return [target_axes.contour(contour.grid_x, contour.grid_y, contour.grid_z.T, levels=[0, contour.sigma],
                                    colors="gray", **_kwargs),
                target_axes.contourf(contour.grid_x, contour.grid_y, contour.grid_z.T, levels=[0, contour.sigma],
                                     colors=contour_color, label=label, **_kwargs)]

    # -- public properties

    @property
    def figures(self):
        """The ``matplotlib`` figures managed by this object."""
        return list(self._figures)

    # -- public methods

    # - get numeric profiles/contours

    def get_profile(self, parameter):
        """
        Calculate and return a profile of the cost function in a parameter.

        :param parameter: name of the parameter to profile in
        :type parameter: str
        :return: two-dimensional array of *x* (parameter) values and *y* (cost function) values
        :rtype: two-dimensional array of float
        """
        _kwargs = dict(bins=self._profile_kwargs['points'], bound=self._profile_kwargs['bound'],
                       args=None, subtract_min=self._profile_kwargs['subtract_min'])
        self._fit._check_dynamic_error_compatibility()
        return self._fit._fitter.profile(parameter, **_kwargs)  # TODO fix for single fit inside multifit

    def get_contours(self, parameter_1, parameter_2, smoothing_sigma=None):
        """
        Calculate and return a list of contours (one for each requested sigma value).

        :param parameter_1: name of the first contour parameter
        :type parameter_1: str
        :param parameter_2:  name of the second contour parameter
        :type parameter_2: str
        :param smoothing_sigma: sigma for smoothing the contour with a gaussian filter
        :type smoothing_sigma: int or float or complex or Iterable
        :return: list of tuples of the form (sigma, contour)
        :rtype: list of 2-tuples of float and 2d-array
        """
        if smoothing_sigma is None:
            smoothing_sigma = self._contour_kwargs['smoothing_sigma']
        _contours = []
        for _cl_obj in self._contour_kwargs['confidence_levels']:
            _contour_method_kwargs = self._contour_kwargs.get('method_kwargs', dict())
            if _contour_method_kwargs is None:
                _contour_method_kwargs = dict()
            # TODO fix for single fit inside multifit
            _cont = self._fit._fitter.contour(parameter_1, parameter_2, sigma=_cl_obj.sigma,
                                              **_contour_method_kwargs)

            # smooth contours if requested
            if smoothing_sigma > 0 and _cont is not None:
                from scipy.ndimage.filters import gaussian_filter
                _cont[0] = gaussian_filter(_cont[0], smoothing_sigma, mode='wrap')
                _cont[1] = gaussian_filter(_cont[1], smoothing_sigma, mode='wrap')

            _contours.append((_cl_obj, _cont))
        self._fit._check_dynamic_error_compatibility()
        return _contours

    # - plot profiles/contours

    def plot_profile(self, parameter, target_axes=None,
                     show_parabolic=True,
                     show_grid=True,
                     show_legend=True,
                     show_fit_minimum=True,
                     show_error_span=True,
                     show_ticks=True,
                     label_ticks_in_sigma=True,
                     label_fit_minimum=True):
        """
        Plot the profile cost function for a parameter.

        :param parameter: name of the parameter to profile in
        :type parameter: str
        :param target_axes: ``Axes`` object (if ``None``, a new figure is created)
        :type target_axes: ``matplotlib`` ``Axes`` or ``None``
        :param show_parabolic: if ``True``, a parabolic approximation of the profile near the minimum is also drawn
        :type show_parabolic: bool`
        :param show_grid: if ``True``, a grid is drawn
        :type show_grid: bool
        :param show_legend: if ``True``, the legend is displayed
        :param show_fit_minimum: if ``True``, the fit minumum is shown as a marker with error bars
        :type show_fit_minimum: bool
        :type show_legend: bool
        :param show_error_span: if ``True``, the parameter error region is shaded
        :type show_error_span: bool
        :param show_ticks: if ``True``, *x* and *y* ticks are displayed
        :type show_ticks: bool
        :param label_ticks_in_sigma: if ``True``, label ticks are in units of 1 sigma
        :type label_ticks_in_sigma: bool
        :param label_fit_minimum: if ``True``, the parameter value and the 1 sigma error
            will be shown as an annotation
        :type label_fit_minimum: bool

        :return: figure containing the plot result
        :rtype: `matplotlib.figure.Figure`
        """
        with rc_context(kafe2_rc):
            if target_axes is None:
                _fig, _gs = self._make_figure_gs(1, 1)
                _axes = plt.subplot(_gs[0, 0])
            else:
                _axes = target_axes

            _par_val = self._fit.parameter_name_value_dict[parameter]
            _par_id = list(self._fit.parameter_name_value_dict.keys()).index(parameter)
            _par_err = self._fit.parameter_errors[_par_id]
            _cost_function_min = self._fit.cost_function_value
            _par_formatted_name = self._parameters_formatted_names[_par_id]

            _x, _y = self.get_profile(parameter)

            _profile_artist = self._plot_profile_xy(_axes, _x, _y,
                                                    label="profile %s" % (self._cost_function_formatted_name,))

            _y_offset = _cost_function_min if not self._profile_kwargs['subtract_min'] else 0.0

            _parabola_artist = None
            if show_parabolic:
                _parabola_artist = self._plot_parabolic_cost(_axes,
                                                             _x,
                                                             quad_coeff=1. / (_par_err**2),
                                                             x_offset=_par_val,
                                                             y_offset=_y_offset,
                                                             label="parabolic approximation")

            _minimum_artist = None
            if show_fit_minimum:
                _minimum_artist = self._plot_minimum(_axes,
                                                     x=_par_val,
                                                     y=_y_offset,
                                                     xerr=_par_err, yerr=None,
                                                     label="fit minimum")

            if label_fit_minimum:
                sigma_str = r"$\sigma_{{{}}} = {:%s}$" % _float_template
                _axes.annotate(
                    '\n'.join([
                        r"$\langle {}\rangle = {}$".format(
                            _par_formatted_name.strip('$'),
                            ScalarBaseFormatter(_par_err, n_significant_digits=2)(_par_val)
                        ),
                        sigma_str.format(
                            _par_formatted_name.strip('$'),
                            _par_err
                        )]),
                    xy=(_par_val, 1),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -10),
                    textcoords='offset points',
                    ha='center',
                    va='top'
                )

            _err_span_artist = None
            if show_error_span:
                _xmin, _xmax = _par_val-_par_err, _par_val+_par_err
                _err_span_artist = self._plot_error_span_on_profile(_axes, xmin=_xmin, xmax=_xmax,
                                                                    label="parameter error")

            _axes.set_xlabel(_par_formatted_name)
            _axes.set_ylabel(self._cost_function_formatted_name if not self._profile_kwargs['subtract_min']
                             else '$\\Delta$%s' % self._cost_function_formatted_name)

            if show_legend:
                _axes.legend(loc='center')

            if show_grid:
                _axes.grid('on')

            if show_ticks:
                _loc_x = SigmaLocator(central_value=_par_val, sigma=_par_err)
                _axes.xaxis.set_major_locator(_loc_x)
                if label_ticks_in_sigma:
                    _form_x = SigmaFormatter(central_value=_par_val, sigma=_par_err)
                    _axes.xaxis.set_major_formatter(_form_x)
                else:
                    _form_x = ScalarFormatter(sigma=_par_err, n_significant_digits=2)
                    _axes.xaxis.set_major_formatter(_form_x)

                _loc_y = plticker.MaxNLocator(5)
                _axes.yaxis.set_major_locator(_loc_y)
            else:
                _axes.set_xticks([])
                _axes.set_yticks([])

            # if own figure created -> set tight layout
            if target_axes is None:
                _maybe_set_tight_layout(_fig)

            return _axes.get_figure()

    def plot_contours(self, parameter_1, parameter_2, target_axes=None,
                      show_grid=True,
                      show_legend=True,
                      show_fit_minimum=True,
                      show_ticks=True,
                      label_ticks_in_sigma=True,
                      naming_convention='sigma'):
        """
        Plot the contour for a parameter pair.

        :param parameter_1: name of the parameter appearing on the *x* axis
        :type parameter_1: str
        :param parameter_2:  name of the parameter appearing on the *y* axis
        :type parameter_2: str
        :param target_axes: ``Axes`` object (if ``None``, a new figure is created)
        :type target_axes: ``matplotlib`` ``Axes` or ``None``
        :param show_grid: if ``True``, a grid is drawn
        :type show_grid: bool
        :param show_legend: if ``True``, the legend is displayed
        :type show_legend: bool
        :param show_fit_minimum: if ``True``, the fit minimum is shown as a marker with error bars
        :type show_fit_minimum: bool
        :param show_ticks: if ``True``, *x* and *y* ticks are displayed
        :type show_ticks: bool
        :param label_ticks_in_sigma: if ``True``, label ticks are in units of 1 sigma
        :type label_ticks_in_sigma: bool
        :param naming_convention: if ``'sigma'`` the contour is labelled in sigma, if ``'cl'`` the contour is labelled
                                  in confidence level
        :type naming_convention: str

        :return: figure containing the plot result
        :rtype: `matplotlib.figure.Figure`
        """
        with rc_context(kafe2_rc):
            if target_axes is None:
                _fig, _gs = self._make_figure_gs(1, 1)
                _axes = plt.subplot(_gs[0, 0])
            else:
                _axes = target_axes

            _par_1_id = list(self._fit.parameter_name_value_dict.keys()).index(parameter_1)
            _par_2_id = list(self._fit.parameter_name_value_dict.keys()).index(parameter_2)
            _par_1_formatted_name = self._parameters_formatted_names[_par_1_id]
            _par_2_formatted_name = self._parameters_formatted_names[_par_2_id]

            if naming_convention.lower() == 'cl':
                _use_cl_in_label = True
            elif naming_convention.lower() == 'sigma':
                _use_cl_in_label = False
            else:
                raise ContoursProfilerException("Unknown contour naming convention '%s'! "
                                                "Must be one of: ('cl', 'sigma')"
                                                % (naming_convention,))

            _cl_contour_pairs = self.get_contours(parameter_1, parameter_2)

            _contour_artists = []
            for _cl_contour_pair, _prop_cycler in zip(_cl_contour_pairs, rcParams["axes.prop_cycle"]):
                _cl, _contour_xy = _cl_contour_pair
                _artists = []
                if _contour_xy is not None:
                    _label = "%s contour" % (_cl.cl_latex_string if _use_cl_in_label else _cl.sigma_latex_string)
                    _artists = self._plot_contour_xy(_axes, _contour_xy, label=_label,
                                                     contour_color=_prop_cycler["color"])
                _contour_artists += _artists

            _par_1_val = self._fit.parameter_name_value_dict[parameter_1]
            _par_2_val = self._fit.parameter_name_value_dict[parameter_2]
            _par_1_err = self._fit.parameter_errors[_par_1_id]
            _par_2_err = self._fit.parameter_errors[_par_2_id]

            _minimum_artist = None
            if show_fit_minimum:
                _minimum_artist = self._plot_minimum(_axes, x=_par_1_val, y=_par_2_val,
                                                     xerr=_par_1_err, yerr=_par_2_err,
                                                     label="fit minimum")

            _axes.set_xlabel(_par_1_formatted_name)
            _axes.set_ylabel(_par_2_formatted_name)

            if show_legend:
                _axes.legend(loc='best')

            if show_grid:
                _axes.grid('on')

            if show_ticks:
                _loc_x = SigmaLocator(central_value=_par_1_val, sigma=_par_1_err)
                _loc_y = SigmaLocator(central_value=_par_2_val, sigma=_par_2_err)
                _axes.xaxis.set_major_locator(_loc_x)
                _axes.yaxis.set_major_locator(_loc_y)
                if label_ticks_in_sigma:
                    _form_x = SigmaFormatter(central_value=_par_1_val, sigma=_par_1_err)
                    _form_y = SigmaFormatter(central_value=_par_2_val, sigma=_par_2_err)
                    _axes.xaxis.set_major_formatter(_form_x)
                    _axes.yaxis.set_major_formatter(_form_y)
                else:
                    _form_x = ScalarFormatter(sigma=_par_1_err, n_significant_digits=2)
                    _form_y = ScalarFormatter(sigma=_par_2_err, n_significant_digits=2)
                    _axes.xaxis.set_major_formatter(_form_x)
                    _axes.yaxis.set_major_formatter(_form_y)
            else:
                _axes.set_xticks([])
                _axes.set_yticks([])

            # if own figure created -> set tight layout
            if target_axes is None:
                _maybe_set_tight_layout(_fig)

            return _axes.get_figure()

    def plot_profiles_contours_matrix(self,
                                      parameters=None,
                                      show_grid_for=None,
                                      show_ticks_for='all',
                                      show_fit_minimum_for='contours',
                                      show_legend=True,
                                      show_parabolic_profiles=True,
                                      show_error_span_profiles=False,
                                      full_matrix=False,
                                      label_ticks_in_sigma=True,
                                      contour_naming_convention='sigma'):
        """
        Plot all profiles and contours to subplots arranges in a matrix-like fashion.

        :param parameters: parameters for which to display profiles and contours. If ``None``, all parameters.
        :type parameters: list of parameter names or ``None``
        :param show_grid_for: subplots for which to show a grid
        :type show_grid_for: ``None``, ``'profiles'``, ``'contours'`` or ``'all'``
        :param show_ticks_for: subplots for which to show ticks
        :type show_ticks_for: ``None``, ``'profiles'``, ``'contours'`` or ``'all'``
        :param show_fit_minimum_for: subplots for which to show the fit minimum
        :type show_fit_minimum_for: ``None``, ``'profiles'``, ``'contours'`` or ``'all'``
        :param show_legend: if ``True``, the legend is displayed
        :type show_legend: bool
        :param show_parabolic_profiles: if ``True``, a parabolic approximation of the profile near the minimum is also
                                        drawn
        :type show_parabolic_profiles: bool
        :param show_error_span_profiles: if ``True``, the parameter uncertainty region is shaded in the
                                         profile plots
        :type show_error_span_profiles: bool
        :param full_matrix: if ``True``, contour subplots are also shown above the main diagonal
        :type full_matrix: bool
        :param label_ticks_in_sigma: if ``True``, label ticks are in units of 1 sigma
        :type label_ticks_in_sigma: bool
        :param contour_naming_convention: if ``'sigma'`` the contour is labelled in sigma, if ``'cl'`` the contour is
                                          labelled in confidence level
        :type contour_naming_convention: str

        :return: figure containing the plot result
        :rtype: `matplotlib.figure.Figure`
        """
        with rc_context(kafe2_rc):
            _par_names = parameters
            if _par_names is None:
                _par_names = list(self._fit.parameter_name_value_dict.keys())
            else:
                # check if there are any unknown parameters
                _unknown_parameters = set(_par_names) - set(self._fit.parameter_name_value_dict.keys())
                if _unknown_parameters:
                    raise ContoursProfilerException("Unknown parameters: {}".format(_unknown_parameters))

            # # check if any parameters are fixed and exclude them from the matrix:
            # # TODO: public interface for querying parameter status
            # _free_pars = set(self._fit._fitter._fit_pars)
            # _par_names = [_par for _par in _par_names if _par in _free_pars]

            _npar = len(_par_names)
            _fig, _gs = self._make_figure_gs(_npar, _npar)

            _show_spec_options = ('all', 'profiles', 'contours')

            if show_grid_for is not None and show_grid_for not in _show_spec_options:
                raise ContoursProfilerException("Unknown specification '%s' for 'show_grid_for'. "
                                                "Expected: one of %r" % (show_grid_for, _show_spec_options))
            if show_ticks_for is not None and show_ticks_for not in _show_spec_options:
                raise ContoursProfilerException("Unknown specification '%s' for 'show_ticks_for'. "
                                                "Expected: one of %r" % (show_ticks_for, _show_spec_options))
            if show_fit_minimum_for is not None and show_fit_minimum_for not in _show_spec_options:
                raise ContoursProfilerException("Unknown specification '%s' for 'show_fit_minimum_for'. "
                                                "Expected: one of %r" % (show_fit_minimum_for, _show_spec_options))

            # determine which plot elements to show for each subplot type
            _show_grid_profiles = show_grid_for in ('all', 'profiles')
            _show_grid_contours = show_grid_for in ('all', 'contours')
            _show_ticks_profiles = show_ticks_for in ('all', 'profiles')
            _show_ticks_contours = show_ticks_for in ('all', 'contours')
            _show_minimum_profiles = show_fit_minimum_for in ('all', 'profiles')
            _show_minimum_contours = show_fit_minimum_for in ('all', 'contours')

            _all_legend_handles = tuple()
            _all_legend_labels = tuple()

            # draw profiles to subplots on the diagonal
            _subplots = np.empty((_npar, _npar), dtype=Axes)  # store subplot system in numpy array
            for row in six.moves.range(_npar):
                _axes = _subplots[row, row] = _fig.add_subplot(_gs[row, row])
                self.plot_profile(_par_names[row], target_axes=_axes,
                                  show_parabolic=show_parabolic_profiles,
                                  show_grid=_show_grid_profiles,
                                  show_legend=False,
                                  show_fit_minimum=_show_minimum_profiles,
                                  show_error_span=show_error_span_profiles,
                                  label_ticks_in_sigma=label_ticks_in_sigma,
                                  show_ticks=_show_ticks_profiles)

                if show_legend:
                    _hs, _ls = _axes.get_legend_handles_labels()
                    _all_legend_handles += tuple(_hs)
                    _all_legend_labels += tuple(_ls)

            # draw contours to subplots in the lower (and possibly upper) triangle
            for row in six.moves.range(_npar):
                for col in six.moves.range(row):
                    _axes = _subplots[row, col] = _fig.add_subplot(_gs[row, col])
                    self.plot_contours(_par_names[col], _par_names[row],
                                       target_axes=_axes,
                                       show_grid=_show_grid_contours,
                                       show_legend=False,
                                       show_fit_minimum=_show_minimum_contours,
                                       show_ticks=_show_ticks_contours,
                                       label_ticks_in_sigma=label_ticks_in_sigma,
                                       naming_convention=contour_naming_convention)

                    if show_legend:
                        _hs, _ls = _axes.get_legend_handles_labels()
                        _all_legend_handles += tuple(_hs)
                        _all_legend_labels += tuple(_ls)

                    if full_matrix:
                        _axes = _subplots[col, row] = _fig.add_subplot(_gs[col, row])
                        self.plot_contours(_par_names[row], _par_names[col],
                                           target_axes=_axes,
                                           show_grid=_show_grid_contours,
                                           show_legend=False,
                                           show_fit_minimum=_show_minimum_contours,
                                           show_ticks=_show_ticks_contours,
                                           label_ticks_in_sigma=label_ticks_in_sigma,
                                           naming_convention=contour_naming_convention)

            # post-processing: join column x axes and row y axes (where applicable)
            for i in six.moves.range(_npar):
                _pf_axes = _subplots[i, i]

                _ct_axes_col = np.concatenate([_subplots[:i, i], _subplots[i+1:, i]])
                _ct_axes_col = [_ax for _ax in _ct_axes_col if _ax is not None]

                _pf_axes.get_shared_x_axes().join(_pf_axes, *_ct_axes_col)
                _pf_axes.autoscale()

                _ct_axes_row = np.concatenate([_subplots[i, :i], _subplots[i, i+1:]])
                _ct_axes_row = [_ax for _ax in _ct_axes_row if _ax is not None]
                if len(_ct_axes_row) > 1:
                    _ct_axes_row[0].get_shared_y_axes().join(_ct_axes_row[0], *_ct_axes_row[1:])
                    _ct_axes_row[0].autoscale()

            # more post-processing: adjust axis and tick labels
            for row, _row_plots in enumerate(_subplots):
                for col, _plot in enumerate(_row_plots):
                    # skip empty plots
                    if not _plot:
                        continue

                    _plot.set_xlim(_linear_range_transform(_plot.get_xlim(), factor=1.01))
                    _plot.set_ylim(_linear_range_transform(_plot.get_ylim(), factor=1.01))

                    # adjust y plot range to match x (in sigma)
                    if col != row:
                        _x_lim = _plot.get_xlim()
                        _x_min = self._fit.parameter_values[col]
                        _y_min = self._fit.parameter_values[row]
                        _y_over_x_err_ratio = self._fit.parameter_errors[row]/self._fit.parameter_errors[col]
                        _plot.set_ylim((
                            _y_min + (_x_lim[0] - _x_min)*_y_over_x_err_ratio,
                            _y_min + (_x_lim[1] - _x_min)*_y_over_x_err_ratio
                        ))

                    # hide tick labels except for outer plots
                    if col != 0 and col != row:
                        # not leftmost column or profile -> hide y tick and axis labels
                        _plot.set_ylabel(None)
                        for _label in _plot.get_yticklabels():
                            _label.set_visible(False)
                    if row != _npar - 1:
                        # not bottom row -> hide x tick and axis labels
                        _plot.set_xlabel(None)
                        for _label in _plot.get_xticklabels():
                            _label.set_visible(False)

                    # rotate long x tick labels to avoid overlap
                    if not label_ticks_in_sigma:
                        for _label in _plot.get_xticklabels():
                            _label.set_rotation(90)

            # align x and y axis labels
            try:
                _fig.align_ylabels(_subplots[:, 0])
                _fig.align_xlabels(_subplots[_npar - 1, :])
            except AttributeError:
                # API call not supported by matplotlib version -> ignore
                pass

            if show_legend:
                # suppress multiple entries for the same label
                _hs = []
                _ls = []
                _seen_labels = set()
                for _h, _l in zip(_all_legend_handles, _all_legend_labels):
                    if _l not in _seen_labels:
                        _hs.append(_h)
                        _ls.append(_l)
                        _seen_labels.add(_l)

                if not full_matrix:
                    # attach legend to invisible Axes in top right corner
                    _ax_for_legend = _fig.add_subplot(_gs[0, _npar - 1])
                    _ax_for_legend.set_visible(False)
                    _leg = _fig.legend(
                        _hs, _ls,
                        loc='upper right',
                        borderpad=0,
                        borderaxespad=0
                    )
                    # patch legend bbox to correspond to axes
                    _leg._bbox_to_anchor = _ax_for_legend.bbox
                else:
                    # place legend outside regular Axes in top right corner
                    _ax_for_legend = _subplots[0, _npar - 1]
                    _leg = _ax_for_legend.legend(
                        _hs, _ls,
                        loc='upper left',
                        bbox_to_anchor=(1.1, 0, 1, 1),
                        borderpad=0,
                        borderaxespad=0
                    )

            # old versions cause problems with 'tight_layout'
            _maybe_set_tight_layout(_fig)

            return _fig
