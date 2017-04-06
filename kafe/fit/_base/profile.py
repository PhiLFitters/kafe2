from ...config import matplotlib as mpl
from . import FitBase
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs


class ContoursProfilerException(Exception):
    pass


class ContoursProfiler(object):

    _DEFAULT_PLOT_PROFILE_KWARGS = dict(marker='', linewidth=2)
    _DEFAULT_PLOT_PARABOLA_KWARGS = dict(marker='', linewidth=2, linestyle='--')
    _DEFAULT_PLOT_HELPER_LINES_KWARGS =  dict(linewidth=1.5, linestyle='--', color='gray', alpha=0.9)
    _DEFAULT_PLOT_FILL_CONTOUR_KWARGS = dict(alpha=0.3, linewidth=2)

    def __init__(self, fit_object,
                 profile_points=100, profile_subtract_min=False, profile_bound=2,
                 contour_points=100, contour_sigma_values=(1.0, 2.0)):

        if not isinstance(fit_object, FitBase):
            raise ContoursProfilerException("Object %r is not a fit object!" % (fit_object,))

        self._fit = fit_object
        self._profile_kwargs = dict(points=profile_points, subtract_min=profile_subtract_min, bound=profile_bound)
        self._contour_kwargs = dict(points=profile_points, sigma_values=contour_sigma_values)

        self._cost_function_formatted_name = "${}$".format(self._fit._cost_function.formatter.latex_name)
        self._parameters_formatted_names = ["${}$".format(pf.latex_name) for pf in self._fit._model_function.argument_formatters]

        self._figures = []

    def _make_figure_gs(self, nrows=1, ncols=1):
        _fig = plt.figure(figsize=(8, 8))  # defaults from matplotlibrc
        _gs = gs.GridSpec(nrows=nrows,
                                    ncols=ncols,
                                    left=0.075,
                                    bottom=0.1,
                                    right=0.925,
                                    top=0.9,
                                    wspace=None,
                                    hspace=None,
                                    height_ratios=None)

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
    def _plot_helper_lines(target_axes, hlines_y, vlines_x):
        _kwargs = ContoursProfiler._DEFAULT_PLOT_HELPER_LINES_KWARGS.copy()

        _xmin, _xmax = target_axes.get_xlim()
        _ymin, _ymax = target_axes.get_ylim()

        _hl_artists = target_axes.hlines(hlines_y, _xmin, _xmax, **_kwargs)
        _vl_artists = target_axes.vlines(vlines_x, _ymin, _ymax, **_kwargs)

        return (_hl_artists, _vl_artists)

    @staticmethod
    def _plot_contour_xy(target_axes, x, y, label):
        _kwargs = ContoursProfiler._DEFAULT_PLOT_FILL_CONTOUR_KWARGS.copy()
        return target_axes.fill(x, y, label=label, **_kwargs)


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
        return self._fit._fitter.profile(parameter, **_kwargs)

    def get_contours(self, parameter_1, parameter_2):
        """
        Calculate and return a list of contours (one for each requested sigma value).

        :param parameter_1: name of the first contour parameter
        :type parameter_1: str
        :param parameter_2:  name of the second contour parameter
        :type parameter_2: str
        :return: list of tuples of the form (sigma, contour)
        :rtype: list of 2-tuples of float and 2d-array
        """
        _contours = []
        for _sigma in self._contour_kwargs['sigma_values']:
            _kwargs = dict(numpoints=self._contour_kwargs['points'], sigma=_sigma)
            _contours.append((_sigma, self._fit._fitter.contour(parameter_1, parameter_2, **_kwargs)))
        return _contours

    # - plot profiles/contours

    def plot_profile(self, parameter, target_axes=None,
                     show_grid=True, show_legend=True, show_helper_lines=True, show_ticks=True):
        """
        Plot the profile cost function for a parameter.

        :param parameter: name of the parameter to profile in
        :type parameter: str
        :param target_axes: ``Axes`` object (if ``None``, a new figure is created)
        :type target_axes: ``matplotlib`` ``Axes` or ``None``
        :param show_grid: if ``True``, a grid is drawn
        :type show_grid: bool
        :param show_legend: if ``True``, the legend is displayed
        :type show_legend: bool
        :param show_helper_lines: if ``True``, a number of horizontal and vertical helper lines are
                            displayed at the minimum and the parameter uncertainty
        :type show_helper_lines: bool
        :param show_ticks: if ``True``, *x* and *y* ticks are displayed
        :type show_ticks: bool
        :return: 3-tuple with lists containin the profile, parabola and helper line artists
        :rtype: tuple of lists of ``matplotlib`` artists
        :return:
        """
        if target_axes is None:
            _fig, _gs = self._make_figure_gs(1, 1)
            _axes = plt.subplot(_gs[0, 0])
        else:
            _axes = target_axes


        _par_val = self._fit.parameter_name_value_dict[parameter]
        _par_id = self._fit.parameter_name_value_dict.keys().index(parameter)
        _par_err = self._fit.parameter_errors[_par_id]
        _cost_function_min = self._fit.cost_function_value
        _par_formatted_name = self._parameters_formatted_names[_par_id]

        _x, _y = self.get_profile(parameter)

        _profile_artist = self._plot_profile_xy(_axes, _x, _y, label="profile %s" % (self._cost_function_formatted_name,))

        _parabola_artist = self._plot_parabolic_cost(_axes,
                                  _x,
                                  quad_coeff=1. / (_par_err**2),
                                  x_offset=_par_val,
                                  y_offset=_cost_function_min,
                                  label="parabolic approximation")

        _helper_lines_artists = None
        if show_helper_lines:
            _hlines_y = [_cost_function_min, _cost_function_min + 1.0]
            _vlines_x = [_par_val - _par_err, _par_val, _par_val + _par_err]

            _helper_lines_artists = self._plot_helper_lines(_axes, hlines_y=_hlines_y, vlines_x=_vlines_x)

        _axes.set_xlabel(_par_formatted_name)
        _axes.set_ylabel(self._cost_function_formatted_name)

        if show_legend:
            _axes.legend(loc='best')

        if show_grid:
            _axes.grid('on')

        if not show_ticks:
            _axes.set_xticks([])
            _axes.set_yticks([])

        return _profile_artist, _parabola_artist, _helper_lines_artists

    def plot_contours(self, parameter_1, parameter_2, target_axes=None,
                      show_grid=True, show_legend=True, show_ticks=True):
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
        :param show_ticks: if ``True``, *x* and *y* ticks are displayed
        :type show_ticks: bool
        :return: list of artists returned by ``matplotlib``
        :rtype: list
        """
        if target_axes is None:
            _fig, _gs = self._make_figure_gs(1, 1)
            _axes = plt.subplot(_gs[0, 0])
        else:
            _axes = target_axes

        _par_1_id = self._fit.parameter_name_value_dict.keys().index(parameter_1)
        _par_2_id = self._fit.parameter_name_value_dict.keys().index(parameter_2)
        _par_1_formatted_name = self._parameters_formatted_names[_par_1_id]
        _par_2_formatted_name = self._parameters_formatted_names[_par_2_id]

        _sigma_contour_pairs = self.get_contours(parameter_1, parameter_2)

        _contour_artists = []
        for _sigma, _contour_xy in _sigma_contour_pairs:
            _x, _y = _contour_xy
            _artist = self._plot_contour_xy(_axes, _x, _y, label="%g-sigma contour" % (_sigma,))
            _contour_artists.append(_artist)

        _axes.set_xlabel(_par_1_formatted_name)
        _axes.set_ylabel(_par_2_formatted_name)

        if show_legend:
            _axes.legend(loc='best')

        if show_grid:
            _axes.grid('on')

        if not show_ticks:
            _axes.set_xticks([])
            _axes.set_yticks([])

        return _contour_artists


    def plot_profiles_contours_matrix(self,
                                      show_grid_for=None,
                                      show_ticks_for=None,
                                      show_legend=True,
                                      show_helper_lines=False):
        """
        Plot all profiles and contours to subplots arranges in a matrix-like fashion.

        :param show_ticks_for: subplots for which to show a grid
        :type show_grid_for: ``None``, ``'profiles'``, ``'contours'`` or ``'all'``
        :param show_legend: if ``True``, the legend is displayed
        :type show_legend: bool
        :param show_helper_lines: if ``True``, a number of horizontal and vertical helper lines are
                            displayed at the minimum and the parameter uncertainty
        :type show_helper_lines: bool
        :param show_ticks_for: subplots for which to show ticks
        :type show_ticks_for: ``None``, ``'profiles'``, ``'contours'`` or ``'all'``
        """
        _par_names = self._fit.parameter_name_value_dict.keys()

        _npar = len(_par_names)
        _fig, _gs = self._make_figure_gs(_npar, _npar)

        _show_spec_options = ('all', 'profiles', 'contours')

        if show_grid_for is not None and show_grid_for not in _show_spec_options:
            raise ContoursProfilerException("Unknown specification '%s' for 'show_grid_for'. "
                                            "Expected: one of %r" % (show_grid_for, _show_spec_options))
        if show_ticks_for is not None and show_ticks_for not in _show_spec_options:
            raise ContoursProfilerException("Unknown specification '%s' for 'show_ticks_for'. "
                                            "Expected: one of %r" % (show_ticks_for, _show_spec_options))

        _show_grid_profiles = show_grid_for in ('all', 'profiles')
        _show_grid_contours = show_grid_for in ('all', 'contours')
        _show_ticks_profiles = show_ticks_for in ('all', 'profiles')
        _show_ticks_contours = show_ticks_for in ('all', 'contours')


        for row in xrange(_npar):
            _axes = plt.subplot(_gs[row, row])
            self.plot_profile(_par_names[row], target_axes=_axes,
                              show_grid=_show_grid_profiles,
                              show_legend=False,
                              show_helper_lines=show_helper_lines,
                              show_ticks=_show_ticks_profiles)

            _profile_legend_handles_labels = _axes.get_legend_handles_labels()

            for col in xrange(row):
                _axes = plt.subplot(_gs[row, col])
                self.plot_contours(_par_names[col], _par_names[row],
                                   target_axes=_axes,
                                   show_grid=_show_grid_contours,
                                   show_legend=False,
                                   show_ticks=_show_ticks_contours)
                _contour_legend_handles_labels = _axes.get_legend_handles_labels()

        if show_legend:
            _hs = _profile_legend_handles_labels[0] + _contour_legend_handles_labels[0]
            _ls = _profile_legend_handles_labels[1] + _contour_legend_handles_labels[1]
            _fig.legend(_hs, _ls, loc='upper right')

        _gs.tight_layout(_fig)