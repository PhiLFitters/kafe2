from ...config import matplotlib as mpl
from . import FitBase
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs


class ContoursProfilerException(Exception):
    pass


class ContoursProfiler(object):

    def __init__(self, fit_object):
        if not isinstance(fit_object, FitBase):
            raise ContoursProfilerException("Object %r is not a fit object!" % (fit_object,))
        self._fit = fit_object

    def get_profile(self, parameter, bins=20, bound=2, args=None, subtract_min=False):
        return self._fit._fitter.profile(parameter, bins=bins, bound=bound, args=args, subtract_min=subtract_min)

    def get_contour(self, parameter_1, parameter_2, numpoints=20, sigma=1.0):
        return self._fit._fitter.contour(parameter_1, parameter_2, numpoints=numpoints, sigma=sigma)

    def plot_profile(self, parameter_name, bins=20, bound=2, args=None, subtract_min=False, **plot_kwargs):
        _x, _y = self.get_profile(parameter_name, bins=bins, bound=bound, args=args, subtract_min=subtract_min)
        _pf_plt = ProfilePlot(self, parameter_name)
        _pf_plt._plot_profile_xy(_x, _y, **plot_kwargs)
        return _pf_plt

    def plot_contour(self, parameter_1, parameter_2, numpoints=20, sigma=1.0, **plot_kwargs):
        _x, _y = self.get_contour(parameter_1, parameter_2, numpoints=numpoints, sigma=sigma)
        _ct_plt = ContourPlot(self, (parameter_1, parameter_2))
        _ct_plt._plot_contour_xy(_x, _y, **plot_kwargs)
        return _ct_plt


class ContourProfilePlotFigureException(Exception):
    pass


class ContourProfilePlotFigureBase(object):
    def __init__(self):
        self._fig = plt.figure(figsize=(8, 8))  # defaults from matplotlibrc
        # self._figsize = (self._fig.get_figwidth()*self._fig.dpi, self._fig.get_figheight()*self._fig.dpi)
        self._outer_gs = gs.GridSpec(nrows=1,
                                     ncols=1,
                                     left=0.075,
                                     bottom=0.1,
                                     right=0.925,
                                     top=0.9,
                                     wspace=None,
                                     hspace=None,
                                     height_ratios=None)

        # TODO: use these later to maintain absolute margin width on resize
        self._absolute_outer_sizes = dict(left=self._outer_gs.left * self._fig.get_figwidth() * self._fig.dpi,
                                          bottom=self._outer_gs.bottom * self._fig.get_figheight() * self._fig.dpi,
                                          right=self._outer_gs.right * self._fig.get_figwidth() * self._fig.dpi,
                                          top=self._outer_gs.top * self._fig.get_figheight() * self._fig.dpi)

        self._plot_axes_gs = gs.GridSpecFromSubplotSpec(
            nrows=1,
            ncols=1,
            wspace=None,
            hspace=None,
            width_ratios=None,
            # height_ratios=[8, 1],
            subplot_spec=self._outer_gs[0, 0]
        )
        self._main_plot_axes = plt.subplot(self._plot_axes_gs[0, 0])


class ProfilePlot(ContourProfilePlotFigureBase):
    def __init__(self, contours_profiler, parameter_name):
        self._cpf = contours_profiler
        self._par_name = parameter_name
        super(ProfilePlot, self).__init__()

    def _plot_profile_xy(self, x, y, **plot_kwargs):
        self._main_plot_axes.plot(x, y, **plot_kwargs)


class ContourPlot(ContourProfilePlotFigureBase):
    def __init__(self, contours_profiler, parameter_names):
        self._cpf = contours_profiler
        self._par_names = parameter_names
        if len(self._par_names) != 2:
            raise ContourProfilePlotFigureException("'parameter_names' must have length 2!")
        super(ContourPlot, self).__init__()

    def _plot_contour_xy(self, x, y, **plot_kwargs):
        self._main_plot_axes.plot(x, y, **plot_kwargs)