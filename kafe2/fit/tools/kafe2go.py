#!/usr/bin/env python
import argparse
import sys

import matplotlib.pyplot as plt

# do not use relative imports here as the working directory of kafe2go can be anywhere on the system
from kafe2 import Plot
from kafe2.fit._base.fit import FitBase
from kafe2.fit.tools.contours_profiler import ContoursProfiler
from kafe2.fit.xy.plot import XYPlotAdapter


class Kafe2GoException(Exception):
    pass


# TODO documentation
def kafe2go():
    _parser = argparse.ArgumentParser(
        description="Perform a fit with the kafe2 package driven by an input file.\n"
                    "Example files can be found at "
                    "https://github.com/dsavoiu/kafe2/tree/master/examples.\n"
                    "Further information on how to create input files is given at "
                    "https://kafe2.readthedocs.io/en/latest/parts/user_guide.html#kafe2go."
        )

    _parser.add_argument('filename', type=str, nargs='+',
                         help="Name(s) of fit input file(s).")
    _parser.add_argument('-if', '--inputformat',
                         type=str, default='yaml',
                         help="File input format. The default format is yaml.")
    _parser.add_argument('-s', '--saveplot',
                         action='store_true',
                         help="Save plot(s) to file(s). The plot(s) will be saved in the current "
                              "working directory.")
    _parser.add_argument('-pf', '--plotformat',
                         type=str, default='pdf',
                         help="Graphics output file format. E.g. pdf, png, svg, ... "
                              "The default format is pdf.")
    _parser.add_argument('-n', '--noplot',
                         action='store_true',
                         help="Don't show plots on screen.")
    _parser.add_argument('-r', '--ratio',
                         action='store_true',
                         help="Show data/model ratio below the main plot.")
    _parser.add_argument('-a', '--asymmetric',
                         action='store_true',
                         help="Show asymmetric parameter uncertainties when displaying the fit "
                              "information. This affects the fit report to the terminal as well as "
                              "the information box of the plot.")
    _parser.add_argument('-c', '--contours',
                         action='store_true',
                         help="Plot contours and profiles.")
    _parser.add_argument('--grid', type=str, nargs=1, default=[None],
                         help="Add a grid to the contour profiles. Available options are either "
                              "all, contours or profiles.")
    _parser.add_argument('--noband',
                         action='store_true',
                         help="Don't draw the 1-sigma band around the fitted function. "
                              "This will only affect plots of XY-fits.")
    _parser.add_argument('--noinfobox',
                         action='store_true',
                         help="Don't add the model info boxes to the plot(s).")
    _parser.add_argument('--separate',
                         action='store_true',
                         help="Create a separate figure for each fit when plotting.")
    _parser.add_argument('--noreport',
                         action='store_true',
                         help="Don't print fit report(s) to the terminal after fitting.")

    if len(sys.argv) == 1:  # print help message if no input given
        _parser.print_help()
        sys.exit(1)
    _args = _parser.parse_args()

    _filenames = _args.filename
    _band = not _args.noband
    _contours = _args.contours
    _report = not _args.noreport
    _infobox = not _args.noinfobox
    _ratio = _args.ratio
    _asymmetric = _args.asymmetric
    _separate = _args.separate
    _input_format = _args.inputformat
    _plot_format = _args.plotformat
    _save_plot = _args.saveplot
    _show_plot = not _args.noplot
    _grid = _args.grid[0]

    _fits = []

    for _fname in _filenames:
        _fit = FitBase.from_file(_fname, file_format=_input_format)
        _fit.do_fit()
        if _report:
            _fit.report(asymmetric_parameter_errors=_asymmetric)
        _fits.append(_fit)

    if not _band:
        XYPlotAdapter.PLOT_SUBPLOT_TYPES.pop('model_error_band')

    _plot = Plot(fit_objects=_fits, separate_figures=_separate)
    _plot.plot(fit_info=_infobox, asymmetric_parameter_errors=_asymmetric, ratio=_ratio)

    _basenames = [name.rsplit('.', 1)[0] for name in _filenames]

    if _save_plot:
        if len(_plot.figures) == 1:
            names = ['_'.join(_basenames)]
        else:
            names = _basenames
        for fig, name in zip(_plot.figures, names):
            fig.savefig(fname='{}.{}'.format(name, _plot_format), format=_plot_format)

    if _contours:
        for _fit, name in zip(_fits, _basenames):
            _profiler = ContoursProfiler(_fit)
            _profiler.plot_profiles_contours_matrix(show_grid_for=_grid)
            if _save_plot:
                for i, fig in enumerate(_profiler.figures):
                    fig.savefig(fname='{}_contours_{}.{}'.format(name, i, _plot_format),
                                format=_plot_format)

    if _show_plot:
        plt.show()


if __name__ == '__main__':
    kafe2go()
