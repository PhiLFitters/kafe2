#!/usr/bin/env python
import argparse, sys
import matplotlib.pyplot as plt
from kafe2 import Plot
from kafe2.fit._base.fit import FitBase
from kafe2.fit.tools.contours_profiler import ContoursProfiler


class Kafe2GoException(Exception):
    pass

#TODO documentation

def kafe2go():
    _parser = argparse.ArgumentParser(description='Perform a fit with the kafe2 package driven by input file')
    
    _parser.add_argument('filenames', type=str, nargs='+',
                         help="name(s) of fit input file(s)")
    _parser.add_argument('-if', '--inputformat',
                         type=str, default='yaml',
                         help="file input format, default=yaml")
    _parser.add_argument('-c', '--contours', 
                         action='store_true',
                         help="plot contours and profiles")
    _parser.add_argument('--noreport',
                         action='store_true',
                         help="don't print fit report after fitting")
    _parser.add_argument('-n', '--noplot',
                         action='store_true',
                         help="don't show plots on screen")
    _parser.add_argument('-r', '--ratio',
                         action='store_true',
                         help="Show the data/model ratio below the main plot")
    # TODO: check how to handle this with multiple fits in one plot
    # _parser.add_argument('--noband',
    #                      action='store_true',
    #                      help="don't draw 1-sigma band around function")
    _parser.add_argument('--noinfobox',
                         action='store_true',
                         help="don't add model info boxes to plots")
    _parser.add_argument('--separate_figures', '-sf',
                         action='store_true',
                         help="create a separate figure for each fit when plotting")
    _parser.add_argument('-pf', '--plotformat',
                         type=str, default='pdf',
                         help="graphics output format, default=pdf")
    _parser.add_argument('-s', '--saveplot', 
                         action='store_true',
                         help="save plot(s) in file(s)")

    if len(sys.argv) == 1:  # print help message if no input given
        _parser.print_help()
        sys.exit(1)
    _args = _parser.parse_args()

    _filenames = _args.filenames
    # _band = not _args.noband
    _contours = _args.contours
    _report = not _args.noreport
    _infobox = not _args.noinfobox
    _ratio = _args.ratio
    _separate = _args.separate_figures
    _input_format = _args.inputformat
    _plot_format = _args.plotformat
    _save_plot = _args.saveplot
    _show_plot = not _args.noplot

    _fits = []

    for _fname in _filenames:
        _fit = FitBase.from_file(_fname, format=_input_format)
        _fit.do_fit()
        if _report:
            _fit.report()
        _fits.append(_fit)
    
    _plot = Plot(fit_objects=_fits, separate_figures=_separate)
    
    # if not _band:
    #     _plot._defined_plot_types.remove('model_error_band')
    #     _plot.PLOT_SUBPLOT_TYPES.pop('model_error_band', None)
        
    _plot.plot(with_fit_info=_infobox, with_ratio=_ratio)

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
            _profiler.plot_profiles_contours_matrix()
            if _save_plot:
                for i, fig in enumerate(_profiler.figures):
                    fig.savefig(fname='{}_contours_{}.{}'.format(name, i, _plot_format), format=_plot_format)
    
    if _show_plot:
        plt.show()


if __name__ == '__main__':
    kafe2go()
