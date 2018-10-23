#!/usr/bin/env python
import argparse, sys
import matplotlib.pyplot as plt
from kafe.fit._base.fit import FitBase
from kafe.fit._base.profile import ContoursProfiler


class Kafe2GoException(Exception):
    pass

#TODO documentation

def kafe2go():
    _parser = argparse.ArgumentParser(description = 'Perform a fit with the kafe2 package driven by input file')
    
    _parser.add_argument('filename', type=str,
      help="name of fit input file")
    _parser.add_argument('-c', '--contours', 
      action='store_true',
      help="plot contours and profiles")
    _parser.add_argument('-if','--inputformat', 
      type=str, default='yaml',
      help="graphics output format, default=pdf")
    _parser.add_argument('--noband', 
      action='store_true',
      help="don't draw 1-sigma band around function")
    _parser.add_argument('--noinfobox', 
      action='store_true',
      help="don't add model info boxes to plots")
    _parser.add_argument('--nolatex', 
      action='store_true',
      help="don't format model functions and model parameters as LaTeX")
    _parser.add_argument('--noreport', 
      action='store_true',
      help="don't print fit report after fitting")
    _parser.add_argument('-n', '--noplot', 
      action='store_true',
      help="don't show plots on screen")
    _parser.add_argument('-pf','--plotformat', 
      type=str, default='pdf',
      help="graphics output format, default=pdf")
    _parser.add_argument('-s', '--saveplot', 
      action='store_true',
      help="save plot(s) in file(s)")

    if len(sys.argv)==1:  # print help message if no input given
        _parser.print_help()
        sys.exit(1)
    _args = _parser.parse_args()

    _filename = _args.filename
    _band = not _args.noband
    _contours = _args.contours
    _report = not _args.noreport
    _infobox = not _args.noinfobox
    _input_format = _args.inputformat
    _latex = not _args.nolatex
    _plot_format = _args.plotformat
    _save_plot = _args.saveplot
    _show_plot = not _args.noplot
    
    _fit = FitBase.from_file(_filename, format=_input_format)
    
    _fit.do_fit()
    
    if _report:
        _fit.report()
    
    _plot = _fit.generate_plot()
    
    if not _band:
        _plot._defined_plot_types.remove('model_error_band')
        _plot.PLOT_SUBPLOT_TYPES.pop('model_error_band', None)
        
    _plot.plot()
    
    if _infobox:
        _plot.show_fit_info_box(format_as_latex=_latex)

    if _save_plot:
        _basename = _filename[:_filename.rfind('.')]
        plt.savefig(fname='%s.%s' % (_basename, _plot_format), format=_plot_format)
    
    if _contours:
        _profiler = ContoursProfiler(_fit)
        _profiler.plot_profiles_contours_matrix()
        
        if _save_plot:
            plt.savefig(fname='%s_contours.%s' % (_basename, _plot_format), format=_plot_format)
    
    if _show_plot:
        plt.show()

if __name__ == '__main__':
    kafe2go()