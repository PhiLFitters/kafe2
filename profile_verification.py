from kafe2.fit._base.profile import ContoursProfiler
from kafe2.fit.xy.fit import XYFit
import numpy as np
import matplotlib.pyplot as plt

KAFE_PLOT = False

_x = [-1.0, 1.0]
_y = [-1.0, 1.0]
_x_err = np.sqrt(2.0)
_y_err = np.sqrt(2.0)

def model_function(x, a):
    return x * a

_fit_iminuit = XYFit(xy_data=[_x, _y], model_function=model_function, minimizer='iminuit')

_fit_iminuit.add_simple_error('x', _x_err)
_fit_iminuit.add_simple_error('y', _y_err)

_fit_iminuit.do_fit()

#_plot = _fit_iminuit.generate_plot()
#_plot.plot()
#_plot.show_fit_info_box(format_as_latex=True)

_profiler_iminuit = ContoursProfiler(_fit_iminuit)
_a_profile_iminuit = _profiler_iminuit.get_profile('a')

_fit_scipy = XYFit(xy_data=[_x, _y], model_function=model_function, minimizer='scipy')

_fit_scipy.add_simple_error('x', _x_err)
_fit_scipy.add_simple_error('y', _y_err)

_fit_scipy.do_fit()

#_plot = _fit_scipy.generate_plot()
#_plot.plot()
#_plot.show_fit_info_box(format_as_latex=True)

_profiler_scipy = ContoursProfiler(_fit_scipy)
_a_profile_scipy = _profiler_scipy.get_profile('a')

if KAFE_PLOT:
    _profiler_iminuit.plot_profile('a', label_ticks_in_sigma=False)
    _profiler_scipy.plot_profile('a', label_ticks_in_sigma=False)
else:
    plt.plot(_a_profile_iminuit[0], _a_profile_iminuit[1], '-', label='iminuit')
    plt.plot(_a_profile_scipy[0], _a_profile_scipy[1], '--', label='scipy')
    _a = _a_profile_scipy[0]
    plt.plot(_a, 1 - 2.0 * _a / (1.0 + _a ** 2), '.', label='analytical')
    plt.legend(loc='best')

plt.show()
