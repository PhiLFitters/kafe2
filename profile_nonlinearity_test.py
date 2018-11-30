import numpy as np
import matplotlib.pyplot as plt
from kafe2 import XYFit
from kafe2.fit._base.profile import ContoursProfiler


np.random.seed(4568)

NUM_DATAPOINTS = 11

def quadratic_model(x, a, b, c):
    return a * x ** 2 + b * x + c


def linear_model(x, b, c):
    return b * x + c

_a_0 = 1.5
_b_0 = 1.5
_c_0 = 1.5
_x_err = 0.2
_y_err = 0.1

_x_0 = np.arange(NUM_DATAPOINTS)
_x_jitter = np.random.normal(loc=0, scale=_x_err, size=NUM_DATAPOINTS)

_y_0 = quadratic_model(_x_0 + _x_jitter, _a_0, _b_0, _c_0)
#_y_0 = linear_model(_x_0 + _x_jitter, _b_0, _c_0)
#_y_0 = linear_model(_x_0, _b_0, _c_0)
_y_jitter = np.random.normal(loc=0, scale=_y_err, size=NUM_DATAPOINTS)
_y_data = _y_0 + _y_jitter

_fit = XYFit(xy_data=[_x_0, _y_data], minimizer='iminuit', model_function='quadratic')

_fit.add_simple_error('x', _x_err)
_fit.add_simple_error('y', _y_err)


_fit.do_fit()

_plot = _fit.generate_plot()
_plot.plot()
_plot.show_fit_info_box(format_as_latex=True)

_profiler = ContoursProfiler(_fit)
_profiler.plot_profiles_contours_matrix(
    show_grid_for='all',
    show_ticks_for='all',
    label_ticks_in_sigma=False
)

plt.show()

