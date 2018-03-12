import numpy as np
import matplotlib.pyplot as plt
from kafe.fit import XYFit, XYPlot

# a simple quadratic 'xy' model (3 parameters)
def xy_model(x, a, b, c):
    return a * x ** 2 + b * x + c

# a simple 'indexed' model: i-th value is xy_model(i, a, b, c)
def idx_model(a, b, c):
    return xy_model(np.arange(10), a, b, c)

def example_xy_fit():
    """Workflow for a kafe fit to an 'XY' data set"""

    # set the quadratic model parameters to numeric values
    a0, b0, c0 = 0.1, 0.2, 3.3

    # compute pseudo-data for 'xy' model: same as 'Indexed', but 'x' is part of data!
    idx_data0 = idx_model(a0, b0, c0) + np.random.normal(0, 1, 10)
    xydata0 = np.array([np.arange(10), idx_data0])


    # -- do some kafe fits: XYFit

    fits = []  # store our kafe 'Fit' objects here
    for i in xrange(4):
        # shift 'y' data in 'y' direction, so the different fits do not overlap
        _y_data_with_offset = xydata0[1] + i
        _xydata_with_offset = np.array([xydata0[0], _y_data_with_offset])

        # initialize an 'IndexedFit'
        f = XYFit(xy_data=_xydata_with_offset,
                  model_function=xy_model)

        # give parameters (trivial) LaTeX names
        f.assign_parameter_latex_names(a='a', b='b', c='c')

        # assign strings for the function expression
        f.assign_model_function_expression("{0}*{x}^2 + {1}*{x} + {2}")
        f.assign_model_function_latex_expression(r"{0}\,{x}^2 + {1}\,{x} + {2}")

        # add an error source to the 'Fit' object error model
        f.add_simple_error('y', 0.25, correlation=0.01)  # all points have a (Gaussian) uncertainty in 'y' of +/-1.0
        f.add_simple_error('x', 0.25, correlation=0.01)

        # do the fit
        f.do_fit()

        # store the result
        fits.append(f)


    # create the plots
    p = XYPlot(fit_objects=fits)
    p.plot()
    p.show_fit_info_box(format_as_latex=True)

example_xy_fit()
plt.show()
