import numpy as np
import matplotlib.pyplot as plt
from kafe.fit.xy_multi import *

NUM_POINTS = 10
a0, b0, c0, d0 = 0.25, 0.6, 0.7, -10.0
Y_ERR = 1.0
X_ERR = 0.1

# a simple quadratic 'xy' model (3 parameters)
def quad_model(x, b, c, d):
    return b * x ** 2 + c * x + d

def quad_model2(x, b, c, d):
    return b * x ** 2 + c * x + d + 20.0

def cube_model(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def example_xy_fit():
    """Workflow for a kafe fit to an 'XY' data set"""

    # set the quadratic model parameters to numeric values

    # compute pseudo-data for 'xy' model: same as 'Indexed', but 'x' is part of data!
    x0 = np.arange(NUM_POINTS)
    y0 = quad_model(x0 + np.random.normal(0, X_ERR, NUM_POINTS), b0, c0, d0) + np.random.normal(0, Y_ERR, NUM_POINTS)
    x1 = np.arange(NUM_POINTS) + 0.5
    #idx_data1 = quad_model2(np.arange(NUM_POINTS), b0, c0, d0) + np.random.normal(0, 0.001, NUM_POINTS)
    y1 = cube_model(x1 + np.random.normal(0, X_ERR, NUM_POINTS), a0, b0, c0, d0) + np.random.normal(0, Y_ERR, NUM_POINTS)


    # -- do some kafe fits: XYFit

    fits = []  # store our kafe 'Fit' objects here

    # initialize an 'IndexedFit'
    f = XYMultiFit(xy_data=[[x0, y0], [x1, y1]],
              model_function=[quad_model, cube_model])
#    f = XYFit(xy_data=[[np.arange(NUM_POINTS)], [idx_data0]],
#              model_function=[quad_model])

    # give parameters (trivial) LaTeX names
    f.assign_parameter_latex_names(a='a', b='b', c='c', d='d')

    # assign strings for the function expression
    f.assign_model_function_expression("{0}*{x}^2 + {1}*{x} + {2}", 0)
    f.assign_model_function_latex_expression(r"{0}\,{x}^2 + {1}\,{x} + {2}", 0)
    f.assign_model_function_expression("{0}*{x}^3 + {1}*{x}^2 + {2}*{x} + {3}", 1)
    f.assign_model_function_latex_expression(r"{0}\,{x}^3 + {1}\,{x}^2 + {2}\,{x} + {3}", 1)

    # add an error source to the 'Fit' object error model
    f.add_simple_error('y', Y_ERR, correlation=0.0)  # all points have a (Gaussian) uncertainty in 'y' of +/-1.0
    f.add_simple_error('x', X_ERR, correlation=0.0)

    # do the fit
    f.do_fit()
    f.report()
    
    # store the result
    fits.append(f)


    # create the plots
    p = XYMultiPlot(fit_objects=fits, separate_plots=False)
    p.plot()
    p.show_fit_info_box(format_as_latex=True)

#example_xy_fit()
#plt.show()