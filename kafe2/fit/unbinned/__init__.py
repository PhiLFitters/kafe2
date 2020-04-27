"""This submodule provides the necessary objects for parameter estimation from unbinned datasets. Those fits are used
when there are two few data points to fill a histogram. Unbinned fits are essentially the ground truth for histogram
fits as there is no modification of the data points through the binning. As each data point is considered, performing an
unbinned fit can take much longer than a histogram fit, where each bin is considered.

:synopsis: This submodule provides the necessary objects for parameter estimation from unbinned datasets.

.. moduleauthor:: Cedric Verstege <cedric.verstege@student.kit.edu>
"""

from .container import *
from .cost import *
from .fit import *
from .model import *
from .plot import *
