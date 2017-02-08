"""
Handle global kafe configuration
"""

import os

# -- import matplotlib and source matplotlibrc file
import matplotlib
matplotlib.rc_file(os.path.join(__path__[0], 'kafe.matplotlibrc.conf'))
