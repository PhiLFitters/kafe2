"""
Handle global kafe configuration
"""

import os
import yaml

# -- read in kafe global configuration from
with open(os.path.join(__path__[0], 'kafe.yaml')) as _f:
    kc = yaml.load(_f)

# -- import matplotlib and source matplotlibrc file
import matplotlib
matplotlib.rc_file(os.path.join(__path__[0], 'kafe.matplotlibrc.conf'))
