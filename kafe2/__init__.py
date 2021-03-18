import logging

from ._version_info import __version__

from .fit import *

logger = logging.getLogger(__name__)
logging.basicConfig()

del logging
