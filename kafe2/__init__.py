import logging

__version__ = "2.8.0"
from .fit import *

logger = logging.getLogger(__name__)
logging.basicConfig()

del logging
