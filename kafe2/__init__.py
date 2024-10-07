import logging

__version__ = "2.9.0"
from .fit import *  # noqa: F401, F403

logger = logging.getLogger(__name__)
logging.basicConfig()

del logging
