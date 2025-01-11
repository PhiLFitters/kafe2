import logging

__version__ = "2.10.0-pre1"
from .fit import *  # noqa: F401, F403

logger = logging.getLogger(__name__)
logging.basicConfig()

del logging
