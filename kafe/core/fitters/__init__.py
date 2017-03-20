import abc
#import pyximport

#pyximport.install()

from .nexus import Nexus
# from .nexus_cython import Nexus as NexusCython
#from simple_fitter import SimpleFitter
from .nexus_fitter import NexusFitter

__all__ = ['Nexus', 'SimpleFitter', 'NexusFitter']

AVAILABLE_FITTERS = {'nexus': NexusFitter,
                     #'simple': SimpleFitter,
                    }

# TODO: replace with config-based solution
DEFAULT_FITTER = NexusFitter


def get_fitter(fitter_spec):
    global AVAILABLE_FITTERS
    fitter_spec = fitter_spec.lower()
    return AVAILABLE_FITTERS.get(fitter_spec, DEFAULT_FITTER)


class FitterBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def do_fit(self): pass


    @abc.abstractmethod
    def _fcn_wrapper(self): pass
