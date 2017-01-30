import abc

from .nexus_fitter import NexusFitter
#from .simple_fitter import SimpleFitter

__all__ = ['SimpleFitter', 'NexusFitter']

class FitterBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def do_fit(self): pass


    @abc.abstractmethod
    def _fcn_wrapper(self): pass