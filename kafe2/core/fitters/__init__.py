import abc
import six
#import pyximport

#pyximport.install()

from ...config import kc

# from .nexus_cython import Nexus as NexusCython
#from simple_fitter import SimpleFitter




__all__ = ['get_fitter']

AVAILABLE_FITTERS = dict()


from .nexus import Nexus
from .nexus_fitter import NexusFitter, NexusFitterException
__all__ += ['Nexus', 'NexusFitter', 'NexusFitterException']

AVAILABLE_FITTERS.update({
    'nexus_fitter': NexusFitter,
})

# try:
#     from .simple_fitter import SimpleFitter, SimpleFitterException
#     __all__ += ['SimpleFitter', 'SimpleFitterException']
#
#     AVAILABLE_FITTERS.update({
#         'simple': SimpleFitter,
#     })
# except ImportError:
#     pass


def get_fitter(fitter_spec):
    global AVAILABLE_FITTERS

    # for 'None', return the default fitter
    if fitter_spec is None:
        _fitter = AVAILABLE_FITTERS.get(kc['core']['fitters']['default_fitter'], None)
    else:
        fitter_spec = fitter_spec.lower()
        _fitter = AVAILABLE_FITTERS.get(fitter_spec, None)

    if _fitter is None:
        raise ValueError("Unknown fitter '{}'! Available: {}".format(fitter_spec, AVAILABLE_FITTERS.keys()))

    return _fitter


@six.add_metaclass(abc.ABCMeta)
class FitterBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """

    @abc.abstractproperty
    def do_fit(self): pass


    @abc.abstractmethod
    def _fcn_wrapper(self): pass
