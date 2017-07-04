import abc

from ...config import kc

__all__ = ['get_minimizer']

AVAILABLE_MINIMIZERS = dict()

try:
    from .scipy_optimize_minimizer import MinimizerScipyOptimize
    __all__.append('MinimizerScipyOptimize')
    AVAILABLE_MINIMIZERS.update({
        'scipy': MinimizerScipyOptimize,
        'scipy.optimize': MinimizerScipyOptimize,
    })
except ImportError:
    pass

try:
    from .iminuit_minimizer import MinimizerIMinuit
    __all__.append('MinimizerIMinuit')
    AVAILABLE_MINIMIZERS.update({
        'iminuit': MinimizerIMinuit,
    })
except ImportError:
    pass

try:
    from .root_tminuit_minimizer import MinimizerROOTTMinuit
    __all__.append('MinimizerROOTTMinuit')
    AVAILABLE_MINIMIZERS.update({
        'minuit': MinimizerROOTTMinuit,
        'root::tminuit': MinimizerROOTTMinuit,
        'root': MinimizerROOTTMinuit,
    })
except ImportError:
    pass

# raise if no minimizers can be imported
if not AVAILABLE_MINIMIZERS:
    raise RuntimeError("Fatal error: no minimizers found! Please ensure that "
                       "at least one of the following Python packages is installed: "
                       "['scipy', 'iminuit', 'ROOT']")

def get_minimizer(minimizer_spec=None):
    global AVAILABLE_MINIMIZERS
    # for 'None', return the default minimizer
    if minimizer_spec is None:
        _minimizer = AVAILABLE_MINIMIZERS.get(kc['core']['minimizers']['default_minimizer'], None)
    else:
        minimizer_spec = minimizer_spec.lower()
        _minimizer = AVAILABLE_MINIMIZERS.get(minimizer_spec, None)

    if _minimizer is None:
        raise ValueError("Unknown minimizer '{}'! Available: {}".format(minimizer_spec, AVAILABLE_MINIMIZERS.keys()))

    return _minimizer


class MinimizerBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def minimize(self): pass

    @abc.abstractproperty
    def hessian(self): pass

    @abc.abstractproperty
    def hessian_inv(self): pass

    @abc.abstractproperty
    def function_value(self): pass