import abc

__all__ = ['get_minimizer']

from scipy_optimize_minimizer import MinimizerScipyOptimize
__all__.append('MinimizerScipyOptimize')

try:
    from iminuit_minimizer import MinimizerIMinuit
    __all__.append('MinimizerIMinuit')
except ImportError:
    pass

try:
    from root_tminuit_minimizer import MinimizerROOTTMinuit
    __all__.append('MinimizerROOTTMinuit')
except ImportError:
    pass


# AVAILABLE_MINIMIZERS = {'iminuit': MinimizerIMinuit,
#                         'minuit': MinimizerROOTTMinuit,
#                         'root::tminuit': MinimizerROOTTMinuit,
#                         'root': MinimizerROOTTMinuit,
#                         'scipy': MinimizerScipyOptimize,
#                         'scipy.optimize': MinimizerScipyOptimize}
#
# # TODO: replace with config-based solution
# DEFAULT_MINIMIZER = MinimizerIMinuit
#
#
# def get_minimizer(minimizer_spec):
#     global AVAILABLE_MINIMIZERS
#     minimizer_spec = minimizer_spec.lower()
#     return AVAILABLE_MINIMIZERS.get(minimizer_spec, DEFAULT_MINIMIZER)


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