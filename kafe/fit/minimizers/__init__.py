import abc

from iminuit_minimizer import MinimizerIMinuit

__all__ = ['MinimizerIMinuit']

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