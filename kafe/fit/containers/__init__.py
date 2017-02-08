import abc


class DataContainerException(Exception):
    pass

class DataContainerBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def size(self): pass

    @abc.abstractproperty
    def data(self): pass

    @abc.abstractproperty
    def err(self): pass

    @abc.abstractproperty
    def cov_mat(self): pass

    @abc.abstractproperty
    def cov_mat_inverse(self): pass

    @property
    def has_errors(self):
        return True if self._error_dicts else False


class ParametricModelBaseMixin(object):
    """
    Mixin class. Defines additional properties and methods to be 'mixed into' another class.
    """
    def __init__(self, model_func, model_parameters, *args, **kwargs):
        # print "ParametricModelBaseMixin.__init__(model_func=%r, model_parameters=%rb, args=%r, kwargs=%r)" % (model_func, model_parameters, args, kwargs)
        self._model_function_handle = model_func
        self.parameters = model_parameters
        super(ParametricModelBaseMixin, self).__init__(*args, **kwargs)

    @property
    def parameters(self):
        return self._model_parameters

    @parameters.setter
    def parameters(self, parameters):
        self._model_parameters = parameters
        self._pm_calculation_stale = True


# public interface of submodule 'kafe.fit.containers'

from .indexed import IndexedContainer, IndexedParametricModel
from .histogram import HistContainer, HistParametricModel
from .xy import XYContainer, XYParametricModel

__all__ = ['HistContainer', 'HistParametricModel',
           'IndexedContainer', 'IndexedParametricModel',
           'XYContainer', 'XYParametricModel']
