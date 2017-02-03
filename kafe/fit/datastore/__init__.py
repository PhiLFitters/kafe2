import abc


class DataContainerException(Exception):
    pass

class DataContainerBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta


# public interface of submodule 'kafe.fit.datastore'

from .indexed import IndexedContainer, IndexedParametricModel
from .histogram import HistContainer, HistParametricModel
from .xy import XYContainer, XYParametricModel

__all__ = ['HistContainer', 'HistParametricModel',
           'IndexedContainer', 'IndexedParametricModel',
           'XYContainer', 'XYParametricModel']
