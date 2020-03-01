import abc
import six

from .._base import GenericDReprBase
# import fit classes
from ...indexed import IndexedFit
from ...xy import XYFit
from ...histogram import HistFit

__all__ = ["FitDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class FitDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'fit'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistFit: 'histogram',
        IndexedFit: 'indexed',
        XYFit: 'xy'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistFit,
        'indexed': IndexedFit,
        'xy': XYFit
    }

    def __init__(self, fit=None):
        self._kafe_object = fit
        super(FitDReprBase, self).__init__()
