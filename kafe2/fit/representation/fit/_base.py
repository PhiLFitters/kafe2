import abc
import six

from kafe2.fit.indexed import IndexedFit
from kafe2.fit.xy import XYFit
from kafe2.fit.histogram import HistFit
from kafe2.fit.representation._base import GenericDReprBase
from kafe2.fit.xy_multi.fit import XYMultiFit

__all__ = ["FitDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class FitDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'fit'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistFit: 'histogram',
        IndexedFit: 'indexed',
        XYFit: 'xy',
        XYMultiFit: 'xy_multi'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistFit,
        'indexed': IndexedFit,
        'xy': XYFit,
        'xy_multi': XYMultiFit
    }

    def __init__(self, fit=None):
        self._kafe_object = fit
        super(FitDReprBase, self).__init__()
