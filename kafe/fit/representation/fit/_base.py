import abc

from kafe.fit.indexed import IndexedFit
from kafe.fit.xy import XYFit
from kafe.fit.histogram import HistFit
from kafe.fit.representation._base import GenericDReprBase
from kafe.fit.xy_multi.fit import XYMultiFit

__all__ = ["FitDReprBase"]


class FitDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
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
