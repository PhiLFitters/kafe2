import abc

from kafe.fit.indexed import IndexedFit
from kafe.fit.xy import XYFit
from kafe.fit.histogram import HistFit
from kafe.fit.representation._base import GenericDReprBase

__all__ = ["FitDReprBase"]


class FitDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = 'fit'

    _FIT_CLASS_TO_TYPE_NAME = {
        IndexedFit: 'indexed',
        XYFit: 'xy',
        HistFit: 'histogram'
    }
    _FIT_TYPE_NAME_TO_CLASS = {
        'indexed': IndexedFit,
        'xy': XYFit,
        'histogram': HistFit
    }

    def __init__(self, fit=None):
        self._fit = fit
        super(FitDReprBase, self).__init__()
