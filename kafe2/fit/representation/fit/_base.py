import abc

import six

# import fit classes
from ...custom import CustomFit
from ...histogram import HistFit
from ...indexed import IndexedFit
from ...unbinned import UnbinnedFit
from ...xy import XYFit
from .._base import GenericDReprBase

__all__ = ["FitDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class FitDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = "fit"

    _CLASS_TO_OBJECT_TYPE_NAME = {
        CustomFit: "custom",
        HistFit: "histogram",
        IndexedFit: "indexed",
        UnbinnedFit: "unbinned",
        XYFit: "xy",
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        "custom": CustomFit,
        "histogram": HistFit,
        "indexed": IndexedFit,
        "unbinned": UnbinnedFit,
        "xy": XYFit,
    }

    def __init__(self, fit=None):
        self._kafe_object = fit
        super(FitDReprBase, self).__init__()
