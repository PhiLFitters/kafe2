import abc
import six

from .._base import GenericDReprBase
# import data container classes
from ...histogram import HistContainer
from ...indexed import IndexedContainer
from ...unbinned import UnbinnedContainer
from ...xy import XYContainer

__all__ = ["DataContainerDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class DataContainerDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'container'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistContainer: 'histogram',
        IndexedContainer: 'indexed',
        UnbinnedContainer: 'unbinned',
        XYContainer: 'xy'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistContainer,
        'indexed': IndexedContainer,
        'unbinned': UnbinnedContainer,
        'xy': XYContainer
    }

    def __init__(self, data_container=None):
        self._kafe_object = data_container
        super(DataContainerDReprBase, self).__init__()
