import abc
import six

from kafe2.fit.histogram import HistContainer
from kafe2.fit.indexed import IndexedContainer
from kafe2.fit.xy import XYContainer
from kafe2.fit.xy_multi import XYMultiContainer
from kafe2.fit.representation._base import GenericDReprBase

__all__ = ["DataContainerDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class DataContainerDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'container'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistContainer: 'histogram',
        IndexedContainer: 'indexed',
        XYContainer: 'xy',
        XYMultiContainer: 'xy_multi'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistContainer,
        'indexed': IndexedContainer,
        'xy': XYContainer,
        'xy_multi': XYMultiContainer
    }

    def __init__(self, data_container=None):
        self._kafe_object = data_container
        super(DataContainerDReprBase, self).__init__()