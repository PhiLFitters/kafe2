import abc

from kafe.fit.histogram import HistContainer
from kafe.fit.indexed import IndexedContainer
from kafe.fit.xy import XYContainer
from kafe.fit.xy_multi import XYMultiContainer
from kafe.fit.representation._base import GenericDReprBase

__all__ = ["DataContainerDReprBase"]


class DataContainerDReprBase(GenericDReprBase):
    __metaclass__ = abc.ABCMeta
    OBJECT_TYPE_NAME = 'container'

    _CONTAINER_CLASS_TO_TYPE_NAME = {
        HistContainer: 'histogram',
        IndexedContainer: 'indexed',
        XYContainer: 'xy',
        XYMultiContainer: 'xy_multi'
    }
    _CONTAINER_TYPE_NAME_TO_CLASS = {
        'histogram': HistContainer,
        'indexed': IndexedContainer,
        'xy': XYContainer,
        'xy_multi': XYMultiContainer
    }

    def __init__(self, data_container=None):
        self._container = data_container
        super(DataContainerDReprBase, self).__init__()