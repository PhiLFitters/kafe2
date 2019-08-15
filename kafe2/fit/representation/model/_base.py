import abc
import six

from kafe2.fit.histogram import HistModelFunction, HistParametricModel
from kafe2.fit.indexed import IndexedModelFunction, IndexedParametricModel
from kafe2.fit.xy import XYModelFunction, XYParametricModel
from kafe2.fit.xy_multi import XYMultiModelFunction, XYMultiParametricModel
from kafe2.fit.representation._base import GenericDReprBase

__all__ = ["ModelFunctionDReprBase", "ParametricModelDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class ModelFunctionDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model_function'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistModelFunction: 'histogram',
        IndexedModelFunction: 'indexed',
        XYModelFunction: 'xy',
        XYMultiModelFunction: 'xy_multi'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistModelFunction,
        'indexed': IndexedModelFunction,
        'xy': XYModelFunction,
        'xy_multi': XYMultiModelFunction
    }

    def __init__(self, model_function=None):
        self._kafe_object = model_function
        super(ModelFunctionDReprBase, self).__init__()


@six.add_metaclass(abc.ABCMeta)
class ParametricModelDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistParametricModel: 'histogram',
        IndexedParametricModel: 'indexed',
        XYParametricModel: 'xy',
        XYMultiParametricModel: 'xy_multi'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistParametricModel,
        'indexed': IndexedParametricModel,
        'xy': XYParametricModel,
        'xy_multi': XYMultiParametricModel
    }

    def __init__(self, parametric_model=None):
        self._kafe_object = parametric_model
        super(ParametricModelDReprBase, self).__init__()

