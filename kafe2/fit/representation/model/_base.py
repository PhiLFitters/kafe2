import abc
import six

from kafe2.fit.histogram import HistModelFunction, HistParametricModel
from kafe2.fit.indexed import IndexedModelFunction, IndexedParametricModel
from kafe2.fit.xy import XYParametricModel
from ..._base import ModelFunctionBase
from kafe2.fit.representation._base import GenericDReprBase

__all__ = ["ModelFunctionDReprBase", "ParametricModelDReprBase"]


@six.add_metaclass(abc.ABCMeta)
class ModelFunctionDReprBase(GenericDReprBase):
    BASE_OBJECT_TYPE_NAME = 'model_function'

    _CLASS_TO_OBJECT_TYPE_NAME = {
        HistModelFunction: 'histogram',
        IndexedModelFunction: 'indexed',
        ModelFunctionBase: 'base'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistModelFunction,
        'indexed': IndexedModelFunction,
        'xy': ModelFunctionBase,  # type from fit is passed to model function, needs to be resolved
        'base': ModelFunctionBase
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
        XYParametricModel: 'xy'
    }
    _OBJECT_TYPE_NAME_TO_CLASS = {
        'histogram': HistParametricModel,
        'indexed': IndexedParametricModel,
        'xy': XYParametricModel
    }

    def __init__(self, parametric_model=None):
        self._kafe_object = parametric_model
        super(ParametricModelDReprBase, self).__init__()
