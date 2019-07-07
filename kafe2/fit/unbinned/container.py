import numpy as np

from ..indexed import IndexedContainer
from ..indexed.container import IndexedContainerException

__all__ = ["UnbinnedContainer"]


class UnbinnedContainerException(IndexedContainerException):
    pass


class UnbinnedContainer(IndexedContainer):
    """
    This object is a specialized data container for a series of data points.

    """
    def __init__(self, data, dtype=float):
        """
        Construct a container for unbinned data:

        :param data: a one-dimensional array of measurements
        :type data: iterable of type <dtype>
        :param dtype: data type of the measurements
        :type dtype: type
        """
        super(UnbinnedContainer, self).__init__(data=data, dtype=dtype)

    # for now everything is the same to the indexed container
