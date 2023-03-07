from ..indexed import IndexedContainer

__all__ = ["UnbinnedContainer"]


class UnbinnedContainer(IndexedContainer):
    """
    This object is a specialized data container for series of measurements.

    """

    def __init__(self, data, dtype=float):
        """
        Construct a container for indexed data:

        :param data: a one-dimensional array of measurements
        :type data: iterable of type <dtype>
        :param dtype: data type of the measurements
        :type dtype: type
        """
        super(UnbinnedContainer, self).__init__(data, dtype)

    def add_error(self):
        raise NotImplementedError("Unbinned fits don't support errors")

    def add_matrix_error(self):
        raise NotImplementedError("Unbinned fits don't support errors")
