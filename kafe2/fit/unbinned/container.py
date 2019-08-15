from ..indexed import IndexedContainer
from ..indexed.container import IndexedContainerException

__all__ = ["UnbinnedContainer"]


class UnbinnedContainerException(IndexedContainerException):
    pass


class UnbinnedContainer(IndexedContainer):
    pass
