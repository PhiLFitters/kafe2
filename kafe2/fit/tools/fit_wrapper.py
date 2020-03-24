import numpy as np

from ..indexed import IndexedContainer, IndexedFit
from ..histogram import HistContainer, HistFit
from ..unbinned import UnbinnedContainer, UnbinnedFit
from ..xy import XYContainer, XYFit


__all__ = ['Fit']


def Fit(data_container, model_function=None):
    """A convenience wrapper for simple fit creation. For more control over the fit creation use the corresponding Fit
    classes.

    :param data_container: A :py:obj:`~kafe2.fit._base.DataContainerBase`-derived data container, containing the data
                           used in the fit.
    :type data_container: IndexedContainer or HistContainer or UnbinnedContainer or XYContainer or list
    :param model_function: The model function used in the fit.
    """
    container_to_fit = {IndexedContainer: IndexedFit,
                        HistContainer: HistFit,
                        UnbinnedContainer: UnbinnedFit,
                        XYContainer: XYFit,
                        list: XYFit,
                        np.ndarray: XYFit}
    try:
        fit_class = container_to_fit[type(data_container)]
    except KeyError:
        raise TypeError("Unknown or unsupported data container. Supported types are {}".format(container_to_fit.keys()))
    # other errors will raise during creation of the fit object
    if model_function is None:
        return fit_class(data_container)  # use default model function
    return fit_class(data_container, model_function)
