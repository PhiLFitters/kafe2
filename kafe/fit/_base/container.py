import abc


class DataContainerException(Exception):
    pass


class DataContainerBase(object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    types of data containers.

    It stores measurement data and uncertainties.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def size(self):
        """The size of the data (number of measurement points)"""
        pass

    @abc.abstractproperty
    def data(self):
        """A numpy array containing the data values"""
        pass

    @abc.abstractproperty
    def err(self):
        """A numpy array containing the pointwise data uncertainties"""
        pass

    @abc.abstractproperty
    def cov_mat(self):
        """A numpy matrix containing the covariance matrix of the data"""
        pass

    @abc.abstractproperty
    def cov_mat_inverse(self):
        """A numpy matrix containing inverse of the data covariance matrix (or ``None`` if not invertible)"""
        pass

    @property
    def has_errors(self):
        """``True`` if at least one uncertainty source is defined for the data container"""
        return True if self._error_dicts else False