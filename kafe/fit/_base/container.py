import abc


class DataContainerException(Exception):
    pass


class DataContainerBase(object):
    """
    Purely abstract class. Defines the minimal interface required by all specializations.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def size(self): pass

    @abc.abstractproperty
    def data(self): pass

    @abc.abstractproperty
    def err(self): pass

    @abc.abstractproperty
    def cov_mat(self): pass

    @abc.abstractproperty
    def cov_mat_inverse(self): pass

    @property
    def has_errors(self):
        return True if self._error_dicts else False