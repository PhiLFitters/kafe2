import abc

from ..io import InputFileHandle, OutputFileHandle

__all__ = ["DataContainerBase", "DataContainerException"]


class DataContainerException(Exception):
    pass


class DataContainerBase(object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    types of data containers.

    It stores measurement data and uncertainties.
    """
    __metaclass__ = abc.ABCMeta

    @classmethod
    def from_file(cls, filename, format=None):
        """Read container from file"""
        from ..representation import get_reader

        _basename_ext = filename.split('.')
        if len(_basename_ext) > 1:
            _basename, _ext = _basename_ext[:-1], _basename_ext[-1]
        else:
            _basename, _ext = _basename_ext[0], None

        if format is None and _ext is None:
            raise DataContainerException("Cannot detect file format from "
                                         "filename '{}' and no format specified!".format(filename))
        else:
            _format = format or _ext  # choose 'format' if specified, otherwise use filename extension

        _reader_class = get_reader('container', _format)
        _container = _reader_class(InputFileHandle(filename=filename)).read()

        # check if the container is the right type (do not check if calling from DataContainerBase)
        if not _container.__class__ == cls and not cls == DataContainerBase:
            raise DataContainerException("Cannot import '{}' from file '{}': file contains wrong container "
                                         "type '{}'!".format(cls.__name__, filename, _container.__class__.__name__))
        return _container

    def to_file(self, filename, format=None):
        """Write container to file"""
        from ..representation import get_writer

        _basename_ext = filename.split('.')
        if len(_basename_ext) > 1:
            _basename, _ext = _basename_ext[:-1], _basename_ext[-1]
        else:
            _basename, _ext = _basename_ext[0], None

        if format is None and _ext is None:
            raise DataContainerException("Cannot detect file format from "
                                         "filename '{}' and no format specified!".format(filename))
        else:
            _format = format or _ext  # choose 'format' if specified, otherwise use filename extension

        _writer_class = get_writer('container', _format)
        _writer_class(self, OutputFileHandle(filename=filename)).write()


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