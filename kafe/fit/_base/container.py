from __future__ import print_function

import abc
import numpy as np

from ...tools import random_alphanumeric
from ...core.error import SimpleGaussianError, MatrixGaussianError
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

    # -- private methods

    def _add_error_object(self, name, error_object, **additional_error_dict_keys):
        """create a new entry <name> under self._error_dicts,
        with keys err=<ErrorObject> and arbitrary additional keys"""
        _name = name
        if _name is not None and _name in self._error_dicts:
            raise DataContainerException("Cannot create error source with name '{}': "
                                         "there is already an error source registered under that name!".format(_name))
        # be paranoid about name collisions
        while _name is None or _name in self._error_dicts:
            _name = random_alphanumeric(size=8)

        additional_error_dict_keys.setdefault('enabled', True)  # enable error source, unless explicitly disabled
        _new_err_dict = dict(err=error_object, **additional_error_dict_keys)
        self._error_dicts[_name] = _new_err_dict
        self._clear_total_error_cache()
        return _name

    def _get_error_dict_raise(self, error_name):
        """return a dictionary containing the error object for error 'name' and additional information"""
        _err_dict = self._error_dicts.get(error_name, None)
        if _err_dict is None:
            raise DataContainerException("No error with name '{}'!".format(error_name))
        return _err_dict

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

    # -- public methods

    # error-related methods

    def add_simple_error(self, err_val,
                         name=None, correlation=0, relative=False, reference=None):
        """
        Add a simple uncertainty source to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param err_val: pointwise uncertainty/uncertainties for all data points
        :type err_val: float or iterable of float
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param correlation: correlation coefficient between any two distinct data points
        :type correlation: float
        :param relative: if ``True``, **err_val** will be interpreted as a *relative* uncertainty
        :type relative: bool
        :param reference: the data values to use when computing absolute errors from relative ones (and vice-versa)
        :type reference: iterable of float or ``None``
        :return: error name
        :rtype: str
        """
        try:
            err_val.ndim   # will raise if simple float
        except AttributeError:
            err_val = np.asarray(err_val, dtype=float)

        if err_val.ndim == 0:  # if dimensionless numpy array (i.e. float64), add a dimension
            err_val = np.ones(self.size) * err_val

        _err = SimpleGaussianError(err_val=err_val, corr_coeff=correlation,
                                   relative=relative, reference=reference)

        _name = self._add_error_object(name=name, error_object=_err)
        return _name

    def add_matrix_error(self, err_matrix, matrix_type,
                         name=None, err_val=None, relative=False, reference=None):
        """
        Add a matrix uncertainty source to the data container.
        Returns an error id which uniquely identifies the created error source.

        :param err_matrix: covariance or correlation matrix
        :param matrix_type: one of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``
        :type matrix_type: str
        :param name: unique name for this uncertainty source. If ``None``, the name
                     of the error source will be set to a random alphanumeric string.
        :type name: str or ``None``
        :param err_val: the pointwise uncertainties (mandatory if only a correlation matrix is given)
        :type err_val: iterable of float
        :param relative: if ``True``, the covariance matrix and/or **err_val** will be interpreted
                         as a *relative* uncertainty
        :type relative: bool
        :param reference: the data values to use when computing absolute errors from relative ones (and vice-versa)
        :type reference: iterable of float or ``None``
        :return: error name
        :rtype: str
        """
        _err = MatrixGaussianError(err_matrix=err_matrix, matrix_type=matrix_type, err_val=err_val,
                                   relative=relative, reference=reference)

        _name = self._add_error_object(name=name, error_object=_err)
        return _name

    def disable_error(self, error_name):
        """
        Temporarily disable an uncertainty source so that it doesn't count towards calculating the
        total uncertainty.

        :param error_name: error name
        :type error_name: str
        """
        _err_dict = self._get_error_dict_raise(error_name)
        _err_dict['enabled'] = False
        self._clear_total_error_cache()

    def enable_error(self, error_name):
        """
        (Re-)Enable an uncertainty source so that it counts towards calculating the
        total uncertainty.

        :param error_name: error name
        :type error_name: str
        """
        _err_dict = self._get_error_dict_raise(error_name)
        _err_dict['enabled'] = True
        self._clear_total_error_cache()

    def get_error(self, error_name):
        """
        Return the uncertainty object object holding the uncertainty.

        NOTE: If you modify this object, the changes will not be reflected
        in the total error calculation until the error cache is cleared.
        This can be forced by calling e.g.
        :py:meth:`~kafe.fit._base.container.DataContainerBase.enable_error`.

        :param error_name: error name
        :type error_name: str
        :return: error object
        :rtype: `~kafe.core.error.GausianErrorBase`-derived
        """
        return self._get_error_dict_raise(error_name)

    def get_total_error(self):
        """
        Get the error object representing the total uncertainty.

        :return: error object representing the total uncertainty
        :rtype: :py:class:`~kafe.core.error.MatrixGaussianError`
        """
        if self._total_error is None:
            self._calculate_total_error()
        return self._total_error

    # IO-related methods

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