from __future__ import print_function

import abc
import numpy as np
import six

from copy import copy

from ...tools import random_alphanumeric
from ...core.error import SimpleGaussianError, MatrixGaussianError
from kafe2.fit.io.file import FileIOMixin

__all__ = ["DataContainerBase", "DataContainerException"]


class DataContainerException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class DataContainerBase(FileIOMixin, object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    types of data containers.

    It stores measurement data and uncertainties.
    """

    def __init__(self):
        self._error_dicts = dict()
        self._total_error = None
        super(DataContainerBase, self).__init__()

    # -- private methods
    
    @classmethod
    def _get_base_class(cls):
        return DataContainerBase

    @classmethod
    def _get_object_type_name(cls):
        return 'container'

    @abc.abstractmethod
    def _calculate_total_error(self):
        pass

    @abc.abstractmethod
    def _clear_total_error_cache(self):
        pass

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

    def _get_error_by_name_raise(self, error_name):
        """return a dictionary containing the error object for error 'name' and additional information"""
        _err_dict = self._error_dicts.get(error_name, None)
        if _err_dict is None:
            raise DataContainerException("No error with name '{}'!".format(error_name))
        return _err_dict

    @abc.abstractproperty
    def size(self):
        """The size of the data (number of measurement points)"""
        return 0

    @abc.abstractproperty
    def data(self):
        """A numpy array containing the data values"""
        return np.empty(tuple())

    @abc.abstractproperty
    def err(self):
        """A numpy array containing the pointwise data uncertainties"""
        return np.empty(tuple())

    @abc.abstractproperty
    def cov_mat(self):
        """A numpy matrix containing the covariance matrix of the data"""
        return np.array(np.empty(tuple()))

    @abc.abstractproperty
    def cov_mat_inverse(self):
        """A numpy matrix containing inverse of the data covariance matrix (or ``None`` if not invertible)"""
        return np.array(np.empty(tuple()))

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
        _err_dict = self._get_error_by_name_raise(error_name)
        _err_dict['enabled'] = False
        self._clear_total_error_cache()

    def enable_error(self, error_name):
        """
        (Re-)Enable an uncertainty source so that it counts towards calculating the
        total uncertainty.

        :param error_name: error name
        :type error_name: str
        """
        _err_dict = self._get_error_by_name_raise(error_name)
        _err_dict['enabled'] = True
        self._clear_total_error_cache()

    def get_matching_errors(self, matching_criteria=None, matching_type='equal'):
        """
        Return a list of uncertainty objects fulfilling the specified
        matching criteria.

        Valid keys for ``matching_criteria``:

            * ``name`` (the unique error name)
            * ``type`` (either ``simple`` or ``matrix``)
            * ``correlated`` (bool, only matches simple errors!)

        NOTE: The error objects contained in the dictionary are not copies,
        but the original error objects.
        Modifying them is possible, but not recommended. If you do modify any
        of them, the changes will not be reflected in the total error calculation
        until the error cache is cleared. This can be done by calling the
        private method
        :py:meth:`~kafe2.fit._base.container.DataContainerBase._clear_total_error_cache`.

        :param matching_criteria: key-value pairs specifying matching criteria.
                                  The resulting error array will only contain
                                  error objects matching *all* provided criteria.
                                  If ``None``, all error objects are returned.
        :type matching_criteria: dict or ``None``
        :param matching_type: how to perform the matching. If ``'equal'``, the
                              value in ``matching_criteria`` is checked for equality
                              against the stored value. If ``'regex', the
                              value in ``matching_criteria`` is interpreted as a regular
                              expression and is matched against the stored value.
        :type matching_type: ``'equal'`` or ``'regex'``
        :return: list of error objects
        :rtype: dict mapping error name to `~kafe2.core.error.GausianErrorBase`-derived
        """
        _result = copy(self._error_dicts)

        if matching_criteria is None:
            matching_criteria = dict()

        if matching_type == 'regex':
            raise NotImplementedError("Matching type 'regex' not yet implemented!")
        elif matching_type != 'equal':
            raise NotImplementedError("Unknown matching type: '{}'! "
                                      "Available: ['equals']".format(matching_type))

        for _crit_key, _crit_value in six.iteritems(matching_criteria):
            # go through all errors, removing those that don't match
            for _error_name, _error_dict in list(_result.items()):  # do not use an iterator!
                if _crit_key == 'name' and _error_name == _crit_value:
                    continue
                elif _crit_key == 'type':
                    # type 'simple'
                    _err_obj = _error_dict['err']
                    if ((_crit_value == 'simple' and isinstance(_err_obj, SimpleGaussianError)) or
                        (_crit_value == 'matrix' and isinstance(_err_obj, MatrixGaussianError))):
                        continue
                elif _crit_key == 'correlated':
                    _err_obj = _error_dict['err']
                    try:
                        if _crit_value == bool(_err_obj.corr_coeff != 0):
                            continue
                    except AttributeError:
                        pass  # error is a MatrixGaussianError and will not be matched
                else:
                    # check the error dict keys for a match
                    _cmp_val = _error_dict.get(_crit_key, None)
                    if _cmp_val == _crit_value and _cmp_val is not None:
                        continue

                # no if clause above executed -> no match, remove error element
                _result.pop(_error_name)

                # exit loop if no remaining errors to examine
                if not _result:
                    break

            # exit loop if no remaining errors to examine
            if not _result:
                break

        return {_name: _entry['err'] for _name, _entry in six.iteritems(_result)}

    def get_error(self, error_name):
        """
        Return the uncertainty object holding the uncertainty.

        NOTE: If you modify this object, the changes will not be reflected
        in the total error calculation until the error cache is cleared.
        This can be forced by calling
        :py:meth:`~kafe2.fit._base.container.DataContainerBase.enable_error`.

        :param error_name: error name
        :type error_name: str
        :return: error object
        :rtype: `~kafe2.core.error.GausianErrorBase`-derived
        """
        return self._get_error_by_name_raise(error_name)

    def get_total_error(self):
        """
        Get the error object representing the total uncertainty.

        :return: error object representing the total uncertainty
        :rtype: :py:class:`~kafe2.core.error.MatrixGaussianError`
        """
        if self._total_error is None:
            self._calculate_total_error()
        return self._total_error
