from __future__ import print_function

import abc
from copy import copy

import numpy as np
import six

from ..io.file import FileIOMixin
from ...core.error import SimpleGaussianError, MatrixGaussianError
from ...tools import random_alphanumeric  # relative import of kafe2.tools not kafe2.fit.tools

__all__ = ["DataContainerBase", "DataContainerException"]


class DataContainerException(Exception):
    pass


@six.add_metaclass(abc.ABCMeta)
class DataContainerBase(FileIOMixin, object):
    """This is a purely abstract class implementing the minimal interface required by all types of data containers.

    It stores measurement data and uncertainties.
    """

    def __init__(self):
        self._error_dicts = dict()
        self._total_error = None
        self._label = None
        self._axis_labels = (None, None)
        self._on_error_change_callback = None
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

        axis = additional_error_dict_keys.get("axis", None)
        if axis is not None:
            if axis == "x":
                additional_error_dict_keys["axis"] = 0
            elif axis == "y":
                additional_error_dict_keys["axis"] = 1

        _new_err_dict = dict(err=error_object, **additional_error_dict_keys)
        self._error_dicts[_name] = _new_err_dict
        self._on_error_change()
        return _name

    def _get_error_by_name_raise(self, error_name):
        """return a dictionary containing the error object for error 'name' and additional information"""
        _err_dict = self._error_dicts.get(error_name, None)
        if _err_dict is None:
            raise DataContainerException("No error with name '{}'!".format(error_name))
        return _err_dict

    def _on_error_change(self):
        self._clear_total_error_cache()
        if self._on_error_change_callback is not None:
            self._on_error_change_callback()

    @property
    def label(self):
        """The label describing the dataset.

        :rtype: str or None
        """
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def axis_labels(self):
        """The axis labels describing the dataset.

        :rtype: tuple[str or None, str or None]
        """
        return self._axis_labels

    @axis_labels.setter
    def axis_labels(self, labels):
        self._axis_labels = labels

    @property
    def x_label(self):
        """The x-axis label.

        :rtype: str or None
        """
        return self._axis_labels[0]

    @x_label.setter
    def x_label(self, label):
        _, _y_label = self._axis_labels
        self._axis_labels = (label, _y_label)

    @property
    def y_label(self):
        """The y-axis label.

        :rtype: str or None
        """
        return self._axis_labels[1]

    @y_label.setter
    def y_label(self, label):
        _x_label, _ = self._axis_labels
        self._axis_labels = (_x_label, label)

    @property
    @abc.abstractmethod
    def size(self):
        """The size of the data (number of measurement points).

        :rtype: int
        """
        return 0

    @property
    @abc.abstractmethod
    def data(self):
        """A numpy array containing the data values.

        :rtype: numpy.ndarray[float]
        """
        return np.empty(tuple())

    @property
    @abc.abstractmethod
    def err(self):
        """A numpy array containing the pointwise data uncertainties.

        :rtype: numpy.ndarray[float]
        """
        return np.empty(tuple())

    @property
    @abc.abstractmethod
    def cov_mat(self):
        """A numpy matrix containing the covariance matrix of the data.

        :rtype: numpy.ndarray[numpy.ndarray[float]]"""
        return np.array(np.empty(tuple()))

    @property
    @abc.abstractmethod
    def cov_mat_inverse(self):
        """A numpy matrix containing inverse of the data covariance matrix (or :py:obj`None` if not invertible).

        :rtype: numpy.ndarray[numpy.ndarray[float]] or None
        """
        return np.array(np.empty(tuple()))

    @property
    def has_errors(self):
        """:py:obj:`True` if at least one uncertainty source is defined for the data container.

        :rtype: bool
        """
        return True if self._error_dicts else False

    # -- public methods

    # error-related methods

    def add_error(self, err_val, name=None, correlation=0, relative=False, reference=None):
        """Add an uncertainty source to the data container.

        :param err_val: Pointwise uncertainty/uncertainties for all data points.
        :type err_val: float or numpy.ndarray[float]
        :param name: Unique name for this uncertainty source. If :py:obj:`None`, the name of the error source will be
                     set to a random alphanumeric string.
        :type name: str or None
        :param correlation: Correlation coefficient between any two distinct data points.
        :type correlation: float
        :param relative: If :py:obj:`True`, **err_val** will be interpreted as a *relative* uncertainty.
        :type relative: bool
        :param reference: The data values to use when computing absolute errors from relative ones (and vice-versa)
        :type reference: typing.Iterable[float] or None
        :return: An error id which uniquely identifies the created error source.
        :rtype: str
        """
        try:
            err_val.ndim  # will raise if simple float
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
        """Add a matrix uncertainty source to the data container.

        :param err_matrix: Covariance or correlation matrix.
        :param matrix_type: One of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``.
        :type matrix_type: str
        :param name: Unique name for this uncertainty source. If :py:obj`None`, the name of the error source will be set to a
                     random alphanumeric string.
        :type name: str or None
        :param err_val: The pointwise uncertainties (mandatory if only a correlation matrix is given).
        :type err_val: typing.Iterable[float]
        :param relative: If :py:obj:`True`, the covariance matrix and/or **err_val** will be interpreted as a *relative*
                         uncertainty.
        :type relative: bool
        :param reference: the data values to use when computing absolute errors from relative ones (and vice-versa)
        :type reference: typing.Iterable[float] or None
        :return: An error id which uniquely identifies the created error source.
        :rtype: str
        """
        _err = MatrixGaussianError(err_matrix=err_matrix, matrix_type=matrix_type, err_val=err_val,
                                   relative=relative, reference=reference)

        _name = self._add_error_object(name=name, error_object=_err)
        return _name

    def disable_error(self, error_name):
        """Temporarily disable an uncertainty source so that it doesn't count towards calculating the total uncertainty.

        :param error_name: error name
        :type error_name: str
        """
        _err_dict = self._get_error_by_name_raise(error_name)
        _err_dict['enabled'] = False
        self._on_error_change()

    def enable_error(self, error_name):
        """(Re-)Enable an uncertainty source so that it counts towards calculating the total uncertainty.

        :param error_name: error name
        :type error_name: str
        """
        _err_dict = self._get_error_by_name_raise(error_name)
        _err_dict['enabled'] = True
        self._on_error_change()

    def get_matching_errors(self, matching_criteria=None, matching_type='equal'):
        """Return a list of uncertainty objects fulfilling the specified matching criteria.

        Valid keys for **matching_criteria**:
            * ``name`` (the unique error name)
            * ``type`` (either ``simple`` or ``matrix``)
            * ``correlated`` (bool, only matches simple errors!)
            * ``relative`` (bool)

        .. note::
            The error objects contained in the dictionary are not copies, but the original error objects.
            Modifying them is possible, but not recommended.
            If you do modify any of them, the changes will not be reflected in the total error calculation until the
            error cache is cleared.
            This can be done by calling the private method :py:meth:`~_clear_total_error_cache`.

        :param matching_criteria: Key-value pairs specifying matching criteria. The resulting error array will only
                                  contain error objects matching *all* provided criteria.
                                  If :py:obj:`None`, all error objects are returned.
        :type matching_criteria: dict or None
        :param matching_type: How to perform the matching.
                              If ``'equal'``, the value in **matching_criteria** is checked for equality against the
                              stored value.
                              If ``'regex'``, the value in **matching_criteria** is interpreted as a regular expression
                              and is matched against the stored value.
        :type matching_type: str
        :return: Dict mapping error name to :py:obj:`~kafe2.core.error.GaussianErrorBase`-derived error objects.
        :rtype: dict[str, kafe2.core.error.GaussianErrorBase]
        """
        _result = copy(self._error_dicts)

        if matching_criteria is None:
            matching_criteria = dict()

        if matching_type == 'regex':
            raise NotImplementedError("Matching type 'regex' not yet implemented!")
        if matching_type != 'equal':
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
                elif _crit_key == 'relative':
                    _err_obj = _error_dict['err']
                    if _err_obj.relative == _crit_value:
                        continue
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
        """Return the uncertainty object holding the uncertainty.

        .. note::
            If you modify this object, the changes will not be reflected in the total error calculation until the error
            cache is cleared. This can be forced by calling :py:meth:`~enable_error`.

        :param error_name: error name
        :type error_name: str
        :return: error object
        :rtype: kafe2.core.error.GaussianErrorBase
        """
        return self._get_error_by_name_raise(error_name)

    def get_total_error(self):
        """Get the error object representing the total uncertainty.

        :return: error object representing the total uncertainty
        :rtype: kafe2.core.error.MatrixGaussianError
        """
        if self._total_error is None:
            self._calculate_total_error()
        return self._total_error
