import sys
import warnings
from collections import OrderedDict
from copy import copy

import numpy as np

from .cost import SharedCostFunction, MultiCostFunction
from .._base import FitBase
from ...core import NexusFitter
from ...core.error import SimpleGaussianError, MatrixGaussianError
from ...core.fitters.nexus import Alias, Function, Array
from ...tools import random_alphanumeric

__all__ = ['MultiFit']


class MultiFit(FitBase):
    """A MultiFit combines several regular fits into a combined fit object.
    Calling do_fit on the MultiFit will result in a numerical minimization of the sum of the cost functions of the
    individual fits.
    Parameters with the same name will be unified: their value will be the same across all individual fits.
    """

    _AXES = ()
    _MODEL_ERROR_NODE_NAMES = []

    def __init__(
            self, fit_list, minimizer=None, minimizer_kwargs=None,
            dynamic_error_algorithm="nonlinear"):
        """
        :param fit_list: List or Iterable of the individual fits from which to create the MultiFit.
        :type fit_list: collections.Iterable[FitBase]
        :param minimizer: The minimizer to use for fitting. Either ``'iminuit'``, ``'tminuit'``, ``'scipy'`` or
            ``None``.
        :type minimizer: str or None
        :param minimizer_kwargs: Dictionary with kwargs for the minimizer.
        :type minimizer_kwargs: dict

        :raises TypeError: If **fit_list** is not iterable.
        """
        # Cast Iterable to List, so that indexing works. Indexing is needed for some methods.
        self._fits = list(fit_list)  # will raise TypeError if fit_list is not iterable
        self._shared_error_dicts = dict()
        self._min_x_error = None
        super(MultiFit, self).__init__(
            data=None, model_function=None, cost_function=None, minimizer=minimizer,
            minimizer_kwargs=minimizer_kwargs, dynamic_error_algorithm=dynamic_error_algorithm)

    # -- private methods

    def _get_model_function_argument_formatters(self):
        _included_argument_names = set()
        _combined_argument_formatters = []
        for _fit in self._fits:
            for _argument_formatter in _fit._get_model_function_argument_formatters():
                if _argument_formatter.name not in _included_argument_names:
                    _combined_argument_formatters.append(_argument_formatter)
                    _included_argument_names.add(_argument_formatter.name)
        return _combined_argument_formatters

    def _get_model_function_parameter_formatters(self):
        _included_parameter_names = set()
        _combined_parameter_formatters = []
        for _fit in self._fits:
            for _parameter_formatter in _fit._get_model_function_parameter_formatters():
                if _parameter_formatter.name not in _included_parameter_names:
                    _combined_parameter_formatters.append(_parameter_formatter)
                    _included_parameter_names.add(_parameter_formatter.name)
        return _combined_parameter_formatters

    def _invalidate_total_error_cache(self):
        pass

    def _mark_errors_for_update(self):
        pass

    def _get_parameter_indices(self, singular_fit):
        return [self.parameter_names.index(_parameter_name) for _parameter_name in singular_fit.parameter_names]

    def _update_parameter_formatters(self, update_asymmetric_errors=False):
        for _fit in self._fits:
            _fit._update_parameter_formatters(update_asymmetric_errors=update_asymmetric_errors)

    def _update_singular_fits(self):
        _parameter_name_value_dict = self.parameter_name_value_dict
        for _fit in self._fits:
            _parameter_indices = self._get_parameter_indices(singular_fit=_fit)
            _asymmetric_parameter_errors = self._fitter.asymmetric_fit_parameter_errors_if_calculated
            if _asymmetric_parameter_errors is not None:
                _asymmetric_parameter_errors = _asymmetric_parameter_errors[_parameter_indices]
            _par_cor_mat = self.parameter_cor_mat
            if _par_cor_mat is not None:
                _par_cor_mat = _par_cor_mat[_parameter_indices][:, _parameter_indices]
            _par_cov_mat = self.parameter_cov_mat
            if _par_cov_mat is not None:
                _par_cov_mat = _par_cov_mat[_parameter_indices][:, _parameter_indices]
            _fit._loaded_result_dict = dict(
                did_fit=self.did_fit,
                parameter_errors=self.parameter_errors[_parameter_indices],
                parameter_cor_mat=_par_cor_mat,
                parameter_cov_mat=_par_cov_mat,
                asymmetric_parameter_errors=_asymmetric_parameter_errors
            )
        self._update_parameter_formatters()

    def _init_nexus(self):
        super(MultiFit, self)._init_nexus()

        self._combined_parameter_node_dict = OrderedDict()
        for _i, _fit_i in enumerate(self._fits):
            _original_cost_i = _fit_i._nexus.get('cost')
            _cost_alias_name_i = 'cost%s' % _i
            _cost_alias_i = Alias(ref=_original_cost_i, name=_cost_alias_name_i)
            self._nexus.add(_cost_alias_i, add_children=False)
            for _par_node in _fit_i.parameter_names:
                self._combined_parameter_node_dict[_par_node] = _fit_i._nexus.get(_par_node)
        for _par_node in self._combined_parameter_node_dict.values():
            self._nexus.add(_par_node)
            for _fit in self._fits:
                if _par_node.name in _fit.parameter_names:
                    _fit._nexus.add(node=_par_node, existing_behavior='replace')
        self._nexus.add(
            Array(nodes=self._combined_parameter_node_dict.values(), name='parameter_values'),
            existing_behavior="replace"
        )

        self._y_data_names = []
        for _i, _fit_i in enumerate(self._fits):
            _fit_i._initialize_fitter()

            _x_data_node = _fit_i._nexus.get('x_data')
            if _x_data_node is not None:
                _x_data_name = 'x_data%s' % _i
                self._nexus.add(
                    Alias(ref=_x_data_node, name=_x_data_name),
                    add_children=False)

            _y_data_node = _fit_i._nexus.get('y_data')
            if _y_data_node is not None:
                _y_data_name = 'y_data%s' % _i
                self._nexus.add(
                    Alias(ref=_y_data_node, name=_y_data_name),
                    add_children=False)
                self._y_data_names.append(_y_data_name)

        self._init_nexus_cost()

        self._initialize_fitter()

    def _init_nexus_cost(self):
        """
        initializes nexus nodes needed for cost calculation as well as the cost function object
        """
        if not self._shared_error_dicts:
            _cost_functions = [_fit._cost_function for _fit in self._fits]
            _cost_names = ['cost%s' % _i for _i in range(len(self._fits))]
            self._cost_function = MultiCostFunction(
                singular_cost_functions=_cost_functions, cost_function_names=_cost_names)
        else:
            _chi2_indices = []  # Indices of fits with chi2 cost function
            _data_indices = [0]  # Indices of edges of covariance block in combined matrix
            _cost_functions = []
            _cost_names = []
            _x_cov_mat_names = []
            _derivative_names = []
            _y_model_names = []
            _y_cov_mat_names = []
            _has_shared_x_error = False
            for _err_dict in self._shared_error_dicts.values():
                if _err_dict['axis'] == 'x':
                    _has_shared_x_error = True
                    break
            for _i, _fit_i in enumerate(self._fits):
                if _fit_i._cost_function.is_chi2:
                    _chi2_indices.append(_i)
                    _data_indices.append(_data_indices[-1] + _fit_i.data_size)

                    if _has_shared_x_error:
                        _x_cov_mat_name = 'x_cov_mat%s' % _i
                        self._nexus.add(
                            Alias(ref=_fit_i._nexus.get('x_total_cov_mat'), name=_x_cov_mat_name),
                            add_children=False)
                        _x_cov_mat_names.append(_x_cov_mat_name)

                        _derivatives_name = 'derivatives%s' % _i

                        # Bind fit object (Python is call-by-value), otherwise all derivatives would
                        # use the same fit.
                        def _get_derivatives_func(fit):
                            return lambda: fit._param_model.eval_model_function_derivative_by_x(
                                model_parameters=fit.parameter_values,
                                dx=0.01 * self._min_x_error
                            )

                        self._nexus.add(
                            Function(func=_get_derivatives_func(_fit_i), name=_derivatives_name),
                            add_children=False)
                        self._nexus.add_dependency(
                            name=_derivatives_name, depends_on='parameter_values')
                        _derivative_names.append(_derivatives_name)

                    _y_model_name = 'y_model%s' % _i
                    self._nexus.add(
                        Alias(ref=_fit_i._nexus.get('y_model'), name=_y_model_name),
                        add_children=False)
                    _y_model_names.append(_y_model_name)

                    _y_cov_mat_name = 'y_cov_mat%s' % _i
                    self._nexus.add(
                        Alias(ref=_fit_i._nexus.get('y_total_cov_mat'), name=_y_cov_mat_name),
                        add_children=False)
                    _y_cov_mat_names.append(_y_cov_mat_name)
                else:
                    _cost_functions.append(_fit_i._cost_function)
                    _cost_names.append('cost%s' % _i)

            # Combines 1-dimensional properties by concatenating them.
            def _combine_1d_property(*single_fit_properties):
                _combined_property = np.zeros(shape=_data_indices[-1])
                for _j, (_single_fit_property, _chi2_index) in enumerate(
                        zip(single_fit_properties, _chi2_indices)):
                    _lower = _data_indices[_j]
                    _upper = _data_indices[_j + 1]
                    _combined_property[_lower:_upper] = _single_fit_property
                return _combined_property

            # Combines cov mats of single fits.
            # Shared errors are not added to the main diagonal blocks because they have already been
            # added to the individual fits.
            def _combine_cov_mats(axis_name, *single_fit_properties):
                _combined_property = np.zeros(shape=(_data_indices[-1], _data_indices[-1]))
                for _j, (_single_fit_property, _chi2_index) in enumerate(
                        zip(single_fit_properties, _chi2_indices)):
                    _lower = _data_indices[_j]
                    _upper = _data_indices[_j + 1]
                    _combined_property[_lower:_upper, _lower:_upper] = _single_fit_property
                for _error_dict in self._shared_error_dicts.values():
                    if _error_dict['axis'] == axis_name:
                        _error = _error_dict['err']
                        for _j in range(len(_error.fit_indices)):
                            _lower_1 = _data_indices[_j]
                            _upper_1 = _data_indices[_j + 1]
                            for _k in range(_j):
                                _lower_2 = _data_indices[_k]
                                _upper_2 = _data_indices[_k + 1]
                                _combined_property[_lower_1:_upper_1, _lower_2:_upper_2] += _error.cov_mat
                                _combined_property[_lower_2:_upper_2, _lower_1:_upper_1] += _error.cov_mat
                return _combined_property

            if _has_shared_x_error:
                self._nexus.add_function(
                    func=lambda *p: _combine_cov_mats('x', *p),
                    func_name='x_cov_mat', par_names=_x_cov_mat_names, add_children=False)
                self._nexus.add_function(
                    func=_combine_1d_property, func_name='derivatives', par_names=_derivative_names,
                    add_children=False)

            self._nexus.add_function(
                func=_combine_1d_property, func_name='y_data',
                par_names=self._y_data_names, add_children=False)
            self._nexus.add_alias(name='data', alias_for='y_data')
            self._nexus.add_function(
                func=_combine_1d_property, func_name='y_model',
                par_names=_y_model_names, add_children=False)
            self._nexus.add_alias(name='model', alias_for='y_model')
            self._nexus.add_function(
                func=lambda *p: _combine_cov_mats('y', *p),
                func_name='y_cov_mat', par_names=_y_cov_mat_names, add_children=False)

            if _has_shared_x_error:
                def total_cov_mat_inverse(x_cov_mat, derivatives, y_cov_mat):
                    _cov_mat = y_cov_mat + x_cov_mat * np.outer(derivatives, derivatives)
                    return np.linalg.inv(_cov_mat)
            else:
                def total_cov_mat_inverse(y_cov_mat):
                    return np.linalg.inv(y_cov_mat)
            self._nexus.add_function(total_cov_mat_inverse)
            self._cost_function = SharedCostFunction()

        _cost_function_node = self._nexus.add_function(
            func=self._cost_function,
            func_name=self._cost_function.name,
            par_names=self._cost_function.arg_names,
            existing_behavior="replace"
        )
        self._nexus.add_alias(
            name='cost', alias_for=_cost_function_node.name, existing_behavior='replace')

    def _initialize_fitter(self):
        self._fitter = NexusFitter(
            nexus=self._nexus, parameters_to_fit=list(self._combined_parameter_node_dict.keys()),
            parameter_to_minimize=self._cost_function.name, minimizer=self._minimizer,
            minimizer_kwargs=self._minimizer_kwargs
        )

    def _add_error_object(self, error_object, reference, name=None, axis=None):
        from ..indexed import IndexedFit
        from ..xy import XYFit

        if axis not in [None, 'x', 'y']:
            raise ValueError("axis must be one of: None, 'x', 'y'")
        _data_size_0 = self._fits[error_object.fit_indices[0]].data_size
        for _fit_index in error_object.fit_indices:
            if not self._fits[_fit_index]._cost_function.is_chi2:
                raise ValueError(
                    "Cannot add shared error because cost function of fit %s is not chi2!"
                    % _fit_index)
            if isinstance(self._fits[_fit_index], XYFit) and axis is None:
                raise ValueError("axis=None is ambiguous for fit %s because it is an XYFit!" % axis)
            if isinstance(self._fits[_fit_index], IndexedFit) and axis == 'x':
                raise ValueError(
                    "axis='x' is incompatible with fit %s because it is an IndexedFit!")
            if self._fits[_fit_index].data_size != _data_size_0:
                raise ValueError("Fit %s data_size not the same as data_size of Fit 0!")

        if error_object.relative:
            if not reference == 'data' or reference == 'model':
                raise ValueError(
                    "Error reference must be either 'model' or 'data' but received %s" % reference)
            if axis == 'x' and reference == 'data':
                _node_name = 'x_data%s'
            elif axis == 'y' and reference == 'data':
                _node_name = 'y_data%s'
            elif axis == 'x' and reference == 'model':
                _node_name = 'x_model%s'
                raise ValueError("Shared errors relative to model are not supported.")
            elif axis == 'y' and reference == 'model':
                _node_name = 'y_model%s'
                raise ValueError("Shared errors relative to model are not supported.")
            else:
                raise AssertionError()
            _reference_value = None
            _first_index = None
            for _fit_index in error_object.fit_indices:
                if _reference_value is None:
                    _reference_value = self._nexus.get(_node_name % _fit_index).value
                    _first_index = _fit_index
                else:
                    _other_reference_value = self._nexus.get(_node_name % _fit_index).value
                    if _reference_value.shape != _other_reference_value.shape or np.any(
                            (_reference_value - _other_reference_value) != 0):
                        raise ValueError(
                            "Cannot add a relative error for fits %s and %s because they have "
                            "conflicting references." % (_first_index, _fit_index))
            error_object.reference = _reference_value

        if axis is None:
            axis = 'y'
        elif axis == 'x':
            _min_of_new_error = np.min(error_object.error)
            if self._min_x_error is None:
                self._min_x_error = _min_of_new_error
            else:
                self._min_x_error = min(
                    self._min_x_error, _min_of_new_error
                )

        if name in self._shared_error_dicts:
            raise ValueError("Error with name=%s already exists!" % name)
        if name is None:
            _name_is_unique = False
            while not _name_is_unique:
                name = random_alphanumeric(8)
                if name in self._shared_error_dicts:
                    continue
                for _fit_index in error_object.fit_indices:
                    _fit = self._fits[_fit_index]
                    if name in _fit.data_container._error_dicts:
                        continue
                    if name in _fit._param_model._error_dicts:
                        continue
                _name_is_unique = True

        _error_dict = dict(err=error_object, enabled=True, axis=axis, reference_name=reference)
        self._shared_error_dicts[name] = _error_dict
        for _fit_index in error_object.fit_indices:
            _fit = self._fits[_fit_index]
            if reference == "data":
                _target = _fit.data_container
            elif reference == "model":
                _target = _fit._param_model
            else:
                raise ValueError()
            _target._add_error_object(name=name, error_object=error_object, axis=axis)
        self._on_error_change()
        return name

    def _on_error_change(self):
        self._init_nexus_cost()
        # Make the nexus fetch the new cost function node:
        self._fitter.parameter_to_minimize = self._cost_function.name

    def _set_new_data(self, new_data):
        raise NotImplementedError()

    def _set_new_parametric_model(self):
        raise NotImplementedError()

    def _set_data_as_model_ref(self):
        for _fit in self._fits:
            _fit._set_data_as_model_ref()

    def _iterative_fits_needed(self):
        for _fit in self._fits:
            if _fit._iterative_fits_needed():
                return True
        return False

    def _second_fit_needed(self):
        for _fit in self._fits:
            if _fit._second_fit_needed():
                return True
        return False

    def _pre_fit_iteration(self, first_fit=False):
        for _fit in self._fits:
            _fit._pre_fit_iteration(first_fit)

    def _post_fit_iteration(self, first_fit=False):
        for _fit in self._fits:
            _fit._post_fit_iteration(first_fit)

    # -- public properties

    @property
    def data(self):
        """List of the data of the individual fits."""
        return [_fit.data for _fit in self._fits]

    @data.setter
    def data(self, new_data):
        raise NotImplementedError("Use data setters for individual fits instead.")

    @property
    def data_container(self):
        """List of the data containers of the individual fits."""
        return [_fit.data_container for _fit in self._fits]

    @property
    def data_size(self):
        """Combined size of the data containers of the individual fits.

        :rtype: int
        """
        return np.sum([_fit.data_size for _fit in self._fits])

    @property
    def has_errors(self):
        """``True`` if at least one uncertainty source is defined for any of the individual fits.

        :rtype: bool
        """
        return np.any([_fit.has_errors for _fit in self._fits])

    @property
    def has_data_errors(self):
        """``True`` if at least one uncertainty source is defined for the data of any of the individual fits.

        :rtype: bool
        """
        return np.any([_fit.has_data_errors for _fit in self._fits])

    @property
    def has_model_errors(self):
        """``True`` if at least one uncertainty source is defined for the model of any of the individual fits.

        :rtype: bool
        """
        return np.any([_fit.has_model_errors for _fit in self._fits])

    @property
    def model(self):
        """List of the model values of the individual fits."""
        return [_fit.model for _fit in self._fits]

    @property
    def model_count(self):
        """The number of model functions contained in the multifit. In most cases one for each singular fit.

        :rtype: int
        """
        return np.sum([_fit.model_count for _fit in self._fits])

    @property
    def model_label(self):
        """List of model labels for each singular fit."""
        return [_fit.model_label for _fit in self._fits]

    @model_label.setter
    def model_label(self, label):
        raise NotImplementedError("Use model_label setters for individual fits instead.")

    @property
    def fits(self):
        """List of individual fits on which the MultiFit is based on."""
        return copy(self._fits)  # shallow copy

    @property
    def asymmetric_parameter_errors(self):
        """The current asymmetric parameter uncertainties."""
        _asymm_par_errs = super(MultiFit, self).asymmetric_parameter_errors
        self._update_singular_fits()
        return _asymm_par_errs

    @property
    def ndf(self):
        return self.data_size - len(self._combined_parameter_node_dict.keys()) \
               + len(self._fitter.fixed_parameters)

    @property
    def goodness_of_fit(self):
        if isinstance(self._cost_function, MultiCostFunction):
            _sum = 0.0
            for _fit in self._fits:
                _gof = _fit.goodness_of_fit
                if _gof is None:
                    return None
                _sum += _gof
            return _sum
        else:
            return super(MultiFit, self).goodness_of_fit

    # -- public methods

    def add_matrix_error(self, err_matrix, matrix_type, fits, axis=None, name=None, err_val=None,
                         relative=False, reference='data', **kwargs):
        """Add a matrix uncertainty source for use in the fit.

        :param err_matrix: Covariance or correlation matrix.
        :param matrix_type: One of ``'covariance'``/``'cov'`` or ``'correlation'``/``'cor'``.
        :type matrix_type: str
        :param fits: The indices of the fits to add the error. If ``'all'``, the error is added to all fits.
        :type fits: int or collections.Iterable[int] or str
        :param axis: Axis of the individual fits to add the error to. ``y``/``1`` errors are treated as regular errors
                     for :py:class:`~kafe2.fit.IndexedFit`.
        :type axis: int or str
        :param name: Unique name for this uncertainty source. If ``None``, the name of the error source will be set to a
                     random alphanumeric string.
        :type name: str or None
        :param err_val: The pointwise uncertainties (mandatory if only a correlation matrix is given).
        :type err_val: collections.Iterable[float]
        :param relative: If ``True``, the covariance matrix and/or **err_val** will be interpreted as a *relative*
                         uncertainty.
        :type relative: bool
        :param reference: Either ``'data'`` or ``'model'``. Specifies which reference values to use when calculating
                          absolute errors from relative errors.
        :type reference: str
        :return: An error id which uniquely identifies the created error source.
        :rtype: str
        """
        # TODO relative errors
        if isinstance(fits, int):
            self._fits[fits].add_matrix_error(err_matrix=err_matrix, matrix_type=matrix_type, axis=axis, name=name,
                                              err_val=err_val, relative=relative, reference=reference, **kwargs)
        else:
            if fits == 'all':
                fits = list(range(len(self._fits)))
            _matrix_error = MatrixGaussianError(
                err_matrix=err_matrix, matrix_type=matrix_type, err_val=err_val, relative=relative,
                fit_indices=fits
            )
            return self._add_error_object(error_object=_matrix_error, reference=reference,
                                          name=name, axis=axis)

    def add_error(self, err_val, fits, axis=None, name=None, correlation=0, relative=False, reference='data', **kwargs):
        """Add an uncertainty source to the fit.

        :param err_val: Pointwise uncertainty/uncertainties for all data points.
        :type err_val: float or collections.Iterable[float]
        :param fits: The indices of the fits to add the error. If ``'all'``, the error is added to all fits.
        :type fits: int or collections.Iterable[int] or str
        :param axis: Axis of the individual fits to add the error to. ``y``/``1`` errors are treated as regular errors
                     for :py:class:`~kafe2.fit.IndexedFit`.
        :type axis: int or str
        :param name: Unique name for this uncertainty source. If ``None``, the name of the error source will be set to a
                     random alphanumeric string.
        :type name: str or None
        :param correlation: Correlation coefficient between any two distinct data points.
        :type correlation: float
        :param relative: If ``True``, **err_val** will be interpreted as a *relative* uncertainty.
        :type relative: bool
        :param reference: Either ``'data'`` or ``'model'``. Specifies which reference values to use when calculating
                          absolute errors from relative errors.
        :type reference: str
        :return: An error id which uniquely identifies the created error source.
        :rtype: str
        """
        if isinstance(fits, int):
            self._fits[fits].add_error(
                err_val=err_val, name=name, correlation=correlation, relative=relative,
                axis=axis, reference=reference, **kwargs
            )
        else:
            if fits == 'all':
                fits = list(range(len(self._fits)))
            try:
                err_val.ndim  # will raise if simple float
            except AttributeError:
                err_val = np.asarray(err_val, dtype=float)

            if err_val.ndim == 0:  # if dimensionless numpy array (i.e. float64), add a dimension
                err_val = np.ones(self._fits[fits[0]].data_size) * err_val
                # Consistent data size is checked later.
            _simple_error = SimpleGaussianError(
                err_val=err_val, corr_coeff=correlation, relative=relative, fit_indices=fits
            )
            return self._add_error_object(error_object=_simple_error, reference=reference,
                                          name=name, axis=axis)

    def assign_model_function_expression(self, expression_format_string, fit_index=None):
        """Assign a plain-text-formatted expression string to the model function of one of the individual fits.

        :param expression_format_string: The model function expression.
        :type expression_format_string: str
        :param fit_index: The index specifying the singular fit. If ``None``, all model functions will be changed.
        :type fit_index: int or None
        """
        if fit_index is None:
            for _fit in self._fits:
                _fit.assign_model_function_expression(
                    expression_format_string=expression_format_string)
        else:
            self._fits[fit_index].assign_model_function_expression(
                expression_format_string=expression_format_string)

    def assign_model_function_latex_name(self, latex_name, fit_index=None):
        """Assign a LaTeX-formatted string to be the model function name of one of the individual fits.

        :param latex_name: The LaTeX model function name.
        :type latex_name: str
        :param fit_index: The index specifying the singular fit. If ``None``, all model functions will be changed.
        :type fit_index: int or None
        """
        if fit_index is None:
            for _fit in self._fits:
                _fit.assign_model_function_latex_name(latex_name=latex_name)
        else:
            self._fits[fit_index].assign_model_function_latex_name(latex_name=latex_name)

    def assign_model_function_latex_expression(self, latex_expression_format_string, fit_index=None):
        """Assign a LaTeX-formatted expression string to the model function of one of the individual fits.
        Elements like ``'{par_name}'`` will be replaced automatically with the corresponding LaTeX names for the
        given parameter. These can be set with :py:meth:`~assign_parameter_latex_names`.

        :param latex_expression_format_string: The LaTeX model function expression.
        :type latex_expression_format_string: str
        :param fit_index: The index specifying the singular fit. If ``None``, all model functions will be changed.
        :type fit_index: int or None
        """
        if fit_index is None:
            for _fit in self._fits:
                _fit.assign_model_function_latex_expression(
                    latex_expression_format_string=latex_expression_format_string)
        else:
            self._fits[fit_index].assign_model_function_latex_expression(
                latex_expression_format_string=latex_expression_format_string)

    def assign_parameter_latex_names(self, **par_latex_names_dict):
        _keys = list(par_latex_names_dict.keys())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Could not assign all latex names.*")
            for _fit in self._fits:
                _fit.assign_parameter_latex_names(**par_latex_names_dict)
                for _arg_formatter in _fit._get_model_function_argument_formatters():
                    try:
                        _keys.remove(_arg_formatter.name)
                    except ValueError:
                        pass
        if _keys:
            warnings.warn(
                "Could not assign all parameter latex names to single fits. Leftover: {}".format(
                    _keys))

    def disable_error(self, err_id):
        for _fit in self._fits:
            _fit.disable_error(err_id=err_id)

    def fix_parameter(self, name, value=None):
        self._fitter.fix_parameter(name=name, value=value)
        # get fixed value before setting it in the individual fits
        _val = self._fitter.fixed_parameters[name]
        for fit in self._fits:
            if name not in fit.parameter_names:
                continue  # skip if sub fit is not dependent on the given par
            fit.fix_parameter(name, _val)  # default values might not have been overwritten, use _val

    def release_parameter(self, name):
        self._fitter.release_parameter(name)
        for fit in self._fits:  # update formatters of individual fits
            if name not in fit.parameter_names:
                continue  # skip if sub fit is not dependent on the given par
            fit.release_parameter(name)

    def do_fit(self, asymmetric_parameter_errors=False):
        _fit_result = super(MultiFit, self).do_fit(
            asymmetric_parameter_errors=asymmetric_parameter_errors)
        self._update_singular_fits()
        return _fit_result

    def get_matching_errors(self, fit_index=None, matching_criteria=None, matching_type='equal'):
        """Return a list of uncertainty objects fulfilling the specified matching criteria.

        Valid keys for **matching_criteria**:
            * ``name`` (the unique error name)
            * ``type`` (either ``'simple'`` or ``'matrix'``)
            * ``correlated`` (bool, only matches simple errors!)
            * ``reference`` (either ``'model'`` or ``'data'``)

        .. note::
            The error objects contained in the dictionary are not copies, but the original error objects.
            Modifying them is possible, but not recommended.
            If you do modify any of them, the changes will not be reflected in the total error calculation until the
            error cache is cleared. This can be done by calling the private dataset method
            :py:meth:`~kafe2.fit._base.DataContainerBase._clear_total_error_cache`.

        :param fit_index: Index for which fit the method should be executed. If ``None`` the function will return the
                          matched errors for all singular fits.
        :type fit_index: int or None
        :param matching_criteria: Key-value pairs specifying matching criteria. The resulting error array will only
                                  contain error objects matching *all* provided criteria.
                                  If ``None``, all error objects are returned.
        :type matching_criteria: dict or None
        :param matching_type: How to perform the matching.
                              If ``'equal'``, the value in ``matching_criteria`` is checked for equality against the
                              stored value.
                              If ``'regex'``, the value in ``matching_criteria`` is interpreted as a regular expression
                              and is matched against the stored value.
        :type matching_type: str
        :return: Dict mapping error name to :py:obj:`~kafe2.core.error.GaussianErrorBase`-derived error objects.
        :rtype: dict[str, kafe2.core.error.GaussianErrorBase]
        """
        if fit_index is None:
            _combined_matching_errors = dict()
            for _fit in self._fits:
                _matching_errors = _fit.get_matching_errors(
                    matching_criteria=matching_criteria, matching_type=matching_type)
                for _key in _matching_errors:
                    # FATAL: there is an error with the same name in separate fits
                    assert _key not in _combined_matching_errors
                    _combined_matching_errors[_key] = _matching_errors[_key]
            return _combined_matching_errors
        return self._fits[fit_index].get_matching_errors(matching_criteria=matching_criteria,
                                                         matching_type=matching_type)

    def report(self, output_stream=sys.stdout, show_data=True, show_model=True, show_fit_results=True,
               asymmetric_parameter_errors=False):
        """Print a summary for each fit state and/or the multifit results.

        :param output_stream: The output stream to which the report should be printed.
        :type output_stream: io.TextIOBase
        :param show_data: If ``True``, print out information about the data for each fit.
        :type show_data: bool
        :param show_model: If ``True``, print out information about the parametric model for each fit.
        :type show_model: bool
        :param show_fit_results: If ``True``, print out information about the combined fit results.
        :type show_fit_results: bool
        :param asymmetric_parameter_errors: If ``True``, use two different parameter errors for up/down directions.
        :type asymmetric_parameter_errors: bool
        """
        _indent = ' ' * 4

        for _i, _fit in enumerate(self._fits):
            _header_string = '#########\n'
            if _i > 9:
                _header_string = '#' + _header_string
            if _i > 99:
                _header_string = '#' + _header_string
            output_stream.write(_header_string)
            output_stream.write('# Fit %s #\n' % _i)
            output_stream.write(_header_string)
            output_stream.write('\n')

            if show_data:
                _fit._report_data(output_stream=output_stream, indent=_indent, indentation_level=1)
            if show_model:
                _fit._report_model(output_stream=output_stream, indent=_indent, indentation_level=1)
        self._report_fit_results(output_stream=output_stream, indent=_indent, indentation_level=0,
                                 asymmetric_parameter_errors=asymmetric_parameter_errors)
