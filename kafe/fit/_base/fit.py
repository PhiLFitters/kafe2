import abc
import inspect
import numpy as np
import re
import string

from ...core import get_minimizer, NexusFitter


__all__ = ["FitBase", "FitException"]


class FitException(Exception):
    pass


class FitBase(object):
    """
    This is a purely abstract class implementing the minimal interface required by all
    types of fitters.
    """
    __metaclass__ = abc.ABCMeta

    CONTAINER_TYPE = None
    MODEL_TYPE = None
    EXCEPTION_TYPE = FitException
    RESERVED_NODE_NAMES = None

    # -- private methods

    def _new_data_container(self, *args, **kwargs):
        """create a DataContainer of the right type for this fit"""
        return self.__class__.CONTAINER_TYPE(*args, **kwargs)

    def _new_parametric_model(self, *args, **kwargs):
        """create a ParametricModel of the right type for this fit"""
        return self.__class__.MODEL_TYPE(*args, **kwargs)

    def _validate_model_function_for_fit_raise(self):
        """make sure the supplied model function is compatible with the fit type"""
        # disallow using reserved keywords as model function arguments
        if not self.RESERVED_NODE_NAMES.isdisjoint(set(self._model_function.argspec.args)):
            _invalid_args = self.RESERVED_NODE_NAMES.intersection(set(self._model_function.argspec.args))
            raise self.__class__.EXCEPTION_TYPE(
                "The following names are reserved and cannot be used as model function arguments: %r"
                % (_invalid_args,))

    def _initialize_fitter(self, minimizer=None, minimizer_kwargs=None):
        self._fitter = NexusFitter(nexus=self._nexus,
                                   parameters_to_fit=self._fit_param_names,
                                   parameter_to_minimize=self._cost_function.name,
                                   minimizer=minimizer,
                                   minimizer_kwargs=minimizer_kwargs)


    @staticmethod
    def _latexify_ascii(ascii_string):
        """function computing a fallback LaTeX representation of an plain-text string"""
        _lpn = ascii_string.replace('_', r"\_")
        return r"{\tt %s}" % (_lpn,)

    # -- public properties

    # @abc.abstractproperty
    # def data(self): pass
    #
    # @abc.abstractproperty
    # def model(self): pass
    #
    # @abc.abstractproperty
    # def data_error(self): pass
    #
    # @abc.abstractproperty
    # def data_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def data_cov_mat_inverse(self): pass
    #
    # @abc.abstractproperty
    # def model_error(self): pass
    #
    # @abc.abstractproperty
    # def model_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def model_cov_mat_inverse(self): pass
    #
    # @abc.abstractproperty
    # def total_error(self): pass
    #
    # @abc.abstractproperty
    # def total_cov_mat(self): pass
    #
    # @abc.abstractproperty
    # def total_cov_mat_inverse(self): pass

    @property
    def parameter_values(self):
        """the current parameter values"""
        return list(self.parameter_name_value_dict.values())

    @property
    def parameter_errors(self):
        """the current parameter uncertainties"""
        return self._fitter.fit_parameter_errors

    @property
    def parameter_cov_mat(self):
        """the current parameter covariance matrix"""
        return self._fitter.fit_parameter_cov_mat

    @property
    def parameter_name_value_dict(self):
        """a dictionary mapping each parameter name to its current value"""
        return self._fitter.fit_parameter_values

    @property
    def cost_function_value(self):
        """the current value of the cost function"""
        return self._fitter.parameter_to_minimize_value

    @property
    def data_size(self):
        """the size (number of points) of the data container"""
        return self._data_container.size

    @property
    def has_model_errors(self):
        """``True`` if at least one uncertainty source is defined for the model"""
        return self._param_model.has_errors

    @property
    def has_data_errors(self):
        """``True`` if at least one uncertainty source is defined for the data"""
        return self._data_container.has_errors

    @property
    def has_errors(self):
        """``True`` if at least one uncertainty source is defined for either the data or the model"""
        return True if self.has_data_errors or self.has_model_errors else False

    # -- public methods

    def set_parameter_values(self, **param_name_value_dict):
        """
        Set the fit parameters to new values. Valid keyword arguments are the names
        of the declared fit parameters.

        :param param_name_value_dict: new parameter values
        """
        return self._fitter.set_fit_parameter_values(**param_name_value_dict)

    def set_all_parameter_values(self, param_value_list):
        """
        Set all the fit parameters at the same time.

        :param param_value_list: list of parameter values (mind the order)
        """
        return self._fitter.set_all_fit_parameter_values(param_value_list)

    @abc.abstractmethod
    def add_simple_error(self):
        """
        Add a simple uncertainty source (constructed from pointwise errors and optionally
        a non-negative global correlation coefficient for any two daya points) to the data container.

        :return: int: error_id
        """
        pass

    @abc.abstractmethod
    def add_matrix_error(self):
        """
        Add a matrix uncertainty source (constructed from pointwise errors and a
        correlation matrix or the full point-to-point covariance matrix) to the data container.

        :return: int: error_id
        """
        pass

    def do_fit(self):
        """
        Perform the minimization of the cost function.
        """
        self._fitter.do_fit()
        # update parameter formatters
        for _fpf, _pv, _pe in zip(self._model_function.argument_formatters, self.parameter_values, self.parameter_errors):
            _fpf.value = _pv
            _fpf.error = _pe

    def assign_model_function_expression(self, expression_format_string):
        """Assign a plain-text-formatted expression string to the model function."""
        self._model_function.formatter.expression_format_string = expression_format_string

    def assign_model_function_latex_expression(self, latex_expression_format_string):
        """Assign a LaTeX-formatted expression string to the model function."""
        self._model_function.formatter.latex_expression_format_string = latex_expression_format_string

    def assign_parameter_latex_names(self, **par_latex_names_dict):
        """Assign LaTeX-formatted strings to the model function parameters."""
        for _pf in self._model_function.argument_formatters:
            _pln = par_latex_names_dict.get(_pf.name, None)
            if _pln is not None:
                _pf.latex_name = _pln