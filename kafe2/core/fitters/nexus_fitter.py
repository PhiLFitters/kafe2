from collections import OrderedDict

from ...config import kc
from ..minimizers import get_minimizer


class NexusFitterException(Exception):
    pass


class NexusFitter(object):

    def __init__(self, nexus, parameters_to_fit, parameter_to_minimize, minimizer=None, minimizer_kwargs=None):
        self._nx = nexus

        self.parameters_to_fit = parameters_to_fit
        self.parameter_to_minimize = parameter_to_minimize

        if minimizer is not None:
            _minimizer_class = get_minimizer(minimizer)
        else:
            _minimizer_class = get_minimizer()  # get the default minimizer

        if minimizer_kwargs is None:
            minimizer_kwargs = dict()

        # initialize minimizer
        _par_name_val_map = self.get_fit_parameter_values()
        _par_values = [_par_name_val_map[_pn] for _pn in parameters_to_fit]
        self._minimizer = _minimizer_class(
            parameters_to_fit,
            _par_values,
            [0.1 if _v==0 else 0.1*_v for _v in _par_values],
            self._fcn_wrapper,
            **minimizer_kwargs
        )

        self._fixed_pars = dict()
        self._limited_pars = dict()

        # flags
        self.__minimizing = False  # minimization ongoing?
        self.__state_is_from_minimizer = False


    # -- private methods

    def _get_pars_from_nexus(self, par_names):
        '''return list of nexus nodes for `par_names`'''
        _not_found = []
        _pars = []
        for _pn in par_names:
            _par = self._nx.get(_pn)
            if _par is None:
                _not_found.append(_pn)
            else:
                _pars.append(_par)


        if _not_found:
            raise NexusFitterException(
                "Parameters not registered in Nexus: {}".format(
                    ', '.join(_not_found)
                )
            )
        return _pars

    def _minimize(self, max_calls=None):
        '''run minimizer'''

        if max_calls is None:
            max_calls = kc('core', 'fitters', 'nexus_fitter', 'max_calls')

        self.__minimizing = True
        self._minimizer.minimize(max_calls=max_calls)
        self.__minimizing = False

        # evaluate function one more time with the final parameters,
        # in order to ensure the nexus is up to date
        _par_vals = self._minimizer.parameter_values
        self._fcn_wrapper(*_par_vals)

        self.__state_is_from_minimizer = True

    def _fcn_wrapper(self, *fit_par_value_list):
        # set fit parameter values
        assert(len(fit_par_value_list) == len(self._fit_pars))
        for _par, _new_value in zip(self._fit_pars, fit_par_value_list):
            _par.value = _new_value

        # evaluate function and return value
        return self._min_par.value


    # -- public properties

    @property
    def minimizer(self):
        return self._minimizer

    @property
    def parameters_to_fit(self):
        return self._fit_par_names

    @parameters_to_fit.setter
    def parameters_to_fit(self, fit_parameters):
        self._fit_pars = self._get_pars_from_nexus(fit_parameters)
        self._fit_par_names = tuple(fit_parameters)

    @property
    def parameter_to_minimize(self):
        return self._min_par_name

    @parameter_to_minimize.setter
    def parameter_to_minimize(self, parameter_to_minimize):
        self._min_par = \
            self._get_pars_from_nexus([parameter_to_minimize])[0]
        self._min_par_name = parameter_to_minimize

    @property
    def fit_parameter_cov_mat(self):
        return self._minimizer.cov_mat

    @property
    def fit_parameter_cor_mat(self):
        return self._minimizer.cor_mat

    @property
    def fit_parameter_errors(self):
        return self._minimizer.parameter_errors

    @property
    def asymmetric_fit_parameter_errors(self):
        return self._minimizer.asymmetric_parameter_errors

    @property
    def asymmetric_fit_parameter_errors_if_calculated(self):
        return self._minimizer.asymmetric_parameter_errors_if_calculated

    @property
    def parameter_to_minimize_value(self):
        return self._nx.get(self._min_par_name).value

    @property
    def n_fit_par(self):
        return len(self.parameters_to_fit)

    @property
    def state_is_from_minimizer(self):
        return self.__state_is_from_minimizer

    @property
    def fixed_parameters(self):
        return self._fixed_pars.copy()

    @property
    def limited_parameters(self):
        return self._limited_pars.copy()

    # -- public methods

    def do_fit(self):
        self._minimize()

    def fix_parameter(self, par_name, par_value=None):
        if par_value is not None:
            self.set_fit_parameter_values(**{par_name: par_value})

        self._minimizer.fix(par_name)
        _fixed_par_dict = self.get_fit_parameter_values([par_name])
        self._fixed_pars.update(_fixed_par_dict)

    def release_parameter(self, par_name):
        self._minimizer.release(par_name)
        self._fixed_pars.pop(par_name, None)

    def limit_parameter(self, par_name, par_limits):
        self._minimizer.limit(par_name, par_limits)
        self._limited_pars.update({par_name: par_limits})

    def unlimit_parameter(self, par_name):
        self._minimizer.unlimit(par_name)
        self._limited_pars.pop(par_name, None)

    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, **kwargs):
        if not self.__state_is_from_minimizer:
            raise NexusFitterException(
                "To calculate a contour the do_fit method has to be called first."
            )

        return self._minimizer.contour(parameter_name_1, parameter_name_2, sigma=sigma, **kwargs)

    def profile(self, parameter_name, bins=20, bound=2, args=None, subtract_min=False):
        if not self.__state_is_from_minimizer:
            raise NexusFitterException(
                "To calculate a profile the do_fit method has to be called first."
            )

        return self._minimizer.profile(parameter_name, bins=bins, bound=bound, subtract_min=subtract_min)

    def get_fit_parameter_values(self, parameter_names=None):
        if parameter_names is None:
            parameter_names = self._fit_par_names
        return OrderedDict([
            (_pn, self._nx.get(_pn).value)
            for _pn in parameter_names
        ])

    def set_fit_parameter_values(self, **parameter_value_dict):
        _dict_key_set = set(parameter_value_dict.keys())
        _par_name_set = set(self.parameters_to_fit)

        # test parameter names
        if not _dict_key_set.issubset(_par_name_set):
            _unknown_par_names = _dict_key_set - _par_name_set
            raise NexusFitterException("Cannot set fit parameter values: Unknown fit parameters: %r!"
                                       % (_unknown_par_names,))

        # set values in nexus
        for _par_name, _new_value in parameter_value_dict.items():
            self._nx.get(_par_name).value = _new_value
            self._minimizer.set(_par_name, _new_value)

        # set flags
        self.__state_is_from_minimizer = False

    def set_all_fit_parameter_values(self, fit_par_value_list):
        # test list length
        if not len(fit_par_value_list) == len(self.parameters_to_fit):
            raise NexusFitterException(
                "Cannot set all fit parameter values: "
                "{} fit parameters declared, "
                "but {} provided!".format(
                    self.n_fit_par,
                    len(fit_par_value_list)
                )
            )

        # set values in nexus and minimizer
        for _par_name, _par, _new_value in zip(self._fit_par_names, self._fit_pars, fit_par_value_list):
            _par.value = _new_value
            self._minimizer.set(_par_name, _new_value)

        # set flags
        self.__state_is_from_minimizer = False

    def reset_minimizer(self):
        self._minimizer.reset()
