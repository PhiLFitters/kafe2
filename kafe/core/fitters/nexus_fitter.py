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

        # local cache
        self.__cache_stale = True
        self.__cache_all_parameters_name_value_dict = None
        self.__cache_fit_parameters_name_value_dict = None
        self.__cache_parameter_to_minimize_value = None

        self.__minimizing = False
        _par_name_val_map = self.fit_parameter_values
        if minimizer_kwargs == None:
            minimizer_kwargs = dict()

        if minimizer is not None:
            _minimizer_class = get_minimizer(minimizer)
        else:
            _minimizer_class = get_minimizer()  # get the default minimizer

        self._minimizer = _minimizer_class(parameters_to_fit,
                                           list(_par_name_val_map.values()),
                                           [0.1 if _v==0 else 0.1*_v for _v in _par_name_val_map.values()],
                                           self._fcn_wrapper, **minimizer_kwargs)
        self.__state_is_from_minimizer = False


    # -- private methods

    def _check_parnames_in_par_space_raise(self, fit_pars):
        for _pn in fit_pars:
            if self._nx.get_by_name(_pn) is None:
                raise NexusFitterException("No parameter with name '%s' registered in Nexus (%r)!" % (_pn, self._nx))

    def _renew_all_par_cache(self):
        _fpns = self.parameters_to_fit
        _apnvd = self.__cache_all_parameters_name_value_dict = self._nx.parameter_values_dict
        self.__cache_fit_parameters_name_value_dict = OrderedDict([(_pn, _apnvd[_pn]) for _pn in _fpns])
        self.__cache_parameter_to_minimize_value = _apnvd[self.parameter_to_minimize]

    def _renew_fit_par_cache(self):
        _fpns = self.parameters_to_fit
        _fpvs = self._nx.get_values_by_name(_fpns)
        self.__cache_fit_parameters_name_value_dict = OrderedDict([(_pn, _pv) for _pn, _pv in zip(_fpns, _fpvs)])

    def _renew_par_to_minimize_cache(self):
        _ptmn = self.parameter_to_minimize
        _ptmv = self._nx.get_values_by_name(_ptmn)
        self.__cache_parameter_to_minimize_value = _ptmv

    def _minimize(self, max_calls=None):
        self.__minimizing = True
        if max_calls is None:
            max_calls = kc('core', 'fitters', 'nexus_fitter', 'max_calls')
        self._minimizer.minimize(max_calls=max_calls)
        # opt.minimize(self._fcn_wrapper,
        #              self.fit_parameter_values.values(),
        #              args=(), method=None, jac=None,
        #              hess=None, hessp=None,
        #              bounds=None, constraints=(),
        #              tol=None,
        #              callback=None,
        #              options=None)
        self.__minimizing = False

        # evaluate function one more time with the final parameters,
        # in order to ensure the nexus is up to date
        _par_vals = self._minimizer.parameter_values
        self._fcn_wrapper(*_par_vals)

        self.__state_is_from_minimizer = True
        self.__cache_stale = True


    def _fcn_wrapper(self, *fit_par_value_list):
        # # set parameters to current values
        # if not (len(fit_par_value_list) == 1 and fit_par_value_list[0] is None):
        #     if not len(fit_par_value_list) == self.n_fit_par:
        #         print fit_par_value_list
        #         print self.n_fit_par
        # set parameters to current values
        _par_val_dict = {_pn: _pv for _pn, _pv in zip(self.parameters_to_fit, fit_par_value_list)}
        self._nx.set(**_par_val_dict)
        # evaluate function and return value
        return self._nx.get_values_by_name(self.parameter_to_minimize)

    # -- public properties

    @property
    def minimizer(self):
        return self._minimizer

    @property
    def parameters_to_fit(self):
        return self._fit_pars

    @parameters_to_fit.setter
    def parameters_to_fit(self, fit_parameters):
        self._check_parnames_in_par_space_raise(fit_parameters)
        self._fit_pars = fit_parameters

    @property
    def parameter_to_minimize(self):
        return self._parameter_to_minimize

    @parameter_to_minimize.setter
    def parameter_to_minimize(self, parameter_to_minimize):
        self._check_parnames_in_par_space_raise((parameter_to_minimize,))
        self._parameter_to_minimize = parameter_to_minimize

    @property
    def fit_parameter_values(self):
        # if self.__cache_stale or self.__minimizing:  # allow getting fresh (non-cached) parameters while minimizing
        if self.__cache_stale:
            self._renew_fit_par_cache()
        return self.__cache_fit_parameters_name_value_dict

    @property
    def fit_parameter_cov_mat(self):
        return self._minimizer.cov_mat

    @property
    def fit_parameter_errors(self):
        return self._minimizer.parameter_errors

    @property
    def parameter_to_minimize_value(self):
        # if self.__cache_stale or self.__minimizing:  # allow getting fresh (non-cached) parameters while minimizing
        if self.__cache_stale:
            self._renew_par_to_minimize_cache()
        return self.__cache_parameter_to_minimize_value

    @property
    def n_fit_par(self):
        return len(self.parameters_to_fit)

    # -- public methods

    def do_fit(self):
        self._minimize()

    def fix_parameter(self, par_name, par_value=None):
        if par_value is not None:
            self.set_fit_parameter_values(**{par_name: par_value})

        self._minimizer.fix(par_name)

    def release_parameter(self, par_name):
        self._minimizer.release(par_name)

    def contour(self, parameter_name_1, parameter_name_2, sigma=1.0, **kwargs):
        if not self.__state_is_from_minimizer:
            raise NexusFitterException("To calculate a contour the do_fit method has to be called first.")
        return self._minimizer.contour(parameter_name_1, parameter_name_2, sigma=sigma, **kwargs)

    def profile(self, parameter_name, bins=20, bound=2, args=None, subtract_min=False):
        if not self.__state_is_from_minimizer:
            raise NexusFitterException("To calculate a profile the do_fit method has to be called first.")
        return self._minimizer.profile(parameter_name, bins=bins, bound=bound, subtract_min=subtract_min)

    def set_fit_parameter_values(self, **parameter_value_dict):
        _dict_key_set = set(parameter_value_dict.keys())
        _par_name_set = set(self.parameters_to_fit)
        # test parameter names
        if not _dict_key_set.issubset(_par_name_set):
            _unknown_par_names = _dict_key_set - _par_name_set
            raise NexusFitterException("Cannot set fit parameter values: Unknown fit parameters: %r!"
                                       % (_unknown_par_names,))
        # set values in nexus
        self._nx.set(**parameter_value_dict)
        self._minimizer.set_several(list(parameter_value_dict.keys()), list(parameter_value_dict.values()))
        # set flags
        self.__state_is_from_minimizer = False
        self.__cache_stale = True

    def set_all_fit_parameter_values(self, fit_par_value_list):
        # test list length
        if not len(fit_par_value_list) == len(self.parameters_to_fit):
            raise NexusFitterException("Cannot set all fit parameter values: %d fit parameters declared, "
                                       "but %d provided!"
                                       % (len(self.parameters_to_fit), len(fit_par_value_list)))
        # set values in nexus
        _par_val_dict = {_pn: _pv for _pn, _pv in zip(self.parameters_to_fit, fit_par_value_list)}
        self.set_fit_parameter_values(**_par_val_dict)
