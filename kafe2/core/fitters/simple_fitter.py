from ..minimizers import MinimizerIMinuit

class SimpleFitterException(Exception):
    pass

class SimpleFitter(object):
    def __init__(self, nexus, parameters_to_fit, parameter_to_minimize, minimizer_class=MinimizerIMinuit):
        self._nx = nexus
        self.parameters_to_fit = parameters_to_fit
        self.parameter_to_minimize = parameter_to_minimize

        # local cache
        self.__cache_stale = True
        self.__cache_all_parameters_name_value_dict = None
        self.__cache_fit_parameters_name_value_dict = None
        self.__cache_parameter_to_minimize_value = None

        _par_name_val_map = self.fit_parameter_values
        self._minimizer = minimizer_class(parameters_to_fit,
                                          _par_name_val_map.values(),
                                          [0.1 if _v==0 else 0.1*_v for _v in _par_name_val_map.values()],
                                          self._fcn_wrapper)

    # -- private methods

    def _check_parnames_in_par_space_raise(self, fit_pars):
        for _pn in fit_pars:
            if self._nx.get_by_name(_pn) is None:
                raise NexusFitterException("No parameter with name '%s' registered in ParameterSpace (%r)!" % (_pn, self._nx))

    def _renew_par_cache(self):
        _fpns = self.parameters_to_fit
        _apnvd = self.__cache_all_parameters_name_value_dict = self._nx.parameter_values_dict
        self.__cache_fit_parameters_name_value_dict = OrderedDict([(_pn, _apnvd[_pn]) for _pn in _fpns])
        self.__cache_parameter_to_minimize_value = _apnvd[self.parameter_to_minimize]

    def _minimize(self, max_calls=6000):
        self._minimizer.minimize(max_calls=max_calls)
        # opt.minimize(self._fcn_wrapper,
        #              self.fit_parameter_values.values(),
        #              args=(), method=None, jac=None,
        #              hess=None, hessp=None,
        #              bounds=None, constraints=(),
        #              tol=None,
        #              callback=None,
        #              options=None)
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
        if self.__cache_stale:
            self._renew_par_cache()
        return self.__cache_fit_parameters_name_value_dict

    @property
    def parameter_to_minimize_value(self):
        if self.__cache_stale:
            self._renew_par_cache()
        return self.__cache_parameter_to_minimize_value

    @property
    def n_fit_par(self):
        return len(self.parameters_to_fit)

    # -- public methods

    def do_fit(self):
        self._minimize()
