from scipy.optimize import brentq

class MinimizerBase(object):

    def __init__(self):
        self._save_state_dict = dict()

    def _reset(self):
        pass

    def _save_state(self):
        raise NotImplementedError()

    def _load_state(self):
        raise NotImplementedError()

    def _calculate_asymmetric_parameter_error(self, parameter_name):
        self.minimize()
        self._save_state()
        _target_chi_2 = self.function_value + 1.0
        _min_parameters = self.parameter_values  # (1.061 -0.548)

        _par_index = self.parameter_names.index(parameter_name)
        _par_min = self.parameter_values[_par_index]
        _par_err = self.parameter_errors[_par_index]

        _cut_dn = self._find_chi_2_cut(parameter_name, _par_min - 2 * _par_err, _par_min, _target_chi_2, _min_parameters)
        _err_dn = _par_min - _cut_dn

        _cut_up = self._find_chi_2_cut(parameter_name, _par_min, _par_min + 2 * _par_err, _target_chi_2, _min_parameters)
        _err_up = _cut_up - _par_min
        #print(_cut_dn, _cut_up)  # expected c ~ (-1.1685, 0.0729)
        self._load_state()
        return _err_dn, _err_up

    def _find_chi_2_cut(self, parameter_name, low, high, target_chi_2, min_parameters):
        def _profile(parameter_value):
            self.set_several(self.parameter_names, min_parameters)
            self.set(parameter_name, parameter_value)
            #print('pre %s: %s@%s' % (parameter_name, self.function_value, self.parameter_values))
            self.fix(parameter_name)
            self.minimize()
            _fval = self.function_value
            self.release(parameter_name)
            #print('post %s: %s@%s' % (parameter_name, _fval, self.parameter_values))
            #print()
            return _fval - target_chi_2

        return brentq(f=_profile, a=low, b=high, xtol=self.tolerance)

    @property
    def function_value(self):
        raise NotImplementedError()

    @property
    def asymmetric_parameter_errors(self):
        _asymmetric_errors = (self._calculate_asymmetric_parameter_error(_par_name)
                              for _par_name in self.parameter_names)
        #self.minimize()
        return _asymmetric_errors

    @property
    def parameter_values(self):
        raise NotImplementedError()

    @property
    def parameter_errors(self):
        raise NotImplementedError()

    @property
    def parameter_names(self):
        raise NotImplementedError()

    @property
    def tolerance(self):
        raise NotImplementedError()

    @tolerance.setter
    def tolerance(self, new_tol):
        raise NotImplementedError()

    def set(self, parameter_name, parameter_value):
        raise NotImplementedError()

    def set_several(self, parameter_names, parameter_values):
        raise NotImplementedError()

    def fix(self, parameter_name):
        raise NotImplementedError()

    def release(self, parameter_name):
        raise NotImplementedError()

    def minimize(self):
        raise NotImplementedError()
