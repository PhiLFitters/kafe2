import abc
import unittest2 as unittest
import numpy as np
import six

from scipy import stats

from kafe2.core.minimizers import AVAILABLE_MINIMIZERS
from kafe2.core.fitters import NexusFitterException

from kafe2.config import kc

from kafe2.fit import UnbinnedFit, UnbinnedContainer
from kafe2.fit._base import ModelFunctionException
from kafe2.fit.unbinned.fit import UnbinnedFitException
from kafe2.fit.unbinned.cost import UnbinnedCostFunction_NegLogLikelihood

from kafe2.test.fit.test_fit import AbstractTestFit



def unbinned_model_density(x, tau=2.2, fbg=0.1):
    b = 11.5
    a = 1.
    pdf1 = np.exp(-x / tau) / tau / (np.exp(-a / tau) - np.exp(-b / tau))
    pdf2 = 1. / (b - a)
    return (1 - fbg) * pdf1 + fbg * pdf2


class TestUnbinnedFitBasicInterface(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):

        self._ref_data = np.array([
            7.42, 3.773, 5.968, 4.924, 1.468, 4.664, 1.745, 2.144, 3.836, 3.132, 1.568, 2.352,
            2.132, 9.381, 1.484, 1.181, 5.004, 3.06, 4.582, 2.076, 1.88, 1.337, 3.092, 2.265,
            1.208, 2.753, 4.457, 3.499, 8.192, 5.101, 1.572, 5.152, 4.181, 3.52, 1.344, 10.29,
            1.152, 2.348, 2.228, 2.172, 7.448, 1.108, 4.344, 2.042, 5.088, 1.02, 1.051, 1.987,
            1.935, 3.773, 4.092, 1.628, 1.688, 4.502, 4.687, 6.755, 2.56, 1.208, 2.649, 1.012,
            1.73, 2.164, 1.728, 4.646, 2.916, 1.101, 2.54, 1.02, 1.176, 4.716, 9.671, 1.692,
            9.292, 10.72, 2.164, 2.084, 2.616, 1.584, 5.236, 3.663, 3.624, 1.051, 1.544, 1.496,
            1.883, 1.92, 5.968, 5.89, 2.896, 2.76, 1.475, 2.644, 3.6, 5.324, 8.361, 3.052, 7.703,
            3.83, 1.444, 1.343])

        self._ref_cont = UnbinnedContainer(self._ref_data)

        # reference initial values
        self._ref_initial_pars = np.array([2.2, 0.1])
        self._ref_initial_model = unbinned_model_density(self._ref_data, *self._ref_initial_pars)

        # pre-fit cost value
        self._ref_initial_cost = -2*np.sum(np.log(self._ref_initial_model))

        # reference fit result values
        self._nominal_fit_result_pars = np.array([2.12812181, 0.11117378])
        self._nominal_fit_result_model = unbinned_model_density(self._ref_data, *self._nominal_fit_result_pars)

        self._nominal_fit_result_cost = -2*np.sum(np.log(self._nominal_fit_result_model))

        # helper dict with all reference property values
        self._ref_prop_dict = dict(
            did_fit=False,
            model_count=1,

            parameter_values=self._ref_initial_pars,
            parameter_names=('tau', 'fbg'),
            cost_function_value=self._ref_initial_cost,

            data=self._ref_data,
            model=self._ref_initial_model,
        )

    def _get_fit(self, model_density_function=None, cost_function=None):
        '''convenience'''

        model_density_function = model_density_function or unbinned_model_density

        # TODO: fix default
        cost_function = cost_function or UnbinnedCostFunction_NegLogLikelihood()

        _fit = UnbinnedFit(
            data=self._ref_cont,
            model_density_function=model_density_function,
            cost_function=cost_function,
            minimizer=self.MINIMIZER
        )

        return _fit

    def _get_test_fits(self):
        return {
            'default': \
                self._get_fit(),
        }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            self._ref_prop_dict
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars,
                model=self._nominal_fit_result_model,
                did_fit=True,
                cost_function_value=np.float64(self._nominal_fit_result_cost),
            ),
            fit_names=['default'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-2
        )

    def test_update_data(self):
        _fit = self._get_fit()

        _new_data = np.array([
            1.891075005713712, 2.361504517508682, 1.2026086768765802, 2.0192327689754013,
            6.789827691034451, 6.229060175471942, 5.6827777380476725, 1.3896314892970838,
            1.0379155165227294, 1.1948753489234705, 1.8735602750435354, 2.139305908923123,
            1.935777036301727, 6.108651728684855, 1.5185867568257783, 1.1244705951426477,
            2.2175887045709146, 3.900030680803669, 4.110502050796017, 3.634194797012612,
            4.931709433880977, 3.779552162832593, 3.6401901478360594, 1.7084479392384508,
            10.502957316458524, 1.669761159672512, 2.0378350020970446, 1.4406236513060318,
            2.3226997026175917, 1.2530908142355945, 2.4681349735479428, 2.2697496807176245,
            8.056140162306527, 6.483413760658378, 2.0103631281648733, 1.8484180369382175,
            2.042048282225629, 1.826991555740888, 4.119685425461164, 2.8467560883558485,
            1.8277818983512852, 9.894627016335363, 1.0841480000087294, 3.1732616953846584,
            1.6776961308198861, 1.555131926310589, 1.1806640179159833, 3.052979898534809,
            5.059864773115917, 1.1294288761955027, 2.6979674092930193, 9.82656489928625,
            5.501267559105623, 1.2351970327509707, 1.6078922552697488, 1.582561168396548,
            2.391340419810505, 2.660167843760268, 1.9569767827507456, 1.2776674104800188,
            1.6183122532590568, 4.552555359280379, 4.214077308248084, 1.7815460978157334,
            3.5980732014726056, 1.690396198372471, 4.671077054310741, 1.2339939189715687,
            2.7618415144011874, 2.86744016658739])

        _fit.data = UnbinnedContainer(_new_data)

        self._assert_fit_properties(
            _fit,
            dict(
                data=_new_data,
            )
        )

        _fit.do_fit()

        self._assert_fit_properties(
            _fit,
            dict(
                data=_new_data,
                parameter_values=np.array([1.64489992, 0.1225065]),
            ),
            rtol=1e-2
        )

    def test_reserved_parameter_names_raise(self):
        def dummy_model(x, data):
            pass

        with self.assertRaises(UnbinnedFitException) as _exc:
            UnbinnedFit(data=self._ref_cont,
                    model_density_function=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn('reserved', _exc.exception.args[0])
        self.assertIn('data', _exc.exception.args[0])

    def test_model_no_pars_raise(self):
        def dummy_model():
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            UnbinnedFit(data=self._ref_cont,
                    model_density_function=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn(
            'needs at least one parameter',
            _exc.exception.args[0])

    def test_model_no_pars_beside_x_raise(self):
        def dummy_model(x):
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            UnbinnedFit(data=self._ref_cont,
                    model_density_function=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn(
            'needs at least one parameter besides',
            _exc.exception.args[0])

    def test_model_varargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, *varargs):
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            UnbinnedFit(data=self._ref_cont,
                    model_density_function=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varargs', _exc.exception.args[0])

    def test_model_varkwargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, **varkwargs):
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            UnbinnedFit(data=self._ref_cont,
                    model_density_function=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varkwargs', _exc.exception.args[0])

    def test_model_varargs_varkwargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, *varargs, **varkwargs):
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            UnbinnedFit(data=self._ref_cont,
                    model_density_function=dummy_model,
                    minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varargs', _exc.exception.args[0])
        # TODO: enable when implemented
        #self.assertIn('varkwargs', _exc.exception.args[0])

    def test_report_before_fit(self):
        # TODO: check report content
        _buffer = six.StringIO()
        _fit = self._get_fit()
        _fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")

    def test_report_after_fit(self):
        # TODO: check report content
        _buffer = six.StringIO()
        _fit = self._get_fit()
        _fit.do_fit()
        _fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")
