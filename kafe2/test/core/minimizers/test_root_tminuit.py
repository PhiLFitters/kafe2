import unittest2 as unittest
from kafe2.test.core.minimizers._base import TestMinimizerMixin

_cannot_import_ROOT = False
try:
    from kafe2.core.minimizers.root_tminuit_minimizer import \
        MinimizerROOTTMinuit, MinimizerROOTTMinuitException
except ImportError:
    _cannot_import_ROOT = True


@unittest.skipIf(_cannot_import_ROOT, "Cannot import ROOT")
class TestMinimizerROOTTMinuit(TestMinimizerMixin, unittest.TestCase):
    def _get_minimizer(self, parameter_names, parameter_values, parameter_errors,
                       function_to_minimize):
        return MinimizerROOTTMinuit(
            parameter_names=parameter_names, parameter_values=parameter_values,
            parameter_errors=parameter_errors, function_to_minimize=function_to_minimize
        )

    @property
    def _expected_tolerance(self):
        return 1e-9

    def test_compare_par_values_minimize_fcn3_limit_unlimit_onesided(self):
        with self.assertRaises(MinimizerROOTTMinuitException):
            self.m3.limit('x', (None, 0.7))

    def test_compare_par_values_minimize_fcn3_limit_unlimit_onesided_2(self):
        with self.assertRaises(MinimizerROOTTMinuitException):
            self.m3.limit('x', (3.5, None))
