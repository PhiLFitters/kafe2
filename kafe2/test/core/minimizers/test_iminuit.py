import unittest
from kafe2.test.core.minimizers._base import AbstractMinimizerTest, MinimizerException

_cannot_import_IMinuit = False
try:
    from kafe2.core.minimizers.iminuit_minimizer import MinimizerIMinuit
except (ImportError, SyntaxError, MinimizerException):
    _cannot_import_IMinuit = True


@unittest.skipIf(_cannot_import_IMinuit, "Cannot import iminuit")
class TestMinimizerIMinuit(AbstractMinimizerTest, unittest.TestCase):

    def _get_minimizer(self, parameter_names, parameter_values, parameter_errors,
                       function_to_minimize):
        return MinimizerIMinuit(
            parameter_names=parameter_names, parameter_values=parameter_values,
            parameter_errors=parameter_errors, function_to_minimize=function_to_minimize
        )

    @property
    def _expected_tolerance(self):
        return 1e-2
