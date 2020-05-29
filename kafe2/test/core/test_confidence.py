import unittest2 as unittest

from kafe2.core.confidence import ConfidenceLevel


class TestConfidence(unittest.TestCase):

    def setUp(self):
        self._sigma = [1.0, 2.0, 3.0]
        self._cl_1d = [0.6827, 0.9545, 0.9973]
        self._cl_2d = [0.3935, 0.8647, 0.9889]

    def test_one_way_conversion(self):
        for _sigma, _cl_1d in zip(self._sigma, self._cl_1d):
            self.assertAlmostEqual(
                _cl_1d,
                ConfidenceLevel(n_dimensions=1, sigma=_sigma).cl,
                places=4
            )
            self.assertAlmostEqual(
                _sigma,
                ConfidenceLevel(n_dimensions=1, cl=_cl_1d).sigma,
                places=3
            )
        for _sigma, _cl_2d in zip(self._sigma, self._cl_2d):
            self.assertAlmostEqual(
                _cl_2d,
                ConfidenceLevel(n_dimensions=2, sigma=_sigma).cl,
                places=4
            )
            self.assertAlmostEqual(
                _sigma,
                ConfidenceLevel(n_dimensions=2, cl=_cl_2d).sigma,
                places=3
            )

    def test_round_trip(self):
        for _n_dimensions in range(1, 10):
            for _sigma in [0.5, 1.2, 2.3, 3.4, 4.5]:
                self.assertAlmostEqual(
                    _sigma,
                    ConfidenceLevel(
                        n_dimensions=_n_dimensions,
                        cl=ConfidenceLevel(
                            n_dimensions=_n_dimensions,
                            sigma=_sigma
                        ).cl
                    ).sigma
                )
            for _cl in [0.12, 0.34, 0.56, 0.78, 0.99]:
                self.assertAlmostEqual(
                    _cl,
                    ConfidenceLevel(
                        n_dimensions=_n_dimensions,
                        sigma=ConfidenceLevel(
                            n_dimensions=_n_dimensions,
                            cl=_cl
                        ).sigma
                    ).cl
                )

    def test_constructor_raise(self):
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions=1)
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions=1, cl=0.683, sigma=1.0)
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions=1, cl=-1.0)
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions=1, cl=2.0)
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions=1, sigma=-1)
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions=1.1, sigma=1)
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions="1", sigma=1)
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions=0, sigma=1)
        with self.assertRaises(ValueError):
            ConfidenceLevel(n_dimensions=-1, sigma=1)

    def test_string_represenations(self):
        _cl = ConfidenceLevel(n_dimensions=1, sigma=1)
        self.assertEqual(str(_cl), "<ConfidenceLevel (d=1): 68.27% (1-sigma)>")
        self.assertEqual(_cl.sigma_string, "1-sigma")
        self.assertEqual(_cl.sigma_latex_string, "1$\\sigma$")
        self.assertEqual(_cl.cl_string, "68.27% CL")
        self.assertEqual(_cl.cl_latex_string, "$68.27\\%$ CL")
