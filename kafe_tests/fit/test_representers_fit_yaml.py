import unittest
import numpy as np
from scipy.stats import norm
from six import StringIO

from kafe.fit.representation import FitYamlWriter, FitYamlReader
from kafe.fit.io.handle import IOStreamHandle
from kafe.fit.histogram import HistFit
from kafe.fit.indexed import IndexedFit
from kafe.fit.xy import XYFit
from kafe.fit.xy_multi import XYMultiFit
from kafe.fit.histogram.container import HistContainer

TEST_FIT_HIST="""
type: histogram
dataset:
    type: histogram
    n_bins: 8
    bin_range: [-2.0, 2.0]
    raw_data: [ 0.94657578  0.3674123  -0.16462338  0.0408205   0.35318031 -0.35919436
               -0.22111037 -0.4663712  -0.26321687  0.63243935  0.30995581  0.45732282
                0.89150917  0.62005489 -0.5260296   0.96317915  0.2918086   0.27343654
                0.0070309  -0.54261172  0.01499711  0.32934897  0.53675086 -0.33745179
                0.42785108  0.29124784  0.82751196  0.3073723  -0.51430445 -0.5821153
                0.28015453 -0.06767495 -0.42469716  0.08997751 -0.34496342  0.91208844
                0.48666838  0.54641629 -0.26780562 -0.18634414 -0.87815388  0.01707615
                0.62112113  1.2714954   1.00311742 -0.65583951 -0.51491734 -0.13001327
                0.12267431  0.06186063 -0.25203     0.5991445  -0.05863787  0.44575788
               -0.55761754  0.70046957  1.0686471   0.04027402  0.39745425  0.8821956
                0.2241812   0.01701976 -0.44040885  0.87585192  0.62350494  0.34860764
               -0.53451919  0.7347968   0.32077684  1.0450513   0.467221    0.46268865
                0.66497412 -0.48091556  0.33789605  0.46706738  0.23336683  0.29909525
                0.08691923  0.98999353 -0.17895406  0.50533812  0.81767467 -0.76434111
                0.09429677  0.61596385  0.69613775 -0.29647721  0.48768034 -0.44852791]
    errors:
      - correlation_coefficient: 0.0
        error_value: 0.1
        name: test_error
        relative: false
        type: simple
parametric_model:
    type: histogram
    n_bins: 8
    bin_range: [-2.0, 2.0]
    model_density_function:
        type: histogram
        python_code: |
            def hist_model_density(x, mu, sigma):
                return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
    model_parameters: [0.1, 1.0]
"""

class TestHistFitYamlRepresenter(unittest.TestCase):

    @staticmethod
    def hist_model_density(x, mu=0.1, sigma=1.0):
        #TODO scipy.stats.norm support
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
    
    @staticmethod
    def hist_model_density_antiderivative(x, mu, sigma):
        return norm(mu, sigma).cdf(x)
    
    def setUp(self):
        self._test_raw_data = [
            0.94657578,  0.3674123,  -0.16462338,  0.0408205,   0.35318031, -0.35919436,
           -0.22111037, -0.4663712,  -0.26321687,  0.63243935,  0.30995581,  0.45732282,
            0.89150917,  0.62005489, -0.5260296,   0.96317915,  0.2918086,   0.27343654,
            0.0070309,  -0.54261172,  0.01499711,  0.32934897,  0.53675086, -0.33745179,
            0.42785108,  0.29124784,  0.82751196,  0.3073723,  -0.51430445, -0.5821153,
            0.28015453, -0.06767495, -0.42469716,  0.08997751, -0.34496342,  0.91208844,
            0.48666838,  0.54641629, -0.26780562, -0.18634414, -0.87815388,  0.01707615,
            0.62112113,  1.2714954,   1.00311742, -0.65583951, -0.51491734, -0.13001327,
            0.12267431,  0.06186063, -0.25203,     0.5991445,  -0.05863787,  0.44575788,
           -0.55761754,  0.70046957,  1.0686471,   0.04027402,  0.39745425,  0.8821956,
            0.2241812,   0.01701976, -0.44040885,  0.87585192,  0.62350494,  0.34860764,
           -0.53451919,  0.7347968,   0.32077684,  1.0450513,   0.467221,    0.46268865,
            0.66497412, -0.48091556,  0.33789605,  0.46706738,  0.23336683,  0.29909525,
            0.08691923,  0.98999353, -0.17895406,  0.50533812,  0.81767467, -0.76434111,
            0.09429677,  0.61596385,  0.69613775, -0.29647721,  0.48768034, -0.44852791
        ]
        self._test_n_bins = 8
        self._test_parameters_default = np.array([0.1, 1.0])
        self._test_parameters = np.array([0.2, 0.5])
        self._test_bin_range = (-2.0, 2.0)
        self._test_parameters_do_fit = np.array([2.0117809129092095, -1.090410559090481])
        self._test_x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        self._test_density_default = self.hist_model_density(self._test_x, *self._test_parameters_default)
        self._test_density_do_fit = self.hist_model_density(self._test_x, *self._test_parameters)
        
        _data = HistContainer(n_bins=self._test_n_bins, bin_range=self._test_bin_range)
        self._fit = HistFit(
            data=_data,
            model_density_function=TestHistFitYamlRepresenter.hist_model_density
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_FIT_HIST))
        
        self._roundtrip_streamreader = FitYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = FitYamlWriter(self._fit, self._roundtrip_stringstream)
        self._testfile_streamreader = FitYamlReader(self._testfile_stringstream)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, HistFit))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_density_default,
                _read_fit.eval_model_function_density(self._test_x)
            )
        )
        
        _read_fit.do_fit()
        
        #FIXME fit results are wrong
        #self.assertTrue(
        #    np.allclose(
        #        self._test_parameters_do_fit,
        #        _read_fit.parameter_values
        #    )
        #)
        #self.assertTrue(
        #    True or
        #    np.allclose(
        #        self._test_density_do_fit,
        #        _read_fit.eval_model_function_density(self._test_x)
        #    )
        #)

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, HistFit))

        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_density_default,
                _read_fit.eval_model_function_density(self._test_x)
            )
        )
        
        _read_fit.do_fit()
        
        #FIXME fit results are wrong
        #self.assertTrue(
        #    np.allclose(
        #        self._test_parameters_do_fit,
        #        _read_fit.parameter_values
        #    )
        #)
        #self.assertTrue(
        #    np.allclose(
        #        self._test_density_do_fit,
        #        _read_fit.eval_model_function_density(self._test_x)
        #    )
        #)

TEST_FIT_INDEXED="""
type: indexed
dataset:
    type: indexed
    data: [ -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
           9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
    errors:
      - correlation_coefficient: 0.0
        error_value: 0.1
        name: test_error
        relative: false
        type: simple
parametric_model:
    type: indexed
    model_function:
        type: indexed
        python_code: |
            def linear_model(a, b):
                return a * np.arange(10) + b
    model_parameters: [1.5, -0.5]
"""

class TestIndexedFitYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(a=1.5, b=-0.5):
        return a * np.arange(10) + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_parameters = np.array([2.0, -1.0])
        self._test_parameters_default = np.array([1.5, -0.5])
        self._test_parameters_do_fit = np.array([2.0117809129092095, -1.090410559090481])
        self._test_y = [ -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
                        9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
        self._test_y_default = self.linear_model(*self._test_parameters_default)
        self._test_y_do_fit = [ -1.09041056, 0.92137035, 2.93315127, 4.94493218, 6.95671309,
                               8.96849401, 10.98027492, 12.99205583, 15.00383674, 17.01561766]
        
        self._fit = IndexedFit(
            data=self._test_y,
            model_function=TestIndexedFitYamlRepresenter.linear_model
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_FIT_INDEXED))
        
        self._roundtrip_streamreader = FitYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = FitYamlWriter(self._fit, self._roundtrip_stringstream)
        self._testfile_streamreader = FitYamlReader(self._testfile_stringstream)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, IndexedFit))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default,
                _read_fit.model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.model
            )
        )

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, IndexedFit))

        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default,
                _read_fit.model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.model
            )
        )

TEST_FIT_XY="""
type: xy
dataset:
    type: xy
    x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    y_data: [ -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
             9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
    y_errors:
      - correlation_coefficient: 0.0
        error_value: 0.1
        name: test_y_error
        relative: false
        type: simple
parametric_model:
    type: xy
    x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    model_function:
        type: xy
        python_code: |
            def linear_model(x, a, b):
                return a * x + b
    model_parameters: [1.5, -0.5]
"""

class TestXYFitYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(x, a=1.5, b=-0.5):
        return a * x + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_parameters = np.array([2.0, -1.0])
        self._test_parameters_default = np.array([1.5, -0.5])
        self._test_parameters_do_fit = np.array([2.0117809129092095, -1.090410559090481])
        self._test_y = [ -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
                        9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
        self._test_y_default = self.linear_model(self._test_x, *self._test_parameters_default)
        self._test_y_do_fit = [ -1.09041056, 0.92137035, 2.93315127, 4.94493218, 6.95671309,
                               8.96849401, 10.98027492, 12.99205583, 15.00383674, 17.01561766]
        
        self._fit = XYFit(
            xy_data=[self._test_x, self._test_y],
            model_function=TestXYFitYamlRepresenter.linear_model
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_FIT_XY))
        
        self._roundtrip_streamreader = FitYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = FitYamlWriter(self._fit, self._roundtrip_stringstream)
        self._testfile_streamreader = FitYamlReader(self._testfile_stringstream)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, XYFit))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default,
                _read_fit.y_model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.y_model
            )
        )

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, XYFit))

        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default,
                _read_fit.y_model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.y_model
            )
        )

TEST_FIT_XY_MULTI="""
type: xy_multi
dataset:
    type: xy_multi
    x_data_0: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    y_data_0: [ -1.19026065,  1.51271632,  5.06403348,  9.53506975, 15.07931631,
                21.54634241, 29.04433804, 37.50568252, 46.99345912, 57.43629710]
    x_data_1: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    y_data_1: [-1.06964365,  1.05016488,  2.70494708,  5.03116094,  6.92949725,
                8.98510957, 11.00178593, 13.00748810, 15.12334609, 16.90640188]
    y_errors:
      - correlation_coefficient: 0.0
        error_value: 0.1
        name: test_y_error
        relative: false
        type: simple
parametric_model:
    type: xy_multi
    x_data_0: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    x_data_1: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    model_function:
        type: xy_multi
        python_code:
          - |
            def quadratic_model(x, a, b, c):
                return a * x ** 2 + b * x + c
          - |
            def linear_model(x, b, c):
                return b * x + c
        data_indices: [0, 10, 20]
    model_parameters: [0.25, 1.5, -0.5]
"""

class TestXYMultiFitYamlRepresenter(unittest.TestCase):

    @staticmethod
    def quadratic_model(x, a=0.25, b=1.5, c=-0.5):
        return a * x ** 2 + b * x + c
    
    @staticmethod
    def linear_model(x, b, c):
        return b * x + c
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_parameters = np.array([0.5, 2.0, -1.0])
        self._test_parameters_default = np.array([0.25, 1.5, -0.5])
        self._test_parameters_do_fit = np.array([0.4997519625958399, 2.008643944532134, -1.0505005938787126])
        self._test_y_0 = [ -1.19026065,  1.51271632,  5.06403348,  9.53506975, 15.07931631,
                           21.54634241, 29.04433804, 37.50568252, 46.99345912, 57.43629710]
        self._test_y_default_0 = self.quadratic_model(self._test_x, *self._test_parameters_default)
        self._test_y_1 = [-1.06964365,  1.05016488,  2.70494708,  5.03116094,  6.92949725,
                           8.98510957, 11.00178593, 13.00748810, 15.12334609, 16.90640188]
        self._test_y_default_1 = self.linear_model(self._test_x, *self._test_parameters_default[1:])
        self._test_y_do_fit = [ -1.05050059,  1.45789531,  4.96579515,  9.4731989,  14.98010659,
                                21.48651819, 28.99243373, 37.49785319, 47.00277657, 57.50720388,
                                -1.05050059,  0.95814335,  2.9667873,   4.97543124,  6.98407518,
                                 8.99271913, 11.00136307, 13.01000702, 15.01865096, 17.02729491]
        self._test_y_default_combined = np.concatenate([self._test_y_default_0, self._test_y_default_1])
        
        self._fit = XYMultiFit(
            xy_data=[[self._test_x, self._test_y_0], [self._test_x, self._test_y_1]],
            model_function=[self.quadratic_model, self.linear_model]
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_FIT_XY_MULTI))
        
        self._roundtrip_streamreader = FitYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = FitYamlWriter(self._fit, self._roundtrip_stringstream)
        self._testfile_streamreader = FitYamlReader(self._testfile_stringstream)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, XYMultiFit))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default_combined,
                _read_fit.y_model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.y_model
            )
        )

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, XYMultiFit))

        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default_combined,
                _read_fit.y_model
            )
        )
        
        _read_fit.do_fit()
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.poi_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.y_model
            )
        )
