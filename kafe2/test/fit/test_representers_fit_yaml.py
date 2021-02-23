import unittest2 as unittest
import numpy as np
from scipy.stats import norm
import six
from six import StringIO
import abc

from kafe2.fit.representation import FitYamlWriter, FitYamlReader
from kafe2.fit.io.handle import IOStreamHandle
from kafe2.fit.histogram import HistFit
from kafe2.fit.indexed import IndexedFit
from kafe2.fit.unbinned import UnbinnedFit
from kafe2.fit.xy import XYFit
from kafe2.fit.histogram.container import HistContainer
from kafe2.fit.representation._yaml_base import YamlReaderException


@six.add_metaclass(abc.ABCMeta)
class AbstractTestFitRepresenter(object):
    FIT_CLASS = None
    TEST_FIT = None
    TEST_FIT_SIMPLE = None
    TEST_FIT_MISSING_KEYWORD = None
    TEST_FIT_EXTRA_KEYWORD = None

    @abc.abstractmethod
    def setUp(self):
        pass

    def setup_streams(self):
        self._testfile_stringstream = IOStreamHandle(StringIO(self.TEST_FIT))
        self._testfile_stringstream_simple = IOStreamHandle(StringIO(self.TEST_FIT_SIMPLE))
        self._testfile_streamreader = FitYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_simple = FitYamlReader(self._testfile_stringstream_simple)

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._roundtrip_streamreader = FitYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = FitYamlWriter(self._fit, self._roundtrip_stringstream)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(self.TEST_FIT_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(self.TEST_FIT_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = FitYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = FitYamlReader(self._testfile_stringstream_extra_keyword)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_save_results(self):
        self._fit.do_fit()
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))

        self.assertTrue(self._fit.did_fit == _read_fit.did_fit)
        self.assertTrue(
            np.allclose(
                self._fit.parameter_cov_mat,
                _read_fit.parameter_cov_mat
            )
        )
        self.assertTrue(
            np.allclose(
                self._fit.parameter_errors,
                _read_fit.parameter_errors
            )
        )
        self.assertTrue(
            np.allclose(
                self._fit.parameter_cor_mat,
                _read_fit.parameter_cor_mat
            )
        )


TEST_FIT_HIST="""
type: histogram
dataset:
    type: histogram
    n_bins: 8
    bin_range: [-2.0, 2.0]
    raw_data: [ 0.94657578,  0.3674123,  -0.16462338,  0.0408205,   0.35318031, -0.35919436,
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
                0.09429677,  0.61596385,  0.69613775, -0.29647721,  0.48768034, -0.44852791]
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
parameter_constraints:
      - type: simple
        index: 0
        value: 0.1
        uncertainty: 1.0
      - type: simple
        name: sigma
        value: 1.0
        uncertainty: 0.5
      - type: matrix
        names: [sigma, mu]
        values: [1.3, 2.5]
        matrix: [[1.1, 0.1], [0.1, 2.4]]
"""

TEST_FIT_HIST_MISSING_KEYWORD="""
type: histogram
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

TEST_FIT_HIST_EXTRA_KEYWORD = TEST_FIT_HIST + """
extra_keyword: 3.14
"""

TEST_FIT_HIST_SIMPLE="""
type: histogram
n_bins: 8
bin_range: [-2.0, 2.0]
raw_data: [ 0.94657578,  0.3674123,  -0.16462338,  0.0408205,   0.35318031, -0.35919436,
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
            0.09429677,  0.61596385,  0.69613775, -0.29647721,  0.48768034, -0.44852791]
errors: 0.1
"""


class TestHistFitYamlRepresenter(unittest.TestCase, AbstractTestFitRepresenter):
    FIT_CLASS = HistFit
    TEST_FIT = TEST_FIT_HIST
    TEST_FIT_SIMPLE = TEST_FIT_HIST_SIMPLE
    TEST_FIT_EXTRA_KEYWORD = TEST_FIT_HIST_EXTRA_KEYWORD
    TEST_FIT_MISSING_KEYWORD = TEST_FIT_HIST_MISSING_KEYWORD

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
        self._test_parameters_default_simple = np.array([1.0, 1.0])
        self._test_parameters = np.array([0.2, 0.5])
        self._test_bin_range = (-2.0, 2.0)
        self._test_parameters_do_fit = np.array([0.20292002, 0.50108339])
        self._test_parameters_do_fit_simple = np.array([0.20032218, 0.49665461])
        self._test_x = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        self._test_density_default = self.hist_model_density(self._test_x, *self._test_parameters_default)
        self._test_density_default_simple = self.hist_model_density(self._test_x, *self._test_parameters_default_simple)
        self._test_density_do_fit = self.hist_model_density(self._test_x, *self._test_parameters_do_fit)
        self._test_density_do_fit_simple = self.hist_model_density(self._test_x, *self._test_parameters_do_fit_simple)

        _data = HistContainer(n_bins=self._test_n_bins, bin_range=self._test_bin_range, fill_data=self._test_raw_data)
        self._fit = HistFit(
            data=_data,
            model_density_function=TestHistFitYamlRepresenter.hist_model_density
        )
        self._fit.add_error(err_val=0.1)
        self._fit.add_parameter_constraint("mu", 0.1, 1.0)
        self._fit.add_parameter_constraint("sigma", 1.0, 0.5)
        self._fit.add_matrix_parameter_constraint(
            ["sigma", "mu"], [1.3, 2.5], [[1.1, 0.1], [0.1, 2.4]])

        self.setup_streams()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
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
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_density_do_fit,
                _read_fit.eval_model_function_density(self._test_x)
            )
        )

    def test_read_from_testfile_stream_simple(self):
        _read_fit = self._testfile_streamreader_simple.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default_simple,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_density_default_simple,
                _read_fit.eval_model_function_density(self._test_x)
            )
        )
        
        _read_fit.do_fit()

        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit_simple,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_density_do_fit_simple,
                _read_fit.eval_model_function_density(self._test_x)
            )
        )

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))

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

        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_density_do_fit,
                _read_fit.eval_model_function_density(self._test_x)
            )
        )


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
parameter_constraints:
      - type: simple
        index: 0
        value: 2.0
        uncertainty: 1.0
      - type: simple
        name: b
        value: -1.0
        uncertainty: 0.5
      - type: matrix
        names: [a, b]
        values: [2.05, -0.95]
        matrix: [[1.1, 0.1], [0.1, 2.4]]
"""

TEST_FIT_INDEXED_MISSING_KEYWORD="""
type: indexed
parametric_model:
    type: indexed
    model_function:
        type: indexed
        python_code: |
            def linear_model(a, b):
                return a * np.arange(10) + b
    model_parameters: [1.5, -0.5]
"""

TEST_FIT_INDEXED_EXTRA_KEYWORD = TEST_FIT_INDEXED + """
extra_keyword: 3.14
"""

TEST_FIT_INDEXED_SIMPLE="""
type: indexed
data: [ -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
       9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
errors: 0.1
model_function: |
    def linear_model(a, b):
        return a * np.arange(10) + b
"""


class TestIndexedFitYamlRepresenter(unittest.TestCase, AbstractTestFitRepresenter):
    FIT_CLASS = IndexedFit
    TEST_FIT = TEST_FIT_INDEXED
    TEST_FIT_SIMPLE = TEST_FIT_INDEXED_SIMPLE
    TEST_FIT_EXTRA_KEYWORD = TEST_FIT_INDEXED_EXTRA_KEYWORD
    TEST_FIT_MISSING_KEYWORD = TEST_FIT_INDEXED_MISSING_KEYWORD

    @staticmethod
    def linear_model(a=1.5, b=-0.5):
        return a * np.arange(10) + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_parameters = np.array([2.0, -1.0])
        self._test_parameters_default = np.array([1.5, -0.5])
        self._test_parameters_default_simple = np.array([1.0, 1.0])
        self._test_parameters_do_fit = np.array([2.0115580399995032, -1.0889949779758534])
        self._test_parameters_do_fit_simple = np.array([2.0117809129092095, -1.090410559090481])
        self._test_y = [-1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
                        9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
        self._test_y_default = self.linear_model(*self._test_parameters_default)
        self._test_y_default_simple = self.linear_model(*self._test_parameters_default_simple)
        self._test_y_do_fit = [-1.08899498, 0.92256306, 2.9341211, 4.94567914, 6.95723718, 8.96879522,
                               10.98035326, 12.9919113, 15.00346934, 17.01502738]
        self._test_y_do_fit_simple = [-1.09041056, 0.92137035, 2.93315127, 4.94493218, 6.95671309,
                                      8.96849401, 10.98027492, 12.99205583, 15.00383674, 17.01561766]

        self._fit = IndexedFit(
            data=self._test_y,
            model_function=TestIndexedFitYamlRepresenter.linear_model
        )
        self._fit.set_all_parameter_values(self._test_parameters_default)
        self._fit.add_error(err_val=0.1)
        self._fit.add_parameter_constraint(name='a', value=2.0, uncertainty=1.0)
        self._fit.add_parameter_constraint(name='b', value=-1.0, uncertainty=0.5)
        self._fit.add_matrix_parameter_constraint(names=['a', 'b'], values=[2.05, -0.95],
                                                  matrix=[[1.1, 0.1], [0.1, 2.4]])

        self.setup_streams()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
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

    def test_read_from_testfile_stream_simple(self):
        _read_fit = self._testfile_streamreader_simple.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default_simple,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default_simple,
                _read_fit.model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit_simple,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit_simple,
                _read_fit.model
            )
        )

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))

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
        python_code: |
            def linear_model(x, a, b):
                return a * x + b
    model_parameters: [1.5, -0.5]
parameter_constraints:
      - type: simple
        index: 0
        value: 2.0
        uncertainty: 1.0
      - type: simple
        name: b
        value: -1.0
        uncertainty: 0.5
      - type: matrix
        names: [a, b]
        values: [2.05, -0.95]
        matrix: [[1.1, 0.1], [0.1, 2.4]]
"""

TEST_FIT_XY_MISSING_KEYWORD="""
type: xy
parametric_model:
    type: xy
    x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    model_function:
        python_code: |
            def linear_model(x, a, b):
                return a * x + b
    model_parameters: [1.5, -0.5]
"""

TEST_FIT_XY_EXTRA_KEYWORD = TEST_FIT_XY + """
extra_keyword: 3.14
"""

TEST_FIT_XY_SIMPLE="""
x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
y_data: [ -1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
        9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
model_function: linear
y_errors: 0.1
"""


class TestXYFitYamlRepresenter(unittest.TestCase, AbstractTestFitRepresenter):
    FIT_CLASS = XYFit
    TEST_FIT = TEST_FIT_XY
    TEST_FIT_SIMPLE = TEST_FIT_XY_SIMPLE
    TEST_FIT_EXTRA_KEYWORD = TEST_FIT_XY_EXTRA_KEYWORD
    TEST_FIT_MISSING_KEYWORD = TEST_FIT_XY_MISSING_KEYWORD

    @staticmethod
    def linear_model(x, a=1.5, b=-0.5):
        return a * x + b
    
    def setUp(self):
        self._test_x = np.arange(10)
        self._test_parameters = np.array([2.0, -1.0])
        self._test_parameters_default = np.array([1.5, -0.5])
        self._test_parameters_default_simple = np.array([1.0, 1.0])
        self._test_parameters_do_fit = np.array([2.0115580399995032, -1.0889949779758534])
        self._test_parameters_do_fit_simple = np.array([2.0117809129092095, -1.090410559090481])
        self._test_y = [-1.0804945, 0.97336504, 2.75769933, 4.91093935, 6.98511206,
                        9.15059627, 10.9665515, 13.06741151, 14.95081026, 16.94404467]
        self._test_y_default = self.linear_model(self._test_x, *self._test_parameters_default)
        self._test_y_default_simple = self.linear_model(self._test_x, *self._test_parameters_default_simple)
        self._test_y_do_fit = [-1.08899498, 0.92256306, 2.9341211, 4.94567914, 6.95723718, 8.96879522, 10.98035326,
                               12.9919113, 15.00346934, 17.01502738]

        self._test_y_do_fit_simple = [-1.09041056, 0.92137035, 2.93315127, 4.94493218, 6.95671309,
                                      8.96849401, 10.98027492, 12.99205583, 15.00383674, 17.01561766]

        self._fit = XYFit(
            xy_data=[self._test_x, self._test_y],
            model_function=TestXYFitYamlRepresenter.linear_model
        )
        self._fit.add_error(axis='y', err_val=0.1)
        self._fit.add_parameter_constraint(name='a', value=2.0, uncertainty=1.0)
        self._fit.add_parameter_constraint(name='b', value=-1.0, uncertainty=0.5)
        self._fit.add_matrix_parameter_constraint(names=['a', 'b'], values=[2.05, -0.95],
                                                  matrix=[[1.1, 0.1], [0.1, 2.4]])

        self.setup_streams()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.parameter_values
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
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.y_model
            )
        )

    def test_read_from_testfile_stream_simple(self):
        _read_fit = self._testfile_streamreader_simple.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
        self.assertTrue(
            np.allclose(
                self._test_parameters_default_simple,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_default_simple,
                _read_fit.y_model
            )
        )
        
        _read_fit.do_fit()
        
        self.assertTrue(
            np.allclose(
                self._test_parameters_do_fit_simple,
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit_simple,
                _read_fit.y_model
            )
        )

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))

        self.assertTrue(
            np.allclose(
                self._test_parameters_default,
                _read_fit.parameter_values
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
                _read_fit.parameter_values
            )
        )
        self.assertTrue(
            np.allclose(
                self._test_y_do_fit,
                _read_fit.y_model
            )
        )


TEST_FIT_UNBINNED = """
dataset:
  data: [1.0130614498386674, -1.1349287030839197, -0.10341784343652169, -1.390768642168136, -0.9744105188262357,
         0.6056117355777544, -0.5504328324999447, 0.526390450221898, -1.1119561843051506, 0.8564964441075679,
         -1.4966544904555332, 0.3533780292764612, -2.354860160605102, -1.611507787899526, -0.32604149834680735 ,
         -1.8506525618128302, 0.5587792631003663, 0.8639576320048902, -1.3134430505718173, 2.530614938202886 ,
         -1.1044403418787425, -0.8438885895226831, -0.25603003463019774, -0.4436650881879283, -0.7249153868101906,
         -0.5553144725434371, 0.9361796588314751, 0.9505148635855938, -0.44558819618432904, -0.6020371626047384,
         -0.9187934153258002, -0.7881656831171765, -0.31596343642852404, -0.6590057926479401, 0.8013842370770446,
         -1.3802275845787508, 0.3702055619566078, 0.6100729933463217, 0.9967097423047069, 0.8694021685126967,
         1.6080865393403492, 0.19305357203727197, -0.8846274588919996, 3.0462083207914286, 0.6329501416274081,
         2.1668232578651456, -0.36652740938218903, 0.30092771240660027, 1.7452901977315924, -0.4388534800294709,
         1.3918081589095852, -0.6437491809609527, -0.510287038125732, 0.9606898464368234, 0.41806431328828525,
         0.5957760238098067, -1.2102742486664348, 0.44097254257312235, 0.3412064563213552, -1.7451310421167718,
         0.2081663284234683, -1.1066854012184062, 0.4121145273973152, 1.948498429389019, 1.4469448685636854,
         0.07981317629602669, 1.411704446184254, -0.4610151277847929, 0.9477286127925121, 0.08899785814182719]
fixed_parameters:
    mu: 0.1
parametric_model:
    model_function:
        python_code: |
            def normal_distribution_pdf(x, mu, sigma):
                return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)
    data: [1.0130614498386674, -1.1349287030839197, -0.10341784343652169, -1.390768642168136, -0.9744105188262357,
           0.6056117355777544, -0.5504328324999447, 0.526390450221898, -1.1119561843051506, 0.8564964441075679,
           -1.4966544904555332, 0.3533780292764612, -2.354860160605102, -1.611507787899526, -0.32604149834680735 ,
           -1.8506525618128302, 0.5587792631003663, 0.8639576320048902, -1.3134430505718173, 2.530614938202886 ,
           -1.1044403418787425, -0.8438885895226831, -0.25603003463019774, -0.4436650881879283, -0.7249153868101906,
           -0.5553144725434371, 0.9361796588314751, 0.9505148635855938, -0.44558819618432904, -0.6020371626047384,
           -0.9187934153258002, -0.7881656831171765, -0.31596343642852404, -0.6590057926479401, 0.8013842370770446,
           -1.3802275845787508, 0.3702055619566078, 0.6100729933463217, 0.9967097423047069, 0.8694021685126967,
           1.6080865393403492, 0.19305357203727197, -0.8846274588919996, 3.0462083207914286, 0.6329501416274081,
           2.1668232578651456, -0.36652740938218903, 0.30092771240660027, 1.7452901977315924, -0.4388534800294709,
           1.3918081589095852, -0.6437491809609527, -0.510287038125732, 0.9606898464368234, 0.41806431328828525,
           0.5957760238098067, -1.2102742486664348, 0.44097254257312235, 0.3412064563213552, -1.7451310421167718,
           0.2081663284234683, -1.1066854012184062, 0.4121145273973152, 1.948498429389019, 1.4469448685636854,
           0.07981317629602669, 1.411704446184254, -0.4610151277847929, 0.9477286127925121, 0.08899785814182719]
    model_parameters: [1.5, 1.0]
type: unbinned
"""

TEST_FIT_UNBINNED_EXTRA_KEYWORD = TEST_FIT_UNBINNED + """
extra_keyword: 3.14
"""

TEST_FIT_UNBINNED_SIMPLE = """
type: unbinned
data: [1.0130614498386674, -1.1349287030839197, -0.10341784343652169, -1.390768642168136, -0.9744105188262357,
       0.6056117355777544, -0.5504328324999447, 0.526390450221898, -1.1119561843051506, 0.8564964441075679,
       -1.4966544904555332, 0.3533780292764612, -2.354860160605102, -1.611507787899526, -0.32604149834680735 ,
       -1.8506525618128302, 0.5587792631003663, 0.8639576320048902, -1.3134430505718173, 2.530614938202886 ,
       -1.1044403418787425, -0.8438885895226831, -0.25603003463019774, -0.4436650881879283, -0.7249153868101906,
       -0.5553144725434371, 0.9361796588314751, 0.9505148635855938, -0.44558819618432904, -0.6020371626047384,
       0.5957760238098067, -1.2102742486664348, 0.44097254257312235, 0.3412064563213552, -1.7451310421167718,
       0.2081663284234683, -1.1066854012184062, 0.4121145273973152, 1.948498429389019, 1.4469448685636854,
       0.4645792379289075, 0.08603433240025622, -1.2395727054852086, 0.7213492596560613, 0.3314999995659894,
       0.510785216253824, 0.44208577350556294, -0.6908107349334328, 1.2491129589765388, 0.8363348859267793,
       0.07981317629602669, 1.411704446184254, -0.4610151277847929, 0.9477286127925121, 0.08899785814182719]
"""

TEST_FIT_UNBINNED_MISSING_KEYWORD = """
fixed_parameters:
    mu: 0.1
parametric_model:
    python_code: |
        def normal_distribution_pdf(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)
type: unbinned
"""


class TestUnbinnedFitYamlRepresenter(unittest.TestCase, AbstractTestFitRepresenter):
    FIT_CLASS = UnbinnedFit
    TEST_FIT = TEST_FIT_UNBINNED
    TEST_FIT_SIMPLE = TEST_FIT_UNBINNED_SIMPLE
    TEST_FIT_EXTRA_KEYWORD = TEST_FIT_UNBINNED_EXTRA_KEYWORD
    TEST_FIT_MISSING_KEYWORD = TEST_FIT_UNBINNED_MISSING_KEYWORD

    @staticmethod
    def normal_distribution_pdf(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)

    def setUp(self):
        self._test_data = [1.0130614498386674, -1.1349287030839197, -0.10341784343652169, -1.390768642168136,
                           -0.9744105188262357, 0.6056117355777544, -0.5504328324999447, 0.526390450221898,
                           -1.1119561843051506, 0.8564964441075679, -1.4966544904555332, 0.3533780292764612,
                           -2.354860160605102, -1.611507787899526, -0.32604149834680735, -1.8506525618128302,
                           0.5587792631003663, 0.8639576320048902, -1.3134430505718173, 2.530614938202886,
                           -1.1044403418787425, -0.8438885895226831, -0.25603003463019774, -0.4436650881879283,
                           -0.7249153868101906, -0.5553144725434371, 0.9361796588314751, 0.9505148635855938,
                           -0.44558819618432904, -0.6020371626047384, -0.9187934153258002, -0.7881656831171765,
                           -0.31596343642852404, -0.6590057926479401, 0.8013842370770446, -1.3802275845787508,
                           0.3702055619566078, 0.6100729933463217, 0.9967097423047069, 0.8694021685126967,
                           1.6080865393403492, 0.19305357203727197, -0.8846274588919996, 3.0462083207914286,
                           0.6329501416274081, 2.1668232578651456, -0.36652740938218903, 0.30092771240660027,
                           1.7452901977315924, -0.4388534800294709, 1.3918081589095852, -0.6437491809609527,
                           -0.510287038125732, 0.9606898464368234, 0.41806431328828525, 0.5957760238098067,
                           -1.2102742486664348, 0.44097254257312235, 0.3412064563213552, -1.7451310421167718,
                           0.2081663284234683, -1.1066854012184062, 0.4121145273973152, 1.948498429389019,
                           1.4469448685636854, 0.07981317629602669, 1.411704446184254, -0.4610151277847929,
                           0.9477286127925121, 0.08899785814182719]
        self._test_parameters = np.array([0.1, -1.0])
        self._test_parameters_default = np.array([0.1, 1.0])
        self._test_parameters_default_simple = np.array([1.0, 1.0])
        self._test_parameters_do_fit = np.array([0.1, 1.09727982])
        self._test_parameters_do_fit_simple = np.array([-0.07060323, 1.02375445])

        self._fit = UnbinnedFit(data=self._test_data, model_density_function='normal_distribution_pdf')
        self._fit.fix_parameter('mu', 0.1)

        self.setup_streams()

    def test_read_from_testfile_stream(self):
        _read_fit = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
        self.assertTrue(np.allclose(self._test_parameters_default, _read_fit.parameter_values))
        _read_fit.do_fit()
        self.assertTrue(np.allclose(self._test_parameters_do_fit, _read_fit.parameter_values))

    def test_read_from_testfile_stream_simple(self):
        _read_fit = self._testfile_streamreader_simple.read()

        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
        self.assertTrue(np.allclose(
            self._test_parameters_default_simple, _read_fit.parameter_values))
        _read_fit.do_fit()
        self.assertTrue(np.allclose(
            self._test_parameters_do_fit_simple, _read_fit.parameter_values))

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_fit = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_fit, self.FIT_CLASS))
        self.assertTrue(np.allclose(self._test_parameters_default, _read_fit.parameter_values))
        _read_fit.do_fit()
        self.assertTrue(np.allclose(self._test_parameters_do_fit, _read_fit.parameter_values))
