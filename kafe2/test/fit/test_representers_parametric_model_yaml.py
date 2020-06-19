import unittest2 as unittest
import numpy as np
from six import StringIO

from kafe2.fit._base import ModelFunctionBase
from kafe2.fit.histogram import HistModelFunction
from kafe2.fit.indexed import IndexedModelFunction
from kafe2.fit.representation import ParametricModelYamlWriter, ParametricModelYamlReader
from kafe2.fit.histogram.model import HistParametricModel
from kafe2.fit.indexed.model import IndexedParametricModel
from kafe2.fit.xy.model import XYParametricModel
from kafe2.fit.io.handle import IOStreamHandle
from kafe2.fit.representation._yaml_base import YamlReaderException

TEST_PARAMETRIC_MODEL_HIST="""
type: histogram
n_bins: 5
bin_range: [0, 5]
model_density_function:
    type: histogram
    python_code: |
        def linear_model(x, a, b):
            return a * x + b
model_parameters: [0.0, 0.08]
"""

TEST_PARAMETRIC_MODEL_HIST_MISSING_KEYWORD="""
type: histogram
n_bins: 5
bin_range: [0, 5]
model_parameters: [0.0, 0.08]
"""

TEST_PARAMETRIC_MODEL_HIST_EXTRA_KEYWORD = TEST_PARAMETRIC_MODEL_HIST + """
extra_keyword: 3.14
"""

TEST_PARAMETRIC_MODEL_HIST_WITH_ERRORS = TEST_PARAMETRIC_MODEL_HIST + """
errors:
  - correlation_coefficient: 0.0
    error_value: 0.1
    name: test_error
    relative: false
    type: simple
"""

TEST_PARAMETRIC_MODEL_HIST_LEGACY = TEST_PARAMETRIC_MODEL_HIST + """
model_density_func_antiderivative: null
"""

TEST_PARAMETRIC_MODEL_HIST_ANTIDERIVATIVE = TEST_PARAMETRIC_MODEL_HIST + """
bin_evaluation: |
    def linear_model_antiderivative(x, a, b):
        return 0.5 * a * x ** 2 + b * x
"""

TEST_PARAMETRIC_MODEL_HIST_NUMPY_VECTORIZE = TEST_PARAMETRIC_MODEL_HIST + """
bin_evaluation: |
    @np.vectorize
    def linear_model_antiderivative(x, a, b):
        return 0.5 * a * x ** 2 + b * x
"""


class TestHistParametricModelYamlRepresenter(unittest.TestCase):

    def _assert_read_model_correct(self, read_model, ref_model, errors=False):
        self.assertTrue(isinstance(read_model, HistParametricModel))

        self.assertTrue(np.all(
            read_model.data == ref_model.data
        ))
        if isinstance(ref_model.bin_evaluation, str):
            self.assertEqual(read_model.bin_evaluation, ref_model.bin_evaluation)
            self.assertEqual(read_model.bin_evaluation_string, ref_model.bin_evaluation_string)
        self.assertTrue(np.all(
            read_model.parameters == self._test_model_parameters
        ))
        self.assertTrue(read_model.n_bins == self._test_n_bins)
        self.assertTrue(np.all(
            read_model.bin_range == self._test_bin_range
        ))
        self.assertTrue(np.all(
            read_model.bin_edges == self._test_bin_edges
        ))
        self.assertTrue(np.allclose(
                read_model.eval_model_function_density(self._test_x),
                ref_model.eval_model_function_density(self._test_x)
        ))

        if errors:
            _given_error = self._test_parametric_model_with_errors.get_error('test_error')
            _read_error = read_model.get_error('test_error')
            self.assertTrue(_given_error['enabled'] == _read_error['enabled'])
            self.assertTrue(
                np.allclose(
                    _given_error['err'].error,
                    _read_error['err'].error
                )
            )
            self.assertTrue(_given_error['err'].corr_coeff == _read_error['err'].corr_coeff)

    @staticmethod
    def _get_model_from_string(input_string):
        _testfile_stringstream = IOStreamHandle(StringIO(input_string))
        return ParametricModelYamlReader(_testfile_stringstream).read()

    @staticmethod
    def _get_model_from_roundtrip(model):
        _roundtrip_stringstream = IOStreamHandle(StringIO())
        _roundtrip_streamreader = ParametricModelYamlReader(_roundtrip_stringstream)
        _roundtrip_streamwriter = ParametricModelYamlWriter(model, _roundtrip_stringstream)
        _roundtrip_streamwriter.write()
        _roundtrip_stringstream.seek(0)  # return to beginning
        return _roundtrip_streamreader.read()


    @staticmethod
    def linear_model(x, a, b):
        return a * x + b

    @staticmethod
    def linear_model_antiderivative(x, a, b):
        return 0.5 * a * x ** 2 + b * x

    @staticmethod
    @np.vectorize
    def linear_model_numpy_vectorize(x, a, b):
        return 0.5 * a * x ** 2 + b * x

    def setUp(self):
        self._test_n_bins = 5
        self._test_bin_range = (0, 5)
        self._test_bin_edges = np.arange(6)
        self._test_model_parameters = np.array([0.0, 0.08])
        self._test_x = np.linspace(start=0.0, stop=5.0, num=11, endpoint=True)
        self._test_parametric_model = HistParametricModel(
            n_bins=self._test_n_bins, 
            bin_range=self._test_bin_range,
            model_density_func=HistModelFunction(
                TestHistParametricModelYamlRepresenter.linear_model),
            model_parameters=self._test_model_parameters
        )

        self._test_parametric_model_with_errors = HistParametricModel(
            n_bins=self._test_n_bins, 
            bin_range=self._test_bin_range,
            model_density_func=HistModelFunction(
                TestHistParametricModelYamlRepresenter.linear_model),
            model_parameters=self._test_model_parameters
        )
        self._test_parametric_model_with_errors.add_error(
            err_val=0.1, 
            name='test_error', 
            correlation=0, 
            relative=False
        )
        
        self._test_parametric_model_antiderivative = HistParametricModel(
            n_bins=self._test_n_bins,
            bin_range=self._test_bin_range,
            model_density_func=HistModelFunction(
                TestHistParametricModelYamlRepresenter.linear_model),
            model_parameters=self._test_model_parameters,
            bin_evaluation=TestHistParametricModelYamlRepresenter.linear_model_antiderivative
        )

        self._test_parametric_model_numpy_vectorize = HistParametricModel(
            n_bins=self._test_n_bins,
            bin_range=self._test_bin_range,
            model_density_func=HistModelFunction(
                TestHistParametricModelYamlRepresenter.linear_model),
            model_parameters=self._test_model_parameters,
            bin_evaluation=TestHistParametricModelYamlRepresenter.linear_model_numpy_vectorize
        )

    def test_read_from_testfile_stream(self):
        _read_parametric_model = TestHistParametricModelYamlRepresenter._get_model_from_string(
            TEST_PARAMETRIC_MODEL_HIST)
        self._assert_read_model_correct(_read_parametric_model, self._test_parametric_model)

    def test_read_from_testfile_stream_antiderivative(self):
        _read_parametric_model = TestHistParametricModelYamlRepresenter._get_model_from_string(
            TEST_PARAMETRIC_MODEL_HIST_ANTIDERIVATIVE)
        self._assert_read_model_correct(
            _read_parametric_model, self._test_parametric_model_antiderivative)

    def test_read_from_testfile_stream_numpy_vectorize(self):
        _read_parametric_model = TestHistParametricModelYamlRepresenter._get_model_from_string(
            TEST_PARAMETRIC_MODEL_HIST_NUMPY_VECTORIZE)
        self._assert_read_model_correct(
            _read_parametric_model, self._test_parametric_model_numpy_vectorize)

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            _read_parametric_model = TestHistParametricModelYamlRepresenter._get_model_from_string(
                TEST_PARAMETRIC_MODEL_HIST_MISSING_KEYWORD)

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            _read_parametric_model = TestHistParametricModelYamlRepresenter._get_model_from_string(
                TEST_PARAMETRIC_MODEL_HIST_EXTRA_KEYWORD)

    def test_read_from_testfile_stream_with_errors(self):
        _read_parametric_model = TestHistParametricModelYamlRepresenter._get_model_from_string(
            TEST_PARAMETRIC_MODEL_HIST_WITH_ERRORS)
        self._assert_read_model_correct(
            _read_parametric_model, self._test_parametric_model_with_errors, errors=True)

    def test_read_from_testfile_stream_legacy(self):
        _read_parametric_model = TestHistParametricModelYamlRepresenter._get_model_from_string(
            TEST_PARAMETRIC_MODEL_HIST_LEGACY)
        self._assert_read_model_correct(_read_parametric_model, self._test_parametric_model)

    def test_round_trip_with_stringstream(self):
        _read_parametric_model = self._get_model_from_roundtrip(self._test_parametric_model)
        self._assert_read_model_correct(_read_parametric_model, self._test_parametric_model)

    def test_round_trip_with_stringstream_antiderivative(self):
        _read_parametric_model = self._get_model_from_roundtrip(
            self._test_parametric_model_antiderivative)
        self._assert_read_model_correct(
            _read_parametric_model, self._test_parametric_model_antiderivative)

    def test_round_trip_with_stringstream_numpy_vectorize(self):
        _read_parametric_model = self._get_model_from_roundtrip(
            self._test_parametric_model_numpy_vectorize)
        self._assert_read_model_correct(
            _read_parametric_model, self._test_parametric_model_numpy_vectorize)

    def test_round_trip_with_stringstream_with_errors(self):
        _read_parametric_model = self._get_model_from_roundtrip(
            self._test_parametric_model_with_errors)
        self._assert_read_model_correct(
            _read_parametric_model, self._test_parametric_model_with_errors, errors=True)


TEST_PARAMETRIC_MODEL_INDEXED="""
type: indexed
model_function:
    type: indexed
    python_code: |
        def linear_model(a, b):
            return a * np.arange(6) + b
model_parameters: [1.1, -1.5]
"""

TEST_PARAMETRIC_MODEL_INDEXED_MISSING_KEYWORD="""
type: indexed
model_parameters: [1.1, -1.5]
"""

TEST_PARAMETRIC_MODEL_INDEXED_EXTRA_KEYWORD = TEST_PARAMETRIC_MODEL_INDEXED + """
extra_keyword: 3.14
"""

TEST_PARAMETRIC_MODEL_INDEXED_WITH_ERRORS = TEST_PARAMETRIC_MODEL_INDEXED + """
errors:
  - correlation_coefficient: 0.0
    error_value: 0.1
    name: test_error
    relative: false
    type: simple
"""

class TestIndexedParametricModelYamlRepresenter(unittest.TestCase):

    TEST_X = np.arange(6)
        
    @staticmethod
    def linear_model(a, b):
        return a * np.arange(6) + b
   
    def setUp(self):
        self._test_model_parameters = np.array([1.1, -1.5])
        self._test_parametric_model = IndexedParametricModel(
            IndexedModelFunction(TestIndexedParametricModelYamlRepresenter.linear_model),
            self._test_model_parameters
        )

        self._test_parametric_model_with_errors = IndexedParametricModel(
            IndexedModelFunction(TestIndexedParametricModelYamlRepresenter.linear_model),
            self._test_model_parameters
        )
        self._test_parametric_model_with_errors.add_error(
            err_val=0.1, 
            name='test_x_error', 
            correlation=0, 
            relative=False
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._roundtrip_stringstream_with_errors = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_INDEXED))
        self._testfile_stringstream_with_errors = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_INDEXED_WITH_ERRORS))
        
        self._roundtrip_streamreader = ParametricModelYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamreader_with_errors = ParametricModelYamlReader(self._roundtrip_stringstream_with_errors)
        self._roundtrip_streamwriter = ParametricModelYamlWriter(
            self._test_parametric_model, 
            self._roundtrip_stringstream
        )
        self._roundtrip_streamwriter_with_errors = ParametricModelYamlWriter(
            self._test_parametric_model_with_errors, 
            self._roundtrip_stringstream_with_errors
        )
        self._testfile_streamreader = ParametricModelYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_errors = ParametricModelYamlReader(self._testfile_stringstream_with_errors)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_INDEXED_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_INDEXED_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ParametricModelYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ParametricModelYamlReader(self._testfile_stringstream_extra_keyword)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_write_to_roundtrip_stringstream_with_errors(self):
        self._roundtrip_streamwriter_with_errors.write()

    def test_read_from_testfile_stream(self):
        _read_parametric_model = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_parametric_model, IndexedParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        self.assertTrue(
            np.allclose(
                _read_parametric_model.eval_model_function(),
                self._test_parametric_model.eval_model_function()
            )
        )
        
    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_read_from_testfile_stream_with_errors(self):
        _read_parametric_model = self._testfile_streamreader_with_errors.read()
        self.assertTrue(isinstance(_read_parametric_model, IndexedParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        self.assertTrue(
            np.allclose(
                _read_parametric_model.eval_model_function(),
                self._test_parametric_model_with_errors.eval_model_function()
            )
        )
        
        _given_error = self._test_parametric_model_with_errors.get_error('test_x_error')
        _read_error = _read_parametric_model.get_error('test_error')
        self.assertTrue(_given_error['enabled'] == _read_error['enabled'])
        self.assertTrue(
            np.allclose(
                _given_error['err'].error,
                _read_error['err'].error
            )
        )
        self.assertTrue(_given_error['err'].corr_coeff == _read_error['err'].corr_coeff)
        
    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_parametric_model = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_parametric_model, IndexedParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        self.assertTrue(
            np.allclose(
                _read_parametric_model.eval_model_function(),
                self._test_parametric_model.eval_model_function()
            )
        )
        
    def test_round_trip_with_stringstream_with_errors(self):
        self._roundtrip_streamwriter_with_errors.write()
        self._roundtrip_stringstream_with_errors.seek(0)  # return to beginning
        _read_parametric_model = self._roundtrip_streamreader_with_errors.read()
        self.assertTrue(isinstance(_read_parametric_model, IndexedParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        self.assertTrue(
            np.allclose(
                _read_parametric_model.eval_model_function(),
                self._test_parametric_model_with_errors.eval_model_function()
            )
        )
        
        _given_error = self._test_parametric_model_with_errors.get_error('test_x_error')
        _read_error = _read_parametric_model.get_error('test_x_error')
        self.assertTrue(_given_error['enabled'] == _read_error['enabled'])
        self.assertTrue(
            np.allclose(
                _given_error['err'].error,
                _read_error['err'].error
            )
        )
        self.assertTrue(_given_error['err'].corr_coeff == _read_error['err'].corr_coeff)

TEST_PARAMETRIC_MODEL_XY="""
type: xy
x_data: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
model_function:
    type: xy
    python_code: |
        def linear_model(x, a, b):
            return a * x + b
model_parameters: [1.1, -1.5]
"""

TEST_PARAMETRIC_MODEL_XY_MISSING_KEYWORD="""
type: xy
model_function:
    type: xy
    python_code: |
        def linear_model(x, a, b):
            return a * x + b
model_parameters: [1.1, -1.5]
"""

TEST_PARAMETRIC_MODEL_XY_EXTRA_KEYWORD = TEST_PARAMETRIC_MODEL_XY + """
extra_keyword: 3.14
"""

TEST_PARAMETRIC_MODEL_XY_WITH_ERRORS = TEST_PARAMETRIC_MODEL_XY + """
x_errors:
  - correlation_coefficient: 0.0
    error_value: 0.1
    name: test_x_error
    relative: false
    type: simple
"""

class TestXYParametricModelYamlRepresenter(unittest.TestCase):

    @staticmethod
    def linear_model(x, a, b):
        return a * x + b
   
    def setUp(self):
        self._test_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self._test_model_parameters = np.array([1.1, -1.5])
        self._test_parametric_model = XYParametricModel(
            self._test_x, 
            ModelFunctionBase(TestXYParametricModelYamlRepresenter.linear_model),
            self._test_model_parameters
        )

        self._test_parametric_model_with_errors = XYParametricModel(
            self._test_x, 
            ModelFunctionBase(TestXYParametricModelYamlRepresenter.linear_model),
            self._test_model_parameters
        )
        self._test_parametric_model_with_errors.add_error(
            axis='x',
            err_val=0.1, 
            name='test_x_error', 
            correlation=0, 
            relative=False
        )
        
        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._roundtrip_stringstream_with_errors = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_XY))
        self._testfile_stringstream_with_errors = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_XY_WITH_ERRORS))
        
        self._roundtrip_streamreader = ParametricModelYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamreader_with_errors = ParametricModelYamlReader(self._roundtrip_stringstream_with_errors)
        self._roundtrip_streamwriter = ParametricModelYamlWriter(
            self._test_parametric_model, 
            self._roundtrip_stringstream
        )
        self._roundtrip_streamwriter_with_errors = ParametricModelYamlWriter(
            self._test_parametric_model_with_errors, 
            self._roundtrip_stringstream_with_errors
        )
        self._testfile_streamreader = ParametricModelYamlReader(self._testfile_stringstream)
        self._testfile_streamreader_with_errors = ParametricModelYamlReader(self._testfile_stringstream_with_errors)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_XY_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_PARAMETRIC_MODEL_XY_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ParametricModelYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ParametricModelYamlReader(self._testfile_stringstream_extra_keyword)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_write_to_roundtrip_stringstream_with_errors(self):
        self._roundtrip_streamwriter_with_errors.write()

    def test_read_from_testfile_stream(self):
        _read_parametric_model = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_parametric_model, XYParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        _read_parametric_model.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _read_parametric_model.y,
                self._test_parametric_model.y
            )
        )
        
    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_read_from_testfile_stream_with_errors(self):
        _read_parametric_model = self._testfile_streamreader_with_errors.read()
        self.assertTrue(isinstance(_read_parametric_model, XYParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        _read_parametric_model.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _read_parametric_model.y,
                self._test_parametric_model_with_errors.y
            )
        )
        
        _given_error = self._test_parametric_model_with_errors.get_error('test_x_error')
        _read_error = _read_parametric_model.get_error('test_x_error')
        self.assertTrue(_given_error['axis'] == _read_error['axis'])
        self.assertTrue(_given_error['enabled'] == _read_error['enabled'])
        self.assertTrue(
            np.allclose(
                _given_error['err'].error,
                _read_error['err'].error
            )
        )
        self.assertTrue(_given_error['err'].corr_coeff == _read_error['err'].corr_coeff)
        
    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_parametric_model = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_parametric_model, XYParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        _read_parametric_model.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _read_parametric_model.y,
                self._test_parametric_model.y
            )
        )
        
    def test_round_trip_with_stringstream_with_errors(self):
        self._roundtrip_streamwriter_with_errors.write()
        self._roundtrip_stringstream_with_errors.seek(0)  # return to beginning
        _read_parametric_model = self._roundtrip_streamreader_with_errors.read()
        self.assertTrue(isinstance(_read_parametric_model, XYParametricModel))
        
        self.assertTrue(
            np.allclose(
                _read_parametric_model.parameters,
                self._test_model_parameters
            )
        )
        _read_parametric_model.x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertTrue(
            np.allclose(
                _read_parametric_model.y,
                self._test_parametric_model_with_errors.y
            )
        )
        
        _given_error = self._test_parametric_model_with_errors.get_error('test_x_error')
        _read_error = _read_parametric_model.get_error('test_x_error')
        self.assertTrue(_given_error['axis'] == _read_error['axis'])
        self.assertTrue(_given_error['enabled'] == _read_error['enabled'])
        self.assertTrue(
            np.allclose(
                _given_error['err'].error,
                _read_error['err'].error
            )
        )
        self.assertTrue(_given_error['err'].corr_coeff == _read_error['err'].corr_coeff)
