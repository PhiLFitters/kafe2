import unittest
from six import StringIO

from kafe2.fit.io import IOStreamHandle
from kafe2.fit.representation import ConstraintYamlReader, ConstraintYamlWriter
from kafe2.fit.representation._yaml_base import YamlReaderException
from kafe2.fit._base.constraint import GaussianSimpleParameterConstraint, GaussianMatrixParameterConstraint

TEST_SIMPLE_GAUSSIAN_CONSTRAINT = """
type: simple
index: 3
value: 2.3
uncertainty: 1.2
"""

TEST_SIMPLE_GAUSSIAN_CONSTRAINT_MISSING_KEYWORD = """
type: simple
index: 3
uncertainty: 1.2
"""

TEST_SIMPLE_GAUSSIAN_CONSTRAINT_EXTRA_KEYWORD = TEST_SIMPLE_GAUSSIAN_CONSTRAINT + """
extra_keyword: 3.14
"""


class TestSimpleGaussianConstraintRepresenter(unittest.TestCase):

    def setUp(self):
        self._index = 3
        self._value = 2.3
        self._uncertainty = 1.2
        self._constraint = GaussianSimpleParameterConstraint(index=self._index, value=self._value,
                                                             uncertainty=self._uncertainty)

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_SIMPLE_GAUSSIAN_CONSTRAINT))

        self._roundtrip_streamreader = ConstraintYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ConstraintYamlWriter(self._constraint, self._roundtrip_stringstream)
        self._testfile_streamreader = ConstraintYamlReader(self._testfile_stringstream)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(
            StringIO(TEST_SIMPLE_GAUSSIAN_CONSTRAINT_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(
            StringIO(TEST_SIMPLE_GAUSSIAN_CONSTRAINT_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ConstraintYamlReader(
            self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ConstraintYamlReader(
            self._testfile_stringstream_extra_keyword)

    def _assert_constraints_equal(self, constraint1, constraint2):
        self.assertTrue(constraint1.index == constraint2.index)
        self.assertTrue(constraint1.value == constraint2.value)
        self.assertTrue(constraint1.uncertainty == constraint2.uncertainty)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_constraint = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_constraint, GaussianSimpleParameterConstraint))
        self._assert_constraints_equal(_read_constraint, self._constraint)

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_constraint = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_constraint, GaussianSimpleParameterConstraint))
        self._assert_constraints_equal(_read_constraint, self._constraint)


TEST_MATRIX_GAUSSIAN_CONSTRAINT = """
type: matrix
indices: [3, 0, 1]
values: [2.3, 5.6, 7.8]
cov_mat: [[1.2, 0.1, 0.2], [0.1, 3.4, 0.3], [0.2, 0.3, 9.0]]
"""

TEST_MATRIX_GAUSSIAN_CONSTRAINT_MISSING_KEYWORD = """
type: matrix
values: [2.3, 5.6, 7.8]
cov_mat: [[1.2, 0.1, 0.2], [0.1, 3.4, 0.3], [0.2, 0.3, 9.0]]
"""

TEST_MATRIX_GAUSSIAN_CONSTRAINT_EXTRA_KEYWORD = TEST_MATRIX_GAUSSIAN_CONSTRAINT + """
extra_keyword: 3.14
"""


class TestMatrixGaussianConstraintRepresenter(unittest.TestCase):

    def setUp(self):
        self._indices = [3, 0, 1]
        self._values = [2.3, 5.6, 7.8]
        self._cov_mat = [[1.2, 0.1, 0.2], [0.1, 3.4, 0.3], [0.2, 0.3, 9.0]]
        self._constraint = GaussianMatrixParameterConstraint(indices=self._indices, values=self._values,
                                                             cov_mat=self._cov_mat)

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT))

        self._roundtrip_streamreader = ConstraintYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ConstraintYamlWriter(self._constraint, self._roundtrip_stringstream)
        self._testfile_streamreader = ConstraintYamlReader(self._testfile_stringstream)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(
            StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(
            StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ConstraintYamlReader(
            self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ConstraintYamlReader(
            self._testfile_stringstream_extra_keyword)

    def _assert_constraints_equal(self, constraint_1, constraint_2):
        for _i in range(3):
            self.assertTrue(constraint_1.indices[_i] == constraint_2.indices[_i])
            self.assertTrue(constraint_1.values[_i] == constraint_2.values[_i])
            for _j in range(3):
                self.assertTrue(constraint_1.cov_mat[_i, _j] == constraint_2.cov_mat[_i, _j])

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_constraint = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_constraint, GaussianMatrixParameterConstraint))
        self._assert_constraints_equal(_read_constraint, self._constraint)

    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_constraint = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_constraint, GaussianMatrixParameterConstraint))
        self._assert_constraints_equal(_read_constraint, self._constraint)
