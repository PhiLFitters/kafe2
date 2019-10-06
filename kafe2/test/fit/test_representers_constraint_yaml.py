import unittest2 as unittest
from six import StringIO
import numpy as np

from kafe2.fit.io import IOStreamHandle
from kafe2.fit.representation import ConstraintYamlReader, ConstraintYamlWriter
from kafe2.fit.representation._yaml_base import YamlReaderException
from kafe2.core.constraint import GaussianSimpleParameterConstraint, GaussianMatrixParameterConstraint

TEST_SIMPLE_GAUSSIAN_CONSTRAINT_ABS = """
type: simple
index: 3
value: 2.0
uncertainty: 1.2
"""

TEST_SIMPLE_GAUSSIAN_CONSTRAINT_REL = """
type: simple
index: 3
value: 2.0
uncertainty: 0.6
relative: True
"""

TEST_SIMPLE_GAUSSIAN_CONSTRAINT_MISSING_KEYWORD = """
type: simple
index: 3
uncertainty: 1.2
"""

TEST_SIMPLE_GAUSSIAN_CONSTRAINT_EXTRA_KEYWORD = TEST_SIMPLE_GAUSSIAN_CONSTRAINT_ABS + """
extra_keyword: 3.14
"""


class TestSimpleGaussianConstraintRepresenter(unittest.TestCase):

    def setUp(self):
        self._index = 3
        self._value = 2.0
        self._uncertainty = 1.2
        self._constraint = GaussianSimpleParameterConstraint(index=self._index, value=self._value,
                                                             uncertainty=self._uncertainty)

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream_abs = IOStreamHandle(StringIO(TEST_SIMPLE_GAUSSIAN_CONSTRAINT_ABS))
        self._testfile_stringstream_rel = IOStreamHandle(StringIO(TEST_SIMPLE_GAUSSIAN_CONSTRAINT_REL))

        self._roundtrip_streamreader = ConstraintYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ConstraintYamlWriter(self._constraint, self._roundtrip_stringstream)
        self._testfile_streamreader_abs = ConstraintYamlReader(self._testfile_stringstream_abs)
        self._testfile_streamreader_rel = ConstraintYamlReader(self._testfile_stringstream_rel)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(
            StringIO(TEST_SIMPLE_GAUSSIAN_CONSTRAINT_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(
            StringIO(TEST_SIMPLE_GAUSSIAN_CONSTRAINT_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ConstraintYamlReader(
            self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ConstraintYamlReader(
            self._testfile_stringstream_extra_keyword)

    def _assert_constraints_equal(self, constraint_1, constraint_2):
        self.assertTrue(constraint_1.index == constraint_2.index)
        self.assertTrue(constraint_1.value == constraint_2.value)
        self.assertTrue(constraint_1.uncertainty == constraint_2.uncertainty)
        self.assertTrue(constraint_1.uncertainty_rel == constraint_2.uncertainty_rel)

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_constraint = self._testfile_streamreader_abs.read()
        self.assertTrue(isinstance(_read_constraint, GaussianSimpleParameterConstraint))
        self._assert_constraints_equal(_read_constraint, self._constraint)
        _read_constraint = self._testfile_streamreader_rel.read()
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


TEST_MATRIX_GAUSSIAN_CONSTRAINT_COV_ABS = """
type: matrix
indices: [3, 0, 1]
values: [1.0, 10.0, 100.0]
matrix: [[0.01, 0.01, 0.2], [0.01, 1.0, 3.0], [0.2, 3.0, 100.0]]
"""

TEST_MATRIX_GAUSSIAN_CONSTRAINT_COV_REL = """
type: matrix
indices: [3, 0, 1]
values: [1.0, 10.0, 100.0]
matrix: [[0.01, 0.001, 0.002], [0.001, 0.01, 0.003], [0.002, 0.003, 0.01]]
relative: True
"""

TEST_MATRIX_GAUSSIAN_CONSTRAINT_COR_ABS = """
type: matrix
indices: [3, 0, 1]
values: [1.0, 10.0, 100.0]
matrix: [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]]
matrix_type: cor
uncertainties: [0.1, 1.0, 10.0]
"""

TEST_MATRIX_GAUSSIAN_CONSTRAINT_COR_REL = """
type: matrix
indices: [3, 0, 1]
values: [1.0, 10.0, 100.0]
matrix: [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]]
matrix_type: cor
uncertainties: [0.1, 0.1, 0.1]
relative: True
"""

TEST_MATRIX_GAUSSIAN_CONSTRAINT_MISSING_KEYWORD = """
type: matrix
values: [1.0, 10.0, 100.0]
matrix: [[0.1, 0.1, 2.0], [0.1, 10.0, 30.0], [2.0, 30.0, 1000.0]]
"""

TEST_MATRIX_GAUSSIAN_CONSTRAINT_EXTRA_KEYWORD = TEST_MATRIX_GAUSSIAN_CONSTRAINT_COV_ABS + """
extra_keyword: 3.14
"""


class TestMatrixGaussianConstraintRepresenter(unittest.TestCase):

    def setUp(self):
        self._indices = [3, 0, 1]
        self._values = [1.0, 10.0, 100.0]
        self._cov_mat = [[0.01, 0.01, 0.2], [0.01, 1.0, 3.0], [0.2, 3.0, 100.0]]
        self._constraint = GaussianMatrixParameterConstraint(indices=self._indices, values=self._values,
                                                             matrix=self._cov_mat)

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream_cov_abs = IOStreamHandle(StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT_COV_ABS))
        self._testfile_stringstream_cov_rel = IOStreamHandle(StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT_COV_REL))
        self._testfile_stringstream_cor_abs = IOStreamHandle(StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT_COR_ABS))
        self._testfile_stringstream_cor_rel = IOStreamHandle(StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT_COR_REL))

        self._roundtrip_streamreader = ConstraintYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = ConstraintYamlWriter(self._constraint, self._roundtrip_stringstream)
        self._testfile_streamreader_cov_abs = ConstraintYamlReader(self._testfile_stringstream_cov_abs)
        self._testfile_streamreader_cov_rel = ConstraintYamlReader(self._testfile_stringstream_cov_rel)
        self._testfile_streamreader_cor_abs = ConstraintYamlReader(self._testfile_stringstream_cor_abs)
        self._testfile_streamreader_cor_rel = ConstraintYamlReader(self._testfile_stringstream_cor_rel)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(
            StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(
            StringIO(TEST_MATRIX_GAUSSIAN_CONSTRAINT_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = ConstraintYamlReader(
            self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = ConstraintYamlReader(
            self._testfile_stringstream_extra_keyword)

    def _assert_constraints_equal(self, constraint_1, constraint_2):
        self.assertTrue(np.all(constraint_1.indices == constraint_2.indices))
        self.assertTrue(np.all(constraint_1.values == constraint_2.values))
        self.assertTrue(np.allclose(constraint_1.cov_mat, constraint_2.cov_mat))
        self.assertTrue(np.allclose(constraint_1.cov_mat_rel, constraint_2.cov_mat_rel))
        self.assertTrue(np.allclose(constraint_1.cor_mat, constraint_2.cor_mat))
        self.assertTrue(np.allclose(constraint_1.uncertainties, constraint_2.uncertainties))
        self.assertTrue(np.allclose(constraint_1.uncertainties_rel, constraint_2.uncertainties_rel))

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_constraint = self._testfile_streamreader_cov_abs.read()
        self.assertTrue(isinstance(_read_constraint, GaussianMatrixParameterConstraint))
        self._assert_constraints_equal(_read_constraint, self._constraint)
        _read_constraint = self._testfile_streamreader_cov_rel.read()
        self.assertTrue(isinstance(_read_constraint, GaussianMatrixParameterConstraint))
        self._assert_constraints_equal(_read_constraint, self._constraint)
        _read_constraint = self._testfile_streamreader_cor_abs.read()
        self.assertTrue(isinstance(_read_constraint, GaussianMatrixParameterConstraint))
        self._assert_constraints_equal(_read_constraint, self._constraint)
        _read_constraint = self._testfile_streamreader_cor_rel.read()
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
