import unittest2 as unittest
import numpy as np

from six import StringIO

from kafe2.fit import IndexedContainer, XYContainer, HistContainer
from kafe2.fit.representation import DataContainerYamlWriter, DataContainerYamlReader, DReprError
from kafe2.fit.io.handle import IOStreamHandle
from kafe2.fit.representation._yaml_base import YamlReaderException


# TODO: check that DReprError is raised when appropriate...


TEST_DATASET_INDEXED = """
type: "indexed"
data: [80.429, 80.339]
errors:
- matrix:
  - - 0.1
    - 0.0
  - - 0.0
    - 0.1
  name: ErrorOne
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !matrix |
    0.2   0.1
    0.1   0.2
  name: ErrorTwo
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !symmetric_matrix |
    0.3
    0.1   0.3
  name: ErrorThree
  matrix_type: covariance
  relative: false
  type: matrix
- matrix:
    [[0.2,  0.1],
     [0.1,  0.2]]
  name: ErrorFour
  matrix_type: covariance
  relative: false
  type: matrix
"""

TEST_DATASET_INDEXED_MISSING_KEYWORD = """
type: "indexed"
errors:
- matrix:
  - - 0.1
    - 0.0
  - - 0.0
    - 0.1
  name: ErrorOne
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !matrix |
    0.2   0.1
    0.1   0.2
  name: ErrorTwo
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !symmetric_matrix |
    0.3
    0.1   0.3
  name: ErrorThree
  matrix_type: covariance
  relative: false
  type: matrix
- matrix:
    [[0.2,  0.1],
     [0.1,  0.2]]
  name: ErrorFour
  matrix_type: covariance
  relative: false
  type: matrix
"""

TEST_DATASET_INDEXED_EXTRA_KEYWORD = TEST_DATASET_INDEXED + """
extra_keyword: 3.14
"""

class TestIndexedContainerYamlRepresentation(unittest.TestCase):

    def setUp(self):
        _data = [4, 5, 6, 2, 4, 5., 3.01e-5, 4.4e+45]
        _ndat = len(_data)
        self._container = IndexedContainer(data=_data)

        self._container.add_error(err_val=0.1, name="SUA", correlation=0.0, relative=False)
        self._container.add_error(err_val=0.1, name="SUR", correlation=0.0, relative=True)
        self._container.add_error(err_val=0.1, name="SCA", correlation=1.0, relative=False)
        self._container.add_matrix_error(err_matrix=np.eye(_ndat) * 0.1, name="MCov", relative=False, matrix_type='covariance')
        self._container.add_matrix_error(err_matrix=np.eye(_ndat), name="MCor", relative=False, matrix_type='correlation', err_val=0.1)

        # # xy
        # self._container.add_simple_error(axis='x', err_val=0.1, correlation=0.0, relative=False)
        # self._container.add_simple_error(axis='y', err_val=0.2, correlation=0.0, relative=False)
        # self._container.add_matrix_error(axis='y', err_matrix=np.eye(8) * 0.1, relative=False, matrix_type='covariance')

        # # hist
        # self._container.add_simple_error(err_val=2, correlation=0.5, relative=False)

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_DATASET_INDEXED))

        self._roundtrip_streamreader = DataContainerYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = DataContainerYamlWriter(self._container, self._roundtrip_stringstream)
        self._testfile_streamreader = DataContainerYamlReader(self._testfile_stringstream)

        self._ref_testfile_data = [80.429, 80.339]
        self._ref_testfile_err = [0.89442719,  0.89442719]
        self._ref_testfile_cov_mat = np.array([[0.8, 0.3], [0.3, 0.8]])
        self._ref_testfile_error_names = {'ErrorOne', 'ErrorTwo', 'ErrorThree', 'ErrorFour'}

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_DATASET_INDEXED_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_DATASET_INDEXED_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = DataContainerYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = DataContainerYamlReader(self._testfile_stringstream_extra_keyword)


    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()


    def test_read_from_testfile_stream(self):
        _read_container = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_container, IndexedContainer))
        self.assertTrue(
            np.allclose(
                _read_container.data,
                self._ref_testfile_data
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.err,
                self._ref_testfile_err
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.cov_mat,
                self._ref_testfile_cov_mat,
            )
        )

        # check that the error names are the same
        self.assertEqual(
            set(_read_container._error_dicts.keys()),
            self._ref_testfile_error_names
        )


    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_container = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_container, IndexedContainer))

        # compare data members
        self.assertTrue(
            np.allclose(
                self._container.data,
                _read_container.data
            )
        )

        # compare (total) errors and cov mats
        # TODO: compare individual error components -> nontrivial because error ids change...
        self.assertTrue(
            np.allclose(
                self._container.err,
                _read_container.err
            )
        )

        self.assertTrue(
            np.allclose(
                self._container.cov_mat,
                _read_container.cov_mat
            )
        )

        # check that the error names are the same
        self.assertEqual(
            set(self._container._error_dicts.keys()),
            set(_read_container._error_dicts.keys())
        )


TEST_DATASET_XY = """
type: "xy"
x_data: [5, 17]
y_data: [80.429, 80.339]
y_errors:
- matrix:
  - - 0.1
    - 0.0
  - - 0.0
    - 0.1
  name: YErrorOne
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !matrix |
    0.2   0.1
    0.1   0.2
  name: YErrorTwo
  matrix_type: covariance
  relative: false
  type: matrix
x_errors:
- matrix: !symmetric_matrix |
    0.3
    0.1   0.3
  name: XErrorOne
  matrix_type: covariance
  relative: false
  type: matrix
- matrix:
    [[0.2,  0.1],
     [0.1,  0.2]]
  name: XErrorTwo
  matrix_type: covariance
  relative: false
  type: matrix
"""

TEST_DATASET_XY_MISSING_KEYWORD = """
type: "xy"
y_data: [80.429, 80.339]
y_errors:
- matrix:
  - - 0.1
    - 0.0
  - - 0.0
    - 0.1
  name: YErrorOne
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !matrix |
    0.2   0.1
    0.1   0.2
  name: YErrorTwo
  matrix_type: covariance
  relative: false
  type: matrix
x_errors:
- matrix: !symmetric_matrix |
    0.3
    0.1   0.3
  name: XErrorOne
  matrix_type: covariance
  relative: false
  type: matrix
- matrix:
    [[0.2,  0.1],
     [0.1,  0.2]]
  name: XErrorTwo
  matrix_type: covariance
  relative: false
  type: matrix
"""

TEST_DATASET_XY_EXTRA_KEYWORD = TEST_DATASET_XY + """
extra_keyword: 3.14
"""


class TestXYContainerYamlRepresentation(unittest.TestCase):

    def setUp(self):
        _data = [[0.0, .1, .3], [10, 24, 44]]
        _ndat = len(_data[0])
        self._container = XYContainer(x_data=_data[0], y_data=_data[1])

        self._container.add_error(axis='y', name="ySUA", err_val=0.1, correlation=0.0, relative=False)
        self._container.add_error(axis='y', name="ySUR", err_val=0.1, correlation=0.0, relative=True)
        self._container.add_error(axis='y', name="ySCA", err_val=0.1, correlation=1.0, relative=False)
        self._container.add_matrix_error(axis='y', name="yMCov", err_matrix=np.eye(_ndat) * 0.1, relative=False, matrix_type='covariance')
        self._container.add_matrix_error(axis='y', name="yMCor", err_matrix=np.eye(_ndat), relative=False, matrix_type='correlation', err_val=0.1)

        self._container.add_error(axis='x', name="xSUA", err_val=0.1, correlation=0.0, relative=False)
        self._container.add_error(axis='x', name="xSUR", err_val=0.1, correlation=0.0, relative=True)
        self._container.add_matrix_error(axis='x', name="xMCov", err_matrix=np.eye(_ndat) * 0.1, relative=False, matrix_type='covariance')

        # self._container.add_simple_error(axis='x', err_val=0.1, correlation=0.0, relative=False)
        # self._container.add_simple_error(axis='y', err_val=0.2, correlation=0.0, relative=False)
        # self._container.add_matrix_error(axis='y', err_matrix=np.eye(8) * 0.1, relative=False, matrix_type='covariance')

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_DATASET_XY))

        self._roundtrip_streamreader = DataContainerYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = DataContainerYamlWriter(self._container, self._roundtrip_stringstream)
        self._testfile_streamreader = DataContainerYamlReader(self._testfile_stringstream)

        self._testfile_stringstream_missing_keyword = IOStreamHandle(StringIO(TEST_DATASET_XY_MISSING_KEYWORD))
        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_DATASET_XY_EXTRA_KEYWORD))
        self._testfile_streamreader_missing_keyword = DataContainerYamlReader(self._testfile_stringstream_missing_keyword)
        self._testfile_streamreader_extra_keyword = DataContainerYamlReader(self._testfile_stringstream_extra_keyword)

        self._ref_testfile_x_data = [5, 17]
        self._ref_testfile_y_data = [80.429, 80.339]
        self._ref_testfile_x_err = [0.70710678,  0.70710678]
        self._ref_testfile_y_err = [0.54772256,  0.54772256]
        self._ref_testfile_y_cov_mat = np.array([[0.3, 0.1], [0.1, 0.3]])
        self._ref_testfile_x_cov_mat = np.array([[0.5, 0.2], [0.2, 0.5]])
        self._ref_testfile_error_names = {'XErrorOne', 'XErrorTwo', 'YErrorOne', 'YErrorTwo'}


    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()


    def test_read_from_testfile_stream(self):
        _read_container = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_container, XYContainer))
        self.assertTrue(
            np.allclose(
                _read_container.x,
                self._ref_testfile_x_data
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.y,
                self._ref_testfile_y_data
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.x_err,
                self._ref_testfile_x_err
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.y_err,
                self._ref_testfile_y_err
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.x_cov_mat,
                self._ref_testfile_x_cov_mat,
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.y_cov_mat,
                self._ref_testfile_y_cov_mat,
            )
        )

        # check that the error names are the same
        self.assertEqual(
            set(_read_container._error_dicts.keys()),
            self._ref_testfile_error_names
        )


    def test_read_from_testfile_stream_missing_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_missing_keyword.read()

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_container = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_container, XYContainer))

        # compare data members
        self.assertTrue(
            np.allclose(
                self._container.x,
                _read_container.x
            )
        )
        self.assertTrue(
            np.allclose(
                self._container.y,
                _read_container.y
            )
        )

        # compare (total) errors and cov mats
        # TODO: compare individual error components -> nontrivial because error ids change...
        self.assertTrue(
            np.allclose(
                self._container.x_err,
                _read_container.x_err
            )
        )
        self.assertTrue(
            np.allclose(
                self._container.y_err,
                _read_container.y_err
            )
        )

        self.assertTrue(
            np.allclose(
                self._container.x_cov_mat,
                _read_container.x_cov_mat
            )
        )
        self.assertTrue(
            np.allclose(
                self._container.y_cov_mat,
                _read_container.y_cov_mat
            )
        )

        # check that the error names are the same
        self.assertEqual(
            set(self._container._error_dicts.keys()),
            set(_read_container._error_dicts.keys())
        )


TEST_DATASET_HIST = """
bin_edges:
- 1.0
- 2.0
- 3.0
raw_data:
- 4
- 5
- 3
- 3
- 2
- 1
errors:
- matrix:
  - - 0.1
    - 0.0
  - - 0.0
    - 0.2
  name: ErrorOne
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !matrix |
    0.2   0.1
    0.1   0.3
  name: ErrorTwo
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !symmetric_matrix |
    0.3
    0.1   0.4
  name: ErrorThree
  matrix_type: covariance
  relative: false
  type: matrix
- matrix:
    [[0.2,  0.1],
     [0.1,  0.1]]
  name: ErrorFour
  matrix_type: covariance
  relative: false
  type: matrix
type: histogram
"""

TEST_DATASET_HIST_EXTRA_KEYWORD = TEST_DATASET_HIST + """
extra_keyword: 3.14
"""

TEST_DATASET_HIST_MANUAL_HEIGHTS = """
bin_edges:
- 1.0
- 2.0
- 3.0
bin_heights:
- 1
- 1
underflow: 0
overflow: 4
errors:
- matrix:
  - - 0.1
    - 0.0
  - - 0.0
    - 0.2
  name: ErrorOne
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !matrix |
    0.2   0.1
    0.1   0.3
  name: ErrorTwo
  matrix_type: covariance
  relative: false
  type: matrix
- matrix: !symmetric_matrix |
    0.3
    0.1   0.4
  name: ErrorThree
  matrix_type: covariance
  relative: false
  type: matrix
- matrix:
    [[0.2,  0.1],
     [0.1,  0.1]]
  name: ErrorFour
  matrix_type: covariance
  relative: false
  type: matrix
type: histogram
"""


class TestHistContainerYamlRepresentation(unittest.TestCase):

    def setUp(self):
        _data = [1, 1, 4, 1, 1, 4, 2, 5, 5, 5, 2, 4, 2, 2, 18, 2, 8, 8, 9]
        _nbins = 3
        self._container = HistContainer(n_bins=_nbins,
                                        bin_range=(0.0, 10.),
                                        fill_data=_data)

        self._container.add_error(err_val=0.1, name='SC5A_1', correlation=0.5, relative=False)
        self._container.add_error(err_val=[i / 10. for i in range(1, _nbins + 1)], name='SC5A_2', correlation=0.5, relative=False)
        self._container.add_matrix_error(err_matrix=np.eye(_nbins) * 0.1, name='MCov', relative=False, matrix_type='covariance')
        self._container.add_matrix_error(err_matrix=np.eye(_nbins), name='MCor', relative=False, matrix_type='correlation', err_val=0.03)

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_DATASET_HIST))

        self._roundtrip_streamreader = DataContainerYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = DataContainerYamlWriter(self._container, self._roundtrip_stringstream)
        self._testfile_streamreader = DataContainerYamlReader(self._testfile_stringstream)

        self._testfile_stringstream_extra_keyword = IOStreamHandle(StringIO(TEST_DATASET_HIST_EXTRA_KEYWORD))
        self._testfile_streamreader_extra_keyword = DataContainerYamlReader(self._testfile_stringstream_extra_keyword)

        self._testfile_stringstream_manual_heights = IOStreamHandle(
            StringIO(TEST_DATASET_HIST_MANUAL_HEIGHTS))
        self._testfile_streamreader_manual_heights = DataContainerYamlReader(
            self._testfile_stringstream_manual_heights)

        self._ref_testfile_n_bins = 2
        self._ref_testfile_bin_range = [1.0, 3.0]
        self._ref_testfile_bin_edges = [1.0, 2.0, 3.0]
        self._ref_testfile_data = [1, 1]
        self._ref_testfile_underflow = 0
        self._ref_testfile_overflow = 4
        self._ref_testfile_err = [0.89442719,  1]
        self._ref_testfile_cov_mat = np.array([[0.8, 0.3], [0.3, 1.0]])
        self._ref_testfile_error_names = {'ErrorOne', 'ErrorTwo', 'ErrorThree', 'ErrorFour'}

    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()

    def test_read_from_testfile_stream(self):
        _read_container = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_container, HistContainer))

        self.assertTrue(_read_container.n_bins == self._ref_testfile_n_bins)
        self.assertTrue(
            np.allclose(
                _read_container.bin_range,
                self._ref_testfile_bin_range
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.bin_edges,
                self._ref_testfile_bin_edges
            )
        )

        self.assertTrue(
            np.allclose(
                _read_container.data,
                self._ref_testfile_data
            )
        )
        self.assertEqual(_read_container.underflow, self._ref_testfile_underflow)
        self.assertEqual(_read_container.overflow, self._ref_testfile_overflow)
        self.assertTrue(
            np.allclose(
                _read_container.err,
                self._ref_testfile_err
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.cov_mat,
                self._ref_testfile_cov_mat,
            )
        )

        # check that the error names are the same
        self.assertEqual(
            set(_read_container._error_dicts.keys()),
            self._ref_testfile_error_names
        )

    def test_read_from_testfile_stream_manual_heights(self):
        _read_container = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_container, HistContainer))

        self.assertTrue(_read_container.n_bins == self._ref_testfile_n_bins)
        self.assertTrue(
            np.allclose(
                _read_container.bin_range,
                self._ref_testfile_bin_range
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.bin_edges,
                self._ref_testfile_bin_edges
            )
        )

        self.assertTrue(
            np.allclose(
                _read_container.data,
                self._ref_testfile_data
            )
        )
        self.assertEqual(_read_container.underflow, self._ref_testfile_underflow)
        self.assertEqual(_read_container.overflow, self._ref_testfile_overflow)
        self.assertTrue(
            np.allclose(
                _read_container.err,
                self._ref_testfile_err
            )
        )
        self.assertTrue(
            np.allclose(
                _read_container.cov_mat,
                self._ref_testfile_cov_mat,
            )
        )

        # check that the error names are the same
        self.assertEqual(
            set(_read_container._error_dicts.keys()),
            self._ref_testfile_error_names
        )

    def test_read_from_testfile_stream_extra_keyword(self):
        with self.assertRaises(YamlReaderException):
            self._testfile_streamreader_extra_keyword.read()

    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_container = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_container, HistContainer))

        self.assertTrue(self._container.n_bins == _read_container.n_bins)
        self.assertTrue(
            np.allclose(
                self._container.bin_range,
                _read_container.bin_range
            )
        )
        self.assertTrue(
            np.allclose(
                self._container.bin_edges,
                _read_container.bin_edges
            )
        )

        # compare data members
        self.assertTrue(
            np.allclose(
                self._container.data,
                _read_container.data
            )
        )

        # compare (total) errors and cov mats
        # TODO: compare individual error components -> nontrivial because error ids change...
        self.assertTrue(
            np.allclose(
                self._container.err,
                _read_container.err
            )
        )

        self.assertTrue(
            np.allclose(
                self._container.cov_mat,
                _read_container.cov_mat
            )
        )

        # check that the error names are the same
        self.assertEqual(
            set(self._container._error_dicts.keys()),
            set(_read_container._error_dicts.keys())
        )

    def test_round_trip_with_stringstream_manual_heights(self):
        self._container.set_bins(
            self._container.data, self._container.underflow, self._container.overflow)
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_container = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_container, HistContainer))

        self.assertTrue(self._container.n_bins == _read_container.n_bins)
        self.assertTrue(
            np.allclose(
                self._container.bin_range,
                _read_container.bin_range
            )
        )
        self.assertTrue(
            np.allclose(
                self._container.bin_edges,
                _read_container.bin_edges
            )
        )

        # compare data members
        self.assertTrue(
            np.allclose(
                self._container.data,
                _read_container.data
            )
        )

        # compare (total) errors and cov mats
        # TODO: compare individual error components -> nontrivial because error ids change...
        self.assertTrue(
            np.allclose(
                self._container.err,
                _read_container.err
            )
        )

        self.assertTrue(
            np.allclose(
                self._container.cov_mat,
                _read_container.cov_mat
            )
        )

        # check that the error names are the same
        self.assertEqual(
            set(self._container._error_dicts.keys()),
            set(_read_container._error_dicts.keys())
        )
