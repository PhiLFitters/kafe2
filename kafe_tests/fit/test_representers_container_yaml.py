import unittest
import numpy as np

from StringIO import StringIO

from kafe.fit import IndexedContainer, XYContainer, HistContainer
from kafe.fit.representation import DataContainerYamlWriter, DataContainerYamlReader, DReprError
from kafe.fit.io.handle import IOStreamHandle

# TODO: check that DReprError is raised when appropriate...


TEST_DATASET_INDEXED = """
dataset:
  type: "indexed"
  data: [80.429, 80.339]
  errors:
  - matrix:
    - - 0.1
      - 0.0
    - - 0.0
      - 0.1
    matrix_type: covariance
    relative: false
    type: matrix
  - matrix: !matrix |
      0.2   0.1
      0.1   0.2
    matrix_type: covariance
    relative: false
    type: matrix
  - matrix: !symmetric_matrix |
      0.3
      0.1   0.3
    matrix_type: covariance
    relative: false
    type: matrix
  - matrix:
      [[0.2,  0.1],
       [0.1,  0.2]]
    matrix_type: covariance
    relative: false
    type: matrix
"""


class TestIndexedContainerYamlRepresentation(unittest.TestCase):

    def setUp(self):
        _data = [4, 5, 6, 2, 4, 5., 3.01e-5, 4.4e+45]
        _ndat = len(_data)
        self._container = IndexedContainer(data=_data)

        self._container.add_simple_error(err_val=0.1, correlation=0.0, relative=False)
        self._container.add_simple_error(err_val=0.1, correlation=0.0, relative=True)
        self._container.add_simple_error(err_val=0.1, correlation=1.0, relative=False)
        self._container.add_matrix_error(err_matrix=np.eye(_ndat) * 0.1, relative=False, matrix_type='covariance')
        self._container.add_matrix_error(err_matrix=np.eye(_ndat), relative=False, matrix_type='correlation', err_val=0.1)

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
        self._ref_testfile_cov_mat = np.matrix([[0.8, 0.3], [0.3, 0.8]])


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


TEST_DATASET_XY = """
dataset:
  type: "xy"
  x_data: [5, 17]
  y_data: [80.429, 80.339]
  y_errors:
  - matrix:
    - - 0.1
      - 0.0
    - - 0.0
      - 0.1
    matrix_type: covariance
    relative: false
    type: matrix
  - matrix: !matrix |
      0.2   0.1
      0.1   0.2
    matrix_type: covariance
    relative: false
    type: matrix
  x_errors:
  - matrix: !symmetric_matrix |
      0.3
      0.1   0.3
    matrix_type: covariance
    relative: false
    type: matrix
  - matrix:
      [[0.2,  0.1],
       [0.1,  0.2]]
    matrix_type: covariance
    relative: false
    type: matrix
"""


class TestXYContainerYamlRepresentation(unittest.TestCase):

    def setUp(self):
        _data = [[0.0, .1, .3], [10, 24, 44]]
        _ndat = len(_data[0])
        self._container = XYContainer(x_data=_data[0], y_data=_data[1])

        self._container.add_simple_error(axis='y', err_val=0.1, correlation=0.0, relative=False)
        self._container.add_simple_error(axis='y', err_val=0.1, correlation=0.0, relative=True)
        self._container.add_simple_error(axis='y', err_val=0.1, correlation=1.0, relative=False)
        self._container.add_matrix_error(axis='y', err_matrix=np.eye(_ndat) * 0.1, relative=False, matrix_type='covariance')
        self._container.add_matrix_error(axis='y', err_matrix=np.eye(_ndat), relative=False, matrix_type='correlation', err_val=0.1)

        self._container.add_simple_error(axis='x', err_val=0.1, correlation=0.0, relative=False)
        self._container.add_simple_error(axis='x', err_val=0.1, correlation=0.0, relative=True)
        self._container.add_matrix_error(axis='x', err_matrix=np.eye(_ndat) * 0.1, relative=False, matrix_type='covariance')

        # self._container.add_simple_error(axis='x', err_val=0.1, correlation=0.0, relative=False)
        # self._container.add_simple_error(axis='y', err_val=0.2, correlation=0.0, relative=False)
        # self._container.add_matrix_error(axis='y', err_matrix=np.eye(8) * 0.1, relative=False, matrix_type='covariance')


        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_DATASET_XY))

        self._roundtrip_streamreader = DataContainerYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = DataContainerYamlWriter(self._container, self._roundtrip_stringstream)
        self._testfile_streamreader = DataContainerYamlReader(self._testfile_stringstream)

        self._ref_testfile_x_data = [5, 17]
        self._ref_testfile_y_data = [80.429, 80.339]
        self._ref_testfile_x_err = [0.70710678,  0.70710678]
        self._ref_testfile_y_err = [0.54772256,  0.54772256]
        self._ref_testfile_y_cov_mat = np.matrix([[0.3, 0.1], [0.1, 0.3]])
        self._ref_testfile_x_cov_mat = np.matrix([[0.5, 0.2], [0.2, 0.5]])


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


TEST_DATASET_HIST = """
dataset:
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
    matrix_type: covariance
    relative: false
    type: matrix
  - matrix: !matrix |
      0.2   0.1
      0.1   0.3
    matrix_type: covariance
    relative: false
    type: matrix
  - matrix: !symmetric_matrix |
      0.3
      0.1   0.4
    matrix_type: covariance
    relative: false
    type: matrix
  - matrix:
      [[0.2,  0.1],
       [0.1,  0.1]]
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

        self._container.add_simple_error(err_val=0.1, correlation=0.5, relative=False)
        self._container.add_simple_error(err_val=[i/10. for i in range(1, _nbins+1)], correlation=0.5, relative=False)
        self._container.add_matrix_error(err_matrix=np.eye(_nbins) * 0.1, relative=False, matrix_type='covariance')
        self._container.add_matrix_error(err_matrix=np.eye(_nbins), relative=False, matrix_type='correlation', err_val=0.03)

        self._roundtrip_stringstream = IOStreamHandle(StringIO())
        self._testfile_stringstream = IOStreamHandle(StringIO(TEST_DATASET_HIST))

        self._roundtrip_streamreader = DataContainerYamlReader(self._roundtrip_stringstream)
        self._roundtrip_streamwriter = DataContainerYamlWriter(self._container, self._roundtrip_stringstream)
        self._testfile_streamreader = DataContainerYamlReader(self._testfile_stringstream)

        self._ref_testfile_data = [1, 1]
        self._ref_testfile_err = [0.89442719,  1]
        self._ref_testfile_cov_mat = np.matrix([[0.8, 0.3], [0.3, 1.0]])


    def test_write_to_roundtrip_stringstream(self):
        self._roundtrip_streamwriter.write()


    def test_read_from_testfile_stream(self):
        _read_container = self._testfile_streamreader.read()
        self.assertTrue(isinstance(_read_container, HistContainer))

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


    def test_round_trip_with_stringstream(self):
        self._roundtrip_streamwriter.write()
        self._roundtrip_stringstream.seek(0)  # return to beginning
        _read_container = self._roundtrip_streamreader.read()
        self.assertTrue(isinstance(_read_container, HistContainer))

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