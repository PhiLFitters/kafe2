import unittest
import numpy as np

from kafe.fit.datastore.xy import XYContainer
from kafe.core.error import cov_mat_from_float_list


class TestDatastoreXY(unittest.TestCase):

    def setUp(self):
        self._ref_x_data = [0, 1, 2, 3, 4]
        self._ref_y_data = [3.3, 5.5, 2.2, 8.5, 10.2]
        self.data_xy = XYContainer(x_data=self._ref_x_data, y_data=self._ref_y_data)

        self._ref_x_err_abs_singlevalue = 0.1
        self._ref_y_err_abs_singlevalue = 1.2
        self._ref_x_err_abs_valuearray = np.array([0.1, 0.1, 0.2, 0.4, 0.5])
        self._ref_y_err_abs_valuearray = [1.2, 1.1, 1.2, 1.2, 1.1]

        self._ref_x_err_corr_coeff = 0.1
        self._ref_y_err_corr_coeff = 0.23

        self.data_xy.add_simple_error('x', self._ref_x_err_abs_valuearray, correlation=self._ref_x_err_corr_coeff,
                                      relative=False)
        self.data_xy.add_simple_error('y', self._ref_y_err_abs_valuearray, correlation=self._ref_y_err_corr_coeff,
                                      relative=False)

        self._ref_x_cov_mat = cov_mat_from_float_list(self._ref_x_err_abs_valuearray,
                                                      correlation=self._ref_x_err_corr_coeff).mat
        self._ref_y_cov_mat = cov_mat_from_float_list(self._ref_y_err_abs_valuearray,
                                                      correlation=self._ref_y_err_corr_coeff).mat


    def test_compare_error_reference(self):
        for _err_dict in self.data_xy._error_dicts.values():
            _err_ref_vals = _err_dict['err'].reference
            _axis = _err_dict['axis']
            assert _axis in (0, 1)
            if _axis == 0:
                self.assertTrue(
                    np.all(_err_ref_vals == self._ref_x_data)
                )
            elif _axis == 1:
                self.assertTrue(
                    np.all(_err_ref_vals == self._ref_y_data)
                )


    def test_compare_ref_x_data(self):
        self.assertTrue(np.all(self.data_xy.x == self._ref_x_data))

    def test_compare_ref_y_data(self):
        self.assertTrue(np.all(self.data_xy.y == self._ref_y_data))

    def test_compare_ref_x_err(self):
        self.assertTrue(np.allclose(self.data_xy.x_err, self._ref_x_err_abs_valuearray, atol=1e-10))

    def test_compare_ref_y_err(self):
        self.assertTrue(np.allclose(self.data_xy.y_err, self._ref_y_err_abs_valuearray, atol=1e-10))

    # def test_compare_ref_x_cov_mat(self):
    #     self.assertTrue(np.allclose(self.data_xy.???, self._ref_x_cov_mat, atol=1e-5))
    #
    # def test_compare_ref_y_cov_mat(self):
    #     self.assertTrue(np.allclose(self.data_xy.???, self._ref_y_cov_mat, atol=1e-5))

    def test_compare_ref_total_x_cov_mat(self):
        _err = self.data_xy.get_total_error(0)
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_x_cov_mat, atol=1e-5))

    def test_compare_ref_total_y_cov_mat(self):
        _err = self.data_xy.get_total_error(1)
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_y_cov_mat, atol=1e-5))

    def test_compare_ref_total_y_for_x_plus_y_as_y_err(self):
        self.data_xy.add_simple_error('y', self._ref_x_err_abs_valuearray,
                                      correlation=self._ref_x_err_corr_coeff,
                                      relative=False)
        _err = self.data_xy.get_total_error(1)
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_y_cov_mat + self._ref_x_cov_mat, atol=1e-5))