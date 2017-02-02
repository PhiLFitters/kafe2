import unittest
import numpy as np

from kafe.fit.datastore.xy import IndexedContainer
from kafe.core.error import cov_mat_from_float_list


class TestDatastoreIndexed(unittest.TestCase):

    def setUp(self):
        self._ref_data = [3.3, 5.5, 2.2, 8.5, 10.2]
        self.idx_cont = IndexedContainer(data=self._ref_data)

        self._ref_err_abs_singlevalue = 1.2
        self._ref_err_abs_valuearray = [1.2, 1.1, 1.2, 1.2, 1.1]

        self._ref_err_corr_coeff = 0.23

        self.idx_cont.add_simple_error(self._ref_err_abs_valuearray, correlation=self._ref_err_corr_coeff,
                                       relative=False)

        self._ref_cov_mat = cov_mat_from_float_list(self._ref_err_abs_valuearray,
                                                    correlation=self._ref_err_corr_coeff).mat


    def test_compare_error_reference(self):
        for _err_dict in self.idx_cont._error_dicts.values():
            _err_ref_vals = _err_dict['err'].reference
            self.assertTrue(
                np.all(_err_ref_vals == self._ref_data)
            )

    def test_compare_ref_data(self):
        self.assertTrue(np.all(self.idx_cont.data == self._ref_data))

    def test_compare_ref_err(self):
        self.assertTrue(np.allclose(self.idx_cont.err, self._ref_err_abs_valuearray, atol=1e-10))

    def test_compare_ref_total_cov_mat(self):
        _err = self.idx_cont.get_total_error()
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_cov_mat, atol=1e-5))

    def test_compare_ref_total_err_for_add_err_twice(self):
        self.idx_cont.add_simple_error(self._ref_err_abs_valuearray,
                                       correlation=self._ref_err_corr_coeff,
                                       relative=False)
        _err = self.idx_cont.get_total_error()
        _mat = _err.cov_mat
        self.assertTrue(np.allclose(_mat, self._ref_cov_mat + self._ref_cov_mat, atol=1e-5))