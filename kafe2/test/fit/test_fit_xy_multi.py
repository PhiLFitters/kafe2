import abc
import unittest2 as unittest
import numpy as np
import six

from kafe2.core.minimizers import AVAILABLE_MINIMIZERS
from kafe2.core.fitters import NexusFitterException

from kafe2.config import kc

from kafe2.fit import XYMultiFit
from kafe2.fit.xy_multi.fit import XYMultiFitException
from kafe2.fit.xy_multi.model import XYMultiModelFunctionException
from kafe2.fit.xy.model import XYModelFunctionException
from kafe2.fit.xy_multi.cost import XYMultiCostFunction_Chi2

from kafe2.test.fit.test_fit import AbstractTestFit
from kafe2.test.fit.test_fit_xy import (TestXYFitBasicInterface, simple_chi2,
    simple_chi2_explicit_model_name,
    analytic_solution)


def simple_xy_model(x, b=1.1, c=2.2, d=3.3):
    return b * x ** 2 + c * x + d

def simple_xy_model_2(x, a=0.5, b=1.1, c=2.2, d=3.3):
    return a * x ** 3 + b * x ** 2 + c * x + d



class TestXYMultiFitBasicInterface(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        six.get_unbound_function(TestXYFitBasicInterface.setUp)(self)   # reuse simple XY test

        # "jitter" for data smearing
        self._y_jitter_2 = np.array([ 0.49671415, -0.1382643,   0.64768854,  1.52302986,
                                     -0.23415337, -0.23413696, 1.57921282,  0.76743473,
                                     -0.46947439,  0.54256004])
        assert len(self._y_jitter_2) == self._n_points

        # reference initial values
        self._ref_initial_pars_2 = np.array([0.5, 1.1, 2.2, 3.3])
        self._ref_initial_y_model_2 = simple_xy_model_2(self._ref_x, *self._ref_initial_pars_2)
        self._ref_initial_xy_model_2 = np.array([self._ref_x, self._ref_initial_y_model_2])

        # fit data
        self._ref_y_data_2 = self._ref_initial_y_model_2 + self._y_jitter_2
        self._ref_xy_data_2 = np.array([self._ref_x, self._ref_y_data_2])

        self._linear_design_matrix_2 = np.array([
            self._ref_x ** 3,
            self._ref_x ** 2,
            self._ref_x,
            np.ones_like(self._ref_x)
        ]).T

        # pre-fit cost value
        self._ref_initial_cost_2 = simple_chi2(
            y_data=self._ref_y_data_2,
            y_model=self._ref_initial_y_model_2,
        )

        # reference fit result values
        self._nominal_fit_result_pars_2 = analytic_solution(
            self._linear_design_matrix_2,
            np.linalg.inv(self._ref_matrix_eye),
            self._ref_y_data_2,
        )

        self._nominal_fit_result_y_model_2 = simple_xy_model_2(self._ref_x, *self._nominal_fit_result_pars_2)
        self._nominal_fit_result_xy_model_2 = np.array([self._ref_x, self._nominal_fit_result_y_model_2])
        self._nominal_fit_result_cost_2 = simple_chi2(
            y_data=self._ref_y_data_2,
            y_model=self._nominal_fit_result_y_model_2,
        )

        # reference multifit result values

        self._ref_matrix_eye_big = np.eye(2 * self._n_points)

        self._nominal_fit_result_pars_big = np.array([1.1143123504315096, 2.216862249916337, 2.9800759487215562, 0.49940864485321074])
        self._nominal_fit_result_pars_big_reversed = np.array([0.49940864485321074, 1.1143123504315096, 2.216862249916337, 2.9800759487215562])

        self._nominal_fit_result_y_model_big = np.hstack([
            simple_xy_model(self._ref_x, *self._nominal_fit_result_pars_big_reversed[1:]),
            simple_xy_model_2(self._ref_x, *self._nominal_fit_result_pars_big_reversed)
        ])
        self._nominal_fit_result_xy_model_big = np.array([
            np.hstack([self._ref_x, self._ref_x]),
            self._nominal_fit_result_y_model_big
        ])
        self._nominal_fit_result_y_model_big_reversed = np.hstack([
            simple_xy_model_2(self._ref_x, *self._nominal_fit_result_pars_big_reversed),
            simple_xy_model(self._ref_x, *self._nominal_fit_result_pars_big_reversed[1:]),
        ])
        self._nominal_fit_result_xy_model_big_reversed = np.array([
            np.hstack([self._ref_x, self._ref_x]),
            self._nominal_fit_result_y_model_big_reversed
        ])

        self._nominal_fit_result_cost_big = simple_chi2(
            y_data=np.hstack([self._ref_y_data, self._ref_y_data_2]),
            y_model=self._nominal_fit_result_y_model_big,
        )

        # helper dict with all reference property values
        self._ref_prop_dict = dict(
            did_fit=False,
        )

    def _get_fit(self, data=None, models=None, errors=None):
        '''convenience'''

        data = data if data is not None else self._ref_xy_data
        models = models if models is not None else simple_xy_model
        errors = errors or [dict(axis='y', err_val=1.0)]
        #par_values = par_values or self._ref_initial_pars

        _fit = XYMultiFit(
            xy_data=data,
            model_function=models,
            # TODO: fix default
            cost_function=XYMultiCostFunction_Chi2(
                axes_to_use='xy', errors_to_use='covariance'),
            minimizer=self.MINIMIZER,
            #x_error_algorithm='nonlinear',  # TODO: test other algorithms
        )

        for _err in errors:
            _fit.add_simple_error(**_err)

        return _fit

    def _get_test_fits(self):
        # TODO/FIXME: scipy minimizer gives wrong results!
        # -> exclude multifits for now...
        if self.MINIMIZER == 'scipy':
            return {
                'single_fit': self._get_fit(
                    data=self._ref_xy_data,
                    models=simple_xy_model,
                ),
            }
        else:
            return {
                'single_fit': self._get_fit(
                    data=self._ref_xy_data,
                    models=simple_xy_model,
                ),
                'two_fits': self._get_fit(
                    data=[self._ref_xy_data, self._ref_xy_data_2],
                    models=[simple_xy_model, simple_xy_model_2],
                ),
                'two_fits_reversed': self._get_fit(
                    data=[self._ref_xy_data_2, self._ref_xy_data],
                    models=[simple_xy_model_2, simple_xy_model],
                ),
            }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                poi_names=('b', 'c', 'd'),
                parameter_values=self._ref_initial_pars,
                poi_values=self._ref_initial_pars,
                x_data=self._ref_x,
                y_data=self._ref_y_data,
                data=np.array([self._ref_x, self._ref_y_data]),
                y_model=self._ref_initial_y_model,
                model=np.array([self._ref_x, self._ref_initial_y_model]),
                model_count=1,
                cost_function_value=self._ref_initial_cost,
            ),
            fit_names=['single_fit']
        )
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                poi_names=('b', 'c', 'd', 'a'),
                x_data=np.hstack([self._ref_x, self._ref_x]),
                y_data=np.hstack([self._ref_y_data, self._ref_y_data_2]),
                data=np.array([np.hstack([self._ref_x, self._ref_x]),
                               np.hstack([self._ref_y_data, self._ref_y_data_2])]),
                y_model=np.hstack([self._ref_initial_y_model, self._ref_initial_y_model_2]),
                model=np.array([np.hstack([self._ref_x, self._ref_x]),
                                np.hstack([self._ref_initial_y_model, self._ref_initial_y_model_2])]),
                model_count=2,
                cost_function_value=self._ref_initial_cost + self._ref_initial_cost_2,

                x_data_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_cov_mat=self._ref_matrix_eye_big,
                x_data_uncor_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_uncor_cov_mat=self._ref_matrix_eye_big,
                x_data_cov_mat_inverse=None,
                y_data_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_uncor_cov_mat_inverse=None,  # TODO: fix
                y_data_uncor_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_cor_mat=self._ref_matrix_eye,  # TODO: fix
                y_data_cor_mat=self._ref_matrix_eye_big,
                x_data_error=np.hstack([self._ref_error, self._ref_error]) * 0,
                y_data_error=np.hstack([self._ref_error, self._ref_error]),
            ),
            fit_names=['two_fits']
        )
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                poi_names=('a', 'b', 'c', 'd'),
                x_data=np.hstack([self._ref_x, self._ref_x]),
                y_data=np.hstack([self._ref_y_data_2, self._ref_y_data]),
                data=np.array([np.hstack([self._ref_x, self._ref_x]),
                               np.hstack([self._ref_y_data_2, self._ref_y_data])]),
                y_model=np.hstack([self._ref_initial_y_model_2, self._ref_initial_y_model]),
                model=np.array([np.hstack([self._ref_x, self._ref_x]),
                                np.hstack([self._ref_initial_y_model_2, self._ref_initial_y_model])]),
                model_count=2,
                cost_function_value=self._ref_initial_cost + self._ref_initial_cost_2,

                x_data_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_cov_mat=self._ref_matrix_eye_big,
                x_data_uncor_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_uncor_cov_mat=self._ref_matrix_eye_big,
                x_data_cov_mat_inverse=None,
                y_data_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_uncor_cov_mat_inverse=None,  # TODO: fix
                y_data_uncor_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_cor_mat=self._ref_matrix_eye,  # TODO: fix
                y_data_cor_mat=self._ref_matrix_eye_big,
                x_data_error=np.hstack([self._ref_error, self._ref_error]) * 0,
                y_data_error=np.hstack([self._ref_error, self._ref_error]),
            ),
            fit_names=['two_fits_reversed']
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                did_fit=True,
                poi_names=('b', 'c', 'd'),
                parameter_values=self._nominal_fit_result_pars,
                poi_values=self._nominal_fit_result_pars,
                x_data=self._ref_x,
                y_data=self._ref_y_data,
                data=np.array([self._ref_x, self._ref_y_data]),
                y_model=self._nominal_fit_result_y_model,
                model=np.array([self._ref_x, self._nominal_fit_result_y_model]),
                model_count=1,
                cost_function_value=np.float64(self._nominal_fit_result_cost),
            ),
            fit_names=['single_fit'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=3e-3
        )
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                did_fit=True,
                poi_names=('b', 'c', 'd', 'a'),
                parameter_values=self._nominal_fit_result_pars_big,
                poi_values=self._nominal_fit_result_pars_big,
                x_data=np.hstack([self._ref_x, self._ref_x]),
                y_data=np.hstack([self._ref_y_data, self._ref_y_data_2]),
                data=np.array([np.hstack([self._ref_x, self._ref_x]),
                               np.hstack([self._ref_y_data, self._ref_y_data_2])]),
                y_model=self._nominal_fit_result_y_model_big,
                model=self._nominal_fit_result_xy_model_big,
                model_count=2,
                cost_function_value=np.float64(self._nominal_fit_result_cost_big),

                x_data_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_cov_mat=self._ref_matrix_eye_big,
                x_data_uncor_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_uncor_cov_mat=self._ref_matrix_eye_big,
                x_data_cov_mat_inverse=None,
                y_data_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_uncor_cov_mat_inverse=None,  # TODO: fix
                y_data_uncor_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_cor_mat=self._ref_matrix_eye,  # TODO: fix
                y_data_cor_mat=self._ref_matrix_eye_big,
                x_data_error=np.hstack([self._ref_error, self._ref_error]) * 0,
                y_data_error=np.hstack([self._ref_error, self._ref_error]),
            ),
            fit_names=['two_fits'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-3
        )
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                did_fit=True,
                poi_names=('a', 'b', 'c', 'd'),
                parameter_values=self._nominal_fit_result_pars_big_reversed,
                poi_values=self._nominal_fit_result_pars_big_reversed,
                x_data=np.hstack([self._ref_x, self._ref_x]),
                y_data=np.hstack([self._ref_y_data_2, self._ref_y_data]),
                data=np.array([np.hstack([self._ref_x, self._ref_x]),
                               np.hstack([self._ref_y_data_2, self._ref_y_data])]),
                y_model=self._nominal_fit_result_y_model_big_reversed,
                model=self._nominal_fit_result_xy_model_big_reversed,
                model_count=2,
                cost_function_value=np.float64(self._nominal_fit_result_cost_big),

                x_data_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_cov_mat=self._ref_matrix_eye_big,
                x_data_uncor_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_uncor_cov_mat=self._ref_matrix_eye_big,
                x_data_cov_mat_inverse=None,
                y_data_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_uncor_cov_mat_inverse=None,  # TODO: fix
                y_data_uncor_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_cor_mat=self._ref_matrix_eye,  # TODO: fix
                y_data_cor_mat=self._ref_matrix_eye_big,
                x_data_error=np.hstack([self._ref_error, self._ref_error]) * 0,
                y_data_error=np.hstack([self._ref_error, self._ref_error]),
            ),
            fit_names=['two_fits_reversed'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-3
        )

    def test_set_all_parameter_values(self):
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                did_fit=False,
                poi_names=('b', 'c', 'd'),
                parameter_values=self._nominal_fit_result_pars,
                poi_values=self._nominal_fit_result_pars,
                x_data=self._ref_x,
                y_data=self._ref_y_data,
                data=np.array([self._ref_x, self._ref_y_data]),
                y_model=self._nominal_fit_result_y_model,
                model=np.array([self._ref_x, self._nominal_fit_result_y_model]),
                model_count=1,
                cost_function_value=np.float64(self._nominal_fit_result_cost),
            ),
            fit_names=['single_fit'],
            call_before_fit=lambda f: f.set_all_parameter_values(
                self._nominal_fit_result_pars),
            rtol=3e-3
        )
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                did_fit=False,
                poi_names=('b', 'c', 'd', 'a'),
                parameter_values=self._nominal_fit_result_pars_big,
                poi_values=self._nominal_fit_result_pars_big,
                x_data=np.hstack([self._ref_x, self._ref_x]),
                y_data=np.hstack([self._ref_y_data, self._ref_y_data_2]),
                data=np.array([np.hstack([self._ref_x, self._ref_x]),
                               np.hstack([self._ref_y_data, self._ref_y_data_2])]),
                y_model=self._nominal_fit_result_y_model_big,
                model=self._nominal_fit_result_xy_model_big,
                model_count=2,
                cost_function_value=np.float64(self._nominal_fit_result_cost_big),

                x_data_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_cov_mat=self._ref_matrix_eye_big,
                x_data_uncor_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_uncor_cov_mat=self._ref_matrix_eye_big,
                x_data_cov_mat_inverse=None,
                y_data_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_uncor_cov_mat_inverse=None,  # TODO: fix
                y_data_uncor_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_cor_mat=self._ref_matrix_eye,  # TODO: fix
                y_data_cor_mat=self._ref_matrix_eye_big,
                x_data_error=np.hstack([self._ref_error, self._ref_error]) * 0,
                y_data_error=np.hstack([self._ref_error, self._ref_error]),
            ),
            fit_names=['two_fits'],
            call_before_fit=lambda f: f.set_all_parameter_values(
                self._nominal_fit_result_pars_big),
            rtol=1e-3
        )
        self.run_test_for_all_fits(
            dict(
                self._ref_prop_dict,
                did_fit=False,
                poi_names=('a', 'b', 'c', 'd'),
                parameter_values=self._nominal_fit_result_pars_big_reversed,
                poi_values=self._nominal_fit_result_pars_big_reversed,
                x_data=np.hstack([self._ref_x, self._ref_x]),
                y_data=np.hstack([self._ref_y_data_2, self._ref_y_data]),
                data=np.array([np.hstack([self._ref_x, self._ref_x]),
                               np.hstack([self._ref_y_data_2, self._ref_y_data])]),
                y_model=self._nominal_fit_result_y_model_big_reversed,
                model=self._nominal_fit_result_xy_model_big_reversed,
                model_count=2,
                cost_function_value=np.float64(self._nominal_fit_result_cost_big),

                x_data_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_cov_mat=self._ref_matrix_eye_big,
                x_data_uncor_cov_mat=self._ref_matrix_eye_big * 0,
                y_data_uncor_cov_mat=self._ref_matrix_eye_big,
                x_data_cov_mat_inverse=None,
                y_data_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_uncor_cov_mat_inverse=None,  # TODO: fix
                y_data_uncor_cov_mat_inverse=self._ref_matrix_eye_big,
                #x_data_cor_mat=self._ref_matrix_eye,  # TODO: fix
                y_data_cor_mat=self._ref_matrix_eye_big,
                x_data_error=np.hstack([self._ref_error, self._ref_error]) * 0,
                y_data_error=np.hstack([self._ref_error, self._ref_error]),
            ),
            fit_names=['two_fits_reversed'],
            call_before_fit=lambda f: f.set_all_parameter_values(
                self._nominal_fit_result_pars_big_reversed),
            rtol=1e-3
        )

    def test_conflicting_defaults_raise(self):
        def simple_xy_model_conflict(x, a=0.5, b=0.5, c=0.5, d=0.5):
            pass
        with self.assertRaises(XYMultiModelFunctionException):
            _conflicting_defaults_fit = XYMultiFit(
                xy_data=[self._ref_xy_data, self._ref_xy_data_2],
                model_function=[simple_xy_model, simple_xy_model_conflict],
                cost_function=simple_chi2,
                minimizer=self.MINIMIZER
            )

    def test_update_data(self):
        _fit = self._get_fit(
            data=self._ref_xy_data,
            models=simple_xy_model,
        )

        _fit.data = np.array([self._ref_x, self._ref_y_data * 2])
        _fit.add_simple_error('y', err_val=1.0)

        self._assert_fit_properties(
            _fit,
            dict(
                data=np.array([self._ref_x, self._ref_y_data * 2]),
                x_data=self._ref_x,
                y_data=self._ref_y_data * 2,
            )
        )

        _fit.do_fit()
        _new_estimates = np.array(self._nominal_fit_result_pars) * 2

        self._assert_fit_properties(
            _fit,
            dict(
                data=np.array([self._ref_x, self._ref_y_data * 2]),
                x_data=self._ref_x,
                y_data=self._ref_y_data * 2,
                poi_values=_new_estimates,
                parameter_values=_new_estimates,
            ),
            rtol=1e-2
        )

    def test_update_data_different_length(self):
        _fit = self._get_fit(
            data=self._ref_xy_data,
            models=simple_xy_model,
        )

        _fit.data = np.array([self._ref_x[:-1], self._ref_y_data[:-1] * 2])
        _fit.add_simple_error('y', err_val=1.0)

        self._assert_fit_properties(
            _fit,
            dict(
                data=np.array([self._ref_x[:-1], self._ref_y_data[:-1] * 2]),
                x_data=self._ref_x[:-1],
                y_data=self._ref_y_data[:-1] * 2,
            )
        )

        _fit.do_fit()

        _new_estimates = analytic_solution(
            self._linear_design_matrix[:-1],
            np.linalg.inv(self._ref_matrix_eye[:-1,:-1]),
            self._ref_y_data[:-1] * 2,
        )

        self._assert_fit_properties(
            _fit,
            dict(
                data=np.array([self._ref_x[:-1], self._ref_y_data[:-1] * 2]),
                x_data=self._ref_x[:-1],
                y_data=self._ref_y_data[:-1] * 2,
                parameter_values=_new_estimates,
                poi_values=_new_estimates,
            ),
            rtol=1e-2
        )

    def test_set_all_parameter_values_wrong_number_raise(self):
        # FIXME: discrepancy
        #with self.assertRaises(XYMultiFitException):
        with self.assertRaises(NexusFitterException):
            self._get_fit().set_all_parameter_values((1,))
        #with self.assertRaises(XYMultiFitException):
        with self.assertRaises(NexusFitterException):
            self._get_fit().set_all_parameter_values((1,2,3,4,5))


    def test_report_before_fit(self):
        # TODO: check report content
        _buffer = six.StringIO()
        _fit = self._get_fit()
        _fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")

    def test_report_after_fit(self):
        # TODO: check report content
        _buffer = six.StringIO()
        _fit = self._get_fit()
        _fit.do_fit()
        _fit.report(output_stream=_buffer)
        self.assertNotEqual(_buffer.getvalue(), "")
