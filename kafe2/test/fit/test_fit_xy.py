import abc
import unittest2 as unittest
import numpy as np
import six

from kafe2.core.minimizers import AVAILABLE_MINIMIZERS
from kafe2.core.fitters import NexusFitterException

from kafe2.config import kc

from kafe2.fit._base import ModelFunctionException
from kafe2.fit import XYFit
from kafe2.fit.xy.fit import XYFitException
from kafe2.fit.xy.model import XYParametricModelException
from kafe2.fit.xy.cost import XYCostFunction_Chi2

from kafe2.test.fit.test_fit import AbstractTestFit


def simple_chi2(y_data, y_model):
    return np.sum((y_data - y_model)**2)

def simple_chi2_explicit_model_name(y_data, simple_xy_model):
    return np.sum((y_data - simple_xy_model)**2)

def simple_xy_model(x, a=1.1, b=2.2, c=3.3):
    return a * x ** 2 + b * x + c

def line_xy_model(x, a=3.0, b=0.0):
    return a * x + b


def analytic_solution(des_mat, cov_mat_inv, y_data):
    return (
        np.squeeze(np.asarray(np.linalg.inv(
            des_mat.T.dot(cov_mat_inv).dot(des_mat)
        ).dot(des_mat.T)
         .dot(cov_mat_inv)
         .dot(y_data))))


class TestXYFitBasicInterface(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        self._n_points = 10

        # "jitter" for data smearing
        self._y_jitter = np.array([-0.3193475 , -1.2404198 , -1.4906926 , -0.78832446,
                                   -1.7638106,   0.36664261,  0.49433821,  0.0719646,
                                    1.95670326,  0.31200215])
        assert len(self._y_jitter) == self._n_points

        # reference initial values
        self._ref_initial_pars = np.array([1.1, 2.2, 3.3])
        self._ref_x = np.arange(self._n_points)
        self._ref_initial_y_model = simple_xy_model(self._ref_x, *self._ref_initial_pars)
        self._ref_initial_xy_model = np.array([self._ref_x, self._ref_initial_y_model])

        # fit data
        self._ref_y_data = self._ref_initial_y_model + self._y_jitter
        self._ref_xy_data = np.array([self._ref_x, self._ref_y_data])

        self._linear_design_matrix = np.array([
            self._ref_x ** 2,
            self._ref_x,
            np.ones_like(self._ref_x)
        ]).T

        # pre-fit cost value
        self._ref_initial_cost = simple_chi2(
            y_data=self._ref_y_data,
            y_model=self._ref_initial_y_model,
        )

        # reference matrices/errors
        self._ref_error = np.ones(self._n_points)
        self._ref_matrix_eye = np.eye(self._n_points)

        # reference fit result values
        #self._nominal_fit_result_pars = (1.1351433, 2.13736919, 2.33346549)
        self._nominal_fit_result_pars = analytic_solution(
            self._linear_design_matrix,
            np.linalg.inv(self._ref_matrix_eye),
            self._ref_y_data,
        )

        self._nominal_fit_result_y_model = simple_xy_model(self._ref_x, *self._nominal_fit_result_pars)
        self._nominal_fit_result_xy_model = np.array([self._ref_x, self._nominal_fit_result_y_model])
        self._nominal_fit_result_cost = simple_chi2(
            y_data=self._ref_y_data,
            y_model=self._nominal_fit_result_y_model,
        )

        # helper dict with all reference property values
        self._ref_prop_dict = dict(
            did_fit=False,
            model_count=1,

            parameter_values=self._ref_initial_pars,
            parameter_names=('a', 'b', 'c'),
            cost_function_value=self._ref_initial_cost,

            x_data=self._ref_x,
            y_data=self._ref_y_data,
            data=self._ref_xy_data,
            x_data_error=self._ref_error * 0,
            y_data_error=self._ref_error,
            data_error=self._ref_error,
            x_data_cov_mat=self._ref_matrix_eye * 0,
            y_data_cov_mat=self._ref_matrix_eye,
            data_cov_mat=self._ref_matrix_eye,
            x_data_cov_mat_inverse=None,
            y_data_cov_mat_inverse=self._ref_matrix_eye,
            data_cov_mat_inverse=self._ref_matrix_eye,
            x_data_cor_mat=self._ref_matrix_eye * np.nan,
            y_data_cor_mat=self._ref_matrix_eye,
            data_cor_mat=self._ref_matrix_eye,

            x_model=self._ref_x,
            y_model=self._ref_initial_y_model,
            model=self._ref_initial_xy_model,
            x_model_error=self._ref_error * 0,
            y_model_error=self._ref_error * 0,
            model_error=self._ref_error * 0,
            x_model_cov_mat=self._ref_matrix_eye * 0,
            y_model_cov_mat=self._ref_matrix_eye * 0,
            model_cov_mat=self._ref_matrix_eye * 0,
            x_model_cov_mat_inverse=None,
            y_model_cov_mat_inverse=None,
            model_cov_mat_inverse=None,
            x_model_cor_mat=self._ref_matrix_eye * np.nan,
            y_model_cor_mat=self._ref_matrix_eye * np.nan,
            model_cor_mat=self._ref_matrix_eye * np.nan,

            x_total_error=self._ref_error * 0,
            y_total_error=self._ref_error,
            total_error=self._ref_error,
            x_total_cov_mat=self._ref_matrix_eye * 0,
            y_total_cov_mat=self._ref_matrix_eye,
            total_cov_mat=self._ref_matrix_eye,
            x_total_cov_mat_inverse=None,
            y_total_cov_mat_inverse=self._ref_matrix_eye,
            total_cov_mat_inverse=self._ref_matrix_eye,
            x_total_cor_mat=self._ref_matrix_eye * np.nan,
            y_total_cor_mat=self._ref_matrix_eye,
            total_cor_mat=self._ref_matrix_eye,
        )

    def _get_fit(
            self, model_function=None, cost_function=None, errors=None,
            dynamic_error_algorithm=None):
        '''convenience'''
        model_function = model_function or simple_xy_model
        # TODO: fix default
        cost_function = cost_function or XYCostFunction_Chi2(
            axes_to_use='xy', errors_to_use='covariance')
        errors = errors or [dict(axis='y', err_val=1.0)]
        dynamic_error_algorithm = dynamic_error_algorithm or "nonlinear"

        _fit = XYFit(
            xy_data=self._ref_xy_data,
            model_function=model_function,
            cost_function=cost_function,
            minimizer=self.MINIMIZER,
            dynamic_error_algorithm=dynamic_error_algorithm
        )
        for _err in errors:
            _fit.add_error(**_err)

        return _fit

    def _get_test_fits(self):
        return {
            'default': self._get_fit(),
            'explicit': self._get_fit(cost_function=simple_chi2),
            'explicit_model': self._get_fit(cost_function=simple_chi2_explicit_model_name),
            'relative_errors_data': self._get_fit(errors=[
                dict(axis="y", err_val=1.0/self._ref_y_data, relative=True, reference="data")
            ])
        }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            self._ref_prop_dict
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars,
                y_model=self._nominal_fit_result_y_model,
                model=self._nominal_fit_result_xy_model,
                did_fit=True,
                cost_function_value=self._nominal_fit_result_cost,
            ),
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-2
        )

    def test_set_all_parameter_values(self):
        self.run_test_for_all_fits(
            dict(self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars,
                x_data=self._ref_x,
                x_model=self._ref_x,
                y_data=self._ref_y_data,
                y_model=self._nominal_fit_result_y_model,
                data=self._ref_xy_data,
                model=self._nominal_fit_result_xy_model,
                parameter_names=('a', 'b', 'c'),
                did_fit=False,
                model_count=1,
                cost_function_value=self._nominal_fit_result_cost
            ),
            # set parameters to their final values by hand
            call_before_fit=lambda f: f.set_all_parameter_values(
                self._nominal_fit_result_pars),
        )

    def test_set_all_parameter_values_wrong_number_raise(self):
        # FIXME: discrepancy
        #with self.assertRaises(XYFitException):
        with self.assertRaises(NexusFitterException):
            self._get_fit().set_all_parameter_values((1,))
        #with self.assertRaises(XYFitException):
        with self.assertRaises(NexusFitterException):
            self._get_fit().set_all_parameter_values((1,2,3,4,5))

    def test_parameter_defaults(self):
        def dummy_model(x, a, b, c):
            return x

        _fit = self._get_fit(model_function=dummy_model)

        self._assert_fit_properties(
            _fit,
            dict(
                parameter_values=np.array([1, 1, 1]),
            )
        )

    def test_parameter_partial_defaults(self):
        def dummy_model(x, a, b, c=3.3):
            return x

        _fit = self._get_fit(model_function=dummy_model)

        self._assert_fit_properties(
            _fit,
            dict(
                parameter_values=np.array([1, 1, 3.3]),
            )
        )

    def test_add_parameter_constraint(self):
        _fit = self._get_fit()

        _fit.add_parameter_constraint('c', 1.0, 1.0)
        _constraint_cost = (self._ref_initial_pars[2] - 1.0)**2

        self._assert_fit_properties(
            _fit,
            dict(
                cost_function_value=np.float64(
                    self._ref_initial_cost + _constraint_cost
                )
            )
        )

    def test_add_matrix_parameter_constraint(self):
        _fit = self._get_fit()

        _fit.add_matrix_parameter_constraint(('a', 'c'), np.ones(2), np.eye(2))
        _constraint_cost = (
            (self._ref_initial_pars[0] - 1.0)**2 +
            (self._ref_initial_pars[2] - 1.0)**2
        )

        self._assert_fit_properties(
            _fit,
            dict(
                cost_function_value=np.float64(
                    self._ref_initial_cost + _constraint_cost
                )
            )
        )

    def test_eval_model_function(self):
        _fit = self._get_fit()
        self.assertTrue(np.allclose(
            _fit.eval_model_function(self._ref_x),
            self._ref_initial_y_model
        ))

    def test_update_data(self):
        _fit = self._get_fit()

        _fit.data = np.array([self._ref_x, self._ref_y_data * 2])
        _fit.add_error('y', err_val=1.0)

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
                parameter_values=_new_estimates,
            ),
            rtol=1e-2
        )

    def test_update_data_different_length(self):
        _fit = self._get_fit()

        _fit.data = np.array([self._ref_x[:-1], self._ref_y_data[:-1] * 2])
        _fit.add_error('y', err_val=1.0)

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
            ),
            rtol=1e-2
        )

    def test_reserved_parameter_names_raise(self):
        def dummy_model(x, y_data):
            pass

        with self.assertRaises(XYFitException) as _exc:
            XYFit(xy_data=self._ref_xy_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn('reserved', _exc.exception.args[0])
        self.assertIn('y_data', _exc.exception.args[0])

    def test_model_no_pars_raise(self):
        def dummy_model():
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            XYFit(xy_data=self._ref_xy_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn(
            'needs at least one parameter',
            _exc.exception.args[0])

    def test_model_no_pars_beside_x_raise(self):
        def dummy_model(x):
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            XYFit(xy_data=self._ref_xy_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn(
            'needs at least one parameter besides',
            _exc.exception.args[0])

    def test_model_varargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, *varargs):
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            XYFit(xy_data=self._ref_xy_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varargs', _exc.exception.args[0])

    def test_model_varkwargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, **varkwargs):
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            XYFit(xy_data=self._ref_xy_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varkwargs', _exc.exception.args[0])

    def test_model_varargs_varkwargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, *varargs, **varkwargs):
            pass

        with self.assertRaises(ModelFunctionException) as _exc:
            XYFit(xy_data=self._ref_xy_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varargs', _exc.exception.args[0])
        # TODO: enable when implemented
        #self.assertIn('varkwargs', _exc.exception.args[0])

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

    def test_unknown_cost_function(self):
        with self.assertRaises(ValueError):
            self._get_fit(cost_function="ABC123")

    def test_model_as_model_func_name(self):
        def model(x, a, b):
            return a * x + b
        self._get_fit(model_function=model)

    def test_data_and_cost_incompatible(self):
        with self.assertRaises(XYFitException):
            self._get_fit(cost_function="nll")

    def test_relative_model_error_nonlinear(self):
        _fit_data_err = self._get_fit(errors=[
            dict(axis="y", err_val=1.0, relative=False, reference="data"),
            dict(axis="y", err_val=0.1, relative=True, reference="data")
        ])
        _fit_model_err = self._get_fit(errors=[
            dict(axis="y", err_val=1.0, relative=False, reference="data"),
            dict(axis="y", err_val=0.1, relative=True, reference="model")
        ], dynamic_error_algorithm="nonlinear")
        _fit_data_err.do_fit()
        _fit_model_err.do_fit()
        self._assert_fit_results_equal(_fit_data_err, _fit_model_err, rtol=6e-2)

    def test_relative_model_error_iterative(self):
        _fit_data_err = self._get_fit(errors=[
            dict(axis="y", err_val=1.0, relative=False, reference="data"),
            dict(axis="y", err_val=0.1, relative=True, reference="data")
        ])
        _fit_model_err = self._get_fit(errors=[
            dict(axis="y", err_val=1.0, relative=False, reference="data"),
            dict(axis="y", err_val=0.1, relative=True, reference="model")
        ], dynamic_error_algorithm="iterative")
        _fit_data_err.do_fit()
        _fit_model_err.do_fit()
        self._assert_fit_results_equal(_fit_data_err, _fit_model_err, rtol=1e-2)


class TestXYFitWithSimpleYErrors(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        six.get_unbound_function(TestXYFitBasicInterface.setUp)(self)

    def _get_fit(self, errors=None):
        '''convenience'''

        errors = errors or [dict(axis='y', err_val=1.0)]

        _fit = XYFit(
            xy_data=self._ref_xy_data,
            model_function=simple_xy_model,
            cost_function=XYCostFunction_Chi2(
                axes_to_use='xy', errors_to_use='covariance'),
            minimizer=self.MINIMIZER
        )

        for _err in errors:
            _fit.add_error(**_err)

        return _fit

    def _get_test_fits(self):
        return {
            'default': self._get_fit(),
            'two_errors': self._get_fit(
                errors=[
                    dict(axis='y', err_val=1.0/np.sqrt(2)),
                    dict(axis='y', err_val=1.0/np.sqrt(2))
                ]
            ),
            'named_errors': self._get_fit(
                errors=[
                    dict(axis='y', err_val=1.0/np.sqrt(2), name="MyYDataError",
                         correlation=0, relative=False, reference='data'),
                    dict(axis='y', err_val=1.0/np.sqrt(2), name="MyYModelError",
                         correlation=0, relative=False, reference='model'),
                ]
            )
        }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            self._ref_prop_dict,
            fit_names=['default', 'two_errors'],
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars,
                y_model=self._nominal_fit_result_y_model,
                model=self._nominal_fit_result_xy_model,
                did_fit=True,
                cost_function_value=self._nominal_fit_result_cost,
            ),
            fit_names=['default', 'two_errors'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-2
        )

    def test_get_matching_error_all(self):
        _fit = self._get_test_fits()['named_errors']
        for _mc in (None, dict()):
            _errs = _fit.get_matching_errors(matching_criteria=_mc)
            self.assertEqual(len(_errs), 2)
            self.assertIs(_fit.data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])
            self.assertIs(_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_get_matching_error_name(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(name='MyYDataError'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(_fit.data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])

    def test_get_matching_error_type_simple(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(type='simple'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(_fit.data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])
        self.assertIs(_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_get_matching_error_type_matrix(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(type='matrix'))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_uncorrelated(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(correlated=True))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_correlated(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(correlated=False))
        self.assertEqual(len(_errs), 2)
        self.assertIs(_fit.data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])
        self.assertIs(_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])

    def test_get_matching_error_reference_data(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(reference='data'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(_fit.data_container._error_dicts['MyYDataError']['err'], _errs['MyYDataError'])

    def test_get_matching_error_reference_model(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(reference='model'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(_fit._param_model._error_dicts['MyYModelError']['err'], _errs['MyYModelError'])


class TestXYFitWithMatrixErrors(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        six.get_unbound_function(TestXYFitBasicInterface.setUp)(self)

    def _get_fit(self, errors=None):
        '''convenience'''

        _fit = XYFit(
            xy_data=self._ref_xy_data,
            model_function=simple_xy_model,
            cost_function=XYCostFunction_Chi2(
                axes_to_use='xy', errors_to_use='covariance'),
            minimizer=self.MINIMIZER
        )

        if errors is None:
            _fit.add_matrix_error(axis='y', err_matrix=np.eye(self._n_points), matrix_type='cov')
        else:
            for _err in errors:
                if 'err_matrix' in _err:
                    _fit.add_matrix_error(**_err)
                else:
                    _fit.add_error(**_err)

        return _fit

    def _get_test_fits(self):
        return {
            'default': self._get_fit(),
            'cor_matrix_and_error_vector': self._get_fit(
                errors=[
                    dict(axis='y',
                         err_matrix=np.eye(self._n_points),
                         matrix_type='cor',
                         err_val=1.0
                    )
                ]
            ),
            'two_matrix_errors': self._get_fit(
                errors=[
                    dict(axis='y', err_matrix=np.eye(self._n_points)/2, matrix_type='cov'),
                    dict(axis='y', err_matrix=np.eye(self._n_points)/2, matrix_type='cov')
                ]
            ),
            'one_matrix_one_simple_error': self._get_fit(
                errors=[
                    dict(axis='y', err_matrix=np.eye(self._n_points)/2, matrix_type='cov'),
                    dict(axis='y', err_val=1/np.sqrt(2))
                ]
            ),
            'named_errors': self._get_fit(
                errors=[
                    dict(axis='y', err_val=1.0/np.sqrt(2), name="MySimpleDataError",
                         correlation=0, relative=False, reference='data'),
                    dict(axis='y', err_val=1.0/np.sqrt(2), name="MySimpleModelError",
                         correlation=0, relative=False, reference='model'),
                    dict(axis='y', err_matrix=np.eye(self._n_points)/2, matrix_type='cov',
                         name="MyMatrixDataError",
                         relative=False, reference='data'),
                    dict(axis='y', err_matrix=np.eye(self._n_points)/2, matrix_type='cov',
                         name="MyMatrixModelError",
                         relative=False, reference='model'),
                ]
            )
        }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            self._ref_prop_dict,
            fit_names=['default', 'cor_matrix_and_error_vector',
                       'one_matrix_one_simple_error', 'two_matrix_errors'],
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars,
                y_model=self._nominal_fit_result_y_model,
                model=self._nominal_fit_result_xy_model,
                did_fit=True,
                cost_function_value=self._nominal_fit_result_cost,
            ),
            fit_names=['default', 'cor_matrix_and_error_vector',
                       'one_matrix_one_simple_error', 'two_matrix_errors'],
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-2
        )

    def test_get_matching_error_all(self):
        _fit = self._get_test_fits()['named_errors']
        for _mc in (None, dict()):
            _errs = _fit.get_matching_errors(matching_criteria=_mc)
            self.assertEqual(len(_errs), 4)
            self.assertIs(_fit.data_container._error_dicts['MySimpleDataError']['err'], _errs['MySimpleDataError'])
            self.assertIs(_fit._param_model._error_dicts['MySimpleModelError']['err'], _errs['MySimpleModelError'])
            self.assertIs(_fit.data_container._error_dicts['MyMatrixDataError']['err'], _errs['MyMatrixDataError'])
            self.assertIs(_fit._param_model._error_dicts['MyMatrixModelError']['err'], _errs['MyMatrixModelError'])

    def test_get_matching_error_name(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(name='MySimpleDataError'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(_fit.data_container._error_dicts['MySimpleDataError']['err'], _errs['MySimpleDataError'])

    def test_get_matching_error_type_simple(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(type='simple'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(_fit.data_container._error_dicts['MySimpleDataError']['err'], _errs['MySimpleDataError'])
        self.assertIs(_fit._param_model._error_dicts['MySimpleModelError']['err'], _errs['MySimpleModelError'])

    def test_get_matching_error_type_matrix(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(type='matrix'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(_fit.data_container._error_dicts['MyMatrixDataError']['err'], _errs['MyMatrixDataError'])
        self.assertIs(_fit._param_model._error_dicts['MyMatrixModelError']['err'], _errs['MyMatrixModelError'])

    def test_get_matching_error_uncorrelated(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(correlated=True))
        self.assertEqual(len(_errs), 0)

    def test_get_matching_error_correlated(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(correlated=False))
        # NOTE: passing 'correlated' only matches 'matrix' errors, irrespective of 'True'/'False' value passed
        self.assertEqual(len(_errs), 2)
        self.assertIs(_fit.data_container._error_dicts['MySimpleDataError']['err'], _errs['MySimpleDataError'])
        self.assertIs(_fit._param_model._error_dicts['MySimpleModelError']['err'], _errs['MySimpleModelError'])

    def test_get_matching_error_reference_data(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(reference='data'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(_fit.data_container._error_dicts['MySimpleDataError']['err'], _errs['MySimpleDataError'])
        self.assertIs(_fit.data_container._error_dicts['MyMatrixDataError']['err'], _errs['MyMatrixDataError'])

    def test_get_matching_error_reference_model(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(reference='model'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(_fit._param_model._error_dicts['MySimpleModelError']['err'], _errs['MySimpleModelError'])
        self.assertIs(_fit._param_model._error_dicts['MyMatrixModelError']['err'], _errs['MyMatrixModelError'])


class TestXYFitWithXYErrors(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        self._n_points = 16
        self._default_cost_function = XYCostFunction_Chi2(
            axes_to_use='xy', errors_to_use='covariance')

        # "jitter" for data smearing
        self._y_jitter = np.array([ 0.2991,  1.558 , -0.714 ,  0.825 , -1.157 ,  0.462 ,  0.103 ,
                                    1.167 ,  0.783 , -0.688 , -1.019 ,  0.14  , -0.11  , -0.87  ,
                                    1.81  ,  0.35  ])

        assert len(self._y_jitter) == self._n_points

        # reference initial values
        self._ref_initial_pars = np.array([1.0, 0.0])
        self._ref_x = np.arange(self._n_points)
        self._ref_initial_y_model = line_xy_model(self._ref_x, *self._ref_initial_pars)
        self._ref_initial_xy_model = np.array([self._ref_x, self._ref_initial_y_model])

        # reference matrices/errors
        self._ref_y_error_vector = np.ones(self._n_points) * 0.1
        self._ref_y_error_matrix = np.diag(self._ref_y_error_vector ** 2)
        self._ref_x_error_vector = np.ones(self._n_points)
        self._ref_x_error_matrix = np.diag(self._ref_x_error_vector ** 2)

        def line_xy_model_derivative(x, a=1.0, b=0.0):
            return a

        _fp = line_xy_model_derivative(self._ref_x, *self._ref_initial_pars)
        self._ref_projected_xy_matrix = (
            self._ref_y_error_matrix  +
            self._ref_x_error_matrix * np.outer(_fp, _fp)
        )
        self._ref_projected_xy_errors = np.diag(np.sqrt(self._ref_projected_xy_matrix))

        # fit data
        self._ref_y_data = self._ref_initial_y_model + self._y_jitter
        self._ref_xy_data = np.array([self._ref_x, self._ref_y_data])

        # pre-fit cost value
        self._ref_initial_cost = self._default_cost_function(
            self._ref_y_data,
            self._ref_initial_y_model,
            np.linalg.inv(self._ref_projected_xy_matrix),
            self._ref_initial_pars,
            [],
        )

        # reference fit result values
        self._nominal_fit_result_pars = np.array([1.02590618, -0.00967721])

        self._nominal_fit_result_y_model = line_xy_model(self._ref_x, *self._nominal_fit_result_pars)
        self._nominal_fit_result_xy_model = np.array([self._ref_x, self._nominal_fit_result_y_model])

        self._nominal_fit_result_projected_xy_matrix = 1.0624835 * np.eye(self._n_points)
        self._nominal_fit_result_projected_xy_errors = np.diag(np.sqrt(self._nominal_fit_result_projected_xy_matrix))

        self._nominal_fit_result_cost = self._default_cost_function(
            self._ref_y_data,
            self._nominal_fit_result_y_model,
            np.linalg.inv(self._nominal_fit_result_projected_xy_matrix),
            self._nominal_fit_result_pars,
            [],
        )

        # helper dict with all reference property values
        self._ref_prop_dict = dict(
            parameter_values=self._ref_initial_pars,

            x_data=self._ref_x,
            x_model=self._ref_x,
            y_data=self._ref_y_data,
            y_model=self._ref_initial_y_model,
            data=self._ref_xy_data,
            model=self._ref_initial_xy_model,
            did_fit=False,

            x_data_cov_mat=self._ref_x_error_matrix,
            y_data_cov_mat=self._ref_y_error_matrix,
            x_data_cov_mat_inverse=np.linalg.inv(self._ref_x_error_matrix),
            y_data_cov_mat_inverse=np.linalg.inv(self._ref_y_error_matrix),
            x_data_cor_mat=np.eye(self._n_points),
            y_data_cor_mat=np.eye(self._n_points),
            x_data_error=self._ref_x_error_vector,
            y_data_error=self._ref_y_error_vector,

            cost_function_value=self._ref_initial_cost,

            total_cov_mat=self._ref_projected_xy_matrix,
            total_error=self._ref_projected_xy_errors,
        )

    def _get_fit(self, errors=None, constraints=None, dynamic_error_algorithm=None):
        '''convenience'''

        errors = errors or [dict(axis='y', err_val=1.0)]
        constraints = constraints or []
        dynamic_error_algorithm = dynamic_error_algorithm or "nonlinear"

        _fit = XYFit(
            xy_data=self._ref_xy_data,
            model_function=line_xy_model,
            cost_function=self._default_cost_function,
            minimizer=self.MINIMIZER,
            dynamic_error_algorithm=dynamic_error_algorithm
        )

        for _err in errors:
            _fit.add_error(**_err)
        for _constraint in constraints:
            _fit.add_parameter_constraint(**_constraint)

        _fit.set_all_parameter_values(self._ref_initial_pars)

        return _fit

    def _get_test_fits(self):
        return {
            'nonlinear_with_x_errors': self._get_fit(
                errors=[
                    dict(axis='x', err_val=self._ref_x_error_vector),
                    dict(axis='y', err_val=self._ref_y_error_vector),
                ]
            ),
        }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            self._ref_prop_dict,
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars,
                y_model=self._nominal_fit_result_y_model,
                model=self._nominal_fit_result_xy_model,
                did_fit=True,
                cost_function_value=self._nominal_fit_result_cost,

                total_cov_mat=self._nominal_fit_result_projected_xy_matrix,
                total_error=self._nominal_fit_result_projected_xy_errors,

            ),
            call_before_fit=lambda f: f.do_fit(),
            rtol=1e-3,
            atol=1e-2,
        )

    def test_limit_parameter_raise(self):
        with self.assertRaises(ValueError):
            self._get_fit().limit_parameter("a")

    def test_limit_parameter_raise_if_not_numeric(self):
        with self.assertRaises(TypeError):
            # old tuple syntax no longer supported
            self._get_fit().limit_parameter("a", (0, 1))
        with self.assertRaises(TypeError):
            # non-numeric types invalid (except None)
            self._get_fit().limit_parameter("a", "invalid", "limits")
        with self.assertRaises(TypeError):
            # string representations of numerics should also fail
            self._get_fit().limit_parameter("a", "0.3", "14")

    def test_iterative_linear_and_relative_model_error(self):
        _constraints = [
            dict(name="a", value=1.0, uncertainty=0.1),
            dict(name="b", value=0.3, uncertainty=0.1)
        ]
        _errors_rel_data = [
            dict(axis="x", err_val=0.1, relative=False, reference="data"),
            dict(axis="y", err_val=1.0, relative=False, reference="data"),
            dict(axis="y", err_val=0.1, relative=True, reference="data")
        ]
        _errors_rel_model = [
            dict(axis="x", err_val=0.1, relative=False, reference="data"),
            dict(axis="y", err_val=1.0, relative=False, reference="data"),
            dict(axis="y", err_val=0.1, relative=True, reference="model")
        ]
        _fit_rel_data_nonlinear = self._get_fit(
            errors=_errors_rel_data, constraints=_constraints,
            dynamic_error_algorithm="nonlinear")
        _fit_rel_data_iterative = self._get_fit(
            errors=_errors_rel_data, constraints=_constraints,
            dynamic_error_algorithm="iterative")
        _fit_rel_model_nonlinear = self._get_fit(
            errors=_errors_rel_model, constraints=_constraints,
            dynamic_error_algorithm="nonlinear")
        _fit_rel_model_iterative = self._get_fit(
            errors=_errors_rel_model, constraints=_constraints,
            dynamic_error_algorithm="iterative")
        _fit_rel_data_nonlinear.do_fit()
        _fit_rel_data_iterative.do_fit()
        _fit_rel_model_nonlinear.do_fit()
        _fit_rel_model_iterative.do_fit()

        print("============ rel data nonlinear vs rel data iterative ==========\n")
        self._assert_fit_results_equal(
            _fit_rel_data_nonlinear, _fit_rel_data_iterative, rtol=1e-2)
        print("============ rel data nonlinear vs rel model nonlinear ==========\n")
        self._assert_fit_results_equal(
            _fit_rel_data_nonlinear, _fit_rel_model_nonlinear, rtol=1e-2)
        print("============ rel data nonlinear vs rel model iterative ==========\n")
        self._assert_fit_results_equal(
            _fit_rel_data_nonlinear, _fit_rel_model_iterative, rtol=1e-2)
        print("============ rel model nonlinear vs rel model iterative ==========\n")
        self._assert_fit_results_equal(
            _fit_rel_model_nonlinear, _fit_rel_model_iterative, rtol=1e-2)
        print("============ rel data iterative vs rel model iterative ==========\n")
        self._assert_fit_results_equal(
            _fit_rel_data_iterative, _fit_rel_model_iterative, rtol=1e-2)
        print("============ rel data iterative vs rel model nonlinear ==========\n")
        self._assert_fit_results_equal(
            _fit_rel_data_iterative, _fit_rel_model_nonlinear, rtol=1e-2)

