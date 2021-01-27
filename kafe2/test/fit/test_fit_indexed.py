import abc
import unittest2 as unittest
import numpy as np
import six

from kafe2.core.minimizers import AVAILABLE_MINIMIZERS
from kafe2.core.fitters import NexusFitterException

from kafe2.config import kc

from kafe2.fit import IndexedFit
from kafe2.fit.indexed.fit import IndexedFitException
from kafe2.fit.indexed.model import IndexedModelFunctionException, IndexedParametricModelException
from kafe2.fit.indexed.cost import IndexedCostFunction_Chi2

from kafe2.test.fit.test_fit import AbstractTestFit


def simple_chi2(data, model):
    return np.sum((data - model)**2)

def simple_chi2_explicit_model_name(data, simple_indexed_model):
    return np.sum((data - simple_indexed_model)**2)

def simple_indexed_model(a=1.1, b=2.2, c=3.3):
    x = np.arange(10)
    return a * x ** 2 + b * x + c

def line_indexed_model(a=3.0, b=0.0):
    x = np.arange(10)
    return a * x + b


def analytic_solution(des_mat, cov_mat_inv, data):
    return (
        np.squeeze(np.asarray(np.linalg.inv(
            des_mat.T.dot(cov_mat_inv).dot(des_mat)
        ).dot(des_mat.T)
         .dot(cov_mat_inv)
         .dot(data))))


class TestIndexedFitBasicInterface(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        self._n_points = 10

        # "jitter" for data smearing
        self._jitter = np.array([-0.3193475 , -1.2404198 , -1.4906926 , -0.78832446,
                                   -1.7638106,   0.36664261,  0.49433821,  0.0719646,
                                    1.95670326,  0.31200215])
        assert len(self._jitter) == self._n_points

        # reference initial values
        self._ref_initial_pars = np.array([1.1, 2.2, 3.3])
        self._ref_initial_model = simple_indexed_model(*self._ref_initial_pars)

        # fit data
        self._ref_data = self._ref_initial_model + self._jitter

        self._linear_design_matrix = np.array([
            np.arange(10) ** 2,
            np.arange(10),
            np.ones(10)
        ]).T

        # pre-fit cost value
        self._ref_initial_cost = simple_chi2(
            data=self._ref_data,
            model=self._ref_initial_model,
        )

        # reference matrices/errors
        self._ref_error = np.ones(self._n_points)
        self._ref_matrix_eye = np.eye(self._n_points)

        # reference fit result values
        #self._nominal_fit_result_pars = (1.1351433, 2.13736919, 2.33346549)
        self._nominal_fit_result_pars = analytic_solution(
            self._linear_design_matrix,
            np.linalg.inv(self._ref_matrix_eye),
            self._ref_data,
        )

        self._nominal_fit_result_model = simple_indexed_model(*self._nominal_fit_result_pars)
        self._nominal_fit_result_cost = simple_chi2(
            data=self._ref_data,
            model=self._nominal_fit_result_model,
        )

        # helper dict with all reference property values
        self._ref_prop_dict = dict(
            did_fit=False,
            model_count=1,

            parameter_values=self._ref_initial_pars,
            parameter_names=('a', 'b', 'c'),
            cost_function_value=self._ref_initial_cost,

            data=self._ref_data,
            data_error=self._ref_error,
            data_cov_mat=self._ref_matrix_eye,
            data_cov_mat_inverse=self._ref_matrix_eye,
            data_cor_mat=self._ref_matrix_eye,

            model=self._ref_initial_model,
            model_error=self._ref_error * 0,
            model_cov_mat=self._ref_matrix_eye * 0,
            model_cov_mat_inverse=None,
            model_cor_mat=self._ref_matrix_eye * np.nan,

            total_error=self._ref_error,
            total_cov_mat=self._ref_matrix_eye,
            total_cov_mat_inverse=self._ref_matrix_eye,
            total_cor_mat=self._ref_matrix_eye,
        )

    def _get_fit(
            self, model_function=None, cost_function=None, errors=None,
            dynamic_error_algorithm=None):
        '''convenience'''
        model_function = model_function or simple_indexed_model
        # TODO: fix default
        cost_function = cost_function or IndexedCostFunction_Chi2(
            errors_to_use='covariance')
        errors = errors or [dict(err_val=1.0)]
        dynamic_error_algorithm = dynamic_error_algorithm or "nonlinear"

        _fit = IndexedFit(
            data=self._ref_data,
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
            'relative_errors_data': self._get_fit(
                errors=[dict(err_val=1.0 / self._ref_data, relative=True, reference='data')]
            ),
        }

    def test_initial_state(self):
        self.run_test_for_all_fits(
            self._ref_prop_dict
        )

    def test_fit_results(self):
        self.run_test_for_all_fits(
            dict(self._ref_prop_dict,
                parameter_values=self._nominal_fit_result_pars,
                model=self._nominal_fit_result_model,
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
                data=self._ref_data,
                model=self._nominal_fit_result_model,
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
        #with self.assertRaises(IndexedFitException):
        with self.assertRaises(NexusFitterException):
            self._get_fit().set_all_parameter_values((1,))
        #with self.assertRaises(IndexedFitException):
        with self.assertRaises(NexusFitterException):
            self._get_fit().set_all_parameter_values((1,2,3,4,5))

    def test_parameter_defaults(self):
        def dummy_model(a, b, c):
            return np.arange(10)

        _fit = self._get_fit(model_function=dummy_model)

        self._assert_fit_properties(
            _fit,
            dict(
                parameter_values=np.array([1, 1, 1]),
            )
        )

    def test_parameter_partial_defaults(self):
        def dummy_model(a, b, c=3.3):
            return np.arange(10)

        _fit = self._get_fit(model_function=dummy_model)

        self._assert_fit_properties(
            _fit,
            dict(
                parameter_values=np.array([1, 1, 3.3]),
            )
        )

    def test_update_data(self):
        _fit = self._get_fit()

        _fit.data = self._ref_data * 2
        _fit.add_error(err_val=1.0)

        self._assert_fit_properties(
            _fit,
            dict(
                data=self._ref_data * 2,
            )
        )

        _fit.do_fit()
        _new_estimates = np.array(self._nominal_fit_result_pars) * 2

        self._assert_fit_properties(
            _fit,
            dict(
                data=self._ref_data * 2,
                parameter_values=_new_estimates,
            ),
            rtol=1e-2
        )

    def test_update_data_length_mismatch_raise(self):
        # TODO: update when different length limitation is fixed
        _fit = self._get_fit()
        with self.assertRaises(IndexedParametricModelException):
            _fit.data = np.arange(self._n_points + 1)

    def test_reserved_parameter_names_raise(self):
        def dummy_model(data):
            pass

        with self.assertRaises(IndexedFitException) as _exc:
            IndexedFit(data=self._ref_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn('reserved', _exc.exception.args[0])
        self.assertIn('data', _exc.exception.args[0])

    def dummy_model_no_pars_raise(self):
        def dummy_model():
            pass

        with self.assertRaises(IndexedModelFunctionException) as _exc:
            IndexedFit(data=self._ref_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn(
            'needs at least one parameter',
            _exc.exception.args[0])

    def dummy_model_varargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, *varargs):
            pass

        with self.assertRaises(IndexedModelFunctionException) as _exc:
            IndexedFit(data=self._ref_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varargs', _exc.exception.args[0])

    def dummy_model_varkwargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, **varkwargs):
            pass

        with self.assertRaises(IndexedModelFunctionException) as _exc:
            IndexedFit(data=self._ref_data,
                  model_function=dummy_model,
                  minimizer=self.MINIMIZER)

        self.assertIn('variable', _exc.exception.args[0])
        self.assertIn('varkwargs', _exc.exception.args[0])

    def dummy_model_varargs_varkwargs_raise(self):
        # TODO: raise even without 'par'
        def dummy_model(x, par, *varargs, **varkwargs):
            pass

        with self.assertRaises(IndexedModelFunctionException) as _exc:
            IndexedFit(data=self._ref_data,
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

    def test_relative_model_error_nonlinear(self):
        _fit_data_err = self._get_fit(errors=[
            dict(err_val=1.0, relative=False, reference="data"),
            dict(err_val=0.1, relative=True, reference="data")
        ])
        _fit_model_err = self._get_fit(errors=[
            dict(err_val=1.0, relative=False, reference="data"),
            dict(err_val=0.1, relative=True, reference="model")
        ], dynamic_error_algorithm="nonlinear")
        _fit_data_err.do_fit()
        _fit_model_err.do_fit()
        self._assert_fit_results_equal(_fit_data_err, _fit_model_err, rtol=1e-2, atol=2e-2)

    def test_relative_model_error_iterative(self):
        _fit_data_err = self._get_fit(errors=[
            dict(err_val=1.0, relative=False, reference="data"),
            dict(err_val=0.1, relative=True, reference="data")
        ])
        _fit_model_err = self._get_fit(errors=[
            dict(err_val=1.0, relative=False, reference="data"),
            dict(err_val=0.1, relative=True, reference="model")
        ], dynamic_error_algorithm="iterative")
        _fit_data_err.do_fit()
        _fit_model_err.do_fit()
        self._assert_fit_results_equal(_fit_data_err, _fit_model_err, rtol=1e-2)


class TestIndexedFitWithSimpleErrors(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        six.get_unbound_function(TestIndexedFitBasicInterface.setUp)(self)

    def _get_fit(self, errors=None):
        '''convenience'''

        errors = errors or [dict(err_val=1.0)]

        _fit = IndexedFit(
            data=self._ref_data,
            model_function=simple_indexed_model,
            cost_function=IndexedCostFunction_Chi2(errors_to_use='covariance'),
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
                    dict(err_val=1.0/np.sqrt(2)),
                    dict(err_val=1.0/np.sqrt(2))
                ]
            ),
            'named_errors': self._get_fit(
                errors=[
                    dict(err_val=1.0/np.sqrt(2), name="MyDataError",
                         correlation=0, relative=False, reference='data'),
                    dict(err_val=1.0/np.sqrt(2), name="MyModelError",
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
                model=self._nominal_fit_result_model,
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
            self.assertIs(_fit.data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])
            self.assertIs(_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])

    def test_get_matching_error_name(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(name='MyDataError'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(_fit.data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])

    def test_get_matching_error_type_simple(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(type='simple'))
        self.assertEqual(len(_errs), 2)
        self.assertIs(_fit.data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])
        self.assertIs(_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])

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
        self.assertIs(_fit.data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])
        self.assertIs(_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])

    def test_get_matching_error_reference_data(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(reference='data'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(_fit.data_container._error_dicts['MyDataError']['err'], _errs['MyDataError'])

    def test_get_matching_error_reference_model(self):
        _fit = self._get_test_fits()['named_errors']
        _errs = _fit.get_matching_errors(matching_criteria=dict(reference='model'))
        self.assertEqual(len(_errs), 1)
        self.assertIs(_fit._param_model._error_dicts['MyModelError']['err'], _errs['MyModelError'])


class TestIndexedFitWithMatrixErrors(AbstractTestFit, unittest.TestCase):

    MINIMIZER = 'scipy'

    def setUp(self):
        six.get_unbound_function(TestIndexedFitBasicInterface.setUp)(self)

    def _get_fit(self, errors=None):
        '''convenience'''

        _fit = IndexedFit(
            data=self._ref_data,
            model_function=simple_indexed_model,
            cost_function=IndexedCostFunction_Chi2(
                errors_to_use='covariance'),
            minimizer=self.MINIMIZER
        )

        if errors is None:
            _fit.add_matrix_error(err_matrix=np.eye(self._n_points), matrix_type='cov')
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
                    dict(err_matrix=np.eye(self._n_points),
                         matrix_type='cor',
                         err_val=1.0
                    )
                ]
            ),
            'two_matrix_errors': self._get_fit(
                errors=[
                    dict(err_matrix=np.eye(self._n_points)/2, matrix_type='cov'),
                    dict(err_matrix=np.eye(self._n_points)/2, matrix_type='cov')
                ]
            ),
            'one_matrix_one_simple_error': self._get_fit(
                errors=[
                    dict(err_matrix=np.eye(self._n_points)/2, matrix_type='cov'),
                    dict(err_val=1/np.sqrt(2))
                ]
            ),
            'named_errors': self._get_fit(
                errors=[
                    dict(err_val=1.0/np.sqrt(2), name="MySimpleDataError",
                         correlation=0, relative=False, reference='data'),
                    dict(err_val=1.0/np.sqrt(2), name="MySimpleModelError",
                         correlation=0, relative=False, reference='model'),
                    dict(err_matrix=np.eye(self._n_points)/2, matrix_type='cov',
                         name="MyMatrixDataError",
                         relative=False, reference='data'),
                    dict(err_matrix=np.eye(self._n_points)/2, matrix_type='cov',
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
                model=self._nominal_fit_result_model,
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
