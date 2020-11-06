import abc
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class AbstractTestFit(object):

    MINIMIZER = None

    @abc.abstractmethod
    def setUp(self):
        pass

    @property
    def _fit_result_attributes(self):
        return [
            "cost_function_value", "parameter_names", "parameter_values", "parameter_errors",
            "parameter_cov_mat"
        ]

    def _assert_fit_results_equal(
            self, fit_1, fit_2, rtol=1e-3, atol=1e-6, check_cost_function_value=True,
            fit_2_is_double_fit=False, fit_2_permutation=None):
        for _attr in self._fit_result_attributes:
            if not check_cost_function_value and _attr == "cost_function_value":
                continue
            with self.subTest(attr=_attr):
                _attr_1 = getattr(fit_1, _attr)
                _attr_2 = getattr(fit_2, _attr)
                if fit_2_is_double_fit and _attr_2 is not None:
                    if _attr in ["cost_function_value", "parameter_cov_mat"]:
                        _attr_2 *= 0.5
                    if _attr in ["parameter_errors"] and fit_2.did_fit:
                        _attr_2 *= np.sqrt(2)
                if fit_2_permutation is not None:
                    _attr_2_arr = np.array(_attr_2)
                    if _attr_2_arr.ndim == 1:
                        _attr_2_arr = _attr_2_arr[fit_2_permutation]
                    elif _attr_2_arr.ndim == 2:
                        _attr_2_arr = _attr_2_arr[fit_2_permutation, :][:, fit_2_permutation]
                    if isinstance(_attr_2, np.ndarray):
                        _attr_2 = _attr_2_arr
                    elif isinstance(_attr_2, list):
                        _attr_2 = _attr_2_arr.tolist()
                    elif isinstance(_attr_2, tuple):
                        _attr_2 = tuple(_attr_2_arr)
                self._assert_values_equal(
                    name=_attr, received_value=_attr_2, expected_value=_attr_1,
                    rtol=rtol, atol=atol)

    def _assert_fit_properties(self, fit, properties, rtol=1e-3, atol=1e-6):
        for _attr, _ref_val in properties.items():
            with self.subTest(attr=_attr):
                # retrieve property value
                _attr_val = getattr(fit, _attr)
                self._assert_values_equal(
                    name=_attr, received_value=_attr_val, expected_value=_ref_val,
                    rtol=rtol, atol=atol)

    def _assert_values_equal(self, name, received_value, expected_value, rtol=1e-3, atol=1e-6):
        # check type identical to ref
        self.assertIs(type(received_value), type(expected_value))

        # check value (almost) equal to ref
        try:
            self.assertTrue(
                (np.all(np.isnan(received_value))
                 and np.all(np.isnan(expected_value))
                 and np.array(received_value).shape == np.array(expected_value).shape)
                or
                np.allclose(
                    np.asarray(received_value),
                    np.asarray(expected_value),
                    rtol=rtol,
                    atol=atol,
                )
            )
        except ValueError:
            # most likely array length mismatch
            print("\nCheck failed: attribute {!r} "
                  "should be:\n\t{}\nand is:\n\t{}".format(
                name, expected_value, received_value))
            raise
        except TypeError:
            # types contained in array do not support 'allclose'
            try:
                self.assertEqual(received_value, expected_value)
            except:
                print("\nCheck failed: attribute {!r} "
                      "should be exactly:\n\t{}\nand is:\n\t{}".format(
                    name, expected_value, received_value))
                raise
        except:
            _abs_diffs = np.abs(expected_value - received_value)
            _min_abs_diff, _max_abs_diff = np.min(_abs_diffs), np.max(_abs_diffs)

            _rel_diffs = np.abs(received_value / expected_value - 1)
            _min_rel_diff, _max_rel_diff = np.nanmin(_rel_diffs), np.nanmax(_rel_diffs)

            print('_' * 70 +
                  "\nCheck failed: attribute {!r} "
                  "should be approximately:\n\t{}\n"
                  "within:\n\t{!r}\nand is:\n\t{}\n"
                  "(abs diff between: {:g} and {:g})\n"
                  "(rel diff between: {:g} and {:g})\n".format(
                      name, expected_value, dict(rtol=rtol, atol=atol), received_value,
                      _min_abs_diff, _max_abs_diff,
                      _min_rel_diff, _max_rel_diff) + '^' * 70)
            raise

    def _get_test_fits(self):
        pass

    def run_test_for_all_fits(self, ref_prop_dict, call_before_fit=None, fit_names=None, **kwargs):
        for _fit_name, _fit in self._get_test_fits().items():
            # skip non-requested
            if fit_names is not None and _fit_name not in fit_names:
                continue
            with self.subTest(fit=_fit_name):
                # call a user-supplied function
                if call_before_fit:
                    call_before_fit(_fit)

                # test object properties
                self._assert_fit_properties(
                    _fit,
                    ref_prop_dict,
                    **kwargs
                )
