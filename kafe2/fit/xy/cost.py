import numpy as np

from .._base import (CostFunction, CostFunction_Chi2, CostFunction_NegLogLikelihood,
                     CostFunctionException)
from .._base.cost import CostFunctionBase_Chi2_Nuisance

__all__ = [
    "XYCostFunction_Chi2",
    "XYCostFunction_Chi2_Nuisance",
    "XYCostFunction_NegLogLikelihood"
]

# TODO replace with calls to _base/cost/_generic_chi2_nuisance if possible

def _generic_xy_chi2_nuisance_pointwise(
        x_data, x_model,
        y_data, y_model,
        x_total_error=None,
        y_total_error=None,
        fail_on_zeros=False):
    '''generic implementation of a least-squares cost function for pointwise errors and using nuisance parameters'''

    if y_model.shape != y_data.shape or y_model.shape != x_model.shape or x_model.shape != x_data.shape:
        raise CostFunctionException(
            "x_data, x_model, 'y_data' and 'y_model' must have the same shape! Got %r, %r, %r and %r..."
                                % (x_data.shape, x_model.shape, y_data.shape, y_model.shape))

    _x_res = (x_data - x_model)
    _y_res = (y_data - y_model)

    if x_total_error.any() == 0.0:
        if fail_on_zeros:
            raise CostFunctionException("'x_err' must not contain any zero values!")
    else:
        #with x-erros
        _x_penalties = np.sum(_x_res ** 2 * x_total_error)
        if y_total_error is not None:
            if y_total_error.any() == 0.0:
                if fail_on_zeros:
                    raise CostFunctionException("'y_err' must not contain any zero values!")
                return _y_res.dot(_y_res) + _x_penalties
            _y_res = _y_res / y_total_error
            _chi2 = np.sum(_y_res ** 2)
            return _chi2 + _x_penalties
        else:
            _chi2 = _y_res.dot(_y_res)
            return _chi2 + _x_penalties

    #without x-errors
    if y_total_error is not None:
        if y_total_error.any() == 0.0:
            if fail_on_zeros:
                raise CostFunctionException("'y_err' must not contain any zero values!")
        else:
            _y_res = _y_res / y_total_error
            _chi2 = np.sum(_y_res ** 2)
            return _chi2

    #cost function value without any errors
    return _y_res.dot(_y_res)


def _generic_xy_chi2_nuisance_covariance(
    x_data, x_model,
    y_data, y_model,
    x_uncor_cov_mat_inverse=None,
    y_uncor_cov_mat_inverse=None,
    y_nuisance_cor_design_mat=None,
    y_nuisance_vector=np.array([]),
    fail_on_no_x_matrix=False,
    fail_on_no_y_matrix=False):
    '''generic implementation of a least-squares cost function for covariance-matrix errors and using nuisance parameters'''

    if y_model.shape != y_data.shape or y_model.shape != x_model.shape or x_model.shape != x_data.shape:
        raise CostFunctionException(
            "x_data, x_model, 'y_data' and 'y_model' must have the same shape! Got %r, %r, %r and %r..."
                                % (x_data.shape, x_model.shape, y_data.shape, y_model.shape))

    _x_res = (x_data - x_model)
    _y_res = (y_data - y_model)

    if x_uncor_cov_mat_inverse is None:
        if fail_on_no_x_matrix:
            raise np.linalg.LinAlgError("Uncorrelated X Covariance matrix is singular!")
        if y_uncor_cov_mat_inverse is None:
            if y_nuisance_vector.all() == 0.0:
                # raise if uncorrelated matrix is None and the correlated is not None
                raise CostFunctionException('Is not working for only fullcorrelated y-errors')
            if fail_on_no_y_matrix:
                raise np.linalg.LinAlgError("Uncorrelated Y Covariance matrix is singular!")
            # cost function values without any errors
            _chisquare = _y_res.dot(_y_res)
            return _chisquare

        else:
            # with y-errors but without x-errors
            _inner_sum = np.squeeze(np.asarray(y_nuisance_vector.dot(y_nuisance_cor_design_mat)))
            _y_penalties = y_nuisance_vector.dot(y_nuisance_vector)
            _chisquare = (_y_res - _inner_sum).dot(y_uncor_cov_mat_inverse).dot(_y_res - _inner_sum)
            return (_y_penalties + _chisquare)

    else:
        _x_penalties = np.transpose(_x_res).dot(x_uncor_cov_mat_inverse).dot(_x_res)
        if y_uncor_cov_mat_inverse is None:
            if y_nuisance_vector.all() == 0.0:
                raise CostFunctionException('Is not working for only fullcorrelated y-errors')
            if fail_on_no_y_matrix:
                raise np.linalg.LinAlgError("Uncorrelated Y Covariance matrix is singular!")
            # with x-errors but without y-errors
            _chisquare = _y_res.dot(_y_res)
            return (_chisquare + _x_penalties)

        else:
            # with x- and y-errors
            _inner_sum = np.squeeze(np.asarray(y_nuisance_vector.dot(y_nuisance_cor_design_mat)))
            _y_penalties = y_nuisance_vector.dot(y_nuisance_vector)
            _chi2 = (_y_res - _inner_sum).dot(y_uncor_cov_mat_inverse).dot(_y_res - _inner_sum)
            return _chi2 + _x_penalties + _y_penalties


class XYCostFunction_Chi2(CostFunction_Chi2):
    def __init__(self, errors_to_use='covariance', fallback_on_singular=True, axes_to_use='xy'):
        """
        Built-in least-squares cost function for *xy* data.

        :param errors_to_use: which errors to use when calculating :math:`\chi^2`
        :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
        :param axes_to_use: take into account errors for which axes
        :type axes_to_use: ``'y'`` or ``'xy'``
        """
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        if axes_to_use.lower() == 'y':
            self._COV_MAT_INVERSE_NAME = "y_total_cov_mat_inverse"
            self._ERROR_NAME = "y_total_error"
        elif axes_to_use.lower() == 'xy':
            self._COV_MAT_INVERSE_NAME = "projected_xy_total_cov_mat_inverse"
            self._ERROR_NAME = "projected_xy_total_error"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y')")
        super(XYCostFunction_Chi2, self).__init__(
            errors_to_use=errors_to_use, fallback_on_singular=fallback_on_singular)

    def on_no_errors(self):
        if not self._no_errors_warning_printed:
            warnings.warn('No y data errors were specified. Will set y data errors to 1 '
                  'if total error becomes 0.')
            self._no_errors_warning_printed = True


class XYCostFunction_NegLogLikelihood(CostFunction_NegLogLikelihood):
    def __init__(self, data_point_distribution="poisson", ratio=False, axes_to_use="xy"):
        self._DATA_NAME = "y_data"
        self._MODEL_NAME = "y_model"
        if axes_to_use.lower() == 'y':
            self._ERROR_NAME = "y_total_error"
        elif axes_to_use.lower() == 'xy':
            self._ERROR_NAME = "projected_xy_total_error"
        else:
            raise CostFunctionException(
                "Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y')")
        super(XYCostFunction_NegLogLikelihood, self).__init__(
            data_point_distribution=data_point_distribution, ratio=ratio)

    def on_no_errors(self):
        if not self._no_errors_warning_printed:
            warnings.warn('No y data errors were specified. Will set y data errors to 1 '
                  'if total error becomes 0.')
            self._no_errors_warning_printed = True


class XYCostFunction_Chi2_Nuisance(CostFunctionBase_Chi2_Nuisance):

    def __init__(self, axes_to_use='xy', errors_to_use='covariance', fallback_on_singular=True):

        """
        Built-in least-squares cost function with nuisanceparameters for *xy* data.

        :param errors_to_use: which errors to use when calculating :math:`\chi^2`
        :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
        :param axes_to_use: take into account errors for which axes
        :type axes_to_use: ``'y'`` or ``'xy'``
        """

        if errors_to_use is None:
            _chi2_nui = self.chi2_no_error

        elif errors_to_use == 'covariance':
            if axes_to_use == 'y':
                if fallback_on_singular:
                    _chi2_nui = self.chi2_nui_cov_fallback_y
                else:
                    _chi2_nui = self.chi2_nui_cov_y
            elif axes_to_use == 'x':
                if fallback_on_singular:
                    _chi2_nui = self.chi2_nui_cov_fallback_x
                else:
                    _chi2_nui = self.chi2_nui_cov_x
            elif axes_to_use == 'xy':
                if fallback_on_singular:
                    _chi2_nui = self.chi2_nui_cov_fallback_xy
                else:
                    _chi2_nui = self.chi2_nui_cov_xy
            else:
                raise CostFunctionException("Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y', 'x')")
        elif errors_to_use == 'pointwise':
            if axes_to_use == 'y':
                if fallback_on_singular:
                    _chi2_nui = self.chi2_nui_pointwise_fallback_y
                else:
                    _chi2_nui = self.chi2_nui_pointwise_y
            elif axes_to_use == 'x':
                if fallback_on_singular:
                    _chi2_nui = self.chi2_nui_pointwise_fallback_x
                else:
                    _chi2_nui = self.chi2_nui_pointwise_x
            elif axes_to_use == 'xy':
                if fallback_on_singular:
                    _chi2_nui = self.chi2_nui_pointwise_fallback_xy
                else:
                    _chi2_nui = self.chi2_nui_pointwise_xy
            else:
                raise CostFunctionException("Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y', 'x')")
        else:
            raise CostFunctionException("Unknown value '%s' for 'errors_to_use': must be one of (None, 'covariance', 'pointwise')")


        super(CostFunction_Chi2, self).__init__(cost_function=_chi2_nui)

        #set the needed flags
        if errors_to_use is None:
            self._formatter.latex_name = r"\chi^{2}_{nui}"
        elif errors_to_use == 'covariance':
            if axes_to_use == 'y':
                self.set_flag('need_y_nuisance', True)
                self._formatter.latex_name = r"\chi^{2}_{nui}(\sigma_y)"
            elif axes_to_use == 'x':
                self.set_flag('need_x_nuisance', True)
                self._formatter.latex_name = r"\chi^{2}_{nui}(\sigma_x)"
            elif axes_to_use == 'xy':
                self.set_flag('need_x_nuisance', True)
                self.set_flag('need_y_nuisance', True)
                self._formatter.latex_name = r"\chi^{2}_{nui}(\sigma_x,\sigma_y)"
        else:
            if axes_to_use == 'y':
                self._formatter.latex_name = r"\chi^{2}_{nui}(\sigma_y)"
            elif axes_to_use == 'x':
                self._formatter.latex_name = r"\chi^{2}_{nui}(\sigma_x)"
                self.set_flag('need_x_nuisance', True)
            else:
                self._formatter.latex_name = r"\chi^{2}_{nui}(\sigma_x, \sigma_y)"
                self.set_flag('need_x_nuisance', True)

    @staticmethod
    def chi2_no_error(y_data, y_model):
        r"""A least-squares cost function calculated from `y` data and model values,
        without considering uncertainties:

        .. math::
            C = \chi^2 =
                ({\bf d}_{y} - {\bf m}_{y})
                \cdot
                ({\bf d}_{y} - {\bf m}_{y})
                +
                C({\bf p})

        In the above, :math:`{\bf d}_{y}` are the measurements
        :math:`{\bf m}_{y}` are the model predictions,
        :math:`{\bf p}` are the model parameters,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param y_data: `y` measurement data :math:`{\bf d}_{y}`
        :param y_model: `y` model predictions :math:`{\bf m}_{y}`

        :return: cost function value
        """
        return CostFunction_Chi2.chi2_no_errors(data=y_data, model=y_model)

    @staticmethod
    def chi2_nui_cov_y(y_data, y_model, y_total_uncor_cov_mat_inverse, _y_total_nuisance_cor_design_mat,
                       y_nuisance_vector):
        r"""A least-squares cost function which uses nuisance parameters to account for correlated
        `y` uncertainties.

        The cost function is given by:

        .. math::
            C = \chi^2 =
                ({\bf d}_{y} - {\bf m}_{y} - {\bf G}{\bf b})^{\top}
                ({\bf V}_{y}^{\mathrm{uncor}})^{-1}
                ({\bf d}_{y} - {\bf m}_{y} - {\bf G}{\bf b})
                +
                {\bf b}^2
                +
                C({\bf p})

        In the above, :math:`{\bf d}_{y}` are the `y` measurements,
        :math:`{\bf m}_{y}` are the `y` model predictions,
        :math:`{\bf G}` is the design matrix containing the correlated parts of all `y` uncertainties,
        :math:`{\bf V}_{y}^{\mathrm{uncor}}` is the uncorrelated part of the total `y` covariance matrix,
        :math:`{\bf b}` is the vector of nuisance parameters,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param y_data: `y` measurement data :math:`{\bf d}_{y}`
        :param y_model: `y` model predictions :math:`{\bf m}_{y}`
        :param y_total_uncor_cov_mat_inverse: inverse :math:`({\bf V}_{y}^{\mathrm{uncor}})^{-1}` of the uncorrelated part of the total `y` covariance matrix
        :param _y_total_nuisance_cor_design_mat: design matrix :math:`{\bf G}` containing correlated `y` uncertainties
        :param y_nuisance_vector: nuisance parameter vector :math:`{\bf b}`

        :return: cost function value
        """
        return CostFunctionBase_Chi2_Nuisance.chi2_nui_cov(
            data=y_data, model=y_model, total_uncor_cov_mat_inverse=y_total_uncor_cov_mat_inverse,
            total_nuisance_cor_design_mat=_y_total_nuisance_cor_design_mat, nuisance_vector=y_nuisance_vector)

    @staticmethod
    def chi2_nui_cov_fallback_y(y_data, y_model, y_total_uncor_cov_mat_inverse, _y_total_nuisance_cor_design_mat,
                                y_nuisance_vector):
        return CostFunctionBase_Chi2_Nuisance.chi2_nui_cov_fallback(
            data=y_data, model=y_model, total_uncor_cov_mat_inverse=y_total_uncor_cov_mat_inverse,
            total_nuisance_cor_design_mat=_y_total_nuisance_cor_design_mat, nuisance_vector=y_nuisance_vector)

    @staticmethod
    def chi2_nui_cov_x(y_data, y_model, x_total_uncor_cov_mat_inverse, x_model, x_data):
        r"""A least-squares cost function which uses `x` nuisance parameters.
        The cost function is given by:

        .. math::
            C = \chi^2 =
                ({\bf d}_{y} - {\bf m}_{y}) \cdot ({\bf d}_{y} - {\bf m}_{y})
                +
                ({\bf d}_{x} - {\bf m}_{x})^{\top}
                ({\bf V}_{x}^{\mathrm{uncor}})^{-1}
                ({\bf d}_{x} - {\bf m}_{x})
                +
                C({\bf p})

        In the above, :math:`{\bf d}_{y}` are the `y` measurements,
        :math:`{\bf m}_{y}` are the `y` model predictions,
        :math:`{\bf d}_{x}` are the `x` measurements,
        :math:`{\bf m}_{x}` are the `x` model predictions,
        :math:`{\bf V}_{x}^{\mathrm{uncor}}` is the total uncorrelated `x` covariance matrix,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param y_data: `y` measurement data :math:`{\bf d}_{y}`
        :param y_model: `y` model predictions :math:`{\bf m}_{y}`
        :param x_data: `x` measurement data :math:`{\bf d}_{x}`
        :param y_model: `y` model predictions :math:`{\bf m}_{x}`
        :param x_total_uncor_cov_mat_inverse: inverse :math:`({\bf V}_{x}^{\mathrm{uncor}})^{-1}` of the uncorrelated part of the total `x` covariance matrix

        :return: cost function value
        """
        return _generic_xy_chi2_nuisance_covariance(
            x_data=x_data, x_model=x_model,
            y_data=y_data, y_model=y_model,
            x_uncor_cov_mat_inverse=x_total_uncor_cov_mat_inverse,
            fail_on_no_y_matrix=False,
            fail_on_no_x_matrix=True,)

    @staticmethod
    def chi2_nui_cov_fallback_x(y_data, y_model, x_total_uncor_cov_mat_inverse, x_model, x_data):
        return _generic_xy_chi2_nuisance_covariance(
            x_data=x_data, x_model=x_model,
            y_data=y_data, y_model=y_model,
            x_uncor_cov_mat_inverse=x_total_uncor_cov_mat_inverse,
            fail_on_no_y_matrix=False,
            fail_on_no_x_matrix=False)

    @staticmethod
    def chi2_nui_cov_xy(y_data, y_model, y_total_uncor_cov_mat_inverse, _y_total_nuisance_cor_design_mat,
                        y_nuisance_vector, x_total_uncor_cov_mat_inverse, x_model, x_data):
        r"""A least-squares cost function which uses `x` and `y` nuisance parameters
        The cost function is given by:

        .. math::
            C = \chi^2 =
            ({\bf d}_{y} - {\bf m}_{y} - {\bf G}{\bf b})^{\top}
            ({\bf V}_{y}^{\mathrm{uncor}})^{-1}
            ({\bf d}_{y} - {\bf m}_{y} - {\bf G}{\bf b})
            +
            ({\bf d}_{x} - {\bf m}_{x})^{\top}
            ({\bf V}_{x}^{\mathrm{uncor}})^{-1}
            ({\bf d}_{x} - {\bf m}_{x})
            +
            {\bf b}^2
            +
            C({\bf p})

        In the above, :math:`{\bf d}_{y}` are the `y` measurements, :math:`{\bf m}_{y}` are the `y` model predictions,
        :math:`{\bf d}_{x}` are the `x` measurements, :math:`{\bf m}_{x}` are the `x` model predictions,
        :math:`{\bf G}` is the design matrix containing the correlated parts of all `y` uncertainties,
        :math:`{\bf V}_{x}^{\mathrm{uncor}}` is the total uncorrelated `y` covariance matrix,
        :math:`{\bf V}_{x}^{\mathrm{uncor}}` is the total uncorrelated `x` covariance matrix,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param y_data: `y` measurement data :math:`{\bf d}_{y}`
        :param y_model: `y` model predictions :math:`{\bf m}_{y}`
        :param x_data: `x` measurement data :math:`{\bf d}_{x}`
        :param x_model: `x` model predictions :math:`{\bf m}_{x}`
        :param x_total_uncor_cov_mat_inverse: inverse :math:`({\bf V}_{x}^{\mathrm{uncor}})^{-1}` of the uncorrelated part of the total `x` covariance matrix
        :param y_total_uncor_cov_mat_inverse: inverse :math:`({\bf V}_{y}^{\mathrm{uncor}})^{-1}` of the uncorrelated part of the total `y` covariance matrix
        :param _y_total_nuisance_cor_design_mat: design matrix :math:`{\bf G}` containing correlated `y` uncertainties
        :param poi_values: vector of parameters of interest :math:`{\bf p}`
        :param y_nuisance_vector: nuisance parameter vector :math:`{\bf b}`

        :return: cost function value
        """
        return _generic_xy_chi2_nuisance_covariance(
            x_data=x_data, x_model=x_model,
            y_data=y_data, y_model=y_model,
            x_uncor_cov_mat_inverse=x_total_uncor_cov_mat_inverse,
            y_uncor_cov_mat_inverse=y_total_uncor_cov_mat_inverse,
            y_nuisance_cor_design_mat=_y_total_nuisance_cor_design_mat,
            y_nuisance_vector=y_nuisance_vector,
            fail_on_no_x_matrix=True, fail_on_no_y_matrix=True)

    @staticmethod
    def chi2_nui_cov_fallback_xy(y_data, y_model, y_total_uncor_cov_mat_inverse, _y_total_nuisance_cor_design_mat,
                                 y_nuisance_vector, x_total_uncor_cov_mat_inverse, x_model, x_data):

        return _generic_xy_chi2_nuisance_covariance(
            y_data=y_data, y_model=y_model, x_data=x_data, x_model=x_model,
            y_uncor_cov_mat_inverse=y_total_uncor_cov_mat_inverse, y_nuisance_vector=y_nuisance_vector,
            x_uncor_cov_mat_inverse=x_total_uncor_cov_mat_inverse,
            y_nuisance_cor_design_mat=_y_total_nuisance_cor_design_mat, fail_on_no_x_matrix=False)

    @staticmethod
    def chi2_nui_pointwise_y(y_data, y_model, y_total_error):
        r"""A least-squares cost function calculated from `y` data and model values,
        considering pointwise (uncorrelated) uncertainties for each data point:

        .. math::
            C = \chi^2({\bf d}_{y}, {\bf m}_{y}, {\bf \sigma}_{y}) =
                \sum_k \frac{d_{y,k} - m_{y,k}}{\sigma_{y,k}}
                +
                C({\bf p})

        In the above, :math:`{\bf d}_{y}` are the `y` measurements,
        :math:`{\bf m}_{y}` are the `y` model predictions,
        :math:`{\bf \sigma}_{y}` are the pointwise total `y` uncertainties,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param y_data: `y` measurement data :math:`{\bf d}_{y}`
        :param y_model: `y` model predictions :math:`{\bf m}_{y}`
        :param y_total_error: total `y` error vector :math:`{\bf \sigma}_{y}`

        :return: cost function value
        """

        return CostFunctionBase_Chi2_Nuisance.chi2_nui_pointwise(
            data=y_data, model=y_model, total_error=y_total_error)

    @staticmethod
    def chi2_nui_pointwise_fallback_y(y_data, y_model, y_total_error):
        return CostFunctionBase_Chi2_Nuisance.chi2_nui_pointwise_fallback(
            data=y_data, model=y_model, total_error=y_total_error)

    @staticmethod
    def chi2_nui_pointwise_x(y_data, y_model, x_total_error, x_model, x_data):
        r"""A least-squares cost function taking pointwise `x` errors into account.

        The cost function is given by:

        .. math::
            C = \chi^2 =
            \sum_k { \left(\frac{d_{y,k} - m_{y,k}}{\sigma_{y,k}}\right)^2 }
            +
            \sum_k { \left(\frac{d_{x,k} - m_{x,k}}{\sigma_{x,k}}\right)^2 }
            +
            C({\bf p})

        In the above, :math:`d_{y,k}` are the `y` measurements, :math:`m_{y,k}` are the `y` model predictions,
        :math:`d_{x,k}` are the `x` measurements, :math:`m_{x,k}` are the `x` model predictions,
        :math:`\sigma_{y,k}` are the total `y` errors,
        :math:`\sigma_{x,k}` are the total `x` errors,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param y_data: `y` measurement data :math:`{\bf d}_{y}`
        :param y_model: `y` model predictions :math:`{\bf m}_{y}`
        :param x_data: `x` measurement data :math:`{\bf d}_{x}`
        :param x_model: `x` model predictions :math:`{\bf m}_{x}`
        :param x_total_error: total `x` error vector :math:`{\bf \sigma}_{x}`
        :param y_total_error: total `y` error vector :math:`{\bf \sigma}_{y}`

        :return: cost function value
        """
        return _generic_xy_chi2_nuisance_pointwise(
            x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data, x_total_error=x_total_error)

    @staticmethod
    def chi2_nui_pointwise_fallback_x(y_data, y_model, x_total_error, x_model, x_data):
        return _generic_xy_chi2_nuisance_pointwise(
            x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data, x_total_error=x_total_error)

    @staticmethod
    def chi2_nui_pointwise_xy(y_data, y_model, x_total_error, x_model, x_data, y_total_error):
        r"""A least-squares cost function taking pointwise `x` and `y` errors into account.

        The cost function is given by:

        .. math::
            C = \chi^2 =
            \sum_k { \left(\frac{d_{y,k} - m_{y,k}}{\sigma_{y,k}}\right)^2 }
            +
            \sum_k { \left(\frac{d_{x,k} - m_{x,k}}{\sigma_{x,k}}\right)^2 }
            +
            C({\bf p})

        In the above, :math:`{\bf d}_{y,k}` are the `y` measurements, :math:`{\bf m}_{y,k}` are the `y` model predictions,
        :math:`d_{x,k}` are the `x` measurements, :math:`{\bf m}_{x,k}` are the `x` model predictions,
        :math:`\sigma_{y,k}` are the total `y` errors,
        :math:`\sigma_{x,k}` are the total `x` errors,
        and :math:`C({\bf p})` is the additional cost resulting from any constrained parameters.

        :param y_data: `y` measurement data :math:`{\bf d}_{y}`
        :param y_model: `y` model predictions :math:`{\bf m}_{y}`
        :param x_data: `x` measurement data :math:`{\bf d}_{x}`
        :param x_model: `x` model predictions :math:`{\bf m}_{x}`
        :param x_total_error: total `x` error vector :math:`{\bf \sigma}_{x}`
        :param y_total_error: total `y` error vector :math:`{\bf \sigma}_{y}`

        :return: cost function value
        """
        return _generic_xy_chi2_nuisance_pointwise(
            x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data, x_total_error=x_total_error,
            fail_on_zeros=True, y_total_error=y_total_error)

    @staticmethod
    def chi2_nui_pointwise_fallback_xy(y_data, y_model, x_total_error, x_model, x_data, y_total_error):
        return _generic_xy_chi2_nuisance_pointwise(
            x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data, x_total_error=x_total_error,
            fail_on_zeros=False, y_total_error=y_total_error)


STRING_TO_COST_FUNCTION = {
    'chi2': (XYCostFunction_Chi2, {}),
    'chi_2': (XYCostFunction_Chi2, {}),
    'chisquared': (XYCostFunction_Chi2, {}),
    'chi_squared': (XYCostFunction_Chi2, {}),
    'chi2n': (XYCostFunction_Chi2_Nuisance, {}),
    'chi_2_n': (XYCostFunction_Chi2_Nuisance, {}),
    'chisquarednuisance': (XYCostFunction_Chi2_Nuisance, {}),
    'chi_squared_nuisance': (XYCostFunction_Chi2_Nuisance, {}),
    'nll': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'nll-poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll_poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nllpoisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": False}),
    'nll-gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'nll_gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'nllgaussiann': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": False}),
    'negloglikelihood': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'neg_log_likelihood': (XYCostFunction_NegLogLikelihood, {"ratio": False}),
    'nllr': (CostFunction_NegLogLikelihood, {"ratio": True}),
    'nllr-poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllr_poisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllrpoisson': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "poisson", "ratio": True}),
    'nllr-gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'nllr_gaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'nllrgaussian': (XYCostFunction_NegLogLikelihood,
                    {"data_point_distribution": "gaussian", "ratio": True}),
    'negloglikelihoodratio': (XYCostFunction_NegLogLikelihood, {"ratio": True}),
    'neg_log_likelihood_ratio': (XYCostFunction_NegLogLikelihood, {"ratio": True}),
}
