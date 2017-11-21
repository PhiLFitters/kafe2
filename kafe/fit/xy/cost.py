from .._base import CostFunctionBase, CostFunctionBase_Chi2, CostFunctionBase_NegLogLikelihood, CostFunctionBase_NegLogLikelihoodRatio, CostFunctionException
from kafe.fit._base.cost import CostFunctionBase_Chi2_Nuisance
import numpy as np

__all__ = ["XYCostFunction_UserDefined", "XYCostFunction_Chi2", "XYCostFunction_NegLogLikelihood"]

def _generic_xy_chi2_nuisance_pointwise(x_data, x_model, y_model, y_data, y_total_error=None, x_total_error=None, fail_on_zeros=False):
# calculates the costfunction values for ChiSquare with Nuisanceparameters for pointwise errors.

    if y_model.shape != y_data.shape or y_model.shape != x_model.shape or x_model.shape != x_data.shape:
        raise CostFunctionException("x_data, x_model, 'y_data' and 'y_model' must have the same shape! Got %r, %r, %r and %r..."
                                % (x_data.shape, x_model.shape, y_data.shape, y_model.shape))
    _x_res = (x_data - x_model)
    _y_res = (y_data - y_model)

    if x_total_error.any() == 0.0:
        if fail_on_zeros:
            raise CostFunctionException("'x_err' must not contain any zero values!")
        else:
            pass
    else:
        #with x-erros
        _x_penalties = np.sum(_x_res ** 2 * x_total_error)
        if y_total_error is not None:
            if y_total_error.any() == 0.0:
                if fail_on_zeros:
                    raise CostFunctionException("'y_err' must not contain any zero values!")
                else:
                    return _y_res.dot(_y_res) + _x_penalties
            else:
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
                pass
        else:
            _y_res = _y_res / y_total_error
            _chi2 = np.sum(_y_res ** 2)
            return _chi2

    #cost function value without any errors
    return _y_res.dot(_y_res)

def _generic_xy_chi2_nuisance_covaraince(x_data, x_model,  y_data, y_model,
                  x_uncor_cov_mat_inverse=None, y_uncor_cov_mat_inverse=None, y_nuisance_cor_cov_mat=None, y_nuisance_vector=np.array([]),
                  fail_on_no_y_matrix=False, fail_on_no_x_matrix=False):
    """calculates the cost function values for ChiSquare with Nuisanceparameters for pointwise errors."""

    if y_model.shape != y_data.shape or y_model.shape != x_model.shape or x_model.shape != x_data.shape:
        raise CostFunctionException("x_data, x_model, 'y_data' and 'y_model' must have the same shape! Got %r, %r, %r and %r..."
                                % (x_data.shape, x_model.shape, y_data.shape, y_model.shape))

    _x_res = (x_data - x_model)
    _y_res = (y_data - y_model)

    if x_uncor_cov_mat_inverse is None:
        if fail_on_no_x_matrix:
            raise np.linalg.LinAlgError("Uncorrelated X Covariance matrix is singular!")
        else:
            if y_uncor_cov_mat_inverse is None:
                if y_nuisance_vector.all() ==0.0:
                    # raise if uncorrelated matrix is None and the correlated is not None
                    raise CostFunctionException('Is not working for only fullcorrelated y-errors')
                else:
                    if fail_on_no_y_matrix:
                        raise np.linalg.LinAlgError("Uncorrelated Y Covariance matrix is singular!")
                    else:
                        # cost function values without any errors
                        _chisquare = _y_res.dot(_y_res)
                        return _chisquare

            else:
                #with y-errors but without x-errors
                _inner_sum = np.squeeze(np.asarray(y_nuisance_vector.dot(y_nuisance_cor_cov_mat)))
                _y_penalties = y_nuisance_vector.dot(y_nuisance_vector)
                _chisquare = (_y_res - _inner_sum).dot(y_uncor_cov_mat_inverse).dot(_y_res - _inner_sum)[0, 0]
                return (_y_penalties + _chisquare)[0, 0]

    else:
        _x_penalties = np.transpose(_x_res).dot(x_uncor_cov_mat_inverse).dot(_x_res)
        if y_uncor_cov_mat_inverse is None:
            if y_nuisance_vector.all() == 0.0:
                raise CostFunctionException('Is not working for only fullcorrelated y-errors')
            else:
                if fail_on_no_y_matrix:

                    raise np.linalg.LinAlgError("Uncorrelated Y Covariance matrix is singular!")
                else:
                    #with x-errors but without y-errors
                    _chisquare = _y_res.dot(_y_res)
                    return (_chisquare +_x_penalties)[0, 0]

        else:
                #with x- and y-errors
                _inner_sum = np.squeeze(np.asarray(y_nuisance_vector.dot(y_nuisance_cor_cov_mat)))
                _y_penalties = y_nuisance_vector.dot(y_nuisance_vector)
                _chi2 = (_y_res - _inner_sum).dot(y_uncor_cov_mat_inverse).dot(_y_res- _inner_sum)[0, 0]
                return (_chi2 + _x_penalties + _y_penalties)[0, 0]


class XYCostFunction_UserDefined(CostFunctionBase):
    def __init__(self, user_defined_cost_function):
        """
        User-defined cost function for fits to *xy* data.
        The function handle must be provided by the user.

        :param user_defined_cost_function: function handle

        .. note::
            The names of the function arguments must be valid reserved
            names for the associated fit type (:py:obj:`~kafe.fit.XYFit`)!
        """
        super(XYCostFunction_UserDefined, self).__init__(cost_function=user_defined_cost_function)


class XYCostFunction_Chi2(CostFunctionBase_Chi2):
    def __init__(self, errors_to_use='covariance', fallback_on_singular=True, axes_to_use='xy'):
        """
        Built-in least-squares cost function for *xy* data.

        :param errors_to_use: which errors to use when calculating :math:`\chi^2`
        :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
        :param axes_to_use: take into account errors for which axes
        :type axes_to_use: ``'y'`` or ``'xy'``
        """

        if axes_to_use.lower() == 'y':
            super(XYCostFunction_Chi2, self).__init__(errors_to_use=errors_to_use, fallback_on_singular=fallback_on_singular)
        elif axes_to_use.lower() == 'xy':
            _cost_function_description = "chi-square with projected x errors"
            if errors_to_use is None:
                _chi2_func = self.chi2_no_errors
                _cost_function_description += ' (no errors)'
            elif errors_to_use.lower() == 'covariance':
                if fallback_on_singular:
                    _chi2_func = self.chi2_xy_covariance_fallback
                else:
                    _chi2_func = self.chi2_xy_covariance
                _cost_function_description += ' (covariance matrix)'
            elif errors_to_use.lower() == 'pointwise':
                if fallback_on_singular:
                    _chi2_func = self.chi2_xy_pointwise_errors_fallback
                else:
                    _chi2_func = self.chi2_xy_pointwise_errors
                _cost_function_description += ' (pointwise errors)'
            else:
                raise CostFunctionException("Unknown value '%s' for 'errors_to_use': must be one of ('covariance', 'pointwise', None)")
            CostFunctionBase.__init__(self, cost_function=_chi2_func)
            self._formatter.latex_name = "\chi^2"
            self._formatter.name = "chi2"
            self._formatter.description = _cost_function_description
        else:
            raise CostFunctionException("Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y')")

        

    @staticmethod
    def chi2_no_errors(y_data, y_model):
        r"""A least-squares cost function calculated from 'y' data and model values,
        without considering uncertainties:

        .. math::
            C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})\cdot({\bf d} - {\bf m})

        In the above, :math:`{\bf d}` are the measurements and :math:`{\bf m}` are the model
        predictions.

        :param y_data: measurement data
        :param y_model: model values
        :return: cost function value
        """
        return CostFunctionBase_Chi2.chi2_no_errors(data=y_data, model=y_model)

    @staticmethod
    def chi2_covariance(y_data, y_model, y_total_cov_mat_inverse):
        r"""A least-squares cost function calculated from 'y' data and model values,
        considering the covariance matrix of the 'y' measurements.

        .. math::
            C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})^{\top}\,{{\bf V}^{-1}}\,({\bf d} - {\bf m})

        In the above, :math:`{\bf d}` are the measurements, :math:`{\bf m}` are the model
        predictions, and :math:`{{\bf V}^{-1}}` is the inverse of the total covariance matrix.

        :param y_data: measurement data
        :param y_model: model values
        :param y_total_cov_mat_inverse: inverse of the total covariance matrix
        :return: cost function value
        """
        return CostFunctionBase_Chi2.chi2_covariance(data=y_data, model=y_model, total_cov_mat_inverse=y_total_cov_mat_inverse)

    @staticmethod
    def chi2_pointwise_errors(y_data, y_model, y_total_error):
        r"""A least-squares cost function calculated from 'y' data and model values,
        considering pointwise (uncorrelated) uncertainties for each data point:

        .. math::
            C = \chi^2({\bf d}, {\bf m}, {\bf \sigma}) = \sum_k \frac{d_k - m_k}{\sigma_k}

        In the above, :math:`{\bf d}` are the measurements, :math:`{\bf m}` are the model
        predictions, and :math:`{\bf \sigma}` are the pointwise total uncertainties.

        :param y_data: measurement data
        :param y_model: model values
        :param y_total_error: total measurement uncertainties
        :return:
        """
        return CostFunctionBase_Chi2.chi2_pointwise_errors(data=y_data, model=y_model, total_error=y_total_error)

    @staticmethod
    def chi2_xy_covariance(y_data, y_model, projected_xy_total_cov_mat_inverse):
        return CostFunctionBase_Chi2.chi2_covariance(data=y_data, model=y_model, total_cov_mat_inverse=projected_xy_total_cov_mat_inverse)

    @staticmethod
    def chi2_xy_pointwise_errors(y_data, y_model, x_total_error, projected_xy_total_error):
        return CostFunctionBase_Chi2.chi2_pointwise_errors(y_data, y_model, total_error=projected_xy_total_error)

    @staticmethod
    def chi2_pointwise_errors_fallback(y_data, y_model, y_total_error):
        return CostFunctionBase_Chi2.chi2_pointwise_errors_fallback(data=y_data, model=y_model, total_error=y_total_error)

    @staticmethod
    def chi2_covariance_fallback(y_data, y_model, y_total_cov_mat_inverse):
        return CostFunctionBase_Chi2.chi2_covariance_fallback(data=y_data, model=y_model, total_cov_mat_inverse=y_total_cov_mat_inverse)

    @staticmethod
    def chi2_xy_pointwise_errors_fallback(y_data, y_model, projected_xy_total_error):
        return CostFunctionBase_Chi2.chi2_pointwise_errors_fallback(y_data, y_model, total_error=projected_xy_total_error)

    @staticmethod
    def chi2_xy_covariance_fallback(y_data, y_model, projected_xy_total_cov_mat_inverse):
        return CostFunctionBase_Chi2.chi2_covariance_fallback(data=y_data, model=y_model, total_cov_mat_inverse=projected_xy_total_cov_mat_inverse)


class XYCostFunction_NegLogLikelihood(CostFunctionBase_NegLogLikelihood):
    def __init__(self, data_point_distribution='poisson'):
        r"""
        Built-in negative log-likelihood cost function for *xy* data.

        In addition to the measurement data and model predictions, likelihood-fits require a
        probability distribution describing how the measurements are distributed around the model
        predictions.
        This built-in cost function supports two such distributions: the *Poisson* and *Gaussian* (normal)
        distributions.

        In general, a negative log-likelihood cost function is defined as the double negative logarithm of the
        product of the individual likelihoods of the data points.

        :param data_point_distribution: which type of statistics to use for modelling the distribution of individual data points
        :type data_point_distribution: ``'poisson'`` or ``'gaussian'``
        """
        super(XYCostFunction_NegLogLikelihood, self).__init__(data_point_distribution=data_point_distribution)

    @staticmethod
    def nll_gaussian(y_data, y_model, y_total_error):
        r"""A negative log-likelihood function assuming Gaussian statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}, {\bf \sigma}) = -2 \ln \prod_j \mathcal{L}_{\rm Gaussian} (x=d_j, \mu=m_j, \sigma=\sigma_j)

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{1}{\sqrt{2{\sigma_j}^2\pi}} \exp{\left(-\frac{ (d_j-m_j)^2 }{ {\sigma_j}^2}\right)}

        In the above, :math:`{\bf d}` are the measurements, :math:`{\bf m}` are the model predictions, and :math:`{\bf \sigma}`
        are the pointwise total uncertainties.

        :param y_data: measurement data
        :param y_model: model values
        :param y_total_error: total *y* uncertainties for data
        :return: cost function value
        """
        # "translate" the argument names
        return CostFunctionBase_NegLogLikelihood.nll_gaussian(data=y_data, model=y_model, total_error=y_total_error)


    @staticmethod
    def nll_poisson(y_data, y_model):
        r"""A negative log-likelihood function assuming Poisson statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}) = -2 \ln \prod_j \mathcal{L}_{\rm Poisson} (k=d_j, \lambda=m_j)

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{{m_j}^{d_j} \exp(-m_j)}{d_j!}

        In the above, :math:`{\bf d}` are the measurements and :math:`{\bf m}` are the model
        predictions.

        :param y_data: measurement data
        :param y_model: model values
        :return: cost function value
        """
        # "translate" the argument names
        return CostFunctionBase_NegLogLikelihood.nll_poisson(data=y_data, model=y_model)


class XYCostFunction_NegLogLikelihoodRatio(CostFunctionBase_NegLogLikelihoodRatio):
    def __init__(self, data_point_distribution='poisson'):
        r"""
        Built-in negative log-likelihood cost function for *xy* data.

        In addition to the measurement data and model predictions, likelihood-fits require a
        probability distribution describing how the measurements are distributed around the model
        predictions.
        This built-in cost function supports two such distributions: the *Poisson* and *Gaussian* (normal)
        distributions.

        In general, a negative log-likelihood cost function is defined as the double negative logarithm of the
        product of the individual likelihoods of the data points.

        :param data_point_distribution: which type of statistics to use for modelling the distribution of individual data points
        :type data_point_distribution: ``'poisson'`` or ``'gaussian'``
        """
        super(XYCostFunction_NegLogLikelihoodRatio, self).__init__(data_point_distribution=data_point_distribution)

    @staticmethod
    def nllr_gaussian(y_data, y_model, y_total_error):
        r"""A negative log-likelihood function assuming Gaussian statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}, {\bf \sigma}) = -2 \ln \prod_j \mathcal{L}_{\rm Gaussian} (x=d_j, \mu=m_j, \sigma=\sigma_j)

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{1}{\sqrt{2{\sigma_j}^2\pi}} \exp{\left(-\frac{ (d_j-m_j)^2 }{ {\sigma_j}^2}\right)}

        In the above, :math:`{\bf d}` are the measurements, :math:`{\bf m}` are the model predictions, and :math:`{\bf \sigma}`
        are the pointwise total uncertainties.

        :param y_data: measurement data
        :param y_model: model values
        :param y_total_error: total *y* uncertainties for data
        :return: cost function value
        """
        # "translate" the argument names
        return CostFunctionBase_NegLogLikelihoodRatio.nllr_gaussian(data=y_data, model=y_model,
                                                                    total_error=y_total_error)


    @staticmethod
    def nllr_poisson(y_data, y_model):
        r"""A negative log-likelihood function assuming Poisson statistics for each measurement.

        The cost function is given by:

        .. math::
            C = -2 \ln \mathcal{L}({\bf d}, {\bf m}) = -2 \ln \prod_j \mathcal{L}_{\rm Poisson} (k=d_j, \lambda=m_j)

        .. math::
            \rightarrow C = -2 \ln \prod_j \frac{{m_j}^{d_j} \exp(-m_j)}{d_j!}

        In the above, :math:`{\bf d}` are the measurements and :math:`{\bf m}` are the model
        predictions.

        :param y_data: measurement data
        :param y_model: model values
        :return: cost function value
        """
        # "translate" the argument names
        return CostFunctionBase_NegLogLikelihoodRatio.nllr_poisson(data=y_data, model=y_model)

class XYCostFunction_Chi2_Nuisance(CostFunctionBase_Chi2_Nuisance):

    def __init__(self, axes_to_use='xy', errors_to_use='covariance', fall_back_on_singular=True):

        """
              Built-in least-squares cost function with nuisanceparameters for *xy* data.

              :param errors_to_use: which errors to use when calculating :math:`\chi^2`
              :type errors_to_use: ``'covariance'``, ``'pointwise'`` or ``None``
              :param axes_to_use: take into account errors for which axes
              :type axes_to_use: ``'y'`` or ``'xy'``
              """

        if errors_to_use==None:
            _chi2_nui = self.chi2_no_error

        elif errors_to_use == 'covariance':
            if axes_to_use == 'y':
                if fall_back_on_singular:
                    _chi2_nui = self.chi2_nui_cov_fall_y
                else:
                    _chi2_nui = self.chi2_nui_cov_y
            elif axes_to_use == 'x':
                if fall_back_on_singular:
                    _chi2_nui = self.chi2_nui_cov_fall_x
                else:
                    _chi2_nui = self.chi2_nui_cov_x
            elif axes_to_use == 'xy':
                if fall_back_on_singular:
                    _chi2_nui = self.chi2_nui_cov_fall_xy
                else:
                    _chi2_nui = self.chi2_nui_cov_xy
            else:
                raise CostFunctionException("Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y', 'x')")
        elif errors_to_use == 'pointwise':
            if axes_to_use == 'y':
                if fall_back_on_singular:
                    _chi2_nui = self.chi2_nui_pointwise_fall_y
                else:
                    _chi2_nui = self.chi2_nui_pointwise_y
            elif axes_to_use == 'x':
                if fall_back_on_singular:
                    _chi2_nui = self.chi2_nui_pointwise_fall_x
                else:
                    _chi2_nui = self.chi2_nui_pointwise_x
            elif axes_to_use == 'xy':
                if fall_back_on_singular:
                    _chi2_nui = self.chi2_nui_pointwise_fall_xy
                else:
                    _chi2_nui = self.chi2_nui_pointwise_xy
            else:
                raise CostFunctionException("Unknown value '%s' for 'axes_to_use': must be one of ('xy', 'y', 'x')")
        else:
            raise CostFunctionException("Unknown value '%s' for 'errors_to_use': must be one of (None, 'covariance', 'pointwise')")


        super(CostFunctionBase_Chi2, self).__init__(cost_function=_chi2_nui)

        #set the needed flags
        if errors_to_use == None:
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
        """A least-squares cost function calculated from 'y' data and model values,
                without considering uncertainties:

                .. math::
                    C = \chi^2({\bf d}, {\bf m}) = ({\bf d} - {\bf m})\cdot({\bf d} - {\bf m})

                In the above, :math:`{\bf d}` are the measurements and :math:`{\bf m}` are the model
                predictions.

                :param y_data: measurement data
                :param y_model: model values
                :return: cost function value
                """
        return CostFunctionBase_Chi2.chi2_no_errors(data=y_data, model=y_model)

    @staticmethod
    def chi2_nui_cov_y(y_data, y_model, y_total_uncor_cov_mat_inverse, nuisance_y_total_cor_cov_mat, y_nuisance_vector):

        r"""a Chisquare costfunction which uses y-nuisance parameters
                the costfunction is given by:

                C = \chi^2({\bf yd}, {\bf ym} = ({\bf yd} - {\bf ym})*{\bf cor_cov} * {\bf ynui}) {\bf y_uncor_cov_inverse} *
               ({\bf yd} - {\bf ym})*{\bf cor_cov}{\bf ynui}) +  {\bf ynui}^2

        In the above, :math:`{\bf yd}` are the measurements and :math:`{\bf ym}` are the model
        predictions, :math:{\bf cor_cov} is the total nuisance correlated covariance matrix and
        :math:{\bf y_uncor_cov}} is the inverse of the total uncorrelated 'y'covariance matrix
        :mathe:{\bf ynui} is the Y Nuisance-vector

                :param: y_data: measurement data
                :param: y_model model values
                :param: y_total_uncor_cov_mat_inverse: invserse of the uncorrelated covariance matrix
                :param: nuisance_y_total_cor_cov_mat: correlated covariance matrix
                :param: y_nuisance_vector: y_nuisance paramters

               :return: cost function value
              """
        return CostFunctionBase_Chi2_Nuisance.chi2_nui_cov(data=y_data, model=y_model,
                                                       total_uncor_cov_mat_inverse=y_total_uncor_cov_mat_inverse,
                                                       nuisance_total_cor_cov_mat=nuisance_y_total_cor_cov_mat,
                                                       nuisance_vector=y_nuisance_vector)

    @staticmethod
    def chi2_nui_cov_fall_y(y_data, y_model, y_total_uncor_cov_mat_inverse, nuisance_y_total_cor_cov_mat, y_nuisance_vector):
        return CostFunctionBase_Chi2_Nuisance.chi2_nui_cov_fall(data=y_data, model=y_model,
                                                  total_uncor_cov_mat_inverse=y_total_uncor_cov_mat_inverse,
                                                  nuisance_total_cor_cov_mat=nuisance_y_total_cor_cov_mat,
                                                  nuisance_vector=y_nuisance_vector)

    @staticmethod
    def chi2_nui_cov_x(y_data, y_model, x_total_uncor_cov_mat_inverse, x_model, x_data):

        r"""a Chisquare costfunction which uses x-nuisance parameters
         the costfunction is given by:

               C = \chi^2 = ({\bf yd} - {\bf ym}) * ({\bf d} - {\bf ym}) +
               {\bf xd} - {\bf xm}) * {\bf x_uncor_cov_inverse}*{\bf xd} - {\bf xm})

        In the above, :math:`{\bf yd}` are the ymeasurements and :math:`{\bf ym}` are the ymodel
        predictions,:math:`{\bf xd}` are the xmeasurements and :math:`{\bf xm}` are the xmodel
       :math:{\bf x_cor_cov_inverse} is the total uncorrelated 'x' covariance matrix

               :param: y_data: measurement data
               :param: y_model y_model values
               :param: x_data: x_measurement data
               :param: x_model x_model values
               :param: x_total_uncor_cov_mat_inverse: invserse of the uncorrelated 'x' covariance matrix


               :return: cost function value
               """
        return _generic_xy_chi2_nuisance_covaraince(x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data,
                                                    x_uncor_cov_mat_inverse=x_total_uncor_cov_mat_inverse,
                                                    fail_on_no_y_matrix=False,
                                                    fail_on_no_x_matrix=True)

    @staticmethod
    def chi2_nui_cov_fall_x(y_data, y_model, x_total_uncor_cov_mat_inverse, x_model, x_data):
        return _generic_xy_chi2_nuisance_covaraince(x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data,
                                         x_uncor_cov_mat_inverse=x_total_uncor_cov_mat_inverse, fail_on_no_y_matrix=False,
                                         fail_on_no_x_matrix=False)

    @staticmethod
    def chi2_nui_cov_xy(y_data, y_model, y_total_uncor_cov_mat_inverse, nuisance_y_total_cor_cov_mat, y_nuisance_vector,
                x_total_uncor_cov_mat_inverse, x_model, x_data):
        r"""a Chisquare costfunction which uses x- and y-nuisance parameters
        the costfunction is given by:

               C = ({\bf yd} - {\bf ym})*{\bf cor_cov}) * {\bf y_uncor_cov_inverse} *
               ({\bf yd} - {\bf ym})*{\bf cor_cov})) +
               {\bf xd} - {\bf xm}) * {\bf x_uncor_cov_inverse}*{\bf xd} - {\bf xm})

        In the above, :math:`{\bf yd}` are the ymeasurements and :math:`{\bf ym}` are the ymodel
        predictions, :math:{\bf cor_cov} is the total nuisance correlated covariance matrix and
        math:`{\bf xd}` are the xmeasurements and :math:`{\bf xm}` are the xmodel
        :math:{\bf x_cor_cov_inverse} is the total uncorrelated X covariance matrix

                :param: y_data: measurement data
                :param: y_model y_model values
                :param: x_data: x_measurement data
                :param: x_model x_model values
                :param: x_total_uncor_cov_mat_inverse: invserse of the uncorrelated  'x' covariance matrix
                :param: y_total_uncor_cov_mat_inverse: invserse of the uncorrelated  'y' covariance matrix
                :param: nuisance_y_total_cor_cov_mat: correlated covariance matrix
                :param: y_nuisance_vector: y_nuisance paramters

               :return: cost function value
                    """
        return _generic_xy_chi2_nuisance_covaraince(y_data=y_data, y_model=y_model, x_data=x_data, x_model=x_model,
                                         y_uncor_cov_mat_inverse=y_total_uncor_cov_mat_inverse, y_nuisance_vector=y_nuisance_vector,
                                         x_uncor_cov_mat_inverse=x_total_uncor_cov_mat_inverse,
                                         y_nuisance_cor_cov_mat=nuisance_y_total_cor_cov_mat, fail_on_no_x_matrix=True,
                                         fail_on_no_y_matrix=True)

    @staticmethod
    def chi2_nui_cov_fall_xy(y_data, y_model, y_total_uncor_cov_mat_inverse, nuisance_y_total_cor_cov_mat, y_nuisance_vector,
                        x_total_uncor_cov_mat_inverse, x_model, x_data):

        return _generic_xy_chi2_nuisance_covaraince(y_data=y_data, y_model=y_model, x_data=x_data, x_model=x_model,
                                         y_uncor_cov_mat_inverse=y_total_uncor_cov_mat_inverse,
                                         y_nuisance_vector=y_nuisance_vector,
                                         x_uncor_cov_mat_inverse=x_total_uncor_cov_mat_inverse,
                                         y_nuisance_cor_cov_mat=nuisance_y_total_cor_cov_mat, fail_on_no_x_matrix=False,
                                         fail_on_no_y_matrix=False)

    @staticmethod
    def chi2_nui_pointwise_y(y_data, y_model, y_total_error):


         r"""A least-squares cost function calculated from 'y' data and model values,
            considering pointwise (uncorrelated) uncertainties for each data point:

            .. math::
                C = \chi^2({\bf yd}, {\bf ym}, {\bf y\sigma}) = \sum_k \frac{yd_k - ym_k}{\sigma_k}

            In the above, :math:`{\bf yd}` are the y measurements, :math:`{\bf ym}` are the ymodel
            predictions, and :math:`{\bf y\sigma}` are the pointwise total y uncertainties.

            :param y_data: measurement data
            :param y_model: model values
            :param y_total_error: total y measurement uncertainties
            :return cost function value:
                """

         return CostFunctionBase_Chi2_Nuisance.chi2_nui_pointwise(data=y_data, model=y_model,
                                                                  total_error=y_total_error)

    @staticmethod
    def chi2_nui_pointwise_fall_y(y_data, y_model, y_total_error):
        return CostFunctionBase_Chi2_Nuisance.chi2_nui_pointwise_fall(data=y_data, model=y_model,
                                                                      total_error=y_total_error)

    @staticmethod
    def chi2_nui_pointwise_x(y_data, y_model, x_total_error, x_model, x_data):
        r"""A least-squares cost function calculated from 'x' data and model values,
        considering pointwise (uncorrelated) uncertainties for each data point, using 'x' nuisance parameters:

                C = \chi^2 = \sum_k {yd_k - ym_k}^2 + \sum_k { \frac{xd_k -xm_}{x\sigma_k}}

        In the above, :math:`{\bf yd}` are the ymeasurements and :math:`{\bf ym}` are the ymodel
        math:`{\bf xd}` are the xmeasurements and :math:`{\bf xm}` are the xmodel
        and :math:`{\bf x\sigma}` are the pointwise total x uncertainties.

            :param: y_data: measurement data
            :param: y_model y_model values
            :param: x_data: x_measurement data
            :param: x_model x_model values
            :param: x_total_error: total x measurement uncertainties
            """
        return _generic_xy_chi2_nuisance_pointwise(x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data,
                                                   x_total_error=x_total_error, fail_on_zeros=True)

    @staticmethod
    def chi2_nui_pointwise_fall_x(y_data, y_model, x_total_error, x_model, x_data):
        return _generic_xy_chi2_nuisance_pointwise(x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data,
                                        x_total_error=x_total_error, fail_on_zeros=False)

    @staticmethod
    def chi2_nui_pointwise_xy(y_data, y_model, x_total_error, x_model, x_data, y_total_error):
        r"""A least-squares cost function calculated from 'xy' data and model values,
           considering pointwise (uncorrelated) uncertainties for each data point, using 'x' nuisance parameters:

                   C = \chi^2 = \sum_k { \frac{yd_k - ym_k^2}{y\sigma_k}} + \sum_k { \frac{xd_k -xm_}{x\sigma_k}}

           In the above, :math:`{\bf yd}` are the ymeasurements and :math:`{\bf ym}` are the ymodel
           :math:`{\bf xd}` are the xmeasurements and :math:`{\bf xm}` are the xmodel
           :math:`{\bf x\sigma}` are the pointwise total x uncertainties.
           and  {\bf y\sigma}` are the pointwise total y uncertainties

               :param: y_data: measurement data
               :param: y_model y_model values
               :param: x_data: x_measurement data
               :param: x_model x_model values
               :param: x_total_error: total x measurement uncertainties
               :param: y_total_error: total y measurement uncertainties
              """
        return _generic_xy_chi2_nuisance_pointwise(x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data,
                                                   x_total_error=x_total_error, fail_on_zeros=True,
                                                   y_total_error=y_total_error)

    @staticmethod
    def chi2_nui_pointwise_fall_xy(y_data, y_model, x_total_error, x_model, x_data, y_total_error):
        return _generic_xy_chi2_nuisance_pointwise(x_model=x_model, y_model=y_model, x_data=x_data, y_data=y_data,
                                     x_total_error=x_total_error, fail_on_zeros=False, y_total_error=y_total_error)




