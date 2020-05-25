import numpy as np

from .._base import ParametricModelBaseMixin
from .container import UnbinnedContainer, UnbinnedContainerException
from ..util import function_library

__all__ = ['UnbinnedParametricModel', 'UnbinnedParametricModelException']


class UnbinnedParametricModelException(UnbinnedContainerException):
    pass


class UnbinnedParametricModel(ParametricModelBaseMixin, UnbinnedContainer):
    def __init__(self, data, model_density_function=function_library.normal_distribution_pdf,
                 model_parameters=[1.0, 1.0]):

        self.support = np.array(data)

        super(UnbinnedParametricModel, self).__init__(
            # this gets passed to ParametricModelBaseMixin.__init__
            model_func=model_density_function,
            model_parameters=model_parameters,
            # this gets passed to UnbinnedContainer.__init__
            data=model_density_function(
                self.support, *model_parameters)
        )

    # -- private methods

    def _recalculate(self):
        # use parent class setter for 'data'
        UnbinnedContainer.data.fset(self, self.eval_model_function())
        self._pm_calculation_stale = False

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, model_support):
        self._support = model_support
        self._pm_calculation_stale = True

    @property
    def data(self):
        if self._pm_calculation_stale:
            self._recalculate()
        return super(UnbinnedParametricModel, self).data

    @data.setter
    def data(self):
        raise UnbinnedParametricModelException("Parametric model data cannot be set!")

    def eval_model_function(self, support=None, model_parameters=None):
        """
        Evaluate the model function.

        :param support: *x* values of the support points (if ``None``, the model *support* values are used)
        :type support: list or ``None``
        :param model_parameters: values of the model parameters (if ``None``, the current values are used)
        :type model_parameters: list or ``None``
        :return: value(s) of the model function for the given parameters
        :rtype: :py:obj:`numpy.ndarray`
        """
        _x = support if support is not None else self.support
        _pars = model_parameters if model_parameters is not None else self.parameters
        return self._model_function_object(_x, *_pars)
