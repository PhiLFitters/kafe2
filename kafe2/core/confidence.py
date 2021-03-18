from __future__ import print_function

import numpy as np
from scipy.special import gammainccinv, gammaincc

__all__ = ['ConfidenceLevel']


class ConfidenceLevel(object):
    """
    Helper class for handling the conversion from confidence levels to sigma values and vice versa.
    """
    def __init__(self, n_dimensions, cl=None, sigma=None):
        self.ndim = n_dimensions
        if (cl is None and sigma is None) or (cl is not None and sigma is not None):
            raise ValueError("Exactly one out of cl and sigma must be defined!")
        if cl is not None:
            self.cl = cl
        if sigma is not None:
            self.sigma = sigma

    def __str__(self):
        return "<ConfidenceLevel (d=%d): %.4g%% (%.3g-sigma)>" % (
            self.ndim, self.cl*100., self.sigma)

    def _calc_sigma_from_cl(self):
        self._sigma = np.sqrt(2 * gammainccinv(self.ndim/2.0, 1.0 - self.cl))

    def _calc_cl_from_sigma(self):
        self._cl = 1. - gammaincc(self.ndim/2.0, self.sigma**2/2.0)

    @property
    def cl(self):
        if self._cl is None:
            self._calc_cl_from_sigma()
        return self._cl

    @cl.setter
    def cl(self, new_cl):
        if new_cl <= 0 or new_cl >= 1:
            raise ValueError(
                "Confidence level must be greater than 0 and less than 1. Got: %g" % (new_cl,))
        self._cl = float(new_cl)
        self._sigma = None

    @property
    def sigma(self):
        if self._sigma is None:
            self._calc_sigma_from_cl()
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        if new_sigma <= 0:
            raise ValueError("Sigma value must be greater than 0! Got: %g" % (new_sigma,))
        self._sigma = float(new_sigma)
        self._cl = None

    @property
    def ndim(self):
        return self._ndim

    @ndim.setter
    def ndim(self, new_ndim):
        if not isinstance(new_ndim, int):
            raise ValueError(
                "Number of dimensions must be of type int! Received type: %s" % type(new_ndim))
        if new_ndim <= 0:
            raise ValueError(
                "Number of dimensions must be greater 0! Received: %d" % (new_ndim,)
            )
        self._ndim = new_ndim

    @property
    def sigma_string(self):
        return "%.3g-sigma" % (self.sigma,)

    @property
    def sigma_latex_string(self):
        return r"%g$\sigma$" % (self.sigma,)

    @property
    def cl_string(self):
        return "%.4g%% CL" % (self.cl*100,)

    @property
    def cl_latex_string(self):
        return r"$%.4g\%%$ CL" % (self.cl*100,)
