from __future__ import print_function

import numpy as np
from scipy.special import gammainccinv, gammaincc


class ConfidenceLevelError(Exception):
    pass


class ConfidenceLevel(object):
    """
    """
    def __init__(self, n_dimensions):
        self.ndim = n_dimensions
        self._cl = None
        self._sigma = None

    def __str__(self):
        if self._cl is None and self._sigma is None:
            return "<ConfidenceLevel: undefined>"

        return "<ConfidenceLevel (d=%d): %.4g%% (%.3g-sigma)" % (self.ndim, self.cl*100., self.sigma)

    @classmethod
    def from_sigma(cls, n_dimensions, sigma):
        obj = cls(n_dimensions)
        obj.sigma = sigma
        return obj

    @classmethod
    def from_cl(cls, n_dimensions, confidence_level):
        obj = cls(n_dimensions)
        obj.cl = confidence_level
        return obj

    def _calc_sigma_from_cl(self):
        self._sigma = np.sqrt(2 * gammainccinv(self.ndim/2., 1. - self.cl))

    def _calc_cl_from_sigma(self):
        self._cl = 1. - gammaincc(self.ndim/2., self.sigma**2/2.)


    @property
    def cl(self):
        if self._cl is None:
            if self._sigma is None:
                raise ConfidenceLevelError("Cannot get confidence level: neither a sigma "
                                           "value nor a CL have been defined!")
            self._calc_cl_from_sigma()
        return self._cl

    @cl.setter
    def cl(self, new_cl):
        if new_cl <= 0 or new_cl >= 1:
            raise ConfidenceLevelError("Confidence level must be greater "
                                       "than 0 and less than 1. Got: %g"
                                       % (new_cl,))
        self._cl = float(new_cl)
        self._sigma = None

    @property
    def sigma(self):
        if self._sigma is None:
            if self._cl is None:
                raise ConfidenceLevelError("Cannot get sigma value: neither a sigma "
                                           "value nor a CL have been defined!")
            self._calc_sigma_from_cl()
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        if new_sigma <= 0:
            raise ConfidenceLevelError("Sigma value must be greater "
                                        "than 0! Got: %g"
                                        % (new_sigma,))
        self._sigma = float(new_sigma)
        self._cl = None


    @property
    def ndim(self):
        return self._ndim

    @ndim.setter
    def ndim(self, new_ndim):
        _ndim_int = int(new_ndim)
        if _ndim_int <= 0:
            raise ConfidenceLevelError("Number of dimensions must be must be greater "
                                        "than or equal to 1. Got: %d"
                                        % (_ndim_int,))
        if _ndim_int != new_ndim:
            print("Warning: Got fractional number of dimensions (%f)! Coercing to: %d" % (new_ndim, _ndim_int))

        self._ndim = _ndim_int


if __name__ == "__main__":
    _cl_obj = ConfidenceLevel.from_sigma(2, 1.0)
    print(_cl_obj)

    _cl_obj = ConfidenceLevel.from_cl(2, .86)
    print(_cl_obj)

    _cl_obj = ConfidenceLevel.from_sigma(1, 1.0)
    print(_cl_obj)

    _cl_obj = ConfidenceLevel.from_cl(1, .86)
    print(_cl_obj)

    try:
        print(ConfidenceLevel.from_sigma(2.1, .86))
    except ConfidenceLevelError as e:
        print(e)

    try:
        print(ConfidenceLevel.from_sigma(2, 400))
    except ConfidenceLevelError as e:
        print(e)

    try:
        print(ConfidenceLevel.from_sigma(2, -1))
    except ConfidenceLevelError as e:
        print(e)

    try:
        _co = ConfidenceLevel.from_sigma(-2, 400)
        print(_co)
    except ConfidenceLevelError as e:
        print(e)

    try:
        print(ConfidenceLevel.from_sigma(-2, -1))
    except ConfidenceLevelError as e:
        print(e)

    try:
        print(ConfidenceLevel.from_sigma(-2.3, 400))
    except ConfidenceLevelError as e:
        print(e)

    try:
        print(ConfidenceLevel.from_sigma(-2.3, -1))
    except ConfidenceLevelError as e:
        print(e)
