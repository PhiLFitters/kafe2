from __future__ import print_function

import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import six

from kafe.core.error import CovMat


def expand_to_ndim(array, target_ndim, direction='right'):
    """Add axes to an array until the desired number of dimensions is reached.

    :param array: array to expand
    :type array: ``numpy.ndarray``
    :param target_ndim: number of dimensions to reach
    :type target_ndim: int
    :param direction: direction of expansion
    :type direction: 'left' or 'right'
    :return: expanded array
    :rtype: ``numpy.ndarray``
    """
    _new_array = array
    if direction == 'right':
        _append_pos = -1
    elif direction == 'left':
        _append_pos = 0
    else:
        raise ValueError("Unknown direction specification '{}': "
                         "expected 'left' or 'right'!".format(direction))
    while _new_array.ndim < target_ndim:
        _new_array = np.expand_dims(_new_array, _append_pos)
    return _new_array

def broadcast_to_shape(array, shape, scheme='default'):
    """
    Broadcast a ``numpy.ndarray`` to a shape according to different schemes.

    :param array: the array to broadcast
    :type array: ``numpy.ndarray``
    :param shape: the shape against which to broadcast the array
    :type shape: tuple of ``int``
    :param scheme: the name of the scheme
    :type scheme: 'default', 'transposed', 'expand_left' or 'expand_right'

    :return: the broadcasted array
    """
    if scheme == 'default':
        # default numpy broadcasting scheme:
        #  self._shape:  a x b x c
        #        array:      b x 1
        #      ---------------------
        #       result:  a x b x c
        return np.broadcast_to(array, shape=shape)
    elif scheme == 'expand_right_successive':
        # dimensions are added to the array (on the right) until
        # a broadcastable situation is reached
        # target shape to the array shape:
        #  self._shape:      a x b
        #        array:  n x a x 1
        #      ---------------------
        #       result:  n x a x b
        _new_array = array
        while True:
            _can_bcast = all([_i == _j or _i == 1 or _j == 1 for _i, _j in zip(reversed(_new_array.shape), reversed(shape))])
            if _can_bcast:
                break
            _new_array = np.expand_dims(_new_array, -1)
        return _new_array
    elif scheme == 'expand_left_successive':
        # dimensions are added to the array (on the left) until
        # a broadcastable situation is reached
        # target shape to the array shape:
        #  self._shape:  a x b
        #        array:  1 x b x n
        #      ---------------------
        #       result:  a x b x n
        _new_array = array
        while True:
            _can_bcast = all([_i == _j or _i == 1 or _j == 1 for _i, _j in zip(_new_array.shape, shape)])
            if _can_bcast:
                break
            _new_array = np.expand_dims(_new_array, 0)
        return _new_array
    elif scheme == 'transposed':
        # "transposed" numpy broadcasting scheme:
        # (broadcasting is done on the transposed arrays
        #  and the result is transposed back)
        #  self._shape:  a x b x c
        #        array:  a x 1
        #      ---------------------
        #       result:  a x b x c
        return np.broadcast_to(array.T, shape=reversed(shape)).T
    elif scheme == 'expand_right':
        # `self.ndim` axes are added to the array shape (to the right)
        # and then broadcasting is done normally
        #  self._shape:          a x b x c  (self.ndim = 3)
        #        array:  n x m x 1 x 1 x 1
        #      -----------------------------
        #       result:  n x m x a x b x c
        # _new_shape = tuple(len(self._shape) * [1] + list(reversed(array.shape)))
        _new_shape = tuple(list(reversed(shape)) + list(reversed(array.shape)))
        return np.broadcast_to(array.T, _new_shape).T
    elif scheme == 'expand_left':
        # `self.ndim` axes are added to the array shape (to the left)
        # and then the 'transposed' broadcasting is applied
        #  self._shape:  a x b x c          (self.ndim = 3)
        #        array:  1 x 1 x 1 x n x m
        #      -----------------------------
        #       result:  a x b x c x n x m
        _new_shape = tuple(list(shape) + list(array.shape))
        return np.broadcast_to(array, _new_shape)
    else:
        raise EnsembleError("Cannot broadcast array to ensemble shape: unknown scheme '{}'! "
                            "Available: {}".format(scheme, ('default', 'transposed', 'expand_right', 'expand_left')))


class EnsembleError(Exception):
    pass


class EnsembleVariable(object):
    """
    Object for storing a finite sample of realizations of a single (possibly multidimensional) random variable,
    forming a statistical ensemble.
    """

    def __init__(self, ensemble_array, dtype=float,
                 distribution=None, distribution_parameters=None):
        """
        Create an ensemble of realizations of random variables.

        :param ensemble_array: a statistical ensemble containing all realizations of the random variable.
                               **Note**: the size of the first axis is taken to be the sample size. Any
                               remaining array axes are understood to be part of the (multidimensional)
                               random variable itself.
        :type ensemble_array: `numpy.ndarray`
        :param sample_size: the size of the ensemble (number of realizations of the variable)
        :type sample_size: int
        :param variable_shape: the ndarray shape for *one* realization of the random variable
        :typle variable_shape: tuple of int (pass an empty `tuple()` or list `[]` for a scalar variable)
        :param dtype: underlying dtype of ensemble variable
        :type dtype: type
        :param distribution: probability distribution of the variable (e.g. `scipy.stats.norm`)
        :type distribution: a probability distribution from `scipy.stats` or ``None``
        :param distribution_parameters:  mapping of distribution parameter names to values. If values are
                                         sequences/arrays, the shape must match the `variable_shape`
        :type distribution_parameters: dict or ``None``
        """
        self._array = np.asarray(ensemble_array)
        self._total_shape = self._array.shape
        self._size = self._total_shape[0]
        self._shape = tuple()
        if self._array.ndim > 1:
            self._shape = self._total_shape[1:]

        self._dist = None
        if distribution is not None:
            self._dist = EnsembleVariableProbabilityDistribution(
                distribution=distribution,
                parameters=distribution_parameters,
                variable_shape=self._shape
            )

    @property
    def size(self):
        """The size of the sample."""
        return self._size

    @property
    def shape(self):
        """The shape of the random variable."""
        return self._shape

    @property
    def ndim(self):
        """The dimensionality of the random variable."""
        return self._array.ndim - 1  # do not include the sample size dimension

    @property
    def values(self):
        """A (possibly) multidimensional array containing all realization of the ensemble variable."""
        return self._array

    @property
    def mean(self):
        """The mean of the ensemble variable across all realizations."""
        return np.mean(self._array, axis=0)

    @property
    def mean_error(self):
        """The standard error of the mean -> standard deviation/sqrt(N)"""
        return self.std / np.sqrt(self.size)

    @property
    def std(self):
        """The standard deviation of the ensemble variable across all realizations."""
        return np.std(self._array, axis=0)

    @property
    def skew(self):
        """The skew of the ensemble variable across all realizations."""
        return scipy.stats.skew(self._array, axis=0)

    @property
    def kurtosis(self):
        """The kurtosis of the ensemble variable across all realizations."""
        return scipy.stats.kurtosis(self._array, axis=0)

    @property
    def cov_mat(self):
        """The sample covariance matrix (only available for one-dimensional ensemble variables)."""
        if self.ndim == 0:
            # trivial covariance matrix
            return np.matrix([[self.std**2]])
        elif self.ndim == 1:
            return np.cov(self._array.T)

        raise EnsembleError("Cannot calculate covariance matrix: ensemble variable must "
                            "have dimension 1 (got {})".format(self.ndim))

    @property
    def cor_mat(self):
        """The sample correlation matrix (only available for one-dimensional ensemble variables)."""
        if self.ndim == 0:
            # trivial correlation matrix
            return np.matrix([[1.0]])
        if self.ndim == 1:
            return CovMat(self.cov_mat).cor_mat

        raise EnsembleError("Cannot calculate correlation matrix: ensemble variable must "
                            "have dimension 1 (got {})".format(self.ndim))

    @property
    def dist(self):
        """An object representing the (expected) probability distribution of the variable."""
        return self._dist  # can be ``None``

    def set_value(self, index, variable_value):
        """Set the value of the `index`-th realization of the ensemble variable."""
        # TODO (?) validate index and/or value shape?
        try:
            self._array[index, :] = variable_value
        except IndexError:
            # for scalar ensemble variables
            self._array[index] = variable_value


class EnsembleVariableProbabilityDistribution(object):
    """
    Object for storing the probability distribution for an ensemble variable.
    """

    def __init__(self, distribution, parameters, variable_shape):
        """
        Create a probability distribution object for a random variable.

        :param distribution: a ``scipy.stats`` probability distribution
        :type distribution: e.g. ``scipy.stats.norm``
        :param parameters: the parameters of the probability distribution
        :type parameters: ``dict`` mapping parameter names to values (or value arrays)
        :param variable_shape: the shape of the "``numpy.ndarray``-valued" random variable
        :type variable_shape: the shape of the "``numpy.ndarray``-valued" random variable
        """
        # self._dist_func = distribution
        self._shape = variable_shape

        self._dist_param_values_dict = {}
        for _dist_par_name, _dist_par_value in six.iteritems(parameters):
            # wrap lists and/or tuples in numpy.ndarray
            if isinstance(_dist_par_value, collections.Sequence) and not isinstance(_dist_par_value,
                                                                                    six.string_types[0]):
                _dist_par_value = np.array(_dist_par_value)

            if isinstance(_dist_par_value, np.ndarray):
                if _dist_par_value.ndim == 0:
                    _dist_par_value = np.ones(self._shape) * _dist_par_value
                if _dist_par_value.shape != self._shape:
                    raise EnsembleError("Distribution parameter '{}' must be a scalar or have shape "
                                        "{}: got {}".format(_dist_par_name, self._shape, _dist_par_value.shape))
            else:
                _dist_par_value = np.ones(self._shape) * _dist_par_value
            self._dist_param_values_dict[_dist_par_name] = _dist_par_value

        self._dist_func = distribution(**self._dist_param_values_dict)

    @property
    def ndim(self):
        """The dimensionality of the random variable."""
        return len(self._shape)  # do not include the sample size dimension

    @property
    def mean(self):
        """The expectation value of the probability distribution"""
        return self._dist_func.mean()

    @property
    def var(self):
        """The variance of the probability distribution."""
        return self._dist_func.var()

    @property
    def std(self):
        """The standard deviation (square root of the variance) of the probability distribution."""
        return self._dist_func.std()

    @property
    def skew(self):
        """The skewness (third standardized moment) of the probability distribution."""
        return self.standardized_moment(3)

    @property
    def kurtosis(self):
        """The kurtosis (fourth standardized moment) of the probability distribution."""
        return self.standardized_moment(4)

    def standardized_moment(self, n):
        """The n-th standardized moment of the distribution"""
        if self.ndim != 0:
            raise NotImplementedError("Standardized moment calculation not available for non-scalar variables.")
        return self._dist_func.expect(lambda x: ((x - self.mean) / self.std) ** n)

    def moment(self, n):
        """The n-th moment of the distribution"""
        if self.ndim != 0:
            raise NotImplementedError("Moment calculation not available for non-scalar variables.")
        return self._dist_func.moment(n)

    def eval(self, x, x_contains_var_shape=False):
        """
        Evaluate the probability distribution/mass at a given point.

        If `x` is a scalar, the probability distribution/mass function
        will be evaluated at the single point `x` for *all*
        different parameter values specified when constructing the object.
        If the parameter values are specified as *p*-dimensional
        arrays, the resulting array will also have *p* dimensions,
        and the shape will match that of the parameter arrays.

        If `x` is an *n*-dimensional array with shape `x.shape`,
        the probability distribution/mass function
        will be evaluated at every value of `x`.
        In this case, the resulting array will have *p+n* dimensions,
        with the first (innermost) *p* corresponding to the parameter arrays
        and the last (outermost) *n*  corresponding to `x.shape`.

        Alternatively, if the flag ``x_contains_var_shape`` is set, the
        last (outermost) dimensions of ``x`` are assumed to have the
        same shape as the parameter arrays. If this is the case, no
        broadcasting of ``x`` to the parameter arrays is done
        and the resulting array will have the same dimension and shape as
        the input array ``x``.

        :param x: value (array of values) at which to evaluate the pdf/pmf
        :type x: `numpy.ndarray` with the *first* entries of `x.shape`
                 matching the ensemble variable's shape
        :param x_contains_var_shape: if ``True``, the alternate method described above is used.
        :type x_contains_var_shape: bool
        :return:
        """

        x = np.asarray(x)
        _x_ndim = x.ndim

        # x needs to be broadcast together with the parameter arrays
        # GOAL:
        # x shape:                      a  x  b  x  c   (n axes)
        # par shape:        m  x  n                     (p axes)
        # ---
        # resulting y:      m  x  n  x  a  x  b  x  c   (p+n axes)

        # 1) cannot expand par -> transpose x
        # x.T shape:        c  x  b  x  a               (n axes)
        x = x.T

        # 2) expand x.T
        # x.T shape:        c  x  b  x  a  x  1  x  1   (n axes)
        if not x_contains_var_shape:
            x = broadcast_to_shape(x, shape=self._shape, scheme='expand_right')

        # 3) evaluate pdf at x.T -> broadcast on last n axes
        # x.T shape:        c  x  b  x  a  x  1  x  1   (n axes)
        # par shape:                          m  x  n   (p axes)
        # ---
        # resulting y:      c  x  b  x  a  x  m  x  n   (n+p axes)
        _y = None
        if _y is None:
            try:
                _y = self._dist_func.pdf(x)
            except AttributeError:
                pass
        if _y is None:
            try:
                _y = self._dist_func.pmf(x)
            except AttributeError:
                pass

        if _y is None:
            raise EnsembleError("Probability distribution has no method 'pdf' or 'pmf'!")

        # 4) transpose last p axes
        # resulting y:      c  x  b  x  a  x  n  x  m   (n+p axes)
        if not x_contains_var_shape:
            _axes_permutation = list(range(_x_ndim)) + list(reversed(range(_x_ndim, _x_ndim + self.ndim)))
        else:
            _axes_permutation = list(range(_x_ndim))

        _y = np.transpose(_y, _axes_permutation)

        # 5) transpose everything
        # resulting y:      m  x  n  x  a  x  b  x  c   (p+n axes)
        _y = _y.T

        return _y


class EnsembleVariablePlotter(object):
    """
    Object for storing the relevant properties and methods for plotting an ensemble variable.
    """

    _DEFAULT_PLOT_PDF_KWARGS = dict(marker='')
    _DEFAULT_PLOT_EXPECTED_MEAN_KWARGS = dict(linewidth=1, marker='', linestyle='--',
                                              # use second color in default color cycle
                                              color=mpl.rcParams['axes.prop_cycle'].by_key()['color'][1])
    _DEFAULT_PLOT_OBSERVED_MEAN_KWARGS = dict(linewidth=1, marker='', color='k')
    _DEFAULT_PLOT_ONE_SIGMA_BAND_MEAN_KWARGS = dict(color='k', alpha=0.1)

    def __init__(self, ensemble_variable,
                 value_ranges,
                 variable_labels=None,
                 ensemble_label=None):
        """
        Create an object to handle the graphical representation of an ensemble variable.

        :param ensemble_variable: the ensemble variable to be plotted
        :type ensemble_variable: `~kafe.fit.tools.ensemble.EnsembleVariable`
        :param value_ranges: the ...
        :type value_ranges: `numpy.ndarray` (shape must match the ensemble variable shape)
        :param axis_labels: the labels to appear on the axes
        :type axis_labels: `numpy.ndarray` (shape must match the ensemble variable shape)
        """
        self._var = ensemble_variable

        # check variable dimensionality (can only plot scalar, 1D and 2D)
        if self._var.ndim >= 3:
            raise EnsembleError("Cannot create plotter for ensemble variable: "
                                "dimensionality too high ({}>2)!".format(self._var.ndim))

        self._ensemble_label = ensemble_label or "{} pseudoexperiments".format(self._var.size)

        self._value_ranges = np.array(value_ranges)
        if self._value_ranges.shape[:-1] != self._var.shape:
            # prefix dimensions do not match -> expand array
            self._value_ranges = broadcast_to_shape(self._value_ranges, self._var.shape, scheme="expand_left")

        # expand, if needed, to at least 3 axes (needed for nested for loop below)
        self._value_ranges = expand_to_ndim(self._value_ranges, target_ndim=3, direction='left')

        self._variable_labels = np.array(variable_labels)
        if self._variable_labels is not None:
            self._variable_labels = broadcast_to_shape(self._variable_labels, self._var.shape, scheme="expand_left")
            self._variable_labels = expand_to_ndim(self._variable_labels, target_ndim=2, direction='left')

    def plot_hist(self, axes_array, show_y_ticks=False):
        """
        Plot each component of the (multidimensional) ensemble variable as a histogram.



        :param axes_array: an array of ``matplotlib`` axes objects in which to perform the plotting. The shape
                           of the axes array must match that of the ensemble variable. For scalar variables,
                           a single ``Axes`` can be provided.
        :type axes_array: ``numpy.ndarray`` of ``matplotlib`` ``Axes`` objects.

        :return: mapping containing plot metadata and other information
        :rtype: `dict`
        """
        axes_array = np.asarray(axes_array)

        if axes_array.shape != self._var.shape:
            raise EnsembleError("Shape of `axes_array` {} does not "
                                "match ensemble variable shape {}!".format(axes_array.shape, self._var.shape))

        _expected_means = None
        _expected_mean_errors = None
        _pdf_eval_ymax = None

        _nbins = 51  # TODO: config
        _pdf_eval_npts = 101  # TODO: config

        # get the expected mean and its error from the ensemble variable distribution
        _dist = self._var.dist
        if _dist is not None:
            _expected_means = np.atleast_2d(self._var.dist.mean)
            # TODO: presumably only valid/relevant for Gaussian -> solution for non-Gaussian?
            _expected_mean_errors = np.atleast_2d(self._var.dist.std)/np.sqrt(self._var.size)

            # evaluate PDF at pre-computed plot points
            _pdf_eval_x = np.apply_along_axis(
                lambda xminmax: np.linspace(xminmax[0], xminmax[1], _pdf_eval_npts), -1, np.squeeze(self._value_ranges))
            _pdf_eval_y = _dist.eval(_pdf_eval_x,
                                     x_contains_var_shape=True)

            # pad the evaluated PDF coordinates to at least 3 dimensions
            _pdf_eval_x = expand_to_ndim(_pdf_eval_x, 3, direction='left')
            _pdf_eval_y = expand_to_ndim(_pdf_eval_y, 3, direction='left')

            _pdf_eval_ymax = np.max(_pdf_eval_y)

        # get the observed mean (pad to at least 2 dimensions: one scalar per plot in 2D matrix)
        _observed_means = np.atleast_2d(self._var.mean)

        # get value array (pad to at least 3 dimensions: one 1D-array per plot in 2D matrix)
        _var_values = expand_to_ndim(self._var.values, 3, direction='right')

        _all_legend_handles = []
        _all_legend_labels = []


        _plot_result_dict = dict()
        for _index1, _axes in enumerate(np.atleast_2d(axes_array)):
            for _index2, _ax in enumerate(_axes):
                _data = _var_values[:, _index2, _index1]
                assert len(_data) == self._var.size

                _bin_contents, _bin_edges, _ = _ax.hist(
                    _data,
                    bins=_nbins,
                    # range=self._value_ranges[_index], # TODO
                    label=self._ensemble_label
                )

                if _expected_means is not None:
                    # only show observed mean if expected mean is available
                    _ax.annotate(r"$\mu={}$".format(round(_observed_means[_index1, _index2], 2)),
                                 xycoords='data',
                                 xy=(_observed_means[_index1, _index2], 0),
                                 textcoords='offset points',
                                 xytext=(0, 25),
                                 fontsize=12,
                                 horizontalalignment='center',
                                 verticalalignment='bottom',
                                 arrowprops=dict(facecolor='k', shrink=.0)
                                 )
                    if _expected_mean_errors is not None and _expected_mean_errors[_index1, _index2]:
                        _mean_pull = (_observed_means[_index1, _index2] - _expected_means[_index1, _index2]) / _expected_mean_errors[_index1, _index2]
                        _ax.annotate(r"$({:+.2f}\sigma)$".format(round(_mean_pull, 2)),
                                     xycoords='data',
                                     xy=(_observed_means[_index1, _index2], 0),
                                     textcoords='offset points',
                                     xytext=(0, 40),
                                     fontsize=12,
                                     horizontalalignment='center',
                                     verticalalignment='bottom',
                                     arrowprops=dict(arrowstyle='-')
                                     )

                if self._value_ranges is not None:
                    _ax.set_xlim(self._value_ranges[_index1, _index2])

                if _dist is not None:
                    _pdf_label = "expected density"
                    # calculate normalization
                    _mean_bin_width = np.mean(_bin_edges[1:] - _bin_edges[:-1])
                    _n_entries = np.sum(_bin_contents)
                    _plot_prob_density_scale = _mean_bin_width * _n_entries
                    _pdf_x = _pdf_eval_x[_index1, _index2]
                    _pdf_y = _pdf_eval_y[_index1, _index2] * _plot_prob_density_scale

                    _ax.plot(_pdf_x, _pdf_y, label=_pdf_label, **self._DEFAULT_PLOT_PDF_KWARGS)

                if _expected_means is not None:
                    _ax.axvline(_expected_means[_index1, _index2], label="expected mean",
                                **self._DEFAULT_PLOT_EXPECTED_MEAN_KWARGS)
                    if _expected_mean_errors is not None:
                        _ax.axvspan(_expected_means[_index1, _index2] - _expected_mean_errors[_index1, _index2],
                                    _expected_means[_index1, _index2] + _expected_mean_errors[_index1, _index2],
                                    label="standard error of the mean",
                                    **self._DEFAULT_PLOT_ONE_SIGMA_BAND_MEAN_KWARGS)

                # ensure density appears with the same scaling across all axes
                if _pdf_eval_ymax is not None:
                    _ax.set_ylim((0, _pdf_eval_ymax * _plot_prob_density_scale * 1.2))

                # set the x axis label
                if self._variable_labels is not None:
                    _xlabel = self._variable_labels[_index1, _index2]
                    if _xlabel is not None:
                        _ax.set_xlabel(_xlabel)

                # disable the y axis ticks
                if not show_y_ticks:
                    _ax.yaxis.set_ticks([])

                # collect legend handles and labels
                _hs, _ls = _ax.get_legend_handles_labels()
                _all_legend_handles += tuple(_hs)
                _all_legend_labels += tuple(_ls)

        # suppress multiple entries for the same label
        _hs, _ls = [], []
        _seen_labels = set()
        for _h, _l in zip(_all_legend_handles, _all_legend_labels):
            if _l not in _seen_labels:
                _hs.append(_h)
                _ls.append(_l)
                _seen_labels.add(_l)

        _plot_result_dict['legend_handles'] = _hs
        _plot_result_dict['legend_labels'] = _ls

        return _plot_result_dict

