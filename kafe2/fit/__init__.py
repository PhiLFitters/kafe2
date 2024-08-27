r"""The :py:mod:`kafe2.fit` module provides an object-oriented toolkit for estimating model
parameters from data ("fitting").

It distinguishes between a number of different data types:

*  *xy* data (dedicated submodule: :py:mod:`~kafe2.fit.xy`),
*  series of indexed measurements (dedicated submodule: :py:mod:`~kafe2.fit.indexed`),
*  histograms (dedicated submodule: :py:mod:`~kafe2.fit.histogram`),
*  raw 1D data using the method of maximum likelihood ("unbinned fit",
   dedicated submodule: :py:mod:`~kafe2.fit.histogram`), and
*  direct minimization of a cost function (dedicated submodule: :py:mod:`~kafe2.fit.custom`).

Each of the above data types has its own particularities when it comes to fitting.
The main difference is due to the way uncertainties can be defined and
interpreted for each type of data and how the fit results are presented.


XY Data
-------

For *xy* data, one data set consists of a list of :math:`N` distinct :math:`y` measurements
:math:`d_i` with the (discrete) index :math:`i` ranging from :math:`0` to :math:`N-1`.
The measurements were taken at :math:`x` values :math:`x_i`.
For each measurement in the series, one or more uncertainty sources can be defined,
each being a numerical estimate of how much the respective measurement has fluctuated from the
"true values".
Correlations between uncertainties on separate measurements :math:`d_i` and :math:`d_j` can also
be taken into account by using *covariance/correlation matrices*.

Additional uncertainites on :math:`x_i` can also be defined.
When fitting an *xy* model to data they are converted to :math:`y` uncertainties via multiplication
with the derivative of the model function by :math:`x`.
When plotting the result of *xy* fits, the model function is displayed as a continuous
function of :math:`x`, and an *error band* can be computed to reflect the model uncertainty,
as determined by propagating the parameter uncertainties onto the *y* axis.

The following objects are provided for handling *xy* data:

*  :py:obj:`~kafe2.fit.xy.XYContainer`: data container for storing *xy* data
*  :py:obj:`~kafe2.fit.xy.XYParametricModel`: corresponding model
*  :py:obj:`~kafe2.fit.xy.XYFit`: a fit of a parametric model to *xy* data

Indexed data
------------

Compared to *xy* data indexed data no longer has an explicit *x* axis.
The data simply appears as an indexed list of data points.
As a consequence the model function does not expect an independent variable.

The following objects are provided for handling indexed data, as described above:

*  :py:obj:`~kafe2.fit.indexed.IndexedContainer`: data container for storing indexed data
*  :py:obj:`~kafe2.fit.indexed.IndexedParametricModel`: corresponding model
*  :py:obj:`~kafe2.fit.indexed.IndexedFit`: a fit of a parametric model to *indexed* data

Histograms
----------

*kafe2* is also able to handle *histograms*. Histograms organize measurements whose
values can fall anywhere across a continuum of values into a number of discrete regions
or "bins". Typically, the continuous "measurement space" (a closed real interval
:math:`[x_{\rm min}, x_{\rm max}]`)
is subdivided into a sequence of successive intervals at the "bin edges"
:math:`x_{\rm min} < x_1 < x_2 < \ldots < x_{\rm max}`. Whenever a measurement falls into one of
the bins, the value of that histogram bin is incremented by one.
A histogram is completely defined by its bin edges and the bin values.

.. note::
    The bin numbering starts at :math:`1` for the first bin and ends at :math:`N`, where :math:`N`
    is defined as the *size* of the histogram. The bin numbers :math:`0` and :math:`N+1` refer to
    the underflow (below :math:`x_{\rm min}`) and overflow bin (above :math:`x_{\rm max}`),
    respectively.


Defining a parametric model for histograms is not as straightforward as for *xy* and indexed data.
Seeing as they keep track of the number of entries in different intervals of the continuum, the bin
values can be interpreted using probability theory.

As the number of entries approaches infinity, the number of entries :math:`n` in the bin covering an
interval :math:`[a, b)`, divided by the total number of entries :math:`N_{\rm E}`, will approach the
probablity of an event landing in that bin:

.. math::
    \lim_{N_{\rm E}\rightarrow\infty} \frac{n}{N_{\rm E}} = \int_a^b f(x)\,{\rm d}x = F(b) - F(a)

In the above formula, :math:`f(x)` is the *probability density function*,
and :math:`F(x)` is an antiderivative of :math:`f`
(for example the *cumulative distribution function*).

Using the above relation, the model prediction :math:`m` for the bin :math:`[a, b)` can be defined
as:

.. math::
    m = N_{\rm E} \int_a^b f(x)\,{\rm d}x = N_{\rm E} \left(F(b) - F(a)\right)

This means that, for histograms, the model density :math:`f(x)` needs to be specified as the model
function.
The model is then calculated by numerically integrating this function over each bin.

An alternative would be to specify the model density *antiderviative* :math:`F` alongside
the model, so that the model can be calculated as a simple difference, rather than as an integral.

The following objects are provided for handling histograms:

*  :py:obj:`~kafe2.fit.histogram.HistContainer`: data container for storing histograms
*  :py:obj:`~kafe2.fit.histogram.HistParametricModel`: corresponding model
*  :py:obj:`~kafe2.fit.histogram.HistFit`: a fit of a parametric model to histograms

Unbinned
--------

If data is treated as *unbinned* the model function :math:`f(x)` is interpreted as a
*model density function*.
The cost function value :math:`C` is then directly calculated as the negative log-likelihood of the
data given said PDF:

.. math::
    C = - 2 \sum_{i=0}^{N-1} \ln f(x_i) .

An unbinned fit is the edge case of a histogram fit for as the individual bins become
infinitessimally thin.

The following objects are provided for handling unbinned data:

*  :py:obj:`~kafe2.fit.unbinned.UnbinnedContainer`: data container for storing unbinned data
*  :py:obj:`~kafe2.fit.unbinned.UnbinnedParametricModel`: corresponding model
*  :py:obj:`~kafe2.fit.unbinned.UnbinnedFit`: a fit of a parametric model to unbinned data

Custom
------

Lets the user directly define a cost function.
Since this fit type does not have explicit data the fit results cannot be plotted automatically.

The following objects are provided for custom fits:

*  :py:obj:`~kafe2.fit.custom.CustomFit`: a fit for minimizing a cost function

Plots
-----

For creating graphical representations of fits, the :py:obj:`~kafe2.fit.Plot` is provided.
It can be instantiated with any fit object (or list of fit objects) as an argument and will produce
one or more plots accordingly using `matplotlib`.

:synopsis: This module contains specialized objects for storing measurement data,
           defining and fitting parametric models to these data and producing
           graphical representations ("plots") of the result.
           It relies on the :py:mod:`kafe2.core` module for basic functionality.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""

# public interface of submodule 'kafe2.fit'

# WARNING: Changing the order of these imports might break the kafe2 interface
# flake8: noqa F401, F403 (unused import, unable to detect undefined names from '*' import)
# fmt: off
# isort: off
from .custom import *
from .indexed import *
from .histogram import *
from .multi import *
from .tools import *
from .xy import *
from .unbinned import *

from ._base.plot import Plot
from .tools.fit_wrapper import Fit  # import after every fit to avoid import conflicts with other tools

from .util import function_library, custom_fit, hist_fit, indexed_fit, unbinned_fit, xy_fit, plot, \
    k2Fit
# fmt: on
# isort: on
