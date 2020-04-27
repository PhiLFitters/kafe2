r"""The :py:mod:`kafe2.fit` module provides an essential toolkit for estimating model
parameters from data ("fitting").

It distinguishes between a number of different data types:

*  series of indexed measurements (dedicated submodule: :py:mod:`~kafe2.fit.indexed`),
*  *xy* data (dedicated submodule: :py:mod:`~kafe2.fit.xy`), and
*  histograms (dedicated submodule: :py:mod:`~kafe2.fit.histogram`)

Each of the above data types has its own particularities when it comes to fitting.
The main difference is due to the way uncertainties can be defined and
interpreted for each type of data.


Indexed data
------------

For **indexed** data, one data set consists of a list of :math:`N` distinct measurements
:math:`d_i`, with the (discrete) index :math:`i` ranging from :math:`0` to :math:`N-1`.
For each measurement in the series, one or more uncertainty sources can be defined,
each being a numerical estimate of how much the respective measurement fluctuates.
Correlations between uncertainties on separate measurements :math:`d_i` and :math:`d_j` can also
be taken into account by using *covariance/correlation matrices*.

Fits to *indexed* data take these uncertainties and correlations into account when estimating
the model parameters and their uncertainties. When plotting indexed data, measurements are
represented as data points with error bars at :math:`(x=i, y=d_i)`, whereas the model
is indicated by a horizontal line near the corresponding data point.

The following objects are provided for handling *indexed* data, as described above:

*  :py:obj:`~kafe2.fit.indexed.IndexedContainer`: data container for storing *indexed* data
*  :py:obj:`~kafe2.fit.indexed.IndexedParametricModel`: corresponding model:

   - uses a model function (:py:obj:`~kafe2.fit.indexed.model.IndexedModelFunction`) to calculate the model
     predictions and stores the result in an :py:obj:`~kafe2.fit.IndexedContainer`

*  :py:obj:`~kafe2.fit.indexed.IndexedFit`: a fit of a parametric model to *indexed* data:

   - finds the minimum of the *cost function* to find the
     parameter values for which the model best fits the data


XY data
-------

For **xy** data, the same principle as for *indexed* data applies, except each measurement and model prediction
now depends on a continuous real independent variable :math:`x` instead of a discrete index :math:`i`.
In effect, the data now consist of :math:`N` ordered pairs :math:`(x=x_i, y=d_i)`.

In contrast to *indexed* data, where only uncertainties on the measurement could be defined,
for *xy* data there is the additional possibility of defining additional uncertainites on :math:`x`.
These can be handled in a number of different ways when fitting an *xy* model to data.
When plotting the result of *xy* fits, the model function is displayed as a continuous
function of :math:`x`, and an **error band** can be computed to reflect the model uncertainty,
as determined by propagating the data uncertainties.

.. TODO: complete section on x errors

The following objects are provided for handling *xy* data:

*  :py:obj:`~kafe2.fit.xy.XYContainer`: data container for storing *xy* data
*  :py:obj:`~kafe2.fit.xy.XYParametricModel`: corresponding model:

   - uses a model function (:py:obj:`~kafe2.fit.xy.model.XYModelFunction`) to calculate the model predictions and
     stores the result in an :py:obj:`~kafe2.fit.XYContainer`

*  :py:obj:`~kafe2.fit.xy.XYFit`: a fit of a parametric model to *xy* data:

   - finds the minimum of the *cost function* to find the parameter values for which the model best fits the data


Histograms
----------

Finally, *kafe2* is also able to handle **histograms**. Histograms organize measurements whose
values can fall anywhere across a continuum of values into a number of discrete regions
or "bins". Typically, the continuous "measurement space" (a closed real interval :math:`[x_{\rm min}, x_{\rm max}]`)
is subdivided into a sequence of successive intervals at the "bin edges"
:math:`x_{\rm min} < x_1 < x_2 < \ldots < x_{\rm max}`. Whenever a measurement falls into one of the bins, the
value of that histogram bin is incremented by one.
So a histogram is completely defined by its bin edges and the bin values.

.. note::
    The bin numbering starts at :math:`1` for the first bin and ends at :math:`N`, where :math:`N`
    is defined as the *size* of the histogram. The bin numbers :math:`0` and :math:`N+1` refer to the
    underflow (below :math:`x_{\rm min}`) and overflow bin (above :math:`x_{\rm max}`), respectively.


Defining a parametric model for histograms is not as straightforward as for *xy* and *indexed* data.
Seeing as they keep track of the number of entries in different intervals of the continuum, the bin values
can be interpreted using probability theory.

As the number of entries approaches infinity, the number of entries :math:`n` in the bin covering an interval
:math:`[a, b)`, divided by the total number of entries :math:`N_{\rm E}`, will approach the probablity of an
event landing in that bin:

.. math::
    \lim_{N_{\rm E}\rightarrow\infty} \frac{n}{N_{\rm E}} = \int_a^b f(x)\,{\rm d}x = F(b) - F(a)

In the above formula, :math:`f(x)` is the **probability distribution function** (or **probability density**),
and :math:`F(x)` is an antiderivative of :math:`f`
(for example the **cumulative probability distribution function**).

Using the above relation, the model prediction :math:`m` for the bin :math:`[a, b)` can be defined as:

.. math::
    m = N_{\rm E} \int_a^b f(x)\,{\rm d}x = N_{\rm E} \left(F(b) - F(a)\right)

This means that, for histograms, the *model density* :math:`f(x)` needs to be specified.
The model is then calculated by numerically integrating this function over each bin.
When fitting, however, the model needs to be calculated for many different points in parameter space,
which makes this approach very slow (many numerical integrations until the fit converges).

An alternative would be to specify the model density *antiderviative* :math:`F` alongside
the model, so that the model can be calculated as a simple difference, rather than a complicated integral.

The following objects are provided for handling histograms:

*  :py:obj:`~kafe2.fit.histogram.HistContainer`: data container for storing histograms
*  :py:obj:`~kafe2.fit.histogram.HistParametricModel`: corresponding model:

   - uses a model function (:py:obj:`~kafe2.fit.xy.model.HistModelFunction`) to calculate the model predictions
     and stores the result in an :py:obj:`~kafe2.fit.HistContainer`

*  :py:obj:`~kafe2.fit.histogram.HistFit`: a fit of a parametric model to histograms:

   - finds the minimum of the *cost function* to find the parameter values for which the model best fits the data

For creating graphical representations of fits, the :py:obj:`~kafe2.fit.Plot` is provided. It can be instantiated
with any fit object (or list of fit objects) as an argument and will produce one or more plots accordingly
using `matplotlib`.

:synopsis: This module contains specialized objects for storing measurement data,
           defining and fitting parametric models to these data and producing
           graphical representations ("plots") of the result.
           It relies on the :py:mod:`kafe2.core` module for basic functionality.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
"""


# public interface of submodule 'kafe2.fit'


from .indexed import *
from .unbinned import *
from .histogram import *
from .multi import *
from .tools import *
from .util import *
from .xy import *

from ._base.plot import Plot
from .tools.fit_wrapper import Fit  # import after every fit to avoid import conflicts with other tools
