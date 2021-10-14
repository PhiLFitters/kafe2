.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow

************************
Mathematical Foundations
************************

This chapter describes the mathematical foundations on which *kafe2* is built,
in particular the method of maximum likelihood and the profile likelihood method.

When performing a fit (as a physicist) the problem is as follows:
you have some amount of measurement **data** from an experiment that you need to compare to one or
more **models** to figure out which of the models - if any - provides the most accurate description
of physical reality.
You typically also want to know the values of the **parameters** of a model and the degree of
(un)certainty on those values.

For very simple problems you can figure this out **analytically**:
you take a formula and simply plug in your measurements.
However, as your problems become more complex this approach becomes much more difficult -
or even straight up impossible.
For this reason complex problems are typically solved **numerically**:
you use an algorithm to calculate a result that only approximates the analytical solution
but in a way that is much easier to solve.
*kafe2* is a tool for the latter approach.


Cost Functions
==============

In the context of **parameter estimation**, a model :math:`m_i(\bm{p})`
(where :math:`i` is just some index) is a function of the parameters :math:`\bm{p}`.
During the fitting process the parameters :math:`\bm{p}` are varied in such a way that the
agreement between the model and the corresponding data :math:`d_i` becomes "best".
This of course means that we need to somehow define "good" or "bad" agreement - we need a metric.
This metric is called a loss function or **cost function**.
It is defined in such a way that a lower cost function value corresponds with a "better" agreement
between a model and our data.

The cost functions implemented by *kafe2* are based on the **method of maximum likelihood**.
The idea behind this method is to *maximize* the likelihood function :math:`\mathcal{L}({\bm p})`
which represents the probability with which a model
:math:`\bm{m}(\bm{p})` would result in the data :math:`\bm{d}` that we ended up measuring:

.. math::

    \mathcal{L}({\bm p}) = \prod_i P(m_i({\bm p}), d_i),

where :math:`P(m_i({\bm p}), d_i)` describes the probability of measuring the data
point :math:`d_i` given the corresponding model prediction :math:`m_i({\bm p})`.
This approach allows for a **statistical interpretation** of our fit results as we will see later.
Instead of the likelihood described above however, we are instead using twice its negative
logarithm, the so-called **negative log-likelihood** (NLL):

.. math::

   \mathrm{NLL} (\bm{p})
   = - 2 \log \mathcal{L}({\bm p})
   = - 2 \log \left( \prod_i P(m_i({\bm p}), d_i) \right)
   = - 2 \sum_i \log P(m_i({\bm p}), d_i).

This transformation is allowed because logarithms are
**strictly monotonically increasing functions**, and therefore the negative logarithm of any
function will have its global minimum at the same place where the original function is maximal.
The model :math:`\bm{m}({\bm p})` that minimizes the NLL
will therefore also maximize the likelihood.

While the above transformation may seem nonsensical at first, there are important advantages to
calculating the negative log-likelihood over the likelihood:

-   The *product* of the probabilities :math:`\prod_i P(m_i({\bm p}), d_i)` is replaced by a
    *sum* over the logarithms of the probabilities :math:`\sum_i \log P(m_i({\bm p}), d_i)`.
    In the context of a computer program sums are preferable over products because they can be
    calculated more quickly and because a product of many small numbers can lead to
    arithmetic underflow.

-   Because the probabilities :math:`P(m_i({\bm p}), d_i)` oftentimes contain exponential functions,
    calculating their logarithm is actually *faster* because it reduces the number of necessary
    operations.

-   Algorithms for numerical optimization **minimize** functions so they can be directly used to
    optimize the NLL.

As an example, let us consider the likelihood function of data :math:`d_i` that follows a
**normal distribution** around means :math:`\mu_i`
with standard deviations (uncertainties) :math:`\sigma_i`:

.. math::

    \mathcal{L}
    = \prod_i P(\mu_i, \sigma_i, d_i)
    = \prod_i \frac{1}{\sqrt[]{2 \pi} \: \sigma_i}
      \exp \left[ - \frac{1}{2} \left( \frac{d_i - \mu_i}{\sigma_i} \right)^2 \right].

The immediate trouble that we run into with this definition is that we have no idea what the
means :math:`\mu_i` are - after all these are the "true values" that our data deviates from.
However, we can still use this likelihood function by choosing :math:`\mu_i = m_i({\bm p})`.
This is because for an infinite amount of data the model values :math:`m_i({\bm p})`
converge against the true values :math:`\mu_i`
(assuming our model is accurate and our understanding
of the uncertainties :math:`\sigma_i` is correct).

.. note ::
    Conceptually uncertainties are typically associated with the data :math:`\bm{d}` even though
    they represent deviations from the model :math:`\bm{m}({\bm p})`.
    However, because the normal distribution is symmetric this does not have an effect on the
    likelihood function :math:`\mathcal{L}`
    (as long as the uncertainties :math:`\bm{\sigma}` do not depend on the model
    :math:`\bm{m}({\bm p})`).

For the NLL we now find:

.. math::

   \mathrm{NLL}(\bm{p})
   = -2 \log \mathcal{L}({\bm p}) \\
   = - 2 \log \prod_i \frac{1}{\sqrt[]{2 \pi} \: \sigma_i}
    \exp \left[ - \frac{1}{2} \left( \frac{d_i - m_i({\bm p})}{\sigma_i} \right)^2 \right] \\
   = - 2 \sum_i \log \frac{1}{\sqrt[]{2 \pi} \: \sigma_i}
    + \sum_i \left( \frac{d_i - m_i({\bm p})}{\sigma_i} \right)^2 \\
   =: - 2 \log L_\mathrm{max} + \chi^2({\bm p}) .

As we can see the logarithm cancels out the exponential function of the normal
distribution and we are left with two parts:
The first is a part represented by :math:`- 2 \log L_\mathrm{max}` that only depends on the
uncertainties :math:`\sigma_i` but not on the model :math:`m_i({\bm p})` or the data :math:`d_i`.
This is the minimum value the NLL could possibly take on if the model :math:`m_i({\bm p})`
were to exactly fit the data :math:`d_i`.
The second part can be summed up as

.. math::
   \chi^2 (\bm{p}) = \sum_i \left( \frac{d_i - m_i({\bm p})}{\sigma_i} \right)^2.

As mentioned before, this is much easier and faster to calculate than the original
likelihood function.
If the uncertainties :math:`\bm{\sigma}` are constant we can ignore the first part and directly use
:math:`\chi^2({\bm p})` (chi-squared) as our cost function because
we are only interested in differences between cost function values.
This special case of the method of maximum likelihood is known as the **method of least squares**
and it is by far the most common cost function used for fits.

Covariance
==========

The :math:`\chi^2({\bm p})` cost function that we discussed in the previous section assumes
that our data points :math:`d_i` are subject to uncertainties :math:`\sigma_i`.
With this notation we implicitly assumed that our uncertainties are **independent** of one another:
that there is no relationship between the uncertainties of any two individual data points.
Independent uncertainties are frequently caused by random fluctuations in the measurement values,
either because of technical limitations of the experimental setup (electronic noise, mechanical
vibrations) or because of the intrinsic statistical nature of the measured observable (as is typical
in quantum mechanics, e. g. radioactive decays).

However, there are also **correlated** uncertainties that arise due to effects that distort multiple
measurements in the same way.
Such uncertainties can for example be caused by a random imperfection of the measurement device
which affects all measurements equally.
The uncertainties of the measurements taken with such a device are no longer uncorrelated, but
instead have one common uncertainty.

Historically uncertainties have been divided into *statistical* and *systematic* uncertainties.
While this is appropriate when propagating the uncertainties of the input variables by hand it is
not a suitable distinction for a numerical fit.
In *kafe2* multiple uncertainties are combined to construct a so-called **covariance matrix**.
This is a matrix with the pointwise data **variances** :math:`\mathrm{Var}_i`
on its diagonal and the **covariances** :math:`\mathrm{Cov}_{ij}`
between two data points outside the diagonal.
By using this covariance matrix for our fit we can estimate the uncertainty of our model parameters
numerically with no need for propagating uncertainties by hand.

As mentioned before, the diagonal elements of our covariance matrix represent the variances
:math:`\mathrm{Var}_i = \sigma_i^2` of our data points.
They simply represent the uncertainty of a single data point :math:`d_i` while ignoring all other
data points.
An element outside the diagonal at position :math:`(i,j)` represents the covariance
:math:`\mathrm{Cov}_{ij}` between points :math:`d_i` and :math:`d_j`:

.. math ::
   \mathrm{Cov}_{ij}
   = E[ (d_i - E[d_i])(d_j - E[d_j]) ]
   = E[d_i \cdot d_j] - E[d_i] \cdot E[d_j]
   = E[d_i \cdot d_j] - \mu_i \cdot \mu_j,

where :math:`E` is the expected value of a variable.
The covariance :math:`\mathrm{Cov}_{ij}` is a measure of the joint variability of :math:`d_i` and
:math:`d_j` - but for a meaningful interpretation it needs to be considered relative to the
pointwise uncertainties :math:`\sigma_i`.
We therefore define the so-called **Pearson correlation coefficient** :math:`\rho_{ij}` as follows:

.. math ::
   \rho_{ij} = \frac{\mathrm{Cov}_{ij}}{\sigma_i \sigma_j}.

.. only:: html

    The correlation :math:`\rho_{ij}` is normalized to the interval :math:`[-1, 1]`.
    Its absolute value is a measure of how strongly the residuals :math:`r_k = d_k - \mu_k`
    depend on one another.
    In other words, the absolute value of :math:`\rho_{ij}` measures how much information
    you get about :math:`r_i` or :math:`r_j` if you know the other one.
    For :math:`\rho = 0` they are completely independent from one another.
    For :math:`\rho = \pm 1` :math:`r_i` and :math:`r_j` are directly proportional to one
    another with a positive (negative) proportional constant for
    :math:`\rho = +1` (:math:`\rho = -1`).
    Let's look at some toy samples for different values of :math:`\rho_{ij}`:

    .. figure:: ../_static/img/covariance_plot.png

.. only:: latex

    The correlation :math:`\rho_{ij}` is normalized to the interval :math:`[-1, 1]`.
    Its absolute value is a measure of how strongly the residuals :math:`r_k = d_k - \mu_k`
    depend on one another.
    In other words, the absolute value of :math:`\rho_{ij}` measures how much information
    you get about :math:`r_i` or :math:`r_j` if you know the other one.
    For :math:`\rho = 0` they are completely independent from one another.
    For :math:`\rho = \pm 1` :math:`r_i` and :math:`r_j` are directly proportional to one
    another with a positive (negative) proportional constant for
    :math:`\rho = +1` (:math:`\rho = -1`).
    Toy samples for different values of :math:`\rho_{ij}` are shown in :numref:`covariance_plot`.

    .. _covariance_plot:
    .. figure:: ../_static/img/covariance_plot.png

        Toy samples for correlation between residuals :math:`r_i` and :math:`r_j` for different
        values of the correlation coefficient :math:`\rho_{ij}`.
        With an increasing absolute value the shape changes from a circle to a line.

For :math:`\rho_{ij} = 0` the sample forms a circle around (0,0).
As the absolute value of :math:`\rho_{ij}` increases the sample changes its shape to a tilted
ellipse - some combinations of :math:`r_i` and :math:`r_j` become more likely than others.
For :math:`\rho_{ij} = \pm 1` the ellipse becomes a line -
in this degenerate case we really only have one source of uncertainty that affects two data points.

As before, if we have "enough" data we can assume :math:`r_k \approx d_k - m_k(\bm{p})`.
This is useful because it allows us to use a covariance matrix to express the correlations of our
uncertainties in our cost function, as we will see shortly.

Covariance Matrix Construction
******************************

In a physics experiment it is typically necessary to consider more than one source of uncertainty.
Let us consider the following example:
we want to measure Earth's gravitational constant :math:`g` by dropping things from various heights
and timing the time they take to hit the ground with a stopwatch.
We assume an independent uncertainty of :math:`\sigma_{\rm human} = 0.5 s` for each data point
because humans are not able to precisely align pressing the button of a stopwatch with the actual
event.
For one reason or another the stopwatch we're using is also consistently
off by a few percentage points.
To account for this we assume a fully correlated (:math:`\rho_{ij} = 1`) uncertainty of
:math:`\sigma_{\rm watch} = 2 \%` for all data points.
To determine the variance of a single data point we can simply add up the variances
of the uncertainty sources:

.. math::
   {\rm Var}_{\rm total}
   = \sigma_{\rm total}^2
   = {\rm Var}_{\rm human} + {\rm Var}_{\rm watch}
   = \sigma_{\rm human}^2 + \sigma_{\rm watch}^2.

As it turns out we can use the same approach for the covariances:
we can simply add up the covariance matrices of the different uncertainty sources
to calculate a total covariance matrix:

.. math::
   {\bm V}_{\rm total} = {\bm V}_{\rm human} + {\bm V}_{\rm watch}.

The next question would then be how you would determine the covariance matrices for the
individual uncertainty sources.
A useful approach is to split a covariance matrix into a vector of uncertainty :math:`\bm \sigma`
and the corresponding correlation matrix :math:`\bm \rho`:

.. math::
   \bm{V} = (\bm{\sigma} \cdot \bm{\sigma}^T) \circ \bm{\rho},

where :math:`\circ` is the Hadamard product (a.k.a. Schur product).
In other words, the components of :math:`\bm V` are calculated by simply multiplying the
components of :math:`{\bm \sigma} \cdot {\bm \sigma}^T` and :math:`\bm \rho` at
the same position.
If we assume that we have three data points we can express the human uncertainty as follows:

.. math::
   \bm{\sigma}_\mathrm{human} = \begin{pmatrix} 0.5 \\ 0.5 \\ 0.5 \end{pmatrix},
   \quad \bm{\rho}_\mathrm{human} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0\\ 0 & 0 & 1\end{pmatrix},
   \quad \bm{V}_\mathrm{human}
   = \begin{pmatrix} 0.25 & 0 & 0 \\ 0 & 0.25 & 0\\ 0 & 0 & 0.25 \end{pmatrix}.

Because the human uncertainties of the individual data points are completely independent from one
another the covariance/correlation matrix is a diagonal matrix.
On the other hand, given some data points :math:`\bm{d}`
the watch uncertainty is expressed like this:

.. math::
   \bm{\sigma}_\mathrm{watch} = 0.02 \cdot \bm{d}
   = 0.02 \cdot \begin{pmatrix} d_1 \\ d_2 \\ d_3 \end{pmatrix},
   \quad \bm{\rho}_\mathrm{watch} = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1\\ 1 & 1 & 1\end{pmatrix},
   \quad \bm{V}_\mathrm{watch} = 0.0004 \cdot
    \begin{pmatrix} d_1^2 & d_1 d_2 & d_1 d_3 \\
                    d_1 d_2 & d_2^2 & d_2 d_3 \\
                    d_1 d_3 & d_2 d_3 & d_3^2
    \end{pmatrix}.

Because the watch uncertainties of the individual data points are fully correlated all components
of the correlation matrix are equal to 1.
However, this does not necessarily mean that all components of the covariance matrix are also equal.
In this example the watch uncertainty per data point is relative,
meaning that the absolute uncertainty differs from data point to data point.

If we were to visualize the correlations of the uncertainty components described above,
we would find that samples of the human component form a circle
while samples from the watch component form a line.
If we were to visualize the total uncertainty we would end up with the mixed case where the sample
forms an ellipse.

Correlated Least Squares
************************

We previously defined the :math:`\chi^2` cost function like this:

.. math::
   \chi^2 (\bm{p}) = \sum_i \left( \frac{d_i - m_i({\bm p})}{\sigma_i} \right)^2.

This definition is only correct if the uncertainties for each data point are independent.
If we want to consider the correlations between uncertainties we need to use
the covariance matrix :math:`\bm{V}` instead of the pointwise uncertainties :math:`\sigma_i`:

.. math::
   \chi^2 (\bm{p})
   = (\bm{d} - \bm{m}(\bm{p}))^T \cdot \bm{V}^{-1} \cdot (\bm{d} - \bm{m}(\bm{p})).

Notably the division by the uncertainties :math:`\sigma_i` has been replaced by a matrix inversion.
This is because the uncorrelated definition is a special case of the correlated definition.
If the uncertainties are completely uncorrelated then :math:`\bm{V}` is a diagonal matrix.
To invert such a matrix you only need to replace the diagonal elements
:math:`V_{ii}` with :math:`1 / V_{ii}`.

Profile Likelihood
==================

When we perform a fit we are not only interested in the parameter values that fit our data "best",
we also want to determine the uncertainty on our result.
Fortunately likelihood-based cost functions provide a straightforward solution to our problem:
the so-called **profile likelihood method**.
By analyzing how a variation of one or more parameters affects the cost function value relative
to the global cost function minimum we can determine areas that contain the true values
of our parameters with a given **confidence level** of, say 95%.

Profile Likelihood (1 Parameter)
********************************

Let's say we performed a fit and found the global cost function minimum of our
negative log-likelihood cost function with optimal parameters :math:`\hat{a}, \hat{\bm{p}}`
(:math:`a` is just one of the parameters that we consider separately).
Because we have some amount of uncertainty on our input data we end up having some amount of
uncertainty on our fit result as well.
The global cost function minimum is the best fit (according to our cost function).
And because our cost function measures how good a fit is given some parameter value :math:`a`,
investigating how flat or steep the cost function minimum is as a function of :math:`a`
tells us something about this parameter:
if the cost function value increases very sharply when we move away from the cost function minimum
then this tells us that even a small deviation from our fit result would result in a
significantly worse fit, making large deviations unlikely.
Conversely, if the cost function value increases very slowly when we move away
from the cost function minimum then this tells us that a deviation from our
fit result would result in a fit that is only slightly worse than our optimal fit result,
making such a deviation from our fit result quite possible.

We are trying to determine a so-called **confidence interval** for :math:`a`:
an interval that we expect to contain the true value of :math:`a` with a given probability
called the **confidence level** :math:`\mathrm{CL}`.
The relevant metric for determining these intervals can be derived by considering
the cost function increase relative to the global cost function minimum:

.. math::
   \Delta \mathrm{NLL}(a, \bm{p}) = \mathrm{NLL}(a, \bm{p}) - \mathrm{NLL}(\hat{a}, \hat{\bm{p}}).

The obvious problem with this definition is that we need values not only for :math:`a`
but also for all other parameters :math:`\bm{p}` which we aren't actually interested in right now.
So how do we determine the values for these parameters?
The approach of the profile likelihood method is to choose :math:`\bm{p}` in such a way that
:math:`\Delta \mathrm{NLL}(a, \bm{p})` becomes minimal.
In practical terms this means that we fix :math:`a` to several values near the cost function minimum
and then perform a fit over all other parameters for each of these values
(this process is called profiling).
In this context the parameters :math:`\bm{p}` are called **nuisance parameters**:
we don't care about their values (right now) but we need to include them in our fits for
a statistically correct result.
If we were to instead use the optimal parameters :math:`\hat{\bm{p}}` then we would save on
computing time but we would also be neglecting correlations between our fit parameters.
Very often a variation of one fit parameter can (in part) be compensated by varying
one of the other fit parameters.
If we were to use the optimal parameters :math:`\hat{\bm{p}}` instead of performing a fit
the cost function would increase more quickly when we vary :math:`a` so we would end up
*understimating* the uncertainty on :math:`a`.

Now, the question is how to translate a desired confidence level :math:`\mathrm{CL}`
to a difference in cost :math:`\Delta \mathrm{NLL}(a, \bm{p})`.
As it turns out the confidence level for a confidence interval can be calculated
from the probability density function (PDF) of the standard normal distribution:

.. math::
   \mathrm{CL} = \int_{-x_\mathrm{max}}^{x_\mathrm{max}}
   \frac{1}{\sqrt{2 \pi}} e^{-\frac{x^2}{2}},
   \quad x_\mathrm{max} = \sqrt{\Delta \mathrm{NLL}(a, \bm{p})}.

What we're actually interested in however, is the inverse case of calculating
:math:`\Delta \mathrm{NLL}(a, \bm{p})` for a given confidence level.
Because the PDF of the normal distribution cannot be integrated analytically we have to
resort to numerical integration -
SciPy's **percent point function** (`scipy.stats.norm.ppf`)
conveniently provides the function we need.
We can use it to calculate :math:`\Delta \mathrm{NLL}(a, \bm{p})` like this:

.. math::
   \Delta \mathrm{NLL}(a, \bm{p})
   = \left(\mathrm{PPF}\left(\frac{1}{2} + \frac{CL}{2}\right)\right)^2.

The takeaway of this complicated-looking formula is this:
the difference in cost is equal to the square of the "sigma value" of the
commonly used confidence intervals of the normal distribution:
the 1-:math:`\sigma`-interval with :math:`\mathrm{CL} \approx 68\%` corresponds to
:math:`\Delta \mathrm{NLL}(a, \bm{p}) = 1^2 = 1`,
the 2-:math:`\sigma`-interval with :math:`\mathrm{CL} \approx 95\%` corresponds to
:math:`\Delta \mathrm{NLL}(a, \bm{p}) = 2^2 = 4`,
the 3-:math:`\sigma`-interval with :math:`\mathrm{CL} \approx 99.7\%` corresponds to
:math:`\Delta \mathrm{NLL}(a, \bm{p}) = 3^2 = 9`,
and so on.

.. note::
   The above formula is only correct for one dimension.
   If the shared profile likelihood of more than one parameter is examined we will
   need to use a different formula (see below).

The profile likelihood method is very expensive in terms of computation.
For this reason it is not the default in *kafe2*.
Instead the default behavior is to assume that the cost function value increases like

.. math::
   \Delta \mathrm{NLL}(a, \hat{\bm{p}})
   = \mathrm{NLL}(\hat{a}, \hat{\bm{p}}) + \left( \frac{a - \hat{a}}{\sigma_a} \right)^2,

where :math:`\sigma_a` is the **parabolic parameter uncertainty** of :math:`a`.
These are the standard parameter uncertainties provided by *kafe2* (and in fact most fitting tools).
Because every minimum can be approximated by a parabola for sufficiently small scales
(Taylor expansion) the parabolic parameter uncertainties are sufficiently accurate for many
applications - but if you suspect they are not you should check the profiles of the parameters
to make sure the result you extract is actually meaningful.

The easiest way to do this is to set the flag ``asymmetric_parameter_errors = True`` when calling
``FitBase.report()`` or ``Plot.plot()``.
The parabolic parameter uncertainties are then replaced with the edges of the
1-:math:`\sigma`-intervals of the corresponding cost function profiles.
Because these intervals are not necessarily symmetric around the cost function minimum they are
referred to as **asymmetric parameter errors** in *kafe2*
(in *Minuit* they are called Minos errors).

.. only:: html

    Another way to check the profiles is to use the :py:obj:`~.ContoursProfiler` object.
    It is capable of plotting the profiles of parameters (and also their contours, see below).
    As an example, let us look at the profile of the parameter :math:`g`
    from the double slit example:

    .. figure:: ../_static/img/003_double_slit_profile_g.png

.. only:: latex

    Another way to check the profiles is to use the :py:obj:`~.ContoursProfiler` object.
    It is capable of plotting the profiles of parameters (and also their contours, see below).
    :numref:`003_double_slit_profile_g` shows the profile of the parameter :math:`g`
    from the double slit example:

    .. _003_double_slit_profile_g:
    .. figure:: ../_static/img/003_double_slit_profile_g.png

       Profile of parameter :math:`g` from the double slit example.
       The parabolic approximation of the confidence interval is very inaccurate.

The profile of this parameter is very clearly asymmetric and not even close to the
parabolic approximation.
If we had only looked at the parabolic parameter uncertainty our idea of the actual
confidence intervals would be very wrong.

Profile Likelihood (2 parameters)
*********************************

.. only:: html

    In the previous section we learned about the profiles of single fit parameters,
    which serve as a replacement for the uncertainties of single fit parameters.
    In this section we will learn about so-called **contours**,
    which serve as a replacement for the covariance of two fit parameters.
    Conceptually they are very similar.
    A profile defines confidence intervals for a single parameter with a certain likelihood of
    containing the true value of a parameter
    while a contour defines a **confidence region** with a certain likelihood of containing a *pair*
    of parameters.
    Let us start by looking at the contours produced in the double slit example:

    .. figure:: ../_static/img/003_double_slit_contours.png

.. only:: latex

    In the previous section we learned about the profiles of single fit parameters,
    which serve as a replacement for the uncertainties of single fit parameters.
    In this section we will learn about so-called **contours**,
    which serve as a replacement for the covariance of two fit parameters.
    Conceptually they are very similar.
    A profile defines confidence intervals for a single parameter with a certain likelihood of
    containing the true value of a parameter
    while a contour defines a **confidence region** with a certain likelihood of containing a *pair*
    of parameters.
    :numref:`003_double_slit_contours` shows the contours produced
    in the double slit example.

    .. _003_double_slit_contours:
    .. figure:: ../_static/img/003_double_slit_contours.png

        Parameter confidence contours produced in the double slit example.
        Due to the nonlinear model function the contours are heavily distorted.


In this visualization the confidence region inside the contours is colored.
By looking at the legend we find that the contours correspond to
1 :math:`\sigma` and 2 :math:`\sigma`.
Notably the confidence levels of the corresponding confidence regions are *not*
the same as in one dimension.
In one dimension 1 :math:`\sigma` corresponds to roughly 68% while
2 :math:`\sigma` corresponds to roughly 95%.
We could derive these confidence levels by integrating the probability density function
of the standard normal distribution over the interval :math:`[-\sigma , \sigma]`
for a desired :math:`\sigma` value.
In two dimensions we instead integrate the PDF of the uncorrelated standard bivariate
normal distribution over a circle with radius :math:`\sigma` around the origin:

.. math::
   \mathrm{CL}(\sigma)
   = \int_0^\sigma dr \int_0^{2 \pi} d \varphi r \frac{1}{2 \pi} e^{- \frac{r^2}{2}}
   = \int_0^\sigma dr \ r e^{- \frac{r^2}{2}}
   = \left[ -e^{-\frac{r^2}{2}} \right]_0^\sigma
   = 1 - e^{-\frac{\sigma^2}{2}}.

With this formula we now find
:math:`\mathrm{CL}(1) = 39.3\%,\ \mathrm{CL}(2) = 86.4\%,\ \mathrm{CL}(3) = 98.8\%`.

.. note::
   So far there has been no mention of how a contour for a given
   :math:`\Delta \mathrm{NLL}` could be calculated.
   This is because (efficiently) calculating these contours is not straightforward and
   even in *kafe2* this is an area of active development.

.. only:: html

    The parabolic equivalent of a contour is to look at the parameter covariance matrix and to
    extrapolate the correlated distribution of two parameters.
    As with the input uncertainties the confidence region calculated this way will
    *always* be an ellipse.
    For (nearly) linear fits such as the exponential fit from the model functions example the
    calculated contours will then look something like this:

    .. figure:: ../_static/img/002_exponential_contours.png

.. only:: latex

    The parabolic equivalent of a contour is to look at the parameter covariance matrix and to
    extrapolate the correlated distribution of two parameters.
    As with the input uncertainties the confidence region calculated this way will
    *always* be an ellipse.
    :numref:`002_exponential_contours` shows contours for the nearly linear exponential fit
    from the model functions example.

    .. _002_exponential_contours:
    .. figure:: ../_static/img/002_exponential_contours.png

        Parameter confidence contours for the exponential fit from the model functions example.
        The fit is nearly linear on the scale of the uncertainty
        so the confidence region is close to an ellipse.

If the fit were perfectly linear the 1-:math:`\sigma`-contour would reach exactly from
:math:`-\sigma` to :math:`+\sigma`,
while the 2-:math:`\sigma`-contour would reach exactly from :math:`-2 \sigma` to :math:`+2 \sigma`.
As we can see the deviation from this is very small so we can probably use the parameter covariance
matrix (or the parameter uncertainties and the parameter correlation matrix) without issue.
If we require highly precise confidence intervals for our parameters
this might not be acceptable though.

.. note::
   The degree to which confidence intervals/regions are distorted from their parabolic
   approximation depends on the scale at which the profile likelihood is calculated.
   Because every function can be accurately approximated by a linear function at infinitesimally
   small scales (Taylor expansion) the parabolic approximation becomes more accurate
   for small parameter uncertainties.
   Conversely, for large parameter uncertainties the parabolic approximation of the profile
   likelihood becomes less accurate.

Nonlinear Regression
====================

In the previous section we discussed the profile likelihood method and how it can
be used to calculate confidence intervals for our fit parameters.
We also discussed the approximation of these confidence intervals through the use of
parabolic uncertainties.
In this context the term "linear" was used to describe fits where the parabolic uncertainties
are accurate.
This section will define more precisely what was meant by that.

Linear Regression
*****************

Let us assume we have some vector of :math:`N` data points :math:`d_i` with corresponding
constant Gaussian uncertainties :math:`\sigma_i` (that can also be correlated).
**Linear regression** is then defined as a regression analysis (fit) using a model
:math:`m_i(\bm{p})` that is a **linear function** of its :math:`M` parameters :math:`p_j`:

.. math::
   m_i(\bm{p}) = b_i + \sum_{j=1}^M w_{ij} p_j,

where the **weights** :math:`w_{ij}` and **biases** :math:`b_i` are simply real numbers.
Put another way, each model value :math:`m_i` is a linear combination of the
parameter values :math:`p_j` plus some bias :math:`b_i`.
We can express the same relationship as above with a weight matrix :math:`\bm{W}`
and a bias vector :math:`\bm{b}`:

.. math::
   \bm{m}(\bm{p}) = \bm{W} \bm{p} + \bm{b}.

If we now use the method of least squares (:math:`\chi^2` ) to estimate the
optimal fit parameters :math:`\hat{\bm{p}}` we get a very useful property:
the parabolic approximation perfectly describes the uncertainties of the optimal
fit parameters :math:`\hat{\bm{p}}`.
We can therefore skip the (relatively) expensive process of profiling the parameters!

Let us look at some examples for linear regression in the context of *xy* fits since
those are the most common.
Let us therefore assume that we have some *y* data :math:`\bm{d}` measured at
*x* values :math:`\bm{x} = (0, 1, 2)^T`.
The model function most commonly associated with linear regression is the
first degree polynomial :math:`f(x) = a + b x`.
We can thus express our model like this:

.. math::
   \bm{m}(\bm{p})
   = \bm{W} \bm{p}
   = \left( \bm{x}^0, \bm{x}^1 \right) \bm{p}
   = \begin{pmatrix} 1 & 0\\ 1 & 1\\ 1 & 2 \end{pmatrix} \begin{pmatrix} a\\ b \end{pmatrix}
   = a \bm{x}^0 + b \bm{x}^1.

The upper indices of vectors are to be interpreted as powers of said vectors using the
Hadamard/Schur product (component-wise multiplication).
In the above equation we only have a weight matrix :math:`W = \left( \bm{x}^0, \bm{x}^1 \right)`
but no bias vector.
We can clearly see that using the first degree polynomial (a line) as our model function
results in linear regression.
Let's take a look at the third degree polynomial :math:`f(x) = a + b x + c x^2 + d x^3`:

.. math::
   \bm{m}(\bm{p})
   = \bm{W} \bm{p}
   = \left( \bm{x}^0, \bm{x}^1, \bm{x}^2, \bm{x}^3 \right) \bm{p}
   = \begin{pmatrix} 1 & 0 & 0 & 0\\ 1 & 1 & 1 & 1\\ 1 & 2 & 4 & 8\end{pmatrix}
     \begin{pmatrix} a\\ b\\ c\\ d \end{pmatrix}
   = a \bm{x}^0 + b \bm{x}^1 + c \bm{x}^2 + d \bm{x}^3.

Again we find that the model :math:`\bm{m}(\bm{p})` is a linear function
of its parameters :math:`\bm{p}`.
A fit using a third degree polynomial as its model function is therefore also linear regression.
This is even though the model function is *not* a linear function
of the independent variable :math:`x`.
However, this was never required in our definition of linear regression to begin with because
:math:`x` is not one of our fit parameters.
In fact, all *xy* fits using polynomials as model functions fall under linear regression.

Nonlinear Regression
********************

Now that we have defined linear regression, the definition of **nonlinear regression** is
rather easy: a regression analysis (fit) that is not linear regression.
The natural consequence of this is that the parabolic approximation of the uncertainty
of our fit parameters is no longer perfectly accurate.
We will therefore need to resort to the profile likelihood method to calculate confidence intervals.
The most direct example of nonlinear regression is a fit with a model function that is
not a linear function of its parameters, e.g. :math:`f(x) = A \cdot e^{- \lambda x}`.
It is simply not possible to express this function using only a finite weight matrix :math:`\bm{W}`
and a bias vector :math:`\bm{b}`.
We would instead need an infinitely large matrix and infinitely many parameters.
With the same *x* vector :math:`\bm{x} = (0, 1, 2)^T` as before we find:

.. math::
   \bm{m}(\bm{p})
   = A \cdot e^{- \lambda \bm{x}}
   = A \cdot \sum_{k=0}^\infty \frac{(- \lambda \bm{x})^k}{k!}

.. math::
   = A \cdot \begin{pmatrix}
            \bm{x}^0 & -\bm{x}^1 & \frac{\bm{x^2}}{2} & -\frac{\bm{x}^3}{6} & \cdots
    \end{pmatrix} \begin{pmatrix}
            \lambda^0 \\ \lambda^1 \\ \lambda^2 \\ \lambda^3 \\ \vdots
    \end{pmatrix}
   = A \cdot \begin{pmatrix}
            1 & 0 & 0 & 0 & \\
            1 & -1 & \frac{1}{2} & -\frac{1}{6} & \cdots \\
            1 & -2 & 2 & -\frac{4}{3} & \\
    \end{pmatrix} \begin{pmatrix}
            \lambda^0 \\ \lambda^1 \\ \lambda^2 \\ \lambda^3 \\ \vdots
    \end{pmatrix}.

.. note::
   We could of course just cut off the series at some point to approximate the exponential function.
   This would be equivalent to approximating the exponential function with a polynomial.
   The parabolic uncertainties of our fit parameters would then be "accurate" but we would only be
   moving the problem because our model would become less accurate in the process.

Unfortunately, even with a linear model function the fit as a whole can become nonlinear
if certain *kafe2* features are used.
As of right now these features are uncertainties in *x* direction for *xy* fits
and uncertainties relative to the model.
This is because when using those features the uncertainties that we feed to our
negative log-likelihood are no longer constant.
Instead they become a function of the fit parameters: :math:`\sigma_i \rightarrow \sigma_i(\bm{p})`.
As a consequence we have to consider the full Gaussian likelihood rather
than just :math:`\chi^2` to get an unbiased result:

.. math::
   \mathrm{NLL}(\bm{p})
   = - 2 \log L_\mathrm{max}(\bm p) + \chi^2({\bm p})
   = - 2 \sum_i \log \frac{1}{\sqrt[]{2 \pi} \: \sigma_i(\bm{p})} + \chi^2(\bm{p}) \\
   = N \log (2 \pi) + 2 \sum_i^N \log \sigma_i(\bm{p}) + \chi^2(\bm{p})
   =: N \log (2 \pi) + C_\mathrm{det}(\bm{p}) + \chi^2(\bm{p}).

As with our derivation of :math:`\chi^2` we end up with a constant term :math:`N \log (2 \pi)`
which we can ignore because we are only interested in the differences in cost.
We also get a new term :math:`C_\mathrm{det}(\bm{p}) = 2 \sum_i^N \log \sigma_i(\bm{p})` that
we need to consider when our uncertainties depend on our fit parameters.
The new term results in higher cost when the uncertainties increase.
If we didn't add :math:`C_\mathrm{det}(\bm{p})` while handling parameter-dependent uncertainties
we would end up with a bias towards parameter values for which the uncertainties are increased
because those values result in a lower value for :math:`\chi^2`.
The subscript "det" is short for determinant, the reason for which should become clear when we
look at the full Gaussian likelihood with correlated uncertainties represented
by a covariance matrix :math:`\bm{V}(\bm{p})`:

.. math::
   \mathrm{NLL}(\bm{p})
   = - 2 \log L_\mathrm{max}(\bm{p}) + \chi^2(\bm{p})
   = - 2 \log \left[ (2 \pi)^{-\frac{N}{2}}
     \frac{1}{\sqrt{\det \bm{V}(\bm{p})}} \right] + \chi^2(\bm{p}) \\
   = N \log (2 \pi) + \log \det \bm{V}(\bm{p}) + \chi^2(\bm{p})
   =: N \log (2 \pi) + C_\mathrm{det}(\bm{p}) + \chi^2(\bm{p})

The constant term is the same as with the uncorrelated uncertainties but term we're interested in
has changed to :math:`C_\mathrm{det}(\bm{p}) = \log \det \bm{V}(\bm{p})`.
If the uncertainties are uncorrelated then the covariance matrix is diagonal
and the result is equal to the term we found earlier.

.. note::
   Handling correlated uncertainties that are a function of our fit parameters
   :math:`\bm{p}` is  computationally expensive because this means that we need to recalculate
   the inverse (actually Cholesky decomposition) of our covariance many times which has
   complexity :math:`O(N^3)` for :math:`N` data points - on modern hardware
   this is typically not an issue though.

Uncertainties In *x* Direction
------------------------------

Now that we know how to handle parameter-dependent uncertainties we can use this knowledge
to handle a very common problem:
fitting a model with model function :math:`f(x; \bm{p})` to data with *x* values :math:`x_i` and
uncertainties in both the *x* and the *y* direction.
The uncertainties in the *y* direction :math:`\sigma_{y, i}` can be used directly.
For the *x* uncertainties :math:`\sigma_{x, i}` we need a trick:
we project the uncertainties :math:`\sigma_{x, i}` onto the *y* axis by
multiplying them with the corresponding model function derivative by *x* :math:`f'(x_i; \bm{p})`:

.. math::
   \sigma_{xy,i}(\bm{p}) = \sqrt{\sigma_{y,i}^2 + (\sigma_{x,i} \cdot f'(x_i; \bm{p}))^2}.

The formula for the pointwise projected *xy* uncertainties :math:`\bm{\sigma}_{xy}` can
be generalized for the equivalent covariance matrices :math:`\bm{V}_x` and :math:`\bm{V}_y`:

.. math::
   \bm{V}_{xy}(\bm{p})
   = \bm{V}_y + (f'(\bm{x}; \bm{p}) \cdot f'(\bm{x}; \bm{p})^T) \circ \bm{V}_x,

where :math:`\circ` is again the Hadamard product (a.k.a. Schur product) where two matrices
are multiplied on a component-by-component basis.
We are also implicitly assuming that :math:`f'(\bm{x}; \bm{p})` is a vectorized function à la
*NumPy* that returns a vector of derivatives for a vector of x values :math:`\bm{x}`.

Uncertainties Relative To The Model
-----------------------------------

**Relative uncertainties** are very common.
For example, the uncertainties of digital multimeters are
typically specified as a percentage of the reading.
Unfortunately such uncertainties are therefore relative to the true values which we don't know.
The standard approach for handling relative uncertainties is therefore to specify them relative
to the data points :math:`d_i` which we do know.
However, this approach introduces a bias:
if the random fluctuation represented by an uncertainty causes our data :math:`d_i` to have
a reduced (increased) absolute value
then the relative uncertainties are underestimated (overestimated).
This causes a bias towards models with smaller absolute values in our fit because we are giving
data points that randomly happen to have a low absolute value a higher weight than data points
with a high absolute value -
and this bias increases for large relative uncertainties.

The solution for the bias described above is to specify uncertainties relative to the model
:math:`m_i(\bm{p})` rather than the data :math:`d_i`.
Because the model "averages" the fluctuations in our data we no longer give a higher weight to data
that randomly happens to have a lower absolute value.
The price we pay for this is that our total uncertainty becomes a function of our model parameters
:math:`\bm{p}` which results in an increase in computation time as described above.

Gaussian Approximation Of The Poisson Distribution
--------------------------------------------------

*kafe2* has a built-in approximation of the Poisson distribution where the Gaussian uncertainty is
assumed as:

.. math::
   \sigma_i(\bm{p}) = \sqrt{m_i(\bm{p})}.

The rationale for using the square root of the model :math:`m_i(\bm{p})` rather than the square
root of the data
:math:`d_i` is the same as with the relative uncertainties described in the previous section.
The benefit of using this approximation of the Poisson distribution instead of the
Poisson distribution itself is that it is capable of
handling additional Gaussian uncertainties on our data.

Hypothesis Testing
==================

So far we have used cost functions to compare how good or bad certain models
and parameter values fit our data relative to each other -
but we have never discussed how good or bad a fit is in an absolute sense.
Luckily for us there is a metric that we can use:
:math:`\chi^2 / \mathrm{NDF}`, where :math:`\chi^2` is simply the sum of the
squared residuals that we already know and :math:`\mathrm{NDF}` is the
**number of degrees of freedom** that our fit has.
The basic definition of :math:`\mathrm{NDF}` is that it's simply the number
of data points :math:`N_{\bm{d}}` minus the number of parameters :math:`N_{\bm{p}}`:

.. math::
   \mathrm{NDF} = N_{\bm{d}} - N_{\bm{p}}.

Conceptually the number of degrees of freedom are the number of "extra measurements"
over the minimum number of data points needed to fully specify a model with :math:`N_{\bm{p}}`
linearly independent parameters.
If our model is not fully specified then our cost function has multiple
(or even infinitely many) global minima.
For example, a line with model function :math:`f(x; a, b) = a x + b` has two
linearly independent parameters and as such needs at least two data points to be fully
specified.

If our model accurately describes our data, and if our assumptions about
the uncertainties of our data are correct, then :math:`\chi^2 / \mathrm{NDF}`
has an expected value of 1.
If :math:`\chi^2 / \mathrm{NDF}` is smaller (larger) than 1 we might be
overestimating (underestimating) the uncertainties on our data.
If :math:`\chi^2 / \mathrm{NDF}` is much larger than 1 then our model may not
accurately describe our data at all.

To further quantify these rather loose criteria we can make use of **Pearson's chi-squared test**.
This is a statistical test that allows us to calculate the probability
:math:`P(\chi^2, \mathrm{NDF})` with which we can expect to observe deviations from
our model that are at least as large as the deviations that we saw in our data.
To conduct this test we first need to define the so-called :math:`\bm{\chi^2}` **distribution**.
This distribution has a single parameter :math:`k` and when sampling from this distribution,
the samples from :math:`k` standard normal distributions :math:`x_l`
are simply squared and then added up:

.. math::
   \chi^2 (k) = \sum_{l=1}^k x_l^2 .

The deviations of our data relative to its true values (represented by our model) and
normalized to its uncertainties follow such standard normal distributions.
We can therefore expect the sum of the squares of these deviations :math:`\chi^2 (\bm{p})` to follow
a :math:`\chi^2 (k)` distribution with :math:`k = \mathrm{NDF}` -
if our model and our assumptions about the uncertainties of our data are correct.
We can associate the following cumulative distribution function (CDF)
:math:`F(k, x)` with the :math:`\chi^2` distribution:

.. math::
   F(x, k)
   = \frac{\int_0^\frac{x}{2} t^{\frac{k}{2} - 1}
   e^{-t} dt}{\int_0^\infty t^{\frac{k}{2} - 1} e^{-t} dt} .

To calculate the probability :math:`P(\chi^2, \mathrm{NDF})` with which we would expect a
:math:`\chi^2` value larger than what we got for our fit
(i.e. the probability of our fit being worse if
we could somehow "reroll" the deviations on our data)
we can now simply use:

.. math::
   P(\chi^2, \mathrm{NDF}) = 1 - F(\chi^2, \mathrm{NDF}).

In *kafe2* :math:`P(\chi^2, \mathrm{NDF})` is also referred to as the :math:`\chi^2` probability.
We can use this number to determine if deviations from our
assumed model are **statistically significant**.

The concept of :math:`\chi^2 / \mathrm{NDF}` as can be generalized for non-Gaussian likelihoods
where the metric becomes **goodness of fit**
per degree of freedom :math:`\mathrm{GoF} / \mathrm{NDF}`.
For a negative log likelihood :math:`\mathrm{NLL}(\bm{m}(\bm{p}), \bm{d})`
with model :math:`\bm{m}(\bm{p})` and
data :math:`\bm{d}` it is defined like this:

.. math::
   \mathrm{GoF} / \mathrm{NDF}
   = \frac{\mathrm{NLL}(\bm{m}(\hat{\bm{p}}), \bm{d}) - \mathrm{NLL}(\bm{d}, \bm{d})}{\mathrm{NDF}}.

We are subtracting the so-called **saturated likelihood** :math:`\mathrm{NLL}(\bm{d}, \bm{d})`
(the minimum value our NLL could have if our model were to perfectly describe our data)
from the global cost function minimum :math:`\mathrm{NLL}(\bm{m}(\hat{\bm{p}}), \bm{d})`
and then divide this difference by :math:`\mathrm{NDF}`.
As before the expected value of :math:`\mathrm{GoF} / \mathrm{NDF}` is 1
if our model and our assumptions about the uncertainties of our data are correct.

Calculating Data Uncertainties from :math:`\chi^2 / \mathrm{NDF}`
*****************************************************************

Many fitting tools allow users to fit a model to data without specifying any data uncertainties.
This seems to be at odds with our current understanding of Gaussian likelihood-based fits where
we always required our data to have some amount of uncertainty.
So how does this work?
The "solution" is to first give all data points an uncorrelated uncertainty of 1 and to scale
these uncertainties *after* the fit in such a way that :math:`\chi^2 / \mathrm{NDF}` is equal to 1.
This approach has a big problem which makes it unsuitable for physics experiments:
*we cannot do any hypothesis tests* because we are
implicitly assuming that our model is 100% correct.
This goes against the very purpose of many physics experiments where experimenters are trying to
determine if a theoretical model is consistent with experimental data.

For example, at the Large Hadron Collider the standard model of particle physics has undergone very
thorough testing that continues to this day.
So far, no statistically significant deviations from the standard model have been found -
which is actually a bummer for theoretical physicists.
You see, we know for a fact that the standard model is incomplete because
(among other things) it does not include gravity.
If we were to find an area in which the predictions of the standard model are wrong
(beyond the expected uncertainties) this would give theorists an important clue
for a new theory that could potentially fix the problems of the standard model.

Fixing And Constraining Parameters
**********************************

*kafe2* allows users to **fix** fit parameters.
The practical consequence of this is that one of our fit parameters becomes a constant and
is *not* changed during the fit.
Because this effectively lowers the number of fit parameters we have to consider the number
of fixed parameters :math:`N_\mathrm{fixed}` in the calculation of the number of degrees of fredom:

.. math::
   \mathrm{NDF}
   = N_{\bm{d}} - (N_{\bm{p}} - N_\mathrm{fixed})
   = N_{\bm{d}} - N_{\bm{p}} + N_\mathrm{fixed}.

It's also possible to **constrain** fit parameters.
Constraints are effectively direct measurements of our fit parameters and they
increase the cost of our fit if they are not exactly met.
For example, the additional cost :math:`C_\mathrm{con}` of a Gaussian constraint for
fit parameter :math:`a` with mean :math:`\mu_a` and standard deviation :math:`\sigma_a`
can be calculated like this:

.. math::
   C_\mathrm{con} = \left( \frac{a - \mu_a}{\sigma_a} \right)^2.

We can of course generalize this concept to account for correlations between
parameters :math:`\bm{p}` as defined by a covariance matrix :math:`\bm{V}_{\bm{p}}`:

.. math::
   C_\mathrm{con}
   = (\bm{p} - \bm{\mu}_{\bm{p}})^\intercal \bm{V}_{\bm{p}}^{-1} (\bm{p} - \bm{\mu}_{\bm{p}}).

If we define any constraints we are adding more data to our fit.
We therefore also have to increase :math:`\mathrm{NDF}`
by the number of constraints :math:`N_\mathrm{con}`:

.. math::
   \mathrm{NDF}
   = N_{\bm{d}} + N_\mathrm{con} - N_{\bm{p}} + N_\mathrm{fixed}.

A simple parameter constraint that constrains a single parameter counts as one constraint.
On the other hand, a matrix parameter constraint that constrains :math:`n` parameters at once
counts as :math:`n` constraints.

Data/Fit Types
==============

A large percentage of fits can be expressed as an :py:obj:`~.XYFit`.
However, there are cases where an :py:obj:`~.XYFit` is not suitable;
*kafe2* offers alternatives **fit types** for those cases.
Typically these alternative fit types are associated with alternative **data (container) types**
so both concepts are explained simultaneously in this section.
For example, an :py:obj:`~.XYFit` uses an :py:obj:`~.XYContainer` to hold its *xy* data
while a :py:obj:`~.HistFit` uses a :py:obj:`~.HistContainer` to hold and bin its data.

For the following considerations :math:`\bm{p}` always describes the vector of fit parameters.
Unless mentioned otherwise fits calculate their cost from a data vector :math:`\bm{d}`
and a model vector :math:`\bm{m}`.

XYFit
*****

Let's start with the most common fit type: :py:obj:`~.XYFit`.
The data associated with this fit type consists of two vectors of equal length:
a vector of *x* data :math:`\bm{x}` and a vector of *y* data :math:`\bm{d}`.
Our model values are calculated as :math:`\bm{m}(\bm{x}; \bm{p}) = f(\bm{x}; \bm{p})`,
they are a function of our *x* data and our fit parameters.
As the difference in notation implies the *x* and *y* axes are *not* treated in the same way.
The *x* axis is interpreted as the **independent variable** of our fit while the *y* data values
:math:`\bm{d}` and *y* model values :math:`\bm{m}(\bm{x}; (\bm{p}))` are what we ultimately
compare to calculate the negative log-likelihood.

.. note::
   Although we only have a few discreet *x* values for which we have to calculate our model
   :math:`\bm{m}(\bm{x}; \bm{p})`, our model function :math:`f(x; \bm{p})`
   is still expected to be a continuous function of :math:`x`.

A visualization of :py:obj:`~.XYFit` is fairly straightforward:
the *xy** axes of our fix directly correspond to the axes of a plot.

IndexedFit
**********

Conceptually :py:obj:`~.IndexedFit` is a simplified version of :py:obj:`~.XYFit`:
we only have a data vector :math:`\bm{d}` and no independent variable at all.
Instead we calculate the model vector :math:`\bm{m}(\bm{p})`
as a function of just the fit parameters.
In *kafe2* :py:obj:`~.IndexedFit` is visualized by interpreting the indices of the data/model
vectors as *x* values and the corresponding *xth* entry of those vectors as the *y* value.

HistFit
*******

:py:obj:`~.HistFit` handles :math:`N` one-dimensional data points :math:`\bm{x}` by binning them
according to some bin edges :math:`x_0 < ... < x_k < ... < x_K` to form our data vector
:math:`\bm{d} \in \mathbb{R}^K`.
The model function :math:`f(x; \bm{p})` that is fitted to these bins is a
**probability density function** for the observed values :math:`\bm{x}`.
The bin heights :math:`\bm{m}(\bm{p})` predicted by our model are obtained by integrating
:math:`f(x; \bm{p})` over a given bin and multiplying the result with :math:`N`:

.. math::
   m_k(\bm{p}) = N \int_{x_{k-1}}^{x_k} f(t; \bm{p}) dt .

The amplitude of our distribution is therefore *not* one of the fit parameters;
we are effectively fitting a density function to a normalized histogram.

Unlike with :py:obj:`~.XYFit` or :py:obj:`~.IndexedFit` the default distribution assumed for the
data of a :py:obj:`~.HistFit` is the Poisson distribution rather than the normal distribution.

UnbinnedFit
***********

Just like :py:obj:`~.HistFit` an :py:obj:`~.UnbinnedFit` accepts a vector of :math:`N`
one-dimensional data points :math:`\bm{x}` in conjunction with a probability density function
:math:`f(x; \bm{p})` for these values as its model function.
As the name implies the data is not binned.
Instead, because our model function can be interpreted as a probability density we can simply
calculate the negative log-likelihood like this:

.. math::
   \mathrm{NLL}(\bm{p}) = - 2 \sum_{n=1}^N \log f(x_n; \bm{p}).

In *kafe2* :py:obj:`~.UnbinnedFit` is visualized by interpreting the independent variable as the
*x* axis of a plot and the height of the probability density function as the *y* axis.
Additionally, a thin, vertical line is added for each data point to indicate
the density of our data.

CustomFit
*********

Unlike the other fit types discussed so far, :py:obj:`~.CustomFit` does not explicitly use data
:math:`\bm{d}` or a model :math:`\bm{m}`.
Instead the user has to manually define how the cost function value is calculated from the fit
parameters :math:`\bm{p}`.
Because any potential data is outside *kafe2* there is no built-in visualization (plotting)
available except for the fit parameter profiles/contours calculated by :py:obj:`~.ContoursProfiler`.

MultiFit
********

A :py:obj:`~.MultiFit` is constructed from :math:`N` regular fits with cost functions
:math:`C_i(\bm{p})`.
The idea behind :py:obj:`~.MultiFit` is rather simple:
multiple models that share at least one parameter are
simultaneously fitted to their respective data.
In accordance with the method of maximum likelihood the optimal fit parameters are those that make
the observed combination of individual datasets the most likely.
The corresponding cost function can simply be calculated as:

.. math::
   C_\mathrm{multi}(\bm{p}) = \sum_i^N C_i(\bm{p}).

If a :py:obj:`~.MultiFit` is built from several fits that assume Gaussian uncertainties,
it's possible to specify uncertainties that are correlated between those fits.
For example, in the case of two fits that have a fully correlated source of uncertainty expressed
by a covariance matrix :math:`\bm{V}_\mathrm{shared}` the effective covariance matrix
:math:`\bm{V}_\mathrm{multi}` for the :py:obj:`MultiFit` becomes:

.. math::
   \bm{V}_\mathrm{multi} = \begin{pmatrix}
      \bm{V}_\mathrm{shared} & \bm{V}_\mathrm{shared} \\
      \bm{V}_\mathrm{shared} & \bm{V}_\mathrm{shared}
   \end{pmatrix} .

Cost Functions
==============

So far we almost universally assumed that the uncertainties of our data can be described with a
normal distribution.
However, this is not always the case.
For example, the number of radioactive decays in a given time interval
follows a Poisson distribution.
In *kafe2* such distinctions are handled via the **cost function**, the function that in one way or
another calculates a scalar cost from the data, model, and uncertainties of a fit.
This section describes the built-in cost functions that *kafe2** provides.

:math:`\chi^2` Cost Function
****************************

The by far most common cost function used is the :math:`\chi^2` cost function that assumes a normal
distribution for the uncertainties of our data.
In *kafe2* the name is strictly speaking a misnomer because the actual cost calculation considers
the full likelihood rather than just :math:`\chi^2` in order to handle non-constant uncertainties.
For :math:`N` data points :math:`d_i` with corresponding model values :math:`m_i(\bm{p})`
and uncorrelated (but possible non-constant) uncertainties :math:`\sigma_i(\bm{p})`
the cost function value is calculated like this:

.. math::
   \mathrm{NLL}(\bm{p})
   = C_\mathrm{det}(\bm{p}) + \chi^2(\bm{p})
   = \sum_i^N 2 \log \sigma_i(\bm{p}) + \left( \frac{d_i - m_i(\bm{p})}{\sigma_i(\bm{p})} \right)^2.

If the uncertainties are instead correlated as described by a covariance matrix
:math:`\bm{V}(\bm{p})` the cost function value becomes:

.. math::
   \mathrm{NLL}(\bm{p})
   = C_\mathrm{det}(\bm{p}) + \chi^2(\bm{p})
   = \log \det \bm{V}(\bm{p})
    + (\bm{d} - \bm{m}(\bm{p}))^T\: \bm{V}(\bm{p})^{-1}\: (\bm{d} - \bm{m}(\bm{p})).

Poisson Cost Function
*********************

The Poisson cost function assumes - as the name implies - a Poisson distribution for our data.
Compared to the normal distribution the Poisson distribution has two important features:
Firstly the data values :math:`d_i` (but not the model values :math:`m_i(\bm{p})`)
have to be positive integers, and secondly the mean and variance are inherently linked.
We can define the likelihood function :math:`\mathcal{L}(\bm{p})`
of the Poisson distribution like this:

.. math::
   \mathcal{L}(\bm{p}) = \prod_i^N \frac{m_i(\bm{p})^{d_i}\: e^{-m_i(\bm{p})}}{d_i !}.

The negative log-likelihood :math:`\mathrm{NLL}(\bm{p})` thus becomes:

.. math::
   \mathrm{NLL}(\bm{p})
   = - 2 \log \mathcal{L}
   = 2 \sum_i^N m_i(\bm{p}) - d_i \log m_i(\bm{p}) + \frac{d_i (d_i + 1)}{2}.

Notably :math:`\mathrm{NLL}(\bm{p})` depends only on the data :math:`d_i` and the model
:math:`m_i(\bm{p})` but *not* on any specified uncertainties :math:`\bm{\sigma}`.
The advantage is that we don't need to specify any uncertainties -
but the significant disadvantage is that we *can't* specify any uncertainties either.
In such cases the cost function in the following section will need to be used.

Gauss Approximation Cost Function
*********************************

Because a Poisson distribution cannot handle Gaussian data uncertainties the Poisson distribution
is frequently approximated with a normal distribution.
The easiest approach is to simply derive the uncertainties :math:`\sigma_i`
from the data :math:`d_i`:

.. math::
   \sigma_i = \sqrt{d_i}.

However, as described in the previous section about nonlinear regression,
this leads to a bias towards small model values :math:`m_i(\bm{p})`.
In *kafe2* the uncertainties are therefore derived from the model values:

.. math::
   \sigma_i = \sqrt{m_i(\bm{p})}.

Just like before these uncertainties can be easily combined with other sources of uncertainty
by simply adding up the (co)variances.
However, this approach has an important limitation:
it is only valid if the model values :math:`m_i(\bm{p})` are large enough
(something like :math:`m_i(\bm{p}) \ge 10`).
This is because for small model values the asymmetry of the Poisson distribution and the portion
of the normal distribution that resides in the unphysical region with :math:`m_i(\bm{p}) < 0`
are no longer negligible.

Numerical Considerations
========================

The mathematical description of :math:`\chi^2` shown so far makes use of the inverse of the
covariance matrix :math:`\bm{V}^{-1}`.
However, *kafe2* does *not* actually calculate :math:`\bm{V}^{-1}`.
Instead the `Cholesky decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_
:math:`\bm{L} \bm{L}^T = \bm{V}` of the covariance matrix is being used where :math:`\bm{L}` is a
lower triangular matrix.
Calculating :math:`\bm{L}` is much faster than calculating :math:`\bm{V}^{-1}` and it also reduces
the rounding error from floating point operations.

We can always calculate a Cholesky decomposition for a matrix that is symmetrical
and positive-definite.
Obviously a covariance matrix is symmetrical by definition.
And because all eigenvalues of a covariance matrix are (typically) positive a covariance matrix
is (typically) also positive definite.

.. note::
   The eigenvalues of a covariance matrix represent the equivalent variances in a coordinate system
   where said variances are uncorrelated (see
   `principal component analysis <https://en.wikipedia.org/wiki/Principal_component_analysis>`_).
   The eigenvalues are therefore all positive unless the uncertainties of two or more data points
   are fully correlated.
   In this case some of the eigenvalues are 0.
   However, as a consequence the covariance matrix would also no longer have full rank
   so we wouldn't be able to invert it either.

Because :math:`\bm{L}` is a triangular matrix
`solving <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html>`_
the corresponding system of linear equations for the residual vector
:math:`\bm{r} = \bm{d} - \bm{m}` (difference between data and model) can be done very quickly:

.. math::

  \bm{L} \bm{x} = \bm{r} .

With :math:`\bm{x} = \bm{L}^{-1} \bm{r}` we now find:

.. math::

  \chi^2
  = \bm{r}^T \bm{V}^{-1} \bm{r}
  = \bm{r}^T (\bm{L} \bm{L}^T)^{-1} \bm{r}
  = \bm{r}^T \bm{L}^{-T} \bm{L}^{-1} \bm{r}
  = \bm{x}^T \bm{x}.

Because :math:`\bm{L}` is a triangular matrix it can also be used to efficiently calculate
:math:`\log \det(\bm{V})`:

.. math::

  \det (\bm{L}) = \det (\bm{L}^T) = \prod_i^N L_{ii},

.. math::

  C_\mathrm{det}
  = \log \det (\bm{V})
  = \log \det (\bm{L} \bm{L}^T)
  = \log (\det \bm{L} \cdot \det \bm{L}^T)
  = \log (\prod_i^N L_{ii}^2)
  = 2 \sum_i^N \log L_{ii}.
