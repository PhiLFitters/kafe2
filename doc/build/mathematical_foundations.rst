.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow


Theoretical Foundation
######################


This section briefly covers the underlying theoretical concepts 
upon which parameter estimation -and by extension *kafe2*- is based.

Basics notions
==============


Measurements and theoretical models
-----------------------------------

When measuring observable quantities, the results typically
consist of a series of numeric measurement values: the **data**.
Since no measurement is perfect, the data will be subject to 
**measurement uncertainties**. These uncertainties, also known
as errors, are unavoidable and can be divided into two subtypes:

**Statistical uncertainties**, also called random uncertainties,
are caused because of random fluctuations in the measurement values, 
either because of technical limitations of the experimental setup
(electronic noise, mechanical vibrations) or because of the intrinsic 
statistical nature of the measured observable (radioactive decay).

When a measurement is performed more than once the statistical
uncertainties of any two measurements will not be correlated.
This means that the random distortion of the measurement data will
be different for each measurement. If the measurement is repeated
often enough the statistical uncertainties of the single measurements
will cancel each other out and bring down the total statistical
uncertainty of the end result. The measure of statistical uncertainty
is called **precision**.

**Systematic uncertainties** arise due to non-random effects, that
distort all measurements in the same way regardless of how many
measurements are taken. Such uncertainties can for example be caused
by imperfect measurement devices whose displayed values are always
off by a small margin.

When a measurement is performed more than once the systematic
uncertainties of any two measurements will be correlated. This
means that the distortion of the measurement data will be the same
for each measurement. As such, repeating a measurement will **not**
reduce the statistical uncertainty of the end result. The measure of
systematic uncertainty is called **accuracy**.

In order to make meaningful quantitative statements based on measurement
data it is essential to estimate the magnitude and correlation of the
above mentioned uncertainties.


Parameter estimation
--------------------

In **parameter estimation**, the main goal is to obtain
best estimates for the parameters of a theoretical model
by comparing the model predictions to experimental measurements.

This is typically done systematically by defining a quantity which 
expresses how well the model predictions fit the available data.
This quantity is usually called the *loss function*, or the 
**cost function** (the term used in *kafe2* ).
The value of a cost function depends mainly on the measurement data,
the theory predictions, and the uncertainties of the measurements and
the model. Cost functions are designed in such a way that the agreement 
between the measurements :math:`d_i` and the corresponding predictions
:math:`m_i` provided by the model is **best** when the cost function
reaches its **global minimum**.

For a given experiment the data :math:`\textbf{d}` and the parametric
form of the model :math:`\textbf{m}` describing the data are constant.
This means that the cost function :math:`C` then only depends on the
model parameters, denoted here as a vector :math:`\textbf{p}` in parameter
space.

    .. math::

        C = C\left(\textbf{d}, \textbf{m}(\textbf{p})\right) =  C(\textbf{p})

Therefore parameter estimation essentially boils down to finding the
vector of model parameters :math:`\hat{\textbf{p}}` for which the cost
function :math:`C(\textbf{p})` is at its global minimum.
In general this is a multidimensional optimization problem which is
typically solved numerically. There are several algorithms and tools
that can be used for this task:
In *kafe2* the Python package *SciPy* can be used to minimize cost
functions via its ``scipy.optimize`` module.
Alternatively,  *kafe2* can use the *MINUIT* implementations *TMinuit*
or *iminuit* when they're installed.
 
.. TODO: add link to future page with minimizer overview

Choosing a cost function
------------------------

There are many cost functions used in statistics but for estimating
parameters in physics experiments two are of particular importance:
The method of **least squares** and the method of **maximum likelihood**.
Incidentally, these are also the two basic types of cost functions that
have been implemented in *kafe2*.

Least-squares ("chi-squared")
-----------------------------

The method of **least squares** is the standard approach in parameter estimation.
To calculate the value of this cost function the differences between model and data are squared and then added up (hence
the name '*least squares*').
More formally, using the data **covariance matrix** :math:`V`, the cost function can be calculated as follows:

    .. math::

        \chi^2(\textbf{p}) = (\textbf{d} - \textbf{m}(\textbf{p}))^T \ V^{-1} \ (\textbf{d} - \textbf{m}(\textbf{p})).

Since the value of this cost function is usually written as :math:`\chi^2` the method of least squares is also known as
**chi-squared**.
If the uncertainties on the data vector :math:`\textbf{d}` are uncorrelated all elements of :math:`V` except for its
diagonal elements :math:`\sigma_i^2` (the squares of the :math:`i`-th point-wise measurement uncertainties) are equal to
zero.
The above statement can then be simplified to

    .. math::

        \chi^2 = \sum_i \left( \frac{d_i - m_i(\textbf{p})}{\sigma_i} \right)^2.

As we can see it is no longer necessary to invert the covariance matrix :math:`V` to calculate :math:`\chi^2`.
All that's needed is to divide the difference between data :math:`d_i` and model :math:`m_i` by the corresponding
uncertainty :math:`\sigma_i`.
Computation-wise there is no noticeable difference for small datasets (:math:`O(100)` data points).
For very large datasets however, the inversion of the covariance matrix takes significantly longer than simply dividing
by the point-wise uncertainties.


Negative log-likelihood
-----------------------

The method of **maximum likelihood** attempts to find the best estimation for the model parameters :math:`\textbf{p}`
by maximizing the probability with which such model parameters (under the given uncertainties) would result in the
observed data :math:`\textbf{d}`.
More formally, the method of maximum likelihood searches for those values of :math:`\textbf{p}` for which the so-called
**likelihood function** (**likelihood** for short) reaches its global maximum.
Using the probability of making a certain measurement given some values of model parameters :math:`P(\textbf{p})` the
likelihood function can be defined as follows:

    .. math::

        L(\textbf{p}) = \prod_i P_i(\textbf{p}).

However, usually parameter estimation is not performed by using the likelihood, but by using its logarithm, the
so-called **log-likelihood**:

    .. math::

        \log L(\textbf{p}) = \log \left( \prod_i P_i(\textbf{p}) \right) = \sum_i \log P_i(\textbf{p}).

This transformation is allowed because logarithms are **strictly monotonically increasing functions**.
Because of this property the logarithm of any function will have its global maximum as the same spot as the original
function.
The parameter values :math:`\textbf{p}` that maximize the log-likelihood will therefore also maximize the likelihood.

While the above transformation may seem nonsensical at first, there are two important advantages to calculating the
log-likelihood over the likelihood:

Firstly, it replaces the **product** of the probabilities :math:`\prod_i P_i` with a **sum** over the logarithms of the
probabilities :math:`\sum_i \log P_i`.
Because sums can be calculated much more quickly than products this significantly reduces the amount of computation time
needed for large datasets.

Secondly, because the probabilities :math:`P_i` are oftentimes proportional to some exponential function, calculating
their logarithm is actually **faster** because it reduces the number of necessary operations.

At last, since cost functions are always **minimized** the log-likelihood is multiplied with :math:`-1` to receive the
**neg log-likelihood**.

As an example, let us look at the neg log-likelihood of data with uncertainties that assume a normal distribution:

    .. math::

        -\log P(\textbf{p})
        = - \log \prod_i \frac{1}{\sqrt[]{2 \pi} \: \sigma_i} \exp\left(
        \frac{1}{2} \left( \frac{d_i - m_i(\textbf{p})}{\sigma_i} \right)^2\right)
        = - \sum_i \log \frac{1}{\sqrt[]{2 \pi} \: \sigma_i} + \sum_i \frac{1}{2}
        \left( \frac{d_i - m_i(\textbf{p})}{\sigma_i} \right)^2
        = - \log L_\mathrm{max} + \frac{1}{2} \chi^2

As we can see the logarithm cancels out the exponential function of the normal distribution and we are left with two
parts:

The first is a constant part that is represented by :math:`-\log L_\mathrm{max}`.
This is the minimum value the neg log-likelihood could possibly take on if the model :math:`\textbf{m}` were to exactly
fir the data :math:`\textbf{d}`.

The second part can be summed up as :math:`\frac{1}{2} \chi^2`.
As it turns the method of least squares is a special case of the method of maximum likelihood where all data points have
normally distributed uncertainties.

Types of datasets
=================


Handling uncertainties
======================

Gaussian uncertainties
----------------------

Correlations
------------

Other types of uncertainties
----------------------------


Cost functions
==============

Least-squares ("chi-squared") estimator
---------------------------------------

:math:`\chi^2`

Negative log-likelihood estimator
---------------------------------



