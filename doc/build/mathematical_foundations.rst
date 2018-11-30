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
statistical uncertainty is called **accuracy**.

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

For a given experiment the data :math:`{\bf d}` and the parametric
form of the model :math:`{\bf m}` describing the data are constant.
This means that the cost function :math:`C` then only depends on the
model parameters, denoted here as a vector :math:`{\bf p}` in parameter
space.

    .. math::

        C = C\left({\bf d}, {\bf m}({\bf p})\right) =  C({\bf p})

Therefore parameter estimation essentially boils down to finding the
vector of model parameters :math:`{\hat{\bf p}}` for which the cost
function :math:`C({\bf p})` is at its global minimum.
In general this is a multidimensional optimization problem which is
typically solved numerically. There are several algorithms and tools
that can be used for this task:
One example is the *SciPy* package for Python which provides function 
minimization functionality through its ``scipy.optimize`` module.
Another option is *iminuit* (a C++ reimplementation of the Fortran
minimizer *MINUIT*) which can be accessed via the corresponding
Python package.
 
.. TODO: add link to future page with minimizer overview

Choosing a cost function
------------------------

However, apart from cost function optimization, the cost function
first has to be defined in a meaningful way.
There are many ways in which this can be done, but

This is especially important


Least-squares ("chi-squared")
-----------------------------

Negative log-likelihood
-----------------------

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



