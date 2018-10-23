.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow


User Guide
##########


This user guide covers the basics of using *kafe2* for
fitting parametric models to data, including:

  * the basic math behind parameter estimation,
  * the several types of data that *kafe2* can work with,
  * various ways of defining uncertainties on the data
  * various ways of defining a metric for data-model agreement
    (*cost functions*)
  * some examples and applications of the above


Basics notions
==============


Measurements and theoretical models
-----------------------------------

When measuring observable quantities, the results typically
consist of a series of numeric measurement values: the **data**.
Of course, no measurement is perfect, and the data is subject
to measurement **uncertainties**, which can have a number of
reasons.

A common type of uncertainty arises because of random
fluctuations of the measurement values, either because of
technical limitations of the experimental setup
(e.g. electronic noise, mechanical vibrations) or because
of the intrinsic statistical nature of the measured observable.

Being of a random nature, such uncertainties do not correlate
between different measurements and can be reduced by repeating
the measurements a large number of times.

However, distortions of the measurement can also arise due to
non-random effects, such as the calibration of the
tools used to perform the measurement. These
typically affect the measurement in a systematic way,
causing the measured observable value to deviate from the
"true value".
In this case, repeating the measurement a large number of times
does not make the measurement more accurate.

It is an important part of the experimental process to estimate
the magnitude and correlation of these effects, in order to
make meaningful quantitative statements on the basis of the
performed measurements.




Parameter estimation
--------------------

In **parameter estimation**, the main goal is to obtain
best estimates for the parameters of a theoretical model
by comparing the model predictions to experimental measurements.

This is typically done in a systematic way by defining
a quantity which expresses how well the model predictions fit the
available data. This quantity depends mainly on the measurement data,
the theory predictions, and the measurement or model uncertainties.
It is known by a number of names: the *objective function*,
the *loss function*, or the **cost function** (the term used in *kafe*),
and is defined in such a way that the agreement between the
measurements :math:`d_i` and the corresponding predictions :math:`m_i`
delivered by the model, is **best** when the cost function reaches
its **global minimum**.

Since the data :math:`{\bf d}` and the parametric form of the model
:math:`{\bf m}` describing the data are known, the cost function
:math:`C` actually only depends on the model parameters, denoted here
as a vector :math:`{\bf p}` in parameter space.

    .. math::

        C = C\left({\bf d}, {\bf m}({\bf p})\right) =  C({\bf p})

So, for the most part, parameter estimation boils down to finding the
global minimimum of the cost function, with respect to the model parameters
:math:`p_i`.
In general, this is a multidimensional optimization problem which is
typically handled numerically. For this, different algorithms and tools
exist.
One example is *iminuit* (a C++ reimplementation of the
Fortran minimizer *MINUIT*), for which a Python package is also available.
The *SciPy* package for Python also provides function minimization
functionality through its ``scipy.optimize`` module.
 
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



