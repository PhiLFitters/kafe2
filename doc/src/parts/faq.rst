.. meta::
    :description lang=en: kafe2 a Python-package for fitting parametric
                          models to several types of data
    :robots: index, follow

.. _faq:

**********
FAQ
**********


General fitting
===============

.. _faq_no_uncertainties:

**Why do I have to specify errors/uncertainties for my data points in a kafe2 fit?**

**What happens when I do not specify errors/uncertainties in kafe2?**

**Why does kafe2 give warnings about missing errors/uncertainties?**

In general, the goal of a fitting algorithm is to minimize the so-called cost function.
This can, for example, be a :math:`\chi^2` function. This cost function is based on the normal distribution,
containing a mean and an uncertainty, which is why uncertainties are necessary here. Data points with small
uncertainties have a larger influence on the fit result than data points with large uncertainties.
A more detailed explanation of the fitting procedure and cost functions
can be found in the :ref:`Mathematical Foundations <mathematical_foundations>` section.
If no uncertainties are specified, *kafe2* falls back to a plain least-squares fit, which still yields
best-fit parameter values. However, neither the parameter uncertainties nor the goodness of fit can be
calculated meaningfully in this case, which is why *kafe2* warns you.

**Why can I not just set the errors/uncertainties to zero?**

**Why does kafe2 give an infinite cost function warning?**

If one or more data points have an uncertainty of zero, the fit will not work and *kafe2* will give you warnings.
E.g., in the :math:`\chi^2` function in the :ref:`Mathematical Foundations <mathematical_foundations>` section,
the uncertainties appear in the denominator, so an uncertainty of zero results in a division by zero.
The cost function then becomes infinite for every combination of parameters. Finding a minimum is impossible:
the parameters stay at their start values, and their uncertainties cannot be calculated.
Instead, specify a realistic, non-zero uncertainty for every data point, e.g. from the resolution of your measuring device.

This problem is indicated by the following two warnings:

.. warning::
    UserWarning: Setting all data errors to 1 as a fallback. warnings.warn("Setting all data errors to 1 as a fallback.")

.. warning::
    Warning: the cost function has been evaluated as infinite. The fit might not converge correctly.



**Why does kafe2 insist on measurement uncertainties when SciPy does not?**

**What can I do if I do not have any uncertainties from my measurement?**

Some fitting tools, like ``scipy.optimize.curve_fit``, allow users to leave the uncertainties unspecified.
In this case, SciPy sets all uncertainties to one and, after the fit, scales them by a common factor
such that :math:`\chi^2/\mathrm{ndf}` equals one. The parameter uncertainties are then derived from the
scatter of the data points around the model instead of from the measurement uncertainties.
Note that by default (``absolute_sigma=False``) SciPy does this even if you specify uncertainties.
Because :math:`\chi^2/\mathrm{ndf}` is one for every model by construction, this destroys the
absolute goodness of fit: you can no longer tell whether your model describes the data well,
or whether your uncertainties are reasonable (generally an important question for a fit in physics).
*kafe2* always treats the uncertainties as absolute, so the :math:`\chi^2/\mathrm{ndf}` value and the
:math:`\chi^2` probability of the fit are meaningful.
*kafe2* can also fit data without uncertainties (see above), but you are strongly encouraged not to do so:
every physical measurement has uncertainties—both statistical and systematic ones—for instance,
because measuring devices are not perfect.
If you did not record any uncertainties, estimate them, e.g. from the resolution or the specified accuracy
of your measuring device, or from the scatter of repeated measurements.

Fit does not converge
=====================

.. _faq_wrong_fit_result:

**I use a correct model, but the best-fit values are always off from the values I would expect. What could cause this?**

**How do starting values influence kafe2 fits?**

**How can I improve convergence if the fit result seems wrong?**

One reason for this could be unspecified or poorly chosen starting values. The fitting algorithm minimizes the
cost function numerically, which requires a starting value for each parameter.
Ideally, this starting value is already close to the true value of a parameter. 
If the starting value is too far off, the minimizer searches in the wrong region and may only find a local minimum,
while we are, of course, searching for the global one.
The starting values can be defined by specifying default values in the model function, or by using the method :py:meth:`~.FitBase.set_all_parameter_values`.
If the wrapper functions are used, it is also possible to set the variable *p0*.

If the starting values are completely undefined, kafe2 also gives the User a warning in the follwoing form:

.. warning::
    Intial values and step sizes for one or more fit parameters are not well-defined: ['a', 'b'].
    For better convergence set initial values != 0, set initial step sizes via parameter_errors, or limit, fix, or constrain the parameters.

.. _faq_small_parameters:

**The parameter I want to estimate from my fit is small, and the fit doesn't find it. How can I fix this?**

**How can I make the minimizer detect very small parameters?**

**How to adjust the parameter range in a kafe2 fit?**

Another reason for fits failing to converge is the initial step size
used by the numerical minimizer. This is determined by the initial setting
of the parameter uncertainties; if no such values are given, they are
initialized from reasonable guesses based on the initial parameter
settings, typically by taking a fraction of the start values.

Problems arise if the start value is zero, in which case no useful
range for a parameter can be guessed. If the global minimum is very
close to a non-zero starting value, it may be ignored because the
initial step size chosen is too large.

As an example, consider the measurement of a length of 1mm = 0.001m. If
the unit meter is chosen and no start values are specified, the fit will
assume a start value of 1m and a correspondingly very large initial step
size. Therefore, the minimum at 0.001 will possibly be skipped.

In general terms, it is recommended to choose physical units and
parameterizations such that the parameter values are of order one. In
this case, the heuristics implemented in *kafe2* and *iminuit* mostly
work well.

To control the initial step size, initial parameter uncertainties should be
specified, either by setting the variable *dp0* in the wrapper
functions, or by setting the initial uncertainties using the property
:py:attr:`~.FitBase.parameter_errors`.

It is also possible to limit the parameters to some expected range using
the :py:meth:`~.FitBase.limit_parameter` method.
For instance, if you expect a parameter to have a value around
:math:`5 \cdot 10^{-10}`, it does not make sense for the minimizer to search between
0 and 1. Searching between :math:`1 \cdot 10^{-10}` and :math:`1 \cdot 10^{-9}` will improve the
chance of finding the correct minimum.

.. _faq_different_magnitudes:

**The numerical values of my x and y measurements are separated by many orders of magnitude. Can this influence my fit result?**

**How can I handle data whose values that differ by many orders of magnitude?**

Yes, large differences in the scale of the data can cause problems.
This can, e.g., occur when fitting Planck's constant :math:`h`.
Here, the frequency is measured in Hertz (order of :math:`10^{15}`) and the electron energy is measured in Joules (order of :math:`10^{-18}`).
A step on the frequency axis requires very large changes to affect the output significantly, while tiny steps on the
energy axis can already lead to large changes in the output. This imbalance makes it hard for the minimizer to navigate
the parameter space due to numerical instability. 
An easy solution is to rescale the data such that the :math:`x` and :math:`y` values are in a more similar regime. In the case of Planck's constant,
it can already be enough to specify the energy in eV, reducing the difference in orders of magnitude by 19.

.. _faq_histogram_fits:

Histogram Fits
==============

**When should I use a histogram fit?**

**XYFit is too slow for a very large dataset. How can I speed it up?**

When large numbers of data points are given, the usual :py:obj:`~.XYFit` gets computationally very
expensive: *kafe2* works with the full covariance matrix of all data points, so the computing time
grows much faster than the number of data points.
A practical solution to this is to fill the data into a smaller number of bins.
This reduces the number of data points significantly and thus reduces the amount of computing power necessary.
For the now-histogrammed data, a :py:obj:`~.HistFit` can be used.
Note that binning discards some information: if your data are individual measurements of a single quantity
(e.g. decay times), an :py:obj:`~.UnbinnedFit` uses the full information and does not require choosing bins.


**What is the difference between a HistFit and an XYFit?**

**What is special about the cost function in a histogram fit?**

A :py:obj:`~.HistFit` and an :py:obj:`~.XYFit` mainly differ in how they handle the data and statistical uncertainties.
The data has to be passed to the :py:obj:`HistFit` in a :py:obj:`~.HistContainer`.
This container type can also directly histogram your raw data. The default cost function
of the histogram fit is a Poisson likelihood compared to a :math:`\chi^2`-function in the case of an :py:obj:`~.XYFit`.
This is important for the correct handling of statistical uncertainties, especially when dealing with empty bins.
In histogram fits, the statistical uncertainty is directly calculated from the model function by default, which reduces biases.
In contrast, the :py:obj:`~.XYFit` assumes Gaussian uncertainties by default and handles uncertainties point by point.

This of course comes with a tradeoff. Binned data loses some spatial resolution, depending on the size of the bins. So in practice
it has to be decided depending on teh case how much resolution can be given up for better handleable data.


**Do I have to specify errors/uncertainties for my histogram fit?**

**Is it possible to combine Poisson uncertaintes and Gaussian uncertaintes?**

**How does kafe2 handle uncertainties in a histogram fit?**

The statistical uncertainties in a histogram fit are inferred from the model function, so statistical uncertaintes
don't have to be specified in the container. Systematic uncertainties can be added via the usual :py:meth:`~.FitBase.add_error` method.
Since these uncertainties are assumed to be gaussian-distributed, the
cost function of the fit then has to be changed to the gaussian approximation of the poisson distribution by passing
``cost_function="gauss_approximation"`` when creating the :py:obj:`~.HistFit`. The :py:func:`~.hist_fit` wrapper function
does this automatically if uncertaintes are given. This makes it possible to
(approximately) combine Poisson and Gaussian uncertainties.

In principle, both types of cost functions can be used for both types of Fits, the :py:obj:`~.XYFit` and the :py:obj:`~.HistFit`. Just the defaults differ.


**What do I have to watch out for when I use a large number of data points?**

**Why do systematics become important with large datasets?**

**How does the dataset size influence statistical and systematic uncertainties?**

The larger the amount of data, the smaller the statistical uncertainties are (relatively).
So when handling large amounts of data, you may come into a regime where systematic uncertaintes—even though 
they are small—can play a significant role and influence the outcome of your fit. It can be useful to estimate the
statistical uncertainties for a few bins and compare them to possible systematics.

.. _faq_plotting:

Plotting
========

**How can I customize the colors of my plot?**

**Can I change the color of my fit?**

**Is it possible to change the color or shape of my data points in a plot?**

You can fully customize the appearance of your plots.
This is described in detail in the :ref:`plotting section <plotting>` of the User Guide. In general, the :py:meth:`~.Plot.customize`
method can be used to customize the plot style of the data points, the data error bars, the model line,
and the model error band. Besides the color, the shape of markers or the label of an object can also be changed.


**Can I customize the axes of my plot?**

**Can I change an axis label of my plot?**

**How to use a logarithmic scale in kafe2?**

The axis labels of a plot can be manually set using the :py:attr:`~.Plot.x_label` and :py:attr:`~.Plot.y_label` properties.
Furthermore, it is also possible to rescale an axis (e.g., as a logarithmic scale) and to change the plot range using the :py:attr:`~.Plot.x_scale`,
:py:attr:`~.Plot.y_scale`,  :py:attr:`~.Plot.x_range` adn :py:attr:`~.Plot.y_range` properties. Examples can be found in the :ref:`plotting section <plotting>` of the User Guide.

.. _faq_interpretation:

Interpretation
==============

**What exactly does the** :math:`\chi^2` **-probability in the fit result tell me?**

**What is a p-value?**

The :math:`\chi^2` probability is a p-value. It tells you how likely it is—assuming the model you used for the fit is true—to get
this :math:`\chi^2` value or a larger one. In other words, it asks: if my model is correct,
how likely is it to get data this incompatible with my model, or worse? If the value is very close to 1, you might have overestimated
your uncertainties. If the value is close to 0, your model could be wrong, or you might have underestimated your uncertainties.
It is therefore a measure of the absolute goodness of fit of a single model. For more information, see the
:ref:`hypothesis testing section <hypothesis_testing>` in the :ref:`Mathematical Foundations <mathematical_foundations>`.