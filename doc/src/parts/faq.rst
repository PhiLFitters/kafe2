.. meta::
    :description lang=en: kafe2 - a Python-package for fitting parametric
                          models to several types of data with
    :robots: index, follow

.. _faq:

**********
FAQ
**********


General fitting
===============

**Why do I have to specify uncertainties for my datapoints in a kafe2 fit?**

**What happens when I do not specify errors in kafe2?**

**Why does kafe2 give warnings about missing uncertainties?**

In general, the goal of a fitting algorithm is to minimize the so-called cost function.
This can, for example, be a :math:`\chi^2`-function or a negative log-likelihood function.
All of these cost functions include the uncertainties of the datapoints.
A more detailed explanation of the fitting procedure and cost functions
can be found in the :ref:`Mathematical Foundations <mathematical_foundations>` section. Not specifying
an uncertainty would leave this parameter open, and the minimum of the cost function cannot be
determined.


**Why can I not just set the uncertainty to zero?**

**Why does kafe2 give an infinite cost function warning?**

If one or more datapoints have an uncertainty of zero, the fit will not work and will give you warnings.
E.g., in the :math:`\chi^2`-function in section :ref:`Mathematical Foundations <mathematical_foundations>`, the
uncertainties appear in the denominator. Setting any uncertainty to zero will result in a division by zero.
This will make the cost function go to infinity for every combination of parameters. Finding a minimum is impossible,
and the fit cannot converge.



**Why does kafe2 require measurement errors when scipy does not?**

**What can I do if I do not have any errors from my measurement?**

Some fitting algorithms, like those in SciPy, allow users to leave uncertainties unspecified.
In this case, SciPy assumes all errors to be equal to one and in the end rescales them such that
the :math:`\chi^2`/ndf value is close to one.
This destroys your metric to compare different models for an experiment (generally the goal of a fit in physics),
because all models will yield the same goodness-of-fit value. In principle, you could do the same manually using kafe2,
but you are encouraged not to. Every physical measurement yields uncertainties—both systematic and statistical ones—
for instance, because measuring devices are not perfect.


Fit does not converge
========================

**I use a correct model, but the best-fit values are always off from the values I would expect. What could cause this?**

**How do starting values influence kafe2 fits?**

**How can I improve convergence if the fit result seems wrong?**

One reason for this could be unspecified or poorly chosen starting values. The fitting algorithm minimizes the
cost function numerically, which requires a starting value for each parameter.
Ideally, this starting value is already close to the true value of a parameter. 
If the starting value is too far off, the minimizer searches in the wrong region and may only find a local minimum,
while we are, of course, searching for the global one.
The starting values can be defined by specifying default values in the model function, or by using the method :py:meth:`.Fit.set_all_parameter_values`.
If the wrapper functions are used, it is also possible to set the variable *p0*.


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

To control the initial step size, initial uncertainties should be
specified, either by setting the variable *dp0* in the wrapper
functions, or by setting the initial uncertainties using the attribute
:py:meth:`~.Fit.parameter_errors`.

It is also possible to limit the parameters to some expected range using
the :py:meth:`~.Fit.limit_parameter` method.
For instance, if you expect a parameter to have a value around
:math:`5 \cdot 10^{-10}`, it does not make sense for the minimizer to search between
0 and 1. Searching between :math:`1 \cdot 10^{-10}` and :math:`1 \cdot 10^{-9}` will improve the
chance of finding the correct minimum.


**The numerical values of my x and y measurements are separated by many orders of magnitude. Can this influence my fit result?**

**How can I handle parameters that differ by many orders of magnitude?**

Yes, large differences in the scale of the data can cause problems.
This can, e.g., occur when fitting Planck's constant $h$.
Here, the frequency is measured in Hertz (order of :math:`10^{15}`) and the electron energy is measured in Joules (order of :math:`10^{-18}`).
A step on the frequency axis requires very large changes to affect the output significantly, while tiny steps on the
energy axis can already lead to large changes in the output. This imbalance makes it hard for the minimizer to navigate
the parameter space due to numerical instability. 
An easy solution is to rescale the data such that $x$ and $y$ values are in a more similar regime. In the case of Planck's constant,
it can already be enough to specify the energy in eV, reducing the difference in orders of magnitude by 19.


Histogram Fits
==============

**When should I use a histogram fit?**

**Why does my fit take so long?**

**Why is an :py:object:`XYFit` slow for large datasets?**

When large numbers of datapoints are given, the usual :py:object:`XYFit` will get computationally very
expensive. A practical solution to this is to fill the data into a smaller number of bins.
This reduces the number of datapoints significantly and thus reduces the amount of computing power necessary. 
For the now-histogrammed data, a histogram fit can be used.


**What is the difference between a :py:object:`HistFit` and an :py:object:`XYFit`?**

**What is special about the cost function in a histogram fit?**

A :py:object:`HistFit` and an :py:object:`XYFit` mainly differ in how they handle the data and statistical uncertainties.
The data has to be passed to the :py:object:`HistFit` in a :py:object:`HistContainer`.
This container type can also directly histogram your raw data. The default cost function
of the histogram fit is a Poisson likelihood compared to a :math:`\chi^2`-function in the case of an :py:object:`XYFit`.
This is important for the correct handling of statistical uncertainties, especially when dealing with empty bins.
In histogram fits, the statistical uncertainty is directly calculated from the model function by default, which reduces biases.
In contrast, the :py:object:`XYFit` assumes Gaussian uncertainties by default and handles uncertainties point by point.


**Do I have to specify errors for my histogram fit?**

**Is it possible to combine Poisson errors and Gaussian errors?**

**How does kafe2 handle uncertainties in a histogram fit?**

The statistical uncertainties in a histogram fit are inferred from the model function, so statistical errors
don't have to be specified in the container. If systematic uncertainties are added, this can be done
via the usual :py:meth:`add_error` method. Since these uncertainties are assumed to be Gaussian-distributed, the
cost function of the fit now has to be the "Gaussian approximation". This makes it possible to
(approximately) combine Poisson and Gaussian uncertainties.


**What do I have to watch out for when I use a large number of datapoints?**

**Why do systematics become important with large datasets?**

**How does the dataset size influence statistical and systematic uncertainties?**

To reduce computing time, it can be useful to histogram the data and then use a histogram fit. This reduces
the number of effective datapoints in your fit.
Furthermore, the larger the amount of data, the smaller the statistical uncertainties are (relatively).
So when handling large amounts of data, you may come into a regime where systematic errors—even though 
they are small—can play a significant role and influence the outcome of your fit. It can be useful to estimate the
statistical uncertainties for a few bins and compare them to possible systematics.


Plotting
========

**How can I customize the colors of my plot?**

**Can I change the color of my fit?**

**Is it possible to change the color or shape of my datapoints in a plot?**

You can fully customize the appearance of your plots.
This is described in detail in the plotting section of the :ref:`User Guide <_user_guide>`. In general, the :py:meth:`~.Plot.customize`
method can be used to customize the plot style of the datapoints, the data errorbars, the model line,
and the model error band. Besides the color, the shape of markers or the label of an object can also be changed.


**Can I customize the axes of my plot?**

**Can I change an axis label of my plot?**

**How to use a logarithmic scale in kafe2?**

The axis labels of a plot can be manually set using the :py:meth:`~.Plot.x_label` and :py:meth:`~.Plot.y_label` methods.
Furthermore, it is also possible to rescale an axis (e.g., as a logarithmic scale) and to change the plot range using the :py:meth:`~.Plot.x_scale`
and :py:meth:`~.Plot.x_range` methods. Examples can be found in the plotting section of the :ref:`User Guide <_user_guide>`.


Interpretation
==============

**What exactly does the :math:`\chi^2`-probability in the fit result tell me?**

**What is a p-value?**

The :math:`\chi`\ ² value is essentially related to a p-value. It tells you how likely it is—considering the model you used for the fit is true—to get this :math:`\chi^2`-value or a larger one. In other words, it asks: If my model is correct, how likely is it to get data this incompatible with my model, or worse? If the value is very close to 1, you might have overestimated
your uncertainties. If the value is 0, your model could be wrong. This metric can be used to compare how well different 
models describe the observed data. For more information, view the hypothesis-testing section of the
"Mathematical Foundation" in the "User's Guide".