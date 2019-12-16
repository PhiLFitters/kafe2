.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow

.. role:: python(code)
   :language: python


**********
User Guide
**********


This user guide covers the basics of using *kafe2* for
fitting parametric models to data.
Specifically, it teaches users how to specify measurement 
data and uncertainties, how to specify model functions, 
and how to extract the fit results.

Basic Fitting Procedure
=======================

Generally, any simple fit performed with the *kafe2* framework 
can be divided into the following steps:

1. Specifying the data
2. Specifying the uncertainties
3. Specifying the model function
4. Performing the fit
5. Extracting the fit results

This document will gradually introduce the above steps via example code.

Using kafe2go
-------------
Using `kafe2go` is the simplest way of performing a fit. Here all the necessary
information like data and uncertainties is specified in the YAML-Data format.
To perform the fit, simply run

.. code-block:: bash

    kafe2go path/to/fit.yml

Using Python
------------
When using `kafe2` via a `Python` script a fine control of the fitting- and
plotting-procedure is possible. For using `kafe2` inside a `Python` script, import the required
`kafe2` modules:

.. code-block:: python

    from kafe2 import XYFit, Plot


Example 1: Line Fit
===================
The first example is the simplest use of a fitting framework, performing a line fit.
A linear function of the form :math:`f(x;a, b) = a x + b` is made to align with
a series of xy data points that have some uncertainty along the x-axis
and the y-axis.
This example demonstrates how to perform such a line fit in kafe2 and
how to extract the results.

.. figure:: ../_static/img/001_line_fit.png
    :alt: Plot of a line fit performed with kafe2.

kafe2go
-------
To run this example, open a text editor and save the following file contents
as a YAML-file named ``line_fit.yml``.

.. literalinclude:: ../../../examples/001_line_fit/line_fit.yml

Then open a terminal, navigate to the directory where the file is
located and run

.. code-block:: bash

    kafe2go line_fit.yml

Python
------
The same fit can also be performed by using a `Python` script.

.. bootstrap_collapsible::
    :control_type: link
    :control_text: python code

    .. literalinclude:: ../../../examples/001_line_fit/line_fit.py
        :lines: 15-


Example 2: Model Functions
==========================

In experimental physics a line fit will only suffice for a small number
of applications. In most cases you will need a more complex model function
with more parameters to accurately model physical reality.
This example demonstrates how to specify arbitrary model functions for
a kafe2 fit.
When a different function has to be fitted, those functions need to be defined either in the
``yml``-file or the `Python` script.

.. figure:: ../_static/img/002_model_comparison.png
    :alt: Comparison of a line and exponential fit with kafe2.

The expectation value of :math:`\chi^2/\mathrm{ndf}` is 1. Thus the exponential function
more accurately describes the dataset than a line function.

An exponential function is a **non-linear** function! Therefore the fit parameters can be heavily
correlated. To verify this, please create a contour plot of the fitted parameters.
This can be done by appending the ``-c`` or ``--contours`` option to ``kafe2go``. Additionally a
grid can be added to the contour plots with the ``--grid all`` flag.
To achieve the same with a `Python` script, import the ``ContoursProfiler`` with
:python:`from kafe2 import ContoursProfiler` and create a contour plot:

.. literalinclude:: ../../../examples/002_model_functions/model_functions.py
    :lines: 55-57

By creating the contours in a `Python` script the user has a finer control on which parameters to
use as well on the appearance of the contour plot via various keyword arguments.

The according contour plot to the exponential fit shown above looks like this:

.. figure:: ../static/img/002_exponential_contours.png
    :alt: Contour plot corresponding to the exponential fit.

When looking at the :math:`1\sigma` contour it's slightly visible, that the contour is not
perfectly elliptical. But in this case the deformation is very small and is negligible.
If a contour is not elliptical it is good practice to show the contour plot in addition to the
uncertainties from the fit. This illustrates the correlation of the parameters.

kafe2go
-------
Inside a ``yml``-File custom fit functions are defined with the ``model_function`` keyword.
The custom function must be a custom python function. Numpy functions are supported as shown in the example.
For more advanced fit functions, consider using ``kafe2`` inside a `Python` script.

.. bootstrap_collapsible::
    :control_type: link
    :control_text: exponential_fit.yml

    .. literalinclude:: ../../../examples/002_model_functions/exponential_fit.yml
        :emphasize-lines: 39-43

.. bootstrap_collapsible::
    :control_type: link
    :control_text: line_fit.yml

    .. literalinclude:: ../../../examples/002_model_functions/line_fit.yml

To use multiple input files with kafe2go, simply run

.. code-block:: bash

    kafe2go path/to/fit1.yml path/to/fit2.yml

To plot the fits in two separate figures append the ``--separate`` flag to the kafe2go command.

.. code-block:: bash

    kafe2go path/to/fit1.yml path/to/fit2.yml --separate


Python
------
Inside a `Python` script a custom function ist defined like this:

.. literalinclude:: ../../../examples/002_model_functions/model_functions.py
    :lines: 18-27

Those functions are passed on to the Fit objects:

.. literalinclude:: ../../../examples/002_model_functions/model_functions.py
    :lines: 34-36

It' also possible to assign LaTeX expressions to the function and its variables.
The according lines are highlighted in the full code example below.

.. bootstrap_collapsible::
    :control_type: link
    :control_text: python code

    .. literalinclude:: ../../../examples/002_model_functions/model_functions.py
        :lines: 13-
        :emphasize-lines: 26-30

Please note, that the function LaTeX expression needs to contain all parameters present in the
function definition. The placeholders are then automatically replaced by their corresponding LaTeX
names. Due to the way `Python` handles the string formatting, curly braces used in LaTeX need to
be doubled, as in the code example above.
