.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow


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
plotting-procedure is possible. For using `kafe2` inside a `python` script, import the required
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
