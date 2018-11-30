.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow


User Guide
##########


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
.. TODO: link
The full code snippets can be found under /examples. 