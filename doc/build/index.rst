.. meta::
   :description lang=en: kafe2 - a general, Python-based approach to fit a
      model function to two-dimensional data points with correlated
      uncertainties in both dimensions
   :robots: index, follow


.. toctree::
   :hidden:
   :caption: Table of Contents
   :name: mastertoc
   :maxdepth: 2

   self
   installation
   mathematical_foundations
   user_guide
   module_doc

Welcome to *kafe2*, the Karlsruhe Fit Environment
================================================

   *kafe2* is a data fitting framework designed for use in undergraduate
   physics lab courses. It provides a basic Python toolkit for fitting
   models to data as well as visualizing the data and the model function.
   It relies on Python packages such as :py:mod:`numpy` and :py:mod:`matplotlib`,
   and can use the Python interface to the minimizer `Minuit` contained in the data
   analysis framework `ROOT` or in the Python package `iminuit`.

.. .. include::  quickstart.rst
