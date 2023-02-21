.. meta::
   :description lang=en: kafe2 - a general, Python-based approach to fit a
      model function to two-dimensional data points with correlated
      uncertainties in both dimensions
   :robots: index, follow


#######################
**kafe2** documentation
#######################

.. image:: _static/img/badge_kafe2.svg.png
   :width: 128px
   :height: 128px
   :alt: kafe2 logo
   :align: left


Welcome to **kafe2**, the *Karlsruhe Fit Environment 2*!

*kafe2* is a data fitting framework originally designed for use in undergraduate
physics lab courses. It provides a *Python* toolkit for fitting
models to data as well as visualizing the fit results.
A quick rundown of why you'd want to use *kafe2* can be found
`here <https://philfitters.github.io/kafe2/>`_.
The gist of it is that *kafe2* provides a simple, user-friendly interface for
state-of-the-art statistical methods.
It relies on Python packages such as :py:mod:`numpy` and :py:mod:`matplotlib`,
and can use the *Python* interface to the minimizer `Minuit` contained in the data
analysis framework `ROOT` or in the Python package `iminuit`.

The :ref:`first chapter <installing_kafe2>` of this documentation gives detailed installation
instructions.
The :ref:`Beginner's Guide <beginners_guide>` explains basic *kafe2* usage to cover simple cases
(both Python code and kafe2go).
The :ref:`User Guide <user_guide>` and the :ref:`kafe2go Guide <user_guide_kafe2go>` describe
advanced *kafe2* use with Python code or *kafe2go*.
The :ref:`next chapter <mathematical_foundations>` explains the mathematical foundations upon which
*kafe* is built.
While strictly speaking not required to use *kafe2*, reading the theory chapter is strongly
recommended to understand which features to use in a state-of-the-art data analysis
(regardless of whether *kafe2* or another data analysis tool is used).
The :ref:`Developer Guide <developer_guide>` covers topics that are only relevant
if you want to work on *kafe2* as a developer (still very much WIP).
Finally, the :ref:`API Documentation <api_documentation>` provides a full description
of the user-facing *kafe2* application programming interface.

.. toctree::
   :name: mastertoc
   :maxdepth: 2
   :includehidden:

   parts/installation
   parts/beginners_guide
   parts/user_guide
   parts/user_guide_kafe2go
   parts/mathematical_foundations.rst
   parts/developer_guide
   parts/api_documentation/index
