.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow

**********
User Guide
**********

Fitting
=======

Minimizers
----------
Logging
+++++++

To enable the output of the minimizer, set up a logger before calling :py:func:`kafe2.fit._base.FitBase.do_fit`:

.. code-block:: python

    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

This currently only works for the :py:mod:`scipy` and :py:mod:`iminuit` minimizer.
For more detailed information increase the logging level to :py:const:`logging.DEBUG`.
This will give a more verbose output when using :py:mod:`iminuit`.
The logger level should be reset to :py:const:`logging.WARNING` before plotting.
Otherwise :py:mod:`matplotlib` will create logging messages as well.


Plotting
========

Contours Profiler
=================

kafe2go
=======
For using kafe2go, yaml files of the data need to be crated. Examples are given at the User Guide.
To run kafe2go issue this command in the terminal:

.. code-block:: bash

    kafe2go path/to/fit.yml

For more information about the command line arguments run:

.. code-block:: bash

    kafe2go --help