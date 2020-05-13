.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow

.. _user-guide:

**********
User Guide
**********

For performing fits with *kafe2*, the user need to specify the data, model function and
optionally a so-called cost function. In most cases the cost function is either the method of
:ref:`least-squares` or the :ref:`negative-log-likelihood`. All this information is then given to
a :py:obj:`~.FitBase`-derived object. More information is given in the :ref:`fitting`-section.

Then there are multiple ways of displaying and using the fit results. The results can either be
used directly inside a *Python*-script, printed to the terminal, or :ref:`plotted <plotting>`.
For further analysis, the :ref:`contours-profiler` is a very helpful tool to display parameter
correlations.

.. figure:: ../_static/img/kafe2_structure.png
    :alt: General workflow with kafe2.

.. _fitting:

Fitting
=======

.. _minimizers:

Minimizers
----------
Currently the use of three different minimizers is supported. By default :py:mod:`iminuit` is
used. If :py:mod:`iminuit` is not available, *kafe2* falls back to
:py:obj:`scipy.optimize.minimize`.

The usage of a specific minimizer can be set during initialization of any
:py:obj:`~.FitBase`-object with the `minimizer` keyword.
Depending on the installed minimizers this can either be :code:`'iminuit'`, :code:`'scipy'` or
:code:`'root'`.

Additional keywords for the instantiation can be passed as a :py:obj:`dict` via the
`minimizer_kwargs` keyword when creating a fit object derived from :py:obj:`~.FitBase`.


Logging
+++++++
To enable the output of the minimizer, set up a logger before calling :py:func:`~.FitBase.do_fit`:

.. code-block:: python

    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

This currently only works for the :py:mod:`scipy` and :py:mod:`iminuit` minimizer.
For more detailed information increase the logging level to :py:const:`logging.DEBUG`.
This will give a more verbose output when using :py:mod:`iminuit`.
The logger level should be reset to :py:const:`logging.WARNING` before plotting.
Otherwise :py:mod:`matplotlib` will create logging messages as well.

.. _plotting:

Plotting
========

For displaying the results of a Fit, *kafe2* provides a :py:obj:`~.Plot`-class. In the background
a :py:obj:`matplotlib.pyplot.figure`-object is created. This means that all customization possible
with *Matplotlib* can be done with *kafe2*-Plots as well.

The Plot class supports plotting multiple fits at once.

.. code-block:: python

    p = Plot([fit_1, fit_2])

Running the :py:meth:`~.Plot.plot` function will perform the the plot. Customization should be
done before this. After plotting the fits, the according :py:mod:`matplotlib` objects can be
accessed via the :py:attr:`~.Plot.figures` and :py:attr:`~.Plot.axes` properties.

The plot range can be set via the :py:attr:`~.Plot.x_range` and :py:attr:`~.Plot.y_range`
properties:

.. code-block:: python

    p.x_range = (0, 10)
    p.y_range = (-5, 25)

Customize the Plot
------------------
Each graphic element has it's own plotting method and can be customized individually. Available
*plot_types* for XYFits are
:code:`'data', 'model_line', 'model_error_band', 'ratio', 'ratio_error_band'`.
The *plot_types* may differ for different types of fits.

The currently set keywords can be obtained with the :py:meth:`~.Plot.get_keywords` method.
With :py:meth:`~.Plot.customize` new values can be added or existing values can
be modified. Using :code:`'__del__'` will delete the keyword and :code:`'__default__'` will reset
it.

To change the name for the data set and suppress the second output, use the following call:

.. code-block:: python

    p.customize('data', 'label', [(0, "test data"),(1, '__del__')])

Marker type, size and color of the marker and error bars can also be customized:

.. code-block:: python

    p.customize('data', 'marker', [(0, 'o'), (1,'o')])
    p.customize('data', 'markersize', [(0, 5), (1, 5)])
    p.customize('data', 'color', [(0, 'blue'), (1,'blue')]) # note: although 2nd label is suppressed
    p.customize('data', 'ecolor', [(0, 'blue'), (1, 'blue')]) # note: although 2nd label is suppressed

The corresponding values for the model function can also be customized:

.. code-block:: python

    p.customize('model_line', 'color', [(0, 'orange'),(1, 'lightgreen')])
    p.customize('model_error_band', 'label', [(0, r'$\pm 1 \sigma$'),(1, r'$\pm 1 \sigma$')])
    p.customize('model_error_band', 'color', [(0, 'orange'),(1, 'lightgreen')])

It is also possible to change parameters using matplotlib functions.
To change the size of the axis labels, use the following calls:

.. code-block:: python

    import matplotlib as mpl
    mpl.rc('axes', labelsize=20, titlesize=25)

.. _contours-profiler:

Contours Profiler
=================

kafe2go
=======
When using kafe2go yaml files of the data need to be created. Examples can be found in the
:ref:`beginners_guide`.
To run kafe2go issue this command in the terminal:

.. code-block:: bash

    kafe2go path/to/fit.yml

For more information about the command line arguments run:

.. code-block:: bash

    kafe2go --help