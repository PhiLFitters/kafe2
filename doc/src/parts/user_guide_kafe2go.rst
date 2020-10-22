.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow

*************
kafe2go Guide
*************

*kafe2go* is a standalone program for performing fits which comes with *kafe2*.
When using *kafe2go* no programming is required.
Instead all necessary information for performing a fit is defined inside a so-called
`YAML <https://en.wikipedia.org/wiki/YAML>`_ file.
Full examples can be found in the :ref:`beginners_guide`.

To run kafe2go on Linux or MacOS issue this command in the terminal, after installing *kafe2*:

.. code-block:: bash

    kafe2go path/to/fit.yml

When using Windows, please use:

.. code-block:: bash

    kafe2go.py path/to/fit.yml

The output can be customized via command line arguments.
For more information about the command line arguments run:

.. code-block:: bash

    kafe2go --help

Additionally, *YAML* files can also be created from fit objects inside a *Python* program by using
the :py:meth:`~.FitBase.to_file` method and loaded from a *YAML* file with
:py:meth:`~.FitBase.from_file`.

Setting a fit type
==================
By default *kafe2go* assumes an XYFit, where each data point has an x and a y component.
Other types are defined via the ``type`` keyword inside the *YAML* file:

.. code-block:: yaml

    type: histogram

Supported types are ``type: xy``, ``type: histogram``, ``type: unbinned`` and ``type: indexed``.

Specifying the data
===================
The syntax for specifying fit data differs depending on which fit type is being used.

XY Fits
-------
For XY Fits the x and y values are defined separately:

.. code-block:: yaml

    # Data is defined by lists:
    x_data: [1.0, 2.0, 3.0, 4.0]
    # In yaml lists can also be written out like this:
    y_data:
    - 2.3
    - 4.2
    - 7.5
    - 9.4

The uncertainties in x and y direction can be defined like this:

.. code-block:: yaml

    # For errors lists describe pointwise uncertainties.
    # By default the errors will be uncorrelated.
    x_errors: [0.05, 0.10, 0.15, 0.20]

    # Because x_errors is equal to 5% of x_data we could have also defined it like this:
    # x_errors: 5%

    # For errors a single float gives each data point
    # the same amount of uncertainty:
    y_errors: 0.4

In total the above examples represents the following dataset:

+-------------+------------+
| X Values    | Y Values   |
+=============+============+
| 1.0 +- 0.05 | 2.3 +- 0.4 |
+-------------+------------+
| 2.0 +- 0.10 | 4.2 +- 0.4 |
+-------------+------------+
| 3.0 +- 0.15 | 7.5 +- 0.4 |
+-------------+------------+
| 4.0 +- 0.20 | 9.4 +- 0.4 |
+-------------+------------+

.. todo::

    Add more advanced errors for XY Fits

Histogram Fits
--------------
Specifying the data for histogram fits can be done in two different ways:
either by specifying the raw data and bins and let *kafe2* handle the binning or by specifying
the bins and bin heights.

Like with Python code there are two ways of specifying bins.
The first way is to specify equidistant binning with the number of bins and the bin range.
A binning with 10 bins in the range from -5 to 5 can be achieved like this:

.. code-block:: yaml

    n_bins: 10
    bin_range: [-5, 5]

Alternatively the ``bin_edges`` keyword can be used to directly specify bin edges with arbitrary
distances between them:

.. code-block:: yaml

    bin_edges: [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

Filling the bins with raw data is done like this:

.. code-block:: yaml

    raw_data: [-7.5, 1.23, 5.74, 1.9, -0.2, 3.1, -2.75, ...]

Data points lying outside the bin range will be stored in an underflow or overflow bin and are
not considered when performing the fit.

Alternatively the heights of the bins can be set manually. This is done with the ``bin_heights``
keyword:

.. code-block:: yaml

    bin_heights: [7, 21, 25, 42, 54, 51, 39, 28, 20, 12]

.. warning::
    The length of ``bin_heights`` must match the number of bins ``n_bins`` or the length of
    ``bin_edges`` minus one.

The height of the underflow and overflow bin is set via the ``underflow`` and ``overflow`` keywords.

Unbinned and Indexed Fits
-------------------------
Setting the data for unbinned and indexed fits is done via the ``data`` keyword:

.. code-block:: yaml

    data: [7.420, 3.773, 5.968, 4.924, 1.468, 4.664, 1.745, 2.144, 3.836, 3.132, 1.568]

Data label and axis labels
--------------------------
The name of the dataset or its label is set with the ``label`` keyword, axis labels can be set
with the ``x_label`` and ``y_label`` keywords:

.. code-block:: yaml

    label: "My Data"
    x_label: "X Values with latex $\\tau$ (µs)"
    y_label: "$y_0$-label"

Text in between dollar signs will be interpreted as latex code.
The labels are used in the graphical output of *kafe2go*.

.. todo::

    Add errors for HistFits and IndexedFits, implement and add for UnbinnedFits

.. _kafe2go_model_function:

Setting a model function
========================
A model function is defined with the ``model_function`` keyword, followed by Python code as a
string:

.. code-block:: yaml

    model_function: |
      def exponential_model(x, A0=1., x0=5.):
          # Our model is a simple exponential function
          # The kwargs in the function header specify parameter defaults.
          return A0 * np.exp(x/x0)

Note the block style indicate ``|`` which indicates a multiline string and keeps line breaks.

Additionally the output names for the model and its parameters can be changed.
Then the ``model_function`` block gains its own keywords and the Python code is moved to the
``python_code`` sub keyword:

.. code-block:: yaml

    model_function:
      python_code: |
        def exponential_model(x, A0=1., x0=5.):
          # Our model is a simple exponential function
          # The kwargs in the function header specify parameter defaults.
          return A0 * np.exp(x/x0)
      name: "exponential function"
      latex_name: "\\exp"
      expression_string: "{A0} * exp({x}/{x0})"
      latex_expression_string: "{A0} e^{{\\frac{{{x}}}{{{x0}}}}}"
      arg_formatters:
        x: 'x'
        x0: "x_0"
        A0:
          - name: A
          - latex_name: A_0

All other keywords are pretty much self-explanatory:

- ``name`` is the model function name for the terminal output.
  If omitted the function name from the definition will be used.
- ``latex_name`` is the model function name for the graphical output.
  If omitted the function name from the definition will be used.
- ``expression_string`` is the model function expression for the terminal output.
  If omitted, the output won't have any function expression after its name.
  Every parameter name inside curly brackets which matches the parameter names from the function
  definition will be replaced with its formatted version defined in ``arg_formatters``.
- ``latex_expression_string`` is the same as ``expression_string`` but for the graphical output
  and supports latex syntax.
- ``arg_formatters`` defines the replacements for the function parameters.
  If only one string is given (see x and x0 in the example), the default name will be used for the
  terminal output and the given string will be used for the graphical latex output.
  If ``name`` and ``latex_name`` are defined, they are used for terminal and graphical outputs
  (see A0).

.. note::

    Special characters inside the strings need to be escaped. E.g. a single ``\`` needs to be
    ``\\``.

.. note::

    Inside the latex expression string, ``{`` and ``}`` for latex expression like ``\\frac`` need to
    be doubled, because single curly brackets are used for replacing the parameters with their
    respective latex names. E.g. kafe2 tries to replace ``{x0}`` with its latex string ``x_0`` in
    this example.


.. _kafe2go_constraints:

Parameter Constraints
=====================
The parameter constraints specified with *kafe2go* require the same information as those
:ref:`specified with Python code <constraints_guide>`:
parameter names, values, and uncertainties.

Simple Gaussian Constraints
---------------------------
Simple gaussian parameter constraints can be defined by parameter name like this:

.. code-block:: yaml

    parameter_constraints:
      a:
        value: 10.0
        uncertainty: 0.001    # a = 10.0+-0.001
      b:
        value: 0.6
        uncertainty: 0.006
        relative: true        # Make constraint uncertainty relative to value, b = 0.6+-0.6%

The same can also be done with a list and the name keyword:

.. code-block:: yaml

    parameter_constraints:
        - name: a
          value: 10.0
          uncertainty: 0.001    # a = 10.0+-0.001
        - name: b
          value: 0.6
          uncertainty: 0.006
          relative: true        # Make constraint uncertainty relative to value, b = 0.6+-0.6%

Matrix Gaussian Constraints
---------------------------
Matrix constraints can only be specified using the list format since they constrain multiple
parameters, possibly making the first format ambiguous.
Inside a list element, the type can be set with the ``type: matrix`` keyword and a covariance
matrix can be given via the ``matrix`` keyword like this:

.. code-block:: yaml

    parameter_constraints:
      - type: matrix
        names: [a, b]
        values: [1.3, 2.5]
        matrix: [[1.1, 0.1], [0.1, 2.4]]
      - type: simple
        name: c
        value: 5.2
        uncertainty: 0.001

Here ``type: simple`` can be omitted, as a simple gaussian constraint is assumed when no type
is given.

A covariance matrix is assumed by default for matrix constraints. Correlation matrices are
supported as well. The matrix type can be set via the ``matrix_type`` keyword.
Only ``matrix_type: cov`` for covariance matrices and ``matrix_type: cor`` for correlation matrices
are supported.

Fixing and limiting parameters
==============================
Fixing and limiting parameters is just as simple as the following example.
Please use the parameter names as they are defined inside the model function
(see: :ref:`kafe2go_model_function`).

.. code-block:: yaml

    fixed_parameters:
      a: 1
      b: 11.5
    limited_parameters:
      c: [0.0, 1.0]
      d: [-5, 5]

Note that a limited parameter needs a lower and an upper limit. The first value of the list is the
lower limit and the second value is the upper limit.
If a parameter shall be limited to non-negative values, choose 0 as lower and a high value as
upper limit.
If the final fit result is close to or at the parameter limits, the limits should be reassessed,
as the final minimum of the cost function might be a local minimum for the given limits.
