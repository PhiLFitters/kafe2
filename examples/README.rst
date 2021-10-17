This directory contains examples for *kafe2*:
files that illustrate how to use *kafe2* for various use cases.
Because *kafe2* is intended as a tool for beginners these examples don't get into too much technical
detail.
For a more detailed explanation for why things are the way they are consult the
`documentation <https://kafe2.readthedocs.io/en/latest/parts/mathematical_foundations.html>`_.

A quick overview of the examples available:

* **001_line_fit**: Fitting a line to *xy* data points with some uncertainties in the
  *x* and *y* direction.
* **002_model_functions**: How to specify custom model functions.
* **003_profiling**: Considerations for handling non-linear fits.
  Relevant for nonlinear model functions, *x* errors, and relative model errors.
* **004_constraints**: How to specify constraints for model parameters.
  Can be used as an alternative to error propagation.
  Also shows how to limit parameters to specified intervals.
* **005_convenience**: How to use *kafe2* functionality that's not directly related to fitting but
  rather to things like changing plot colors or saving fit results to files.
* **006_advanced_errors**: More complicated uses of uncertainties.
  Specify correlations as either a covariance matrix or through multiple simple errors.
  Enable/disable specific error components.
  Specify errors relative to the model instead of the data.
* **007_cost_functions**: How to define cost functions other than chi squared.
* **008_indexed_fit**: How to fit a model to indexed data.
* **009_histogram_fit**: How to bin one-dimensional data and fit a probability distribution.
* **010_unbinned_fit**: How to fit a probability distribution to data without binning it.
* **011_multifit**: How to handle several fits using the same parameters.
  Either use constraints or a MultiFit.
* **012_custom_fit**: How to define a fit using only a cost function without explicit model or data.

All examples are available as Python scripts.
Some are also available as YAML files intended for *kafe2go* (under progress).
In this directory there are also Jupyter notebooks that cover the same topics as the examples in a
more tutorial-like way.
The Jupyter notebooks are available both in German and in English, the examples are only available
in English.
