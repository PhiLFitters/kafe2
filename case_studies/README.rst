This directory contains case studies for *kafe2*:
files that highlight some aspect of *kafe2* or fitting in general and explain why things are the
way they are.

A quick overview of the case studies available:

* **determinant_cost.py**: How *kafe2* handles the bias introduced by parameter-dependent input
  uncertainties.
* **ensemble_test.py**: How to do an ensemble test with *kafe2* (**not production ready**).
* **extremely_nonlinear_fit.py**: Profile of a simple fit with *x* errors so large that the
  likelihood has several minima.
* **relative_errors.py**: Why *kafe2* has several methods of specifying relative errors,
  how they differ, and whether they're biased.
* **shared_error_vs_separate.py**: Comparison of adding a shared (100% correlated) error to a
  MultiFit vs. adding two separate (uncorrelated errors) to the underlying fits.