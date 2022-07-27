<!--
This is the source for https://kafe2.github.io/index.html,
generated via   pandoc README.md -o index.html
-->

<style> 
  body {
    background-color:white; 
    text-align:justify; 
    max-width:50em;
  }
  h1 {
    align:center;
    background-color:GhostWhite;
  }
  p {
    background-color:white;
  }
</style>

<!-- header -->
<p  style="background-color:AliceBlue;">   </p>
<img src="kafe2_logo.png" width=120
     style= "float: left; margin-right: 10px;" /> <br>
<h1> <i>kafe2</i> - Data Visualisation and Model Fitting </h1>
  > &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
    Link to [**github Repository** of the *kafe2* project](
    https://github.com/PhiLFitters/kafe2). 
<p  style="background-color:AliceBlue;">  <hr>  </p>

<!-- body -->

## **_kafe2_**  

&nbsp; is an open-source <i>Python</i> package designed to provide a flexible
interface for the estimation of model parameters from measured data.
It is the spiritual successor to the original
[*kafe*](https://github.com/dsavoiu/kafe) package.


_kafe2_ offers support for **several types of input data** like series of
indexed measurements, *x-y* value pairs, and histogram data. Different types
of uncertainties are taken into account, like 
**statistical and relative or absolute systematic uncertainties**. 
Arbitrarily complex parametric models to describe the
data can be provided as *Python* code. 

Fitting procedures in _kafe2_ are based on **negative log-likelihood cost-functions**
(*nlL*). As the default and for consistency with other tools, parameter uncertainties 
are determined via the Cramér-Rao-Frechét bound, i.e. based on the 2nd derivatives 
of the *nlL* function with respect to the parameters. Optionally, one and 
two-dimensional profile-likelihood scans can be performed, which provide 
asymmetric confidence intervals for the parameter values.

The graphical result of a fit of two models to data with _kafe2_ is shown 
in the figures below.

![Example: Fit of two models with kafe2](kafe_graph1.png)

  > ![Profile Likelihoods and one- and two-sigma Confidence Contour  
    for the two-parameter fit of the exponential model](kafe_graph2.png)

Handling different kinds of uncertainties is one of the unique features 
of the *kafe2* package. Data points are affected by uncertainties in both 
the *x* and *y* directions, which may be absolute, relative, uncorrelated 
or correlated among all data points or among groups of data points. 
These different kinds of uncertainties are taken into account by construction 
of the overall covariance matrix in each step of the numerical optimization. 
This detail of the numerical procedure is particularly important for relative 
uncertainties and for uncertainties in the *x*-direction, which depend on the 
parameter-dependent model values rather than on the observed values in data. 
Although computationally expensive, this method avoids biases in the predicted 
parameters and leads to an optimal statistical coverage of the confidence 
intervals derived for the parameter estimates.

The *kafe2* package comes with extensive documentation for beginners, advanced
users and programmers wanting to include powerful fits in their own projects.
A large number of examples explain practical applications and may serve as
simple function wrappers for own projects. The stand-alone program _kafe2go_
permits data-driven fits without the need for the user providing _Python_ code; 
the input format describing data and their uncertainties as well as meta-data 
and the fit model is based on the very light-weight  human-readable data 
description language *yaml*.

Click this link to see the [full **documentation**](
https://kafe2.readthedocs.io/en/latest/).
Installation is easy via [PyPi](https://pypi.org/project/kafe2/), just type
`pip3 install kafe2`.
The  *kafe2* code, documentation and application examples are available in the 
[**Repository** of the *kafe2* project](https://github.com/PhiLFitters/kafe2). 

### Dependencies

  - *Python* versions 3.8+ are supported.
  - Numerical aspects are handled by the scientific _Python_ stack
    (*NumPy*, *SciPy*, ...). 
  - Visualizations of the data, the estimated model and of parameter confidence regions 
    are provided by *matplotlib*. 
  - High-dimensional numerical optimization and uncertainty analysis rely on 
    the packages *scipy.optimize* or *iminuit* (based on the Minuit2 C++ package 
    developed and maintained at CERN). 

