<!--
This is the source for https://kafe2.github.io/index.html,
generated via   pandoc README.md -o index.html
-->

<body style="background-color:white;">

<p style="background-color:AliceBlue;">         </p>

<img src="kafe2_logo.png" width=120
     style= "float: left; margin-right: 10px;" /> <br>

<h1 ALIGN="center"; style="background-color:GhostWhite"> 
      <i>kafe2</i> - Data Visualisation and Model Fitting 
</h1>

  > &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
    Link to [**github Repository** of the *kafe2* project](
    https://github.com/PhiLFitters/kafe2). 

<p style="background-color:AliceBlue;"> <hr>  </p>


## **_kafe2_** 
&nbsp; is an open-source <i>Python</i> package designed to provide a flexible
interface for the estimation of model parameters from measured data.
It is the spiritual successor to the original
[*kafe*](https://github.com/dsavoiu/kafe) package.


  - *Python* versions 3.8+ are supported.
  - Numerical aspects are handled by the scientific _Python_ stack
    (*NumPy*, *SciPy*, ...). 
  - Visualizations of the data, the estimated model and of parameter confidence regions 
    are provided by *matplotlib*. 
  - High-dimensional numerical optimization and uncertainty analysis rely on 
    the packages *scipy.optimize* or *iminuit* (based on the Minuit2 C++ package 
    developed and maintained at CERN). 

## Fitting with *kafe2* in a nutshell

*kafe2* offers support for several types of data (including series of
indexed measurements, *x-y* value pairs, and histogram data) and uncertainty 
models, as well as arbitrarily complex parametric models implemented in *Python*.

Fitting procedures in _kafe2_ are based on negative log-likelihood cost-functions
(*nlL*). As the default and for consistency with other tools, uncertainties are 
determined via the Cramér-Rao-Frechét bound, i.e. based on the 2nd derivatives 
of the *nlL* function with respect to the parameters. Optionally, one and
two-dimensional scans of the profile likelihood can be performed, which provide 
asymmetric confidence intervals for the parameter values, as is shown in the 
example below.

![Example: Fit of two models with kafe2](
kafe_graph1.png)

  > ![Profile Likelihoods and Confidence Contour for the
    two-parameter fit of the exponential model](kafe_graph2.png)

Handling different kinds of uncertainties is one of the unique features 
of the *kafe2* package. Data points are affected by uncertainties in the
*x* and *y* directions, which may be absolute, relative, uncorrelated or 
correlated among all or groups of data points. These different kinds of 
uncertainties are taken into account by construction of the overall 
covariance matrix in each step of the numerical optimization. 
This is particularly important for relative uncertainties and 
for uncertainties in the *x*-direction, which depend on the model values rather 
than on the observed values in data. This procedure is numerically expensive, 
but avoids biases in the predicted parameters and leads to an optimal statistical 
coverage of the confidence intervals derived for the parameter estimates.

Click this link to see the [full **documentation**](
https://kafe2.readthedocs.io/en/latest/).
Installation is easy via [PyPi](https://pypi.org/project/kafe2/), just type
`pip3 install kafe2`.
The  *kafe2* code and application examples are available in the 
[**Repository** of the *kafe2* project](https://github.com/PhiLFitters/kafe2). 
