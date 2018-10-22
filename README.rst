.. -*- mode: rst -*-

*************************************
*kafe2* - Karlsruhe Fit Environment 2
*************************************

=====
About
=====

*kafe2* is an open-source Python package designed to provide a flexible 
Python interface for the estimation of model parameters from measured
data. It is the spiritual successor of the original *kafe* package.

*kafe2* offers support for several types of data formats (including series
of indexed measurements, xy value pairs, and histograms) and data
uncertainty models, as well as arbitrarily complex parametric
models. The numeric aspects are handled using the scientific Python
stack (numpy, scipy, ...). Visualization of the data and the estimated
model are provided by matplotlib.

.. note:: kafe2 is still in development and not yet ready for production
          use.


============
Requirements
============

*kafe2* needs some additional Python packages:

* `NumPy <http://www.numpy.org>`_
* `Numdifftools <https://pypi.org/project/Numdifftools/>`
* `SciPy <http://www.scipy.org>`_
* `matplotlib <http://matplotlib.org>`_


Optionally, a function minimizer other than scipy.optimize.minimize can be used.
*kafe2* implements interfaces to two function minimizers and will use them
by default if they're installed:

* *MINUIT*, which is included in *CERN*'s data analysis package `ROOT <http://root.cern.ch>`_ (>= 5.34), or
* `iminuit <https://github.com/iminuit/iminuit>`_ (>= 1.1.1), which is independent of ROOT


Finally, *kafe2* requires a number of external programs:

* Qt4 and the Python bindings PyQt4 are needed because *Qt* is the supported
  interactive frontend for matplotlib. Other frontends are not supported and may cause unexpected behavior.

==========================
Installation notes
==========================

The easiest way to install *kafe2* is via *pip <https://pypi.org/project/pip/>*, which is
already included for Python >= 2.7.9. Installing via *pip* will automatically install the minimal
dependencies.

For Python 2:

    .. code:bash
    
        pip install kafe2

For Python 3:

    .. code:bash
    
        pip3 install kafe2


------------------------
Optional: Install *ROOT*
------------------------

ROOT and its Python bindings can be obtained via the package manager in
Ubuntu/Mint/Debian:

    .. code:: bash

        apt-get install root-system libroot-bindings-python5.34 libroot-bindings-python-dev

Or, in Fedora/RHEL/CentOS:

    .. code:: bash

        yum install root root-python


This setup is usually sufficient. However, you may decide to build ROOT yourself. In this case,
be sure to compile with *PyROOT* support. Additionally, for Python to see the *PyROOT* bindings,
the following environment variables have to be set correctly (:

    .. code:: bash

        export ROOTSYS=<directory where ROOT is installed>
        export LD_LIBRARY_PATH=$ROOTSYS/lib:$PYTHONDIR/lib:$LD_LIBRARY_PATH
        export PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH


For more info, refer to `<http://root.cern.ch/drupal/content/pyroot>`_.


---------------------------
Optional: Install `iminuit`
---------------------------

*iminuit* is a Python wrapper for the Minuit minimizer which is
independent of ROOT. If compiling/installing ROOT is not possible,
this minimizer can be used instead.

To install the *iminuit* package for Python, the `Pip installer
<http://www.pip-installer.org/>`_ is recommended:

    .. code:: bash

        pip install iminuit

If you don't have *Pip* installed, get it from the package manager.

In Ubuntu/Mint/Debian, do:

    .. code:: bash

        apt-get install python-pip

In Fedora/RHEL/CentOS, do:

    .. code:: bash

        yum install python-pip

or use ``easy_install`` (included with `setuptools <https://pypi.python.org/pypi/setuptools>`_):

    .. code:: bash

        easy_install pip

You might also need to install the Python headers for *iminuit* to
compile properly.

In Ubuntu/Mint/Debian, do:

    .. code:: bash

        apt-get install libpython2.7-dev

In Fedora/RHEL/CentOS, do:

    .. code:: bash

        yum install python-devel

