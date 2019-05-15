.. -*- mode: rst -*-

*************************************
kafe2 - Karlsruhe Fit Environment 2
*************************************

=====
About
=====

*kafe2* is an open-source Python package designed to provide a flexible 
Python interface for the estimation of model parameters from measured
data. It is the spiritual successor to the original *kafe* package.

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
* `Numdifftools <https://pypi.org/project/Numdifftools/>`_
* `SciPy <http://www.scipy.org>`_
* `matplotlib <http://matplotlib.org>`_
* `tabulate <https://pypi.org/project/tabulate/>`_


Since *kafe2* relies on *matplotlib* for graphics it might be necessary to install external programs:

* `Tkinter <https://wiki.python.org/moin/TkInter>`_, the default GUI used by *matplotlib*


Optionally, a function minimizer other than ``scipy.optimize.minimize`` can be used.
*kafe2* implements interfaces to two function minimizers and will use them
by default if they're installed:

* *MINUIT*, which is included in *CERN*'s data analysis package `ROOT <http://root.cern.ch>`_ (>= 5.34), or
* `iminuit <https://github.com/iminuit/iminuit>`_ (>= 1.1.1), which is independent of ROOT


==========================
Installation notes (Linux)
==========================

The easiest way to install *kafe2* is via `pip <https://pip.pypa.io/en/stable/>`_, which is
already included for Python >= 2.7.9. Installing via *pip* will automatically install the minimal
dependencies. Please note that commands below should be run as root.

**Note: kafe2 has not yet been added to the Python Packaging Index. The instructions below do not yet work.**

For Python 2:

    .. code:: bash
    
        pip install kafe2


For Python 3:

    .. code:: bash
    
        pip3 install kafe2


If you don't have *pip* installed, get it from the package manager.

In Ubuntu/Mint/Debian, do:

    .. code:: bash

        apt-get install python-pip


In Fedora/RHEL/CentOS, do:

    .. code:: bash

        yum install python-pip


or use ``easy_install`` (included with `setuptools <https://pypi.python.org/pypi/setuptools>`_):

    .. code:: bash

        easy_install pip


You will also need to install *Tkinter* if it didn't already come with your Python distribution.

For Python 2, Ubuntu/Mint/Debian:
	
	.. code:: bash
	
	    apt-get install python-tk


For Python 2, Fedora/RHEL/CentOS:
	
	.. code:: bash
	
	    yum install tkinter


For Python 3, Ubuntu/Mint/Debian:
	
	.. code:: bash
	
	    apt-get install python3-tk


For Python 3, Fedora/RHEL/CentOS:
	
	.. code:: bash
	
	    yum install python3-tkinter


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

You might also need to install the Python headers for *iminuit* to
compile properly.

In Ubuntu/Mint/Debian, do:

    .. code:: bash

        apt-get install libpython2.7-dev

In Fedora/RHEL/CentOS, do:

    .. code:: bash

        yum install python-devel

