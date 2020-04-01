.. -*- mode: rst -*-

*************************************
kafe2 - Karlsruhe Fit Environment 2
*************************************

.. image:: https://readthedocs.org/projects/kafe2/badge/?version=latest
    :target: https://kafe2.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/dsavoiu/kafe2.svg?branch=master
    :target: https://travis-ci.org/dsavoiu/kafe2


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

While *kafe2* supports both Python 2 and Python 3, the use of Python 3 is recommended.

**Note**

kafe2 is currently in a *beta* state. Most features are working as intended. However, bugs
may occur during use. If you encounter any bugs or have an improvement proposal, please let us
know by opening an issue `here <https://github.com/dsavoiu/kafe2/issues>`_.

A user guide can be found on `kafe2.readthedocs.io <https://kafe2.readthedocs.io/en/latest/parts/user_guide.html>`_.


============
Requirements
============

*kafe2* needs some additional Python packages. When *kafe2* is installed via *pip*, those packages
are automatically installed as dependencies:

* `NumPy <http://www.numpy.org>`_
* `Numdifftools <https://pypi.org/project/Numdifftools/>`_
* `SciPy <http://www.scipy.org>`_
* `matplotlib <http://matplotlib.org>`_
* `tabulate <https://pypi.org/project/tabulate/>`_
* `PyYAML <https://pypi.org/project/PyYAML/>`_

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

For Python 2:

.. code:: bash

    pip2 install kafe2



For Python 3:

.. code:: bash

    pip3 install kafe2


If you don't have *pip* installed, get it from the package manager.

In Ubuntu/Mint/Debian, do:

.. code:: bash

    apt-get install python-pip python3-pip


In Fedora/RHEL/CentOS, do:

.. code:: bash

    yum install python2-pip python3-pip


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

**Note: Starting with Ubuntu 16.10, ROOT is no longer available in the official repositories.**

In older versions of Ubuntu (and related Linux distributions), ROOT and its Python bindings
can be obtained via the package manager via:

.. code:: bash

    apt-get install root-system libroot-bindings-python5.34 libroot-bindings-python-dev


Or, in Fedora/RHEL/CentOS:

.. code:: bash

    yum install root root-python


This setup is usually sufficient. However, you may decide to build ROOT yourself. In this case,
be sure to compile with *PyROOT* support. Additionally, for Python to see the *PyROOT* bindings,
the following environment variables have to be set correctly:

.. code:: bash

    export ROOTSYS=<directory where ROOT is installed>
    export LD_LIBRARY_PATH=$ROOTSYS/lib:$PYTHONDIR/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH


For more info, refer to `<http://root.cern.ch/drupal/content/pyroot>`_.


---------------------------
Optional: Install `iminuit`
---------------------------

*iminuit* is a Python wrapper for the Minuit minimizer which is
independent of ROOT. This minimizer can be used instead of ROOT.

To install the *iminuit* package for Python, the `Pip installer
<http://www.pip-installer.org/>`_ is recommended:

.. code:: bash

    pip install iminuit

You might also need to install the Python headers for *iminuit* to
compile properly.

In Ubuntu/Mint/Debian, do:

.. code:: bash

    apt-get install libpython2-dev libpython3-dev

In Fedora/RHEL/CentOS, do:

.. code:: bash

    yum install python2-devel python3-devel

