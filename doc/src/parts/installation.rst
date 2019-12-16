.. meta::
   :description lang=en: kafe - a general, Python-based approach to fit a
      model function to two-dimensional data points with correlated
      uncertainties in both dimensions
   :robots: index, follow


==================
Installing *kafe2*
==================

Requirements
============

*kafe2* needs some additional Python packages:

* `NumPy <http://www.numpy.org>`_
* `Numdifftools <https://pypi.org/project/Numdifftools/>`_
* `SciPy <http://www.scipy.org>`_
* `matplotlib <http://matplotlib.org>`_


Since *kafe2* relies on *matplotlib* for graphics it might be necessary to install external programs:

* `Tkinter <https://wiki.python.org/moin/TkInter>`_, the default GUI used by *matplotlib*


Optionally, a function minimizer other than ``scipy.optimize.minimize`` can be used.
*kafe2* implements interfaces to two function minimizers and will use them
by default if they're installed:

* *MINUIT*, which is included in *CERN*'s data analysis package `ROOT <http://root.cern.ch>`_ (>= 5.34), or
* `iminuit <https://github.com/iminuit/iminuit>`_ (>= 1.1.1), which is independent of ROOT


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


.. todo::

    Update this section


Installation notes (Windows)
============================

*kafe* can be installed under Windows, but requires some additional configuration.

The recommended Python distribution for working with *kafe* under Windows is
`WinPython <https://winpython.github.io/>`_, which has the advantage that it is
portable and comes with a number of useful pre-installed packages. Particularly,
*NumPy*, *SciPy* and *matplotlib* are all pre-installed in *WinPython*, as are
all *Qt*-related dependencies.

Be sure to install *WinPython* version **2.7**, since *kafe* does not currently
run under Python 3.


Install `iminuit`
-----------------

After installing *WinPython*, start 'WinPython Command Prompt.exe' in the
*WinPython* installation directory and run

.. code:: bash

    pip install iminuit


Install `kafe`
--------------

Now *kafe* can be installed from PyPI by running:

.. code:: bash

    pip install kafe

Alternatively, it may be installed directly using *setuptools*. Just run
the following in 'WinPython Command Prompt.exe' after switching to the
directory into which you have downloaded *kafe*:

.. code:: bash

    python setup.py install


Using *kafe* with ROOT under Windows
--------------------------------------

If you want *kafe* to work with ROOT's ``TMinuit`` instead of using
*iminuit*, then ROOT has to be installed. Please note that ROOT releases
for Windows are 32-bit and using the PyROOT bindings on a 64-bit *WinPython*
distribution will not work.

A pre-built verson of ROOT for Windows is available on the ROOT homepage as a Windows
Installer package. The recommended version is
`ROOT 5.34 <https://root.cern.ch/content/release-53434>`_.
During the installation process, select "Add ROOT to the system PATH for all users"
when prompted. This will set the ``PATH`` environment variable to include
the relevant ROOT directories. The installer also sets the ``ROOTSYS`` environment
variable, which points to the directory where ROOT in installed. By default,
this is ``C:\root_v5.34.34``.

Additionally, for Python to find the *PyROOT* bindings, the ``PYTHONPATH``
environment variable must be modified to include the ``bin`` subdirectory
of path where ROOT is installed. On Windows 10, assuming ROOT has been installed
in the default directory (``C:\root_v5.34.34``), this is achieved as follows:

1)  open the Start Menu and start typing "environment variables"
2)  select "Edit the system environment variables"
3)  click the "Environment Variables..." button
4)  in the lower part, under "System variables", look for the "PYTHONPATH" entry

5)  modify/add the "PYTHONPATH" entry:

    * if it doesn't exist, create it by choosing "New...",
      enter PYTHONPATH as the variable name
      and ``C:\root_v5.34.34\bin`` as the variable value
    * if it already exists and contains only one path, edit it via "Edit..." and
      insert ``C:\root_v5.34.34\bin;`` at the beginning of the variable value.
      (Note the semicolon!)
    * if the variable already contains several paths, choosing "Edit..." will
      show a dialog box to manage them. Choose "New" and write
      ``C:\root_v5.34.34\bin``

6)  close all opened dialogs with "OK"


Now you may try to ``import ROOT`` in the *WinPython* interpreter to check
if everything has been set up correctly.

For more information please refer to ROOT's official
`PyROOT Guide <https://root.cern.ch/pyroot>`_.

