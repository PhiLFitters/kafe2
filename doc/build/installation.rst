.. meta::
   :description lang=en: kafe - a general, Python-based approach to fit a
      model function to two-dimensional data points with correlated
      uncertainties in both dimensions
   :robots: index, follow


*****************
Installing *kafe*
*****************

Requirements
============

*kafe* needs some additional Python packages. The recommended versions of these are
as follows:

* `SciPy <http://www.scipy.org>`_ >= 0.12.0
* `NumPy <http://www.numpy.org>`_ >= 1.10.4
* `matplotlib <http://matplotlib.org>`_ >= 1.5.0


Additionally, a function minimizer is needed. *kafe* implements interfaces to two
function minimizers and requires at least one of them to be installed:

* *MINUIT*, which is included in *CERN*'s data analysis package `ROOT <http://root.cern.ch>`_ (>= 5.34), or
* `iminuit <https://github.com/iminuit/iminuit>`_ (>= 1.1.1), which is independent of ROOT (this is the default)


Finally, *kafe* requires a number of external programs:

* Qt4 (>= 4.8.5) and the Python bindings PyQt4 (>= 3.18.1) are needed because *Qt* is the supported
  interactive frontend for matplotlib. Other frontends are not supported and may cause unexpected behavior.
* A *LaTeX* distribution (tested with `TeX Live <https://www.tug.org/texlive/>`_), since *LaTeX* is
  used by matplotlib for typesetting labels and mathematical expressions.
* `dvipng <http://www.nongnu.org/dvipng/>`_ for converting DVI files to PNG graphics


Installation notes (Linux)
==========================


Most of the above packages and programs can be installed through the package manager on most Linux
distributions.

*kafe* was developed for use on Linux desktop systems. Please note that all
commands below should be run as root.


Install *NumPy*, *SciPy* and *matplotlib*
-----------------------------------------

These packages should be available in the package manager.

In Ubuntu/Mint/Debian:

    .. code:: bash

        apt-get install python-numpy python-scipy python-matplotlib

In Fedora/RHEL/CentOS:

    .. code:: bash

        yum install numpy scipy python-matplotlib



Install *ROOT*
--------------

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



Install `iminuit`
-----------------

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


Read the README file for more information on other dependencies
(there should be adequate packages for your Linux distribution
to satisfy these).


Install *kafe*
--------------

To install *kafe* using *Pip*, simply run the helper script as root:

    .. code:: bash

        ./install.sh

To remove kafe using *Pip*, just run the helper script:

    .. code:: bash

        ./uninstall.sh


Alternatively, installing using Python's *setuptools* also works, but may not
provide a clean uninstall. Use this method if installing with *Pip* is not possible:

    .. code:: bash

        python setup.py install


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

