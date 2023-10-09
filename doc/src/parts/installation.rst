.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow

.. _installing_kafe2:

******************
Installing *kafe2*
******************

.. warning::

    *kafe2* versions 2.3.x are the latest versions which support Python 2.
    Python 2 support will be dropped for all future releases.

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
* `iminuit <https://github.com/iminuit/iminuit>`_ (>= 1.5.2), which is independent of ROOT


Installation notes (Linux)
==========================

The easiest way to install *kafe2* is via `pip <https://pip.pypa.io/en/stable/>`_, which is
already included in Python. Installing via *pip* will automatically install the minimal
dependencies. Please note that commands below could be run as root to install them system-wide
or as the current user, optionally with the parameter "--user", to install *kafe2* only locally.

For Python 3:

    .. code:: bash

        pip install kafe2


If you don't have *pip* installed, get it from the package manager.

In Ubuntu/Mint/Debian, do:

    .. code:: bash

        apt-get install python3-pip


In Fedora/RHEL/CentOS, do:

    .. code:: bash

        yum install python3-pip


or use ``easy_install`` (included with `setuptools <https://pypi.python.org/pypi/setuptools>`_):

    .. code:: bash

        easy_install pip


You will also need to install *Tkinter* if it didn't already come with your Python distribution.

For Python 3, Ubuntu/Mint/Debian:

    .. code:: bash

        apt-get install python3-tk


For Python 3, Fedora/RHEL/CentOS:

    .. code:: bash

        yum install python3-tkinter


Optional: Install *ROOT*
------------------------

.. warning::

   On recent *kafe2* versions *ROOT* is not functional as an optimization backend. Use *iminuit* instead.

**Note: Starting with Ubuntu 16.10, *ROOT* is no longer available in the official repositories.**

In older versions of Ubuntu (and related Linux distributions), ROOT and its Python bindings
can be obtained via the package manager via:

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
independent of *ROOT*. This minimizer can be used instead of *ROOT*.

To install the *iminuit* package for Python, the `Pip installer
<http://www.pip-installer.org/>`_ is recommended:

    .. code:: bash

        pip install iminuit

Note that the above command does **not** upgrade *iminuit* if it is already installed;
To do this add the "--upgrade" option to the above command.
*kafe2* officially supports *iminuit* 1.5.4 and the newest version of *iminuit* 2.
To install the legacy version of *iminuit*, do:

    .. code:: bash

        pip install iminuit==1.5.4

Note: the last version of *iminuit* that was usable in combination with *Python* 2.7
was *iminuit* 1.3.10.
The use of *Python* 2.7 in combination with *iminuit* is therefore not officially supported.
You might also need to install the Python headers for *iminuit* to
compile properly.

In Ubuntu/Mint/Debian, do:

    .. code:: bash

        apt-get install libpython3-dev

In Fedora/RHEL/CentOS, do:

    .. code:: bash

        yum install python3-devel


Installation notes (Windows)
============================

.. todo::

    Update and test this section

*kafe2* can be installed under Windows, but requires some additional configuration.

The recommended Python distribution for working with *kafe2* under Windows is
`WinPython <https://winpython.github.io/>`_, which has the advantage that it is
portable and comes with a number of useful pre-installed packages. Particularly,
*NumPy*, *SciPy* and *matplotlib* are all pre-installed in *WinPython*, as are
all *Qt*-related dependencies.


Install `iminuit`
-----------------

After installing *WinPython*, start 'WinPython Command Prompt.exe' in the
*WinPython* installation directory and run

.. code:: bash

    pip install 'iminuit'


Install `kafe2`
---------------

Now *kafe* can be installed from PyPI by running:

.. code:: bash

    pip install kafe2

Alternatively, it may be installed directly using *setuptools*. Just run
the following in 'WinPython Command Prompt.exe' after switching to the
directory into which you have downloaded *kafe2*:

.. code:: bash

    python -m build


Using *kafe* with ROOT under Windows
--------------------------------------

.. warning::

   On recent *kafe2* versions *ROOT* is not functional as an optimization backend. Use *iminuit* instead.

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
