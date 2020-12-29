.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow

.. role:: python(code)


***************
Developer Guide
***************

This developer guide provides information for developers who wish to modify *kafe2*.
It currently only covers tools needed for development.
A description of the software design will be added at some point in the future.

Tools
=====

This section covers software dependencies needed for development and how to use them.
All command line instructions below assume that the current working directory is the root of the
*kafe2* repository.

Requirements
------------

The following software should be installed on your machine:

* `git <https://git-scm.com/>`_
* `Python <https://www.python.org/>`_, both Python 2 *and* Python 3
* `ROOT <https://root.cern.ch/>`_

Additionally, the following Python packages should be installed:

* `NumPy <https://numpy.org/>`_
* `Scipy <https://www.scipy.org/>`_
* `iminuit <https://pypi.org/project/iminuit/>`_ <2
* `matplotlib <https://matplotlib.org/>`_
* `numdifftools <https://pypi.org/project/numdifftools/>`_
* `PyYaml <https://pyyaml.org/>`_
* `six <https://pypi.org/project/six/>`_
* `funcsigs <https://pypi.org/project/funcsigs/>`_
* `tabulate <https://pypi.org/project/tabulate/>`_
* `unittest2 <https://pypi.org/project/unittest2/>`_
* `coverage <https://pypi.org/project/coverage/>`_
* `Sphinx <https://pypi.org/project/Sphinx/>`_
* `Sphinx Bootstrap Theme <https://pypi.org/project/sphinx-bootstrap-theme/>`_
* `mock <https://pypi.org/project/mock/>`_

For support of newer versions of iminuit check out the current
`GitHub issue<https://github.com/dsavoiu/kafe2/issues/163>`_.

To install all of these packages automatically run::

    pip2 install -r dev_dependencies.txt
    pip3 install -r dev_dependencies.txt

Running Unit Tests
------------------

To run unit tests for both Python 2 and Python 3 run::

    python2 -m unittest discover -v -s kafe2/test
    python3 -m unittest discover -v -s kafe2/test

Determining Test Coverage
-------------------------

To determine the test coverage run::

    coverage run

This will run Python3 unit tests and keep track of the lines that were executed.
To print out a general report of the coverage run::

    coverage report

To get the lines of a specific file that were not tested run::

    coverage report -m path/to/file

For a graphical representation of the result an HTML document can be created::

    coverage html

Building the Documentation
--------------------------

To build the documentation run::

    cd doc
    make

For creating only the html or pdf documentation run ``make html`` or ``make latex``.
Cleaning the output directories can be done with ``make clean``.

Coding Style
============

In general the code of *kafe2* tries to follow the guidelines of
`PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_.
We've decided to use a maximum line length of 100 characters for all files.

But please, do not try to enforce this only for the sake of updating the style.
Only update the coding style if you're already performing other changes on a particular section.
There might be some sections in the source code which do not follow the general style guidelines,
this is okay, as long as the code is working and understandable.

Docstrings
----------

Documenting every method is a very good idea in general, so that a user or developer can quickly
and easily understand what a specific code block does and what it is used for.
Sphinx is used for creating the documentation, hence we use the
`Sphinx docstring format <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_.

Since we also support Python 2, we can't use Python's own `type hints <https://www.python
.org/dev/peps/pep-0484/>`_ introduced with Python 3.5.
Modern IDEs can also perform type checking with the help of docstrings.
That's why, in general, we try to include the types inside the docstrings.
