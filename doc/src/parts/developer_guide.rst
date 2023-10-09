.. meta::
   :description lang=en: kafe2 - a Python-package for fitting parametric
                         models to several types of data with
   :robots: index, follow

.. role:: python(code)

.. _developer_guide:

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

In Addition to that a virtual environment for the development purpose can be created by calling::

  make devenv

This does not only create a virtual environment but also installs all the dependencies and
the *kafe2* package in editable mode.

Running Unit Tests
------------------

Running unit tests can now be done from the Makefile by calling::

    make test

This will run Python3 unit tests and use coverage to keep track of the lines that were executed.
To print out a general report of the coverage run::

    coverage report

To get the lines of a specific file that were not tested run::

    coverage report -m path/to/file

For a graphical representation of the result an HTML document can be created::

    coverage html

Building the Documentation
--------------------------

To build the documentation run::

    make docs

For creating only the html or pdf documentation go into the *doc/src/*
directory and run ``make html`` or ``make latex``.
Cleaning the output directories can be done with ``make clean``.

Coding Style
============

In general the code of *kafe2* tries to follow the guidelines of
`PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_.
We've decided to use a maximum line length of 150 characters for all files.

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
