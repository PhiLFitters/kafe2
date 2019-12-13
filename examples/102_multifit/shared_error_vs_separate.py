#!/usr/bin/env python
# -*- coding: utf8 -*-
r"""
Comparing shared errors with separate errors
--------------------------------------------

When creating a :py:object:`~kafe.fit.multi.Multifit` object for our problem there are two ways to
add errors: Errors can either be added to both fits individually or to both fits at once. Note that
these two procedures are **not** equivalent.

When you add an error to two fits individually you are adding two **different** errors. The error
of the first fit is uncorrelated with the error of the second fit.

When you add an error to both fits at once you are adding the **same** error to both fits. The error
of the first fit will be 100% correlated with the error of the second fit.

In our case we have to add x errors to both fits at once because we have measured I and T at the
same time. The error on U (the x error) affected I and T in exactly the same way. If we had measured
I and T independently of one another the error on U would not be (100%) correlated. In this case we
would have to add a different error to each fit.

To illustrate the importance of the above distinction this example will create two MultiFits: one
with a shared x error and one with two separate x errors. Because the error on U will be treated as
entirely uncorrelated the total x error will be overestimated by a factor of sqrt(2). Also, due to
nonlinear behavior introduced by x errors this will give us an incorrect estimate of the resistance
R.
"""

# TODO add code
