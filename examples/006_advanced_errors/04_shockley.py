#!/usr/bin/env python3
# -*- coding: utf8 -*-
"""
Fit of Shockley equation to I-U characteristic of a diode
=========================================================

This is a practical example with a non-trivial covariance matrix
with independent and correlated relative uncertainties in the x- 
and y-direction. *kafe2* supports constructing the full covariance 
matrix with the method 

``add_error(err_val=?, axis=?, correlation=0, relative=False, reference='data')``,

which allows the user to specify the components of the uncertainties 
one after the other. The resulting individual covariance matrices 
are all added to form the full covariance matrix used in the fit. 

Here, we take as an example a typical digital amperemeter or voltmeter. 
Device characteristics are specified as 4000 Counts, +/-(0.5% + 3 digits),
where the calibration uncertainty is correlated among all measurements, 
while the digitisation uncertainties are independent. There often also
is an additional, independent noise component.

The code in this example shows how these uncertainty components for 
a set of voltage and current measurements with a diode are specified. 
Most of the code is needed to specify the uncertainties, the fit 
of the Shockley equation and the output of the results is very similar
to the other examples discussed already.
"""

from kafe2 import Fit, Plot, XYContainer
import numpy as np
import matplotlib.pyplot as plt


# model function to fit
def Shockley(U, I_s=0.5, U0=0.03):
    """Parametrisation of a diode characteristic
    U0 should be limited such that  U/U0<150 to avoid
    exceeding the 64 bit floating point range

    Args:
      - U: Voltage (V)
      - I_s: reverse saturation current (nA)
      - U0: thermal voltage (V) * emission coefficient
    
    Returns:
      - float I: diode current (mA)
  """
    return 1E-6 * I_s * np.exp((U / U0) - 1.)


# measurements:
# voltmeter characteristics: 
#  - voltage, measurement range 2V
voltages = [0.450, 0.470, 0.490, 0.510, 0.530,
            0.550, 0.560, 0.570, 0.580, 0.590, 0.600, 0.610, 0.620, 0.630,
            0.640, 0.645, 0.650, 0.655, 0.660, 0.665]
# - current: 2 measurements in range 200µA, 12 in range 20mA and 6 in range 200mA
currents = [0.056, 0.198,
            0.284, 0.404, 0.739, 1.739, 1.962, 2.849, 3.265, 5.706, 6.474, 7.866, 11.44, 18.98,
            23.35, 27.96, 38.61, 46.73, 49.78, 57.75]

# create a data container    
diode_data = XYContainer(x_data=voltages, y_data=currents)
diode_data.label = 'I vs. U'
diode_data.axis_labels = ['Voltage (V)', 'Current (mA)']

# --- calculate uncertainty components

#  - precision voltmeter: 4000 Counts, +/-(0.5% + 3 digits)
#     - range 2 V
crel_U = 0.005
Udigits = 3
Urange = 2
Ucounts = 4000
deltaU = Udigits * Urange / Ucounts
# - noise contribution delta U = 0.005 V
deltaU_noise = 0.005
# add voltage uncertainties to data object
diode_data.add_error(axis='x', err_val=deltaU)
diode_data.add_error(axis='x', err_val=deltaU_noise)
# note: relative uncertainties w.r.t. model to be added to fit object later

#  - precision amperemeter: 2000 Counts, +/-(1.0% + 3 digits) 
#     - measurement ranges 200µA, 20mA und 200mA 
crel_I = 0.010
Idigits = 3
Icounts = 2000
Irange1 = 0.2
Irange2 = 20
Irange3 = 200
deltaI = np.asarray(2 * [Idigits * Irange1 / Icounts] +
                    12 * [Idigits * Irange2 / Icounts] +
                    6 * [Idigits * Irange3 / Icounts])
#  noise contribution delta I = 0.050 mA
deltaI_noise = 0.050
# add current uncertainties to data object
diode_data.add_error(axis='y', err_val=deltaI)
diode_data.add_error(axis='y', err_val=deltaI_noise)
# note: relative uncertainties w.r.t. model to be added to fit object

# --- start of fit 

# create Fit object
ShockleyFit = Fit(diode_data, model_function=Shockley)
ShockleyFit.model_label = 'Shockley equation'

# add relative errors with reference to model
ShockleyFit.add_error(axis='x', err_val=crel_U, correlation=1.,
                      relative=True, reference='model')
ShockleyFit.add_error(axis='y', err_val=crel_I, correlation=1.,
                      relative=True, reference='model')
# to avoid overflow of 64 bit floats, limit U0
ShockleyFit.limit_parameter('U0', lower=0.005)

ShockleyFit.do_fit()

# create plot object
plotShockleyFit = Plot(ShockleyFit)
plotShockleyFit.plot(asymmetric_parameter_errors=True)

plt.show()
