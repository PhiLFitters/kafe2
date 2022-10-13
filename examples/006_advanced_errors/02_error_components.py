"""
Typically, the uncertainties of the measurement data are much more complex than in the examples
discussed so far. In most cases there are uncertainties in ordinate and abscissa, and in addition to
the independent uncertainties of each data point there are common, correlated uncertainties for all
of them.

With the method add_error() or add_matrix_error() uncertainties can be specified on the 'x' and 'y'
data, either in the form of independent or correlated, relative or absolute uncertainties of all or
groups of measured values or by specifying the complete covariance or correlation matrix. All
uncertainties specified in this way are included in the global covariance matrix for the fit.

As an example, we consider measurements of a cross section as a function of the energy near a
resonance. These are combined measurement data from the four experiments at CERN's LEP accelerator,
which were corrected for effects caused by photon radiation: Measurements of the hadronic cross
section (sigma) as a function of the centre-of-mass energy (E).
"""

from kafe2 import XYContainer, Fit, Plot, ContoursProfiler

# Center-of-mass energy E (GeV):
E = [88.387, 89.437, 90.223, 91.238, 92.059, 93.004, 93.916]  # x data
E_errors = [0.005, 0.0015, 0.005, 0.003, 0.005, 0.0015, 0.005]  # Uncorrelated absolute x errors
ECor_abs = 0.0017  # Correlated absolute x error

# Hadronic cross section with photonic corrections applied (nb):
sig = [6.803, 13.965, 26.113, 41.364, 27.535, 13.362, 7.302]  # y data
sig_errors = [0.036, 0.013, 0.075, 0.010, 0.088, 0.015, 0.045]  # Uncorrelated absolute y errors
sigCor_rel = 0.0007  # Correlated relative y error


# Breit-Wigner with s-dependent width:
def BreitWigner(E, s0=41.0, M_Z=91.2, G_Z=2.5):
    s = E*E
    Msq = M_Z*M_Z
    Gsq = G_Z*G_Z
    return s0*s*Gsq/((s-Msq)*(s-Msq)+(s*s*Gsq/Msq))


BW_data = XYContainer(E, sig)  # Create data container.

# Add errors to data container.
# By default errors are assumed to be absolute and uncorrelated.
# For errors that are relative and/or correlated you need to set the corresponding kwargs.

# Add independent errors:
error_name_sig = BW_data.add_error(axis='x', name='deltaSig', err_val=E_errors)
error_name_E = BW_data.add_error(axis='y', name='deltaE', err_val=sig_errors)

# Add fully correlated, absolute Energy errors:
error_name_ECor = BW_data.add_error(axis='x', name='Ecor', err_val=ECor_abs, correlation=1.)

# Add fully correlated, relative cross section errors:
error_name_sigCor = BW_data.add_error(
    axis='y', name='sigCor', err_val=sigCor_rel, correlation=1., relative=True)

# Note: kafe2 methods that add errors return a name for the added error. If no name is specified
# a random alphanumeric string is assigned automatically. Further down we will use these names to
# enable/disable some of the errors.

# Assign labels for the data and the axes:
BW_data.label = 'QED-corrected hadronic cross-sections'
BW_data.axis_labels = ('CM Energy (GeV)', r'$\sigma_h$ (nb)')
# Note: Because kafe2 containers are copied when a Fit object is created from them assigning labels
# to the original XYContainer after the fit has already been created would NOT work.

BW_fit = Fit(
    BW_data,
    "BreitWigner: x s0 M_Z G_Z -> s0 * x^2 * G_Z^2 / ((x^2 - M_Z^2)^2 + x^4 * G_Z^2 / M_Z^2)"
)

# Uncomment the following two lines to assign data labels after the fit has already been created:
# BW_fit.data_container.label = 'QED-corrected hadronic cross-sections'
# BW_fit.data_container.axis_labels = ('CM Energy (GeV)', r'$\sigma_h$ (nb)')

# Model labels always have to be assigned after the fit has been created:
BW_fit.model_label = 'Beit-Wigner with s-dependent width'

# Set LaTeX names for printout in info-box:
BW_fit.assign_parameter_latex_names(x='E', s0=r'\sigma^0')

# Do the fit:
BW_fit.do_fit()

# Print a report:
BW_fit.report(asymmetric_parameter_errors=True)

# Plot the fit results:
BW_plot = Plot(BW_fit)
BW_plot.y_range = (0, 1.03*max(sig))  # Explicitly set y_range to start at 0.
BW_plot.plot(residual=True, asymmetric_parameter_errors=True)

# Create a contour plot:
ContoursProfiler(BW_fit).plot_profiles_contours_matrix(show_grid_for='contours')

# Investigate the effects of individual error components: disabling the correlated uncertainty on
# energy should decrease the uncertainty of the mass M but have little to no effect otherwise.
print('====== Disabling error component %s ======' % error_name_ECor)
print()
BW_fit.disable_error(error_name_ECor)
BW_fit.do_fit()
BW_fit.report(show_data=False, show_model=False)

BW_plot.show()
