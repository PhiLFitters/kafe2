"""
kafe2 example: Plot Customization
=================================

This example is a cheat sheet for plot/report customization.
It briefly demonstrates methods that modify the optics of kafe2 output.
"""

from kafe2 import XYContainer, XYFit, Plot


# A similar setup to 001_line_fit/line_fit.py
# This code will execute the fit and plot it without any customizations
# We do this to demonstrate the difference to the fully customized plot
def model(x, a, b):
    return a * x + b


xy_data = XYContainer(x_data=[1.0, 2.0, 3.0, 4.0], y_data=[2.3, 4.2, 7.5, 9.4])
xy_data.add_error(axis='x', err_val=0.1)
xy_data.add_error(axis='y', err_val=0.4)

fit = XYFit(xy_data=xy_data, model_function=model)
fit.do_fit()
fit.report()

plot = Plot(fit_objects=fit)
plot.plot(residual=True)
plot.show()

# Now we will look at how to customize a plot
# We create a new XYFit object which will be used for the customization:
customized_fit = XYFit(xy_data=xy_data, model_function=model)
customized_fit.do_fit()

# Non-LaTeX names are used in reports and other text-based output:
customized_fit.assign_parameter_names(x='t', a='alpha', b='beta')
customized_fit.assign_model_function_expression('theta')
customized_fit.assign_model_function_expression("{a} * {x} + {b}")
# Note: the model function expression is formatted as a Python string.
# The names of parameters in curly brackets will be replaced with the specified names.

customized_fit.report()

# Assign the LaTeX-names that are used in the plot info box (legend):
customized_fit.assign_parameter_latex_names(x='t', a='\\alpha', b='\\beta')
customized_fit.assign_model_function_latex_name('\\theta')
customized_fit.assign_model_function_latex_expression('{a} \\cdot {x} + {b}')

# Labels can be set for a fit.
# These labels are then used by all Plots created from said fit.
# If a Plot object also defines labels those labels override the fit labels.

# The labels displayed in the info box:
customized_fit.data_container.label = "data label"
customized_fit.model_label = "model label"

# The labels displayed on the x- and y-axis:
customized_fit.data_container.axis_labels = ["x label", "y label"]

# We also create a new Plot object to demonstrate the customizability:
customized_plot = Plot(fit_objects=customized_fit)

# Plot objects can be modified with the customize method which sets matplotlib.pyplot keywords.
# The first argument specifies the subplot for which to set keywords.
# The second argument specifies which keyword to set.
# The third argument is a list of values to set for the keyword for each fit managed
# by the plot object.
#
# Available keywords can be retrieved with Plot.get_keywords(subplot_type).
# subplot_type is for example 'data', 'model_line', or 'model_error_band'.
customized_plot.customize('data', 'marker', 'X')  # Set the data marker shape
customized_plot.customize('data', 'markersize', 10)  # Set the data marker size
customized_plot.customize('data', 'color', '#600E8F')  # Set the data marker color
customized_plot.customize('data', 'ecolor', '#8F0BDB')  # Set the errorbar color
customized_plot.customize('data', 'label', 'data label')  # Overwrite the data label in the info box

customized_plot.customize('model_line', 'linestyle', ':')  # Set the linestyle for the model line
customized_plot.customize('model_line', 'linewidth', 2)  # Set the line width for the model line
customized_plot.customize('model_line', 'color', '#8F1B0E')  # Set the color of the model line
customized_plot.customize('model_line', 'label',
                          'model line label')  # Overwrite the model label in the info box

customized_plot.customize('model_error_band', 'alpha',
                          0.2)  # Set the alpha value (transparency) for the error band
customized_plot.customize('model_error_band', 'linestyle',
                          '--')  # Set the linestyle for the border of the error band
customized_plot.customize('model_error_band', 'linewidth',
                          3)  # Set the linewidth for the border of the error band
customized_plot.customize('model_error_band', 'color', '#DB1F0B')  # Set the color of the error band
customized_plot.customize('model_error_band', 'label',
                          'model error band label')  # Set the label for the error band

# Analogous to data: set the appearance for the residual
customized_plot.customize('residual', 'marker', 'X')
customized_plot.customize('residual', 'markersize', 10)
customized_plot.customize('residual', 'color', '#600E8F')
customized_plot.customize('residual', 'ecolor', '#8F0BDB')

# Analogous to model error band: set the appearance for the residual error band
customized_plot.customize('residual_error_band', 'alpha', 0.2)
customized_plot.customize('residual_error_band', 'linestyle', '--')
customized_plot.customize('residual_error_band', 'linewidth', 3)
customized_plot.customize('residual_error_band', 'color', '#DB1F0B')
# Note: The labels accept None to hide the entry in the info box
# The subplots accept the 'hide' keyword to hide subplots from the plot

# In addition to the customize method, Plot has a few convenience methods for common operations:
customized_plot.x_range = (0.5, 4.5)  # Set the range of the x-axis
customized_plot.y_range = (2, 10)  # Set the range of the y-axis

customized_plot.x_label = 'x label'  # Overwrite the label of the x-axis
customized_plot.y_label = 'y label'  # Overwrite the label of the y-axis

# The options for the scale of an axis include linear and logarithmic:
customized_plot.x_scale = 'linear'  # Set the scale of the x-axis to linear (default)
customized_plot.y_scale = 'log'  # Set the scale of the y-axis to logarithmic

# Perform the customized plot:
customized_plot.plot(residual=True)
customized_plot.show()
