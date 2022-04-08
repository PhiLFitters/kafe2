"""
kafe2 example: Plot Customization
=================================

This example is a cheat sheet for plot/report customization.
It briefly demonstrates methods that modify the optics of kafe2 output.
"""

from kafe2 import XYContainer, Fit, Plot

# The same setup as in 001_line_fit/line_fit.py :
xy_data = XYContainer(x_data=[1.0, 2.0, 3.0, 4.0],
                      y_data=[2.3, 4.2, 7.5, 9.4])
xy_data.add_error(axis='x', err_val=0.1)
xy_data.add_error(axis='y', err_val=0.4)
line_fit = Fit(data=xy_data)
line_fit.do_fit()

# Non-LaTeX names are used in reports and other text-based output:
line_fit.assign_parameter_names(x='t', a='alpha', b='beta')
line_fit.assign_model_function_expression('theta')
line_fit.assign_model_function_expression("{a} * {x} + {b}")
# Note: the model function expression is formatted as a Python string.
#    The names of parameters in curly brackets will be replaced with the specified latex names.

# You could also just hard-code the parameter names like this:
# line_fit.assign_model_function_expression("alpha * t + beta")

line_fit.report()

# LaTeX names are used in plot info boxes:
line_fit.assign_parameter_latex_names(x='t', a='\\alpha', b='\\beta')
line_fit.assign_model_function_latex_name('\\theta')
line_fit.assign_model_function_latex_expression('{a} \\cdot {x} + {b}')

# Labels can be set for a fit.
# These labels are then used by all Plots created from said fit.
# If a Plot object also defines labels those labels override the fit labels.

# The labels displayed in the info box:
line_fit.data_container.label = "My data label"
line_fit.model_label = "My model label"

# The labels displayed on the x/y axes:
line_fit.data_container.axis_labels = ["My x axis", "My y axis"]

plot = Plot(fit_objects=line_fit)

# Plot objects can be modified with the customize method which sets matplotlib keywords.
# The first argument specifies the subplot for which to set keywords.
# The second argument specifies which keyword to set.
# The third argument is a list of values to set for the keyword for each fit managed
#     by the plot object.

plot.customize('data', 'label', ["My data label 2"])  # Overwrite data label in info box.
# plot.customize('data', 'label', [None])  # Hide data label in info box.
plot.customize('data', 'marker', ['o'])  # Set the data marker shape in the plot.
plot.customize('data', 'markersize', [5])  # Set the data marker size in the plot.
plot.customize('data', 'color', ['blue'])  # Set the data marker color in the plot.
plot.customize('data', 'ecolor', ['gray'])  # Set the data errorbar color in the plot.

plot.customize('model_line', 'label', ['My model label 2'])  # Set the model label in the info box.
# plot.customize('model_line', 'label', [None])  # Hide the model label in the info box.
plot.customize('model_line', 'color', ['lightgreen'])  # Set the model line color in the plot.
plot.customize('model_line', 'linestyle', ['-'])  # Set the style of the model line in the plot.
plot.customize('model_error_band', 'label', [r'$\pm 1 \sigma$'])  # Error band label in info box.
# plot.customize('model_error_band', 'label', [None])  # Hide error band label.
plot.customize('model_error_band', 'color', ['lightgreen'])  # Error band color in plot.
# plot.customize('model_error_band', 'hide', [True])  # Hide error band in plot and info box.

# Available keywords can be retrieved with Plot.get_keywords(subplot_type) .
# subplot_type is for example 'data', 'model_line', or 'model_error_band'.

# In addition to the customize method Plot has a few convenience methods for common operations:

plot.x_range = (0.8, 6)  # Set the x range of the plot.
plot.y_range = (1, 11)  # Set the y range of the plot.

plot.x_label = 'My x axis 2'  # Overwrite the label of the x axis.
plot.y_label = 'My y axis 2'  # Overwrite the label of the y axis.

plot.x_scale = 'log'  # Set the x axis to a logarithmic scale.
plot.y_scale = 'log'  # Set the y axis to a logarithmic scale.

# Finally, perform the plot:
plot.plot(ratio=True)
plot.show()
