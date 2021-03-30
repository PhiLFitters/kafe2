"""
kafe2 example: Accessing Fit Data via Properties
================================================

In the previous kafe2 examples we retrieved kafe2 results in a human-readable form via
reports and plots.
This example demonstrates how these fit results can instead be retrieved as Python variables.
"""

from kafe2 import XYContainer, Fit

# The same setup as in 001_line_fit/line_fit.py :
xy_data = XYContainer(x_data=[1.0, 2.0, 3.0, 4.0],
                      y_data=[2.3, 4.2, 7.5, 9.4])
xy_data.add_error(axis='x', err_val=0.1)
xy_data.add_error(axis='y', err_val=0.4)
line_fit = Fit(data=xy_data)

# First option: retrieve the fit results from the dictionary returned by do_fit.
result_dict = line_fit.do_fit()
# This dictionary contains the same information that would be shown in a report or plot.
# It can also be retrieved via
# result_dict = line_fit.get_result_dict()

# Print contents of result dict:
for key in result_dict:
    if "mat" in key:
        print("%s:" % key)
        print(result_dict[key])
    else:
        print("%s = %s" % (key, result_dict[key]))
    print()
# Note: the asymmetric parameter errors are None because computing them is relatively expensive.
# To calculate them run
# result_dict = line_fit.do_fit(asymmetric_parameter_errors=True)
# or
# result_dict = line_fit.get_result_dict(asymmetric_parameter_errors=True)

# A comparable output to above can be achieved by manually calling and printing fit properties:
print("============ Manual prints below ============")
print()
print("did_fit = %s\n" % line_fit.did_fit)
print("cost = %s\n" % line_fit.cost_function_value)
print("ndf = %s\n" % line_fit.ndf)
print("goodness_of_fit = %s\n" % line_fit.goodness_of_fit)
print("gof/ndf = %s\n" % (line_fit.goodness_of_fit / line_fit.ndf))
print("chi2_probability = %s\n" % line_fit.chi2_probability)
print("parameter_values = %s\n" % line_fit.parameter_values)
print("parameter_name_value_dict = %s\n" % line_fit.parameter_name_value_dict)
print("parameter_cov_mat:\n%s\n" % line_fit.parameter_cov_mat)
print("parameter_errors = %s\n" % line_fit.parameter_errors)
print("parameter_cor_mat:\n%s\n" % line_fit.parameter_cor_mat)
print("asymmetric_parameter_errors:\n%s\n" % line_fit.asymmetric_parameter_errors)
