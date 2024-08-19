"""We had already seen that kafe2 allows the declaration of relative uncertainties which we want to
examine more closely in this example.
Adjustments with relative uncertainties suffer from the fact that the estimate of the parameter
values is distorted.
This is because measured values that fluctuate to smaller values have smaller uncertainties;
the uncertainties are correspondingly greater when the measured values fluctuate upwards.
If the random fluctuations were exactly the other way round, other uncertainties would be assigned.
It would be correct to relate the relative uncertainties to the true values, which we do not know.
Instead, the option reference='model' allows the uncertainties to be dependent on the model value
- still not completely correct, but much better.
"""

from kafe2 import XYContainer, Fit, Plot, ContoursProfiler

x = [19.8, 3.0, 5.1, 16.1, 8.2,  11.7, 6.2, 10.1]
y = [23.2, 3.2, 4.5, 19.9, 7.1, 12.5, 4.5, 7.2]
data = XYContainer(x_data=x, y_data=y)
data.add_error(axis='x', err_val=0.3)
data.axis_labels = ['x-label', 'y-label']

# create fit with relative data uncertainty
linear_fit1 = Fit(data, model_function='linear_model')
linear_fit1.add_error('y', 0.15, relative=True, reference='data')
linear_fit1.data_container.label = "data + relative data error"
linear_fit1.model_label = "linear model"
linear_fit1.do_fit()

# create fit with relative model uncertainty
linear_fit2 = Fit(data, model_function='linear_model')
linear_fit2.add_error('y', 0.15, relative=True, reference='model')
linear_fit2.data_container.label = "data + relative model error"
linear_fit2.model_label = "linear model"
linear_fit2.do_fit()

plot = Plot((linear_fit2, linear_fit1))  # first fit is shown on top of second fit
# assign colors to data...
plot.customize('data', 'marker', ('o', 'o'))
plot.customize('data', 'markersize', (5, 5))
plot.customize('data', 'color', ('red', 'grey'))
plot.customize('data', 'ecolor', ('red', 'grey'))
# ... and model
plot.customize('model_line', 'color', ('orange', 'mistyrose'))
plot.customize('model_error_band', 'label', (r'$\pm 1 \sigma$', r'$\pm 1 \sigma$'))
plot.customize('model_error_band', 'color', ('orange', 'mistyrose'))
plot.plot(pull=True)

cpf1 = ContoursProfiler(linear_fit1)
cpf1.plot_profiles_contours_matrix(show_grid_for='contours')

cpf2 = ContoursProfiler(linear_fit2)
cpf2.plot_profiles_contours_matrix(show_grid_for='contours')

plot.show()
