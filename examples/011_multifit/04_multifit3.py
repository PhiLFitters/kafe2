''' general example for fitting multiple distributions with kafe2
      - define models
      - set up data objects
      - set up fit objects
      - perform fit
      - show and save output
'''
# Imports  #
from kafe2 import XYContainer, Fit, MultiFit, Plot
import numpy as np, matplotlib.pyplot as plt

# -- define two model functions with common parameter p0
#      remark:
#       - p0 might be a material constant,
#         e.g. elastic modulus or specific resistance,
#       - g might be a geometry factor, like length and/or
#         diameter of a sample or a combination of both
#       - o might be a nuisance parapeter, e.g. an off-set from noise
#   Note that constraints on g1, g2 are needed, i.e. external
#    measurents, to give meaningful results

def model1(x, p0=1., g1=1., o1=0):
   return g1 * p0 * x + o1 

def model2(x, p0=1., g2=1., o2=0):
   return g2 * p0 * x + o2 

# Workflow #

# 1. set data 

#   data set 1
x1 = [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]
y1 = [ 1.29, 1.78, 3.32, 3.85,  5.27,  6.00, 7.07, 8.57, 8.95, 10.52 ]
e1 = 0.2
e1x = 0.15
c1 = 1.0
ec1 = 0.05

#   data set 2
x2 = [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10. ]
y2 = [ 0.76, 1.19, 1.71, 2.21, 2.58, 3.01, 3.67, 4.24, 4.69, 4.97 ]
e2 = 0.15
e2x = 0.15
c2 = 0.5
ec2 = 0.05

# 2. convert to kafe2 data structure and add uncertainties

xy_d1 = XYContainer(x1, y1)
xy_d1.add_error('y', e1)                    # independent errors y
xy_d1.add_error('x', e1x)                  # independent errors 
xy_d2 = XYContainer(x2, y2)
xy_d2.add_error('y', e2)                    # independent errors y
xy_d2.add_error('x', e2x)                  # independent errors y

# set meaningful names 
xy_d1.label = 'Beispieldaten (1)'
xy_d1.axis_labels = ['x', 'y (1)']
xy_d2.label = 'Beispieldaten (2)'
xy_d2.axis_labels = ['x', 'y(2) & f(x)']

# 3. create the Fit objects 
xyFit1 = Fit(xy_d1, model1)
xyFit2 = Fit(xy_d2, model2)
# set meaningful names for model
xyFit1.model_label = 'Lineares Modell'
xyFit2.model_label = 'Lineares Modell'
# add the parameter constraints
xyFit1.add_parameter_constraint(name='g1', value = c1 , uncertainty = ec1)
xyFit2.add_parameter_constraint(name='g2', value = c2 , uncertainty = ec2)

# combine the two fit objects to form a MultiFit
multiFit = MultiFit( fit_list=[xyFit1, xyFit2] )

# 4. perform the fit
multiFit.do_fit()

# 5. report fit results
multiFit.report()

# 6. create and draw plots
multiPlot = Plot(multiFit)
##multiPlot = Plot(multiFit, separate_figures=True)
multiPlot.plot(figsize=(13., 7.))

# 7. show or save plots #
##for i, fig in enumerate(multiPlot.figures):
##  fig.savefig("MultiFit-"+str(i)+".pdf")
plt.show()
