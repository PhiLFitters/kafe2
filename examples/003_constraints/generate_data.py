import numpy as np
import matplotlib.pyplot as plt

from kafe2 import XYContainer

err_val_x = 0.001
err_val_y = 0.01
num_datapoints = 121

l, delta_l = 10.0, np.random.randn() * 0.001
r, delta_r = 0.052, np.random.randn() * 0.001
g_e = 9.780  # gravitational pull at the equator
y_0, delta_y_0 = 0.6, np.random.randn() * 0.006  # 0.01 relative
c, delta_c = 0.01, np.random.randn() * 0.0005
x = np.linspace(start=0.0, stop=60.0, num=num_datapoints, endpoint=True)
delta_x = np.random.randn(num_datapoints) * err_val_x

print("T: %s" % (2.0 * np.pi * np.sqrt(l / g)))
print("M: %s" % (4/3 * np.pi * r ** 3 * 7874))


def damped_harmonic_oscillator(x, y_0, l, r, g, c):
    l_total = l + r
    omega_0 = np.sqrt(g / l_total)
    omega_d = np.sqrt(omega_0 ** 2 - c ** 2)
    return y_0 * np.exp(-c * x) * (np.cos(omega_d * x) + c / omega_d * np.sin(omega_d * x))


y = damped_harmonic_oscillator(
    x + delta_x,
    y_0 + delta_y_0,
    l + delta_l,
    r + delta_r,
    g_e,
    c + delta_c
)
y += np.random.randn(num_datapoints) * err_val_y

# Optional: plot the data
#plt.plot(_x, _y, '+')
#plt.show()

data = XYContainer(x_data=x, y_data=y)

data.add_simple_error(axis='x', err_val=err_val_x)
data.add_simple_error(axis='y', err_val=err_val_y)

data.to_file(filename='data.yml')
