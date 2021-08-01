import numpy as np
import matplotlib.pyplot as plt

SAMPLE_SIZE = 150

correlations = (-1.0, -0.8, 0.0, 0.8, 1.0)
plt.figure(figsize=(16.0, 12.0), dpi=200)

for i, correlation in enumerate(correlations):
    ax = plt.subplot(2, 3, i+1)
    ax.set_aspect('equal')
    plt.xlim(-3, +3)
    plt.ylim(-3, +3)
    plt.xlabel(r"$r_i$")
    plt.ylabel(r"$r_j$")
    plt.title(f"$\\rho_{{ij}} = {correlation:.1f}$")

    data = np.random.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1.0, correlation], [correlation, 1.0]],
        size=SAMPLE_SIZE
    )
    data_x = data[:, 0]
    data_y = data[:, 1]

    plt.plot(data_x, data_y, marker='.', linestyle='None')
plt.savefig("covariance_plot.png", bbox_inches="tight")
