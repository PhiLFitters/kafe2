#!/usr/bin/env python3

"""
This file performs a benchmark of chi2 calculation using a QR decomposition (chi2, default)
versus chi2 calculation using a Cholesky decomposition (chi2_fast).
"""

import subprocess
from time import time, sleep
import numpy as np
from kafe2 import XYFit, plot


NUM_POINTS = [1585, 1000, 631, 398, 251, 158, 100]
NUM_SAMPLES = 10


def get_fit(cost_function):
    runtime = []
    for num_points in NUM_POINTS:
        runtime_np = []
        for _ in range(NUM_SAMPLES):
            sleep(1.0)
            t0 = time()
            subprocess.run(["python3", "qr_vs_cholesky_speed_run.py", cost_function, str(num_points)])
            runtime_np.append(time() - t0)
        print(f"Done: cost_function={cost_function} num_points={num_points}")
        runtime.append(runtime_np)
    runtime = np.array(runtime)

    runtime_mean = np.mean(runtime, axis=1)
    runtime_err = np.std(runtime, axis=1) / np.sqrt(NUM_SAMPLES)

    fit = XYFit([NUM_POINTS, runtime_mean], model_function="cubic")
    fit.add_error("y", runtime_err)
    fit.do_fit()

    return fit


fit_qr = get_fit("chi2")
fit_chol = get_fit("chi2_fast")
plot(
    [fit_qr, fit_chol],
    x_label="Num. datapoints",
    y_label="Runtime [s]",
    model_label=["QR", "Cholesky"],
    x_scale="log",
    y_scale="log",
    save=False
)
