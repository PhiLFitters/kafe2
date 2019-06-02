import numpy as np
import matplotlib.pyplot as plt

DELTA_T_CALENDARS = 4857  # Difference between modern and ancient calendar in years
CURRENT_YEAR = 2019  # Current year according to the modern calendar
DAYS_PER_YEAR = 365.25  # Assumed number of days per year
N_A = 6.02214076e23  # Avogadro constant in 1/mol
MOLAR_MASS_C14 = 14.003241  # Molar mass of the Carbon-14 isotope in g/mol
T12_C14 = 5730  # Half-life of the Carbon-14 isotope in years, read as T_1/2
T12_C14_ERROR = 40  # Uncertainty over the actual half life of Carbon-14 in years
INITIAL_C14_CONCENTRATION = 1e-12  # Initial concentration of Carbon-14 at the time of death
INITIAL_C14_CONCENTRATION_ERROR = 0  # Uncertainty over initial Carbon-14 concentration
SAMPLE_MASS = 1.0  # Mass of the carbon samples examined in g
SAMPLE_MASS_ERROR = 0  # Uncertainty on the mass of the carbon samples in g, includes contamination
# x-values are the years the kings died and were mummified
X = np.array([19, 46, 82, 119, 147, 178, 196, 221, 251, 278, 308, 328, 372, 394, 425, 452, 489])

num_x_values = X.shape[0]
delta_t = CURRENT_YEAR + DELTA_T_CALENDARS
true_t12_c14 = T12_C14 + np.random.normal(loc=0.0, scale=T12_C14_ERROR)
true_initial_c14_concentrations = INITIAL_C14_CONCENTRATION + np.random.normal(
    loc=0.0, scale=INITIAL_C14_CONCENTRATION_ERROR, size=num_x_values)
true_sample_masses = SAMPLE_MASS + np.random.normal(loc=0.0, scale=SAMPLE_MASS_ERROR, size=num_x_values)

initial_num_atoms_carbon = N_A * true_sample_masses / MOLAR_MASS_C14
initial_num_atoms_c14 = initial_num_atoms_carbon * true_initial_c14_concentrations
current_num_atoms_c14 = initial_num_atoms_c14 * np.exp(-np.log(2) * (delta_t - X) / true_t12_c14)
current_activity_per_day = current_num_atoms_c14 * np.log(2) / (true_t12_c14 * DAYS_PER_YEAR)
measured_activity_per_day = np.random.poisson(lam=current_activity_per_day, size=num_x_values)
np.savetxt('measured_c14_activity.txt', [X, measured_activity_per_day])
