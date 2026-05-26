from pathlib import Path

import numpy as np
import pandas as pd

TEST_DATA_PATH: Path = Path()/ "input" / "TestData_ZG.csv"

# Miyauchi ODE parameters
r0: float    = 9.26e-6
R_day: float = r0 * 86400
N_max: float = 1e7

# Avrami reference material (brick ZG)
P_ref: float = 0.2909
R_ref: float = 5.5

K_ref_init: float  = 2.19e-6
t1_ref_init: float = 0.94

# Quagliarini et al. (2021) polynomial coefficient matrices
A_coeff_matrix: np.ndarray = np.array([
    [-3.419,      9.2e-2,    -5.7e-3   ],
    [ 8.798e-1,  -3.1032e-2,  2.16e-3  ],
    [-3.98e-2,    2.8023e-3, -2.184e-4 ],
    [ 5e-4,      -5.21e-5,    4.198e-6 ],
])

K_coeff_matrix: np.ndarray = np.array([
    [-1.018137e-6,  2.638435e-5, -1.109344e-3],
    [ 3.832828e-7, -1.056712e-5,  2.729806e-4],
    [-3.978387e-8,  1.173803e-6, -1.081282e-5],
    [ 7.709831e-10,-2.315303e-8,  1.169671e-7],
])

t1_coeff_matrix: np.ndarray = np.array([4.73e-5, -2.88e-4, -2.66e-4])


def calculate_K_temperature(T, P: float, R: float) -> np.ndarray:
    T_array  = np.atleast_1d(np.asarray(T, dtype=float))
    T_vector = np.vstack([np.ones_like(T_array), T_array, T_array**2, T_array**3])
    P_R_vec  = np.array([P, P**3, R**-8])
    K_values = (K_coeff_matrix @ P_R_vec) @ T_vector
    K_values = np.maximum(1e-10, K_values)
    return float(K_values[0]) if np.isscalar(T) else K_values


def calculate_A_temperature(T, P: float, R: float) -> np.ndarray:
    T_array  = np.atleast_1d(np.asarray(T, dtype=float))
    T_vector = np.vstack([np.ones_like(T_array), T_array, T_array**2, T_array**3])
    P_R_vec  = np.array([P**2, R, R**2])
    A_values = (A_coeff_matrix @ P_R_vec) @ T_vector
    A_values = np.maximum(1e-10, A_values)
    return float(A_values[0]) if np.isscalar(T) else A_values


def calculate_t1(P: float, R: float) -> float:
    return float(t1_coeff_matrix @ np.array([P**-8, R, R**2]))


_T_cal      = np.linspace(0, 40, 401)
_A_cal      = calculate_A_temperature(_T_cal, P_ref, R_ref)
_valid_mask = (_T_cal >= 5) & (_T_cal <= 40)

A_max: float      = float(np.max(_A_cal[_valid_mask]))
T_at_A_max: float = float(_T_cal[_valid_mask][np.argmax(_A_cal[_valid_mask])])

t_half_init: float = t1_ref_init + (np.log(2) / K_ref_init) ** 0.25
N0: float  = N_max / (1.0 + np.exp(R_day * t_half_init))
Y_0: float = N0 / N_max


def load_climate_data(csv_path=None):
    if csv_path is None:
        csv_path = TEST_DATA_PATH
    df      = pd.read_csv(csv_path)
    T_data  = df["T [C]"].values
    RH_data = df["RH [%]"].values
    S_data  = df["Short_wave_global [W/m2]"].values
    hours   = np.arange(len(df))
    return T_data, RH_data, S_data, hours
