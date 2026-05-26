"""
Original Exponential Algae Growth Model (Miyauchi's framework)
Nakajima et al. 2020, https://doi.org/10.1016/j.buildenv.2019.106575
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Model parameters ──────────────────────────────────────────────────────────
r0 = 4.63e-6   # Algae breeding rate [1/s]
G0 = 0.5       # Baseline solar coefficient [-]
N0 = 1         # Initial algal population density [cell/m²]

INPUT_FILE  = Path("input/TestData_ZG.csv")
OUTPUT_DIR  = Path("output")


# ── Temperature index g_T ─────────────────────────────────────────────────────
a_growth = 4 * np.log(0.5) / ((20 - 1) / 2) ** 2

# Decline phase [23, 40]: 4th-degree polynomial with 5 constraints
# f(23)=1, f'(23)=0  → a0=1, a1=0 (substituted out)
# f(30)=0, f(40)=-1, f'(40)=0  → solve for [a4, a3, a2]
_u30, _u40 = 7.0, 17.0  # T − 23 at T=30 and T=40
_A_dec = np.array([
    [_u30**4,   _u30**3,   _u30**2 ],   # f(7)  = 0
    [_u40**4,   _u40**3,   _u40**2 ],   # f(17) = −2  (a0=1 already subtracted)
    [4*_u40**3, 3*_u40**2, 2*_u40  ],   # f'(17)= 0
], dtype=float)
_c4, _c3, _c2 = np.linalg.solve(_A_dec, np.array([-1.0, -2.0, 0.0]))


def gT(T):
    T = np.asarray(T, dtype=float)
    scalar = T.ndim == 0
    if scalar:
        T = T[None]
    result = np.zeros_like(T)
    result[(T > 1)  & (T < 20)]  = np.exp(a_growth * (T[(T > 1) & (T < 20)] - 20) ** 2)
    result[(T >= 20) & (T <= 23)] = 1.0
    m = (T > 23) & (T < 40)
    u = T[m] - 23.0
    result[m] = _c4*u**4 + _c3*u**3 + _c2*u**2 + 1.0
    result[T >= 40] = -1.0
    return float(result[0]) if scalar else result


# ── Humidity index g_RH ───────────────────────────────────────────────────────

def gRH(RH):
    RH = np.asarray(RH, dtype=float)
    scalar = RH.ndim == 0
    if scalar:
        RH = RH[None]
    result = np.zeros_like(RH)
    m = (RH >= 90) & (RH <= 100)
    result[m] = (RH[m] - 90) / 10
    result[RH > 100] = 1.0
    return float(result[0]) if scalar else result


# ── Solar index g_S ───────────────────────────────────────────────────────────

def gS(S):
    S = np.asarray(S, dtype=float)
    scalar = S.ndim == 0
    if scalar:
        S = S[None]
    result = np.zeros_like(S)
    m = (S >= 5) & (S < 50)
    result[m] = (S[m] - 5) / 45
    result[(S >= 50) & (S < 600)] = 1.0
    m = (S >= 600) & (S <= 1000)
    result[m] = (1000 - S[m]) / 400
    return float(result[0]) if scalar else result


# ── Comprehensive environmental index G ───────────────────────────────────────
# G calculation rule:
#   T > 30 : G = g_T * g_RH * g_S
#   T ≤ 30 : G = g_T * g_RH * (G0 + (1 - G0) * g_S)

def calculate_G(T, RH, S):
    T_arr = np.asarray(T, dtype=float)
    gt    = gT(T)
    grh   = gRH(RH)
    gs    = gS(S)
    scalar = T_arr.ndim == 0
    if scalar:
        if T > 30:
            return gt * grh * gs
        else:
            return gt * grh * (G0 + (1 - G0) * gs)
    G  = gt * grh * (G0 + (1 - G0) * gs)
    m  = T_arr > 30
    G[m] = gt[m] * grh[m] * gs[m]
    return G


# ── Growth model solver ───────────────────────────────────────────────────────

def solve_algae_growth(T_data, RH_data, S_data, hours, N_initial=N0):
    """
    Inputs:
        T_data  : temperature [°C]
        RH_data : relative humidity [%]
        S_data  : solar irradiance [W/m²]
        hours   : integer hour indices [h]
        N_initial : initial population [cell/m²]

    Outputs:
        N_population : algal population density [cell/m²]
        G_values     : comprehensive environmental index [-]
        gt_values    : temperature index [-]
        grh_values   : humidity index [-]
        gs_values    : solar index [-]

    Assumptions:
        - dN/dt = r0 · G(t) · N;  Euler forward integration, Δt = 1 h
        - T > 30: G = g_T · g_RH · g_S
        - T ≤ 30: G = g_T · g_RH · (G0 + (1−G0)·g_S)
    """
    gt_values  = gT(T_data)
    grh_values = gRH(RH_data)
    gs_values  = gS(S_data)

    G_values = gt_values * grh_values * (G0 + (1 - G0) * gs_values)
    m = T_data > 30
    G_values[m] = gt_values[m] * grh_values[m] * gs_values[m]

    N = np.zeros(len(hours))
    N[0] = N_initial
    for i in range(1, len(hours)):
        dt   = (hours[i] - hours[i - 1]) * 3600
        N[i] = N[i - 1] + r0 * G_values[i - 1] * N[i - 1] * dt

    return N, G_values, gt_values, grh_values, gs_values


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_FILE)
T_data  = df["T [C]"].values
RH_data = df["RH [%]"].values
S_data  = df["Short_wave_global [W/m2]"].values
hours   = np.arange(len(df))


# ── Run simulation ────────────────────────────────────────────────────────────

N_population, G_values, gt_values, grh_values, gs_values = solve_algae_growth(
    T_data, RH_data, S_data, hours
)


# ── Plot results ──────────────────────────────────────────────────────────────

days = hours / 24

fig, axes = plt.subplots(5, 1, figsize=(14, 18))

axes[0].plot(days, N_population, "g-", lw=2.5)
axes[0].set_ylabel("Population [cell/m²]")
axes[0].set_title("Algal Population Growth")
axes[0].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
axes[0].grid(True, alpha=0.3)

ax2t = axes[1].twinx()
axes[1].plot(days, T_data,    "r-",       lw=2,   label="T [°C]")
ax2t.plot(   days, gt_values, color="darkred", lw=1, ls=":", label="$g_T$")
axes[1].set_ylabel("Temperature [°C]", color="r")
ax2t.set_ylabel("$g_T$ [-]", color="darkred")
ax2t.set_ylim(-1, 1.1);  axes[1].set_ylim(-20, 45)
axes[1].set_title("Temperature & $g_T$")
axes[1].grid(True, alpha=0.3)

ax3t = axes[2].twinx()
axes[2].plot(days, RH_data,    "b-",          lw=2,   label="RH [%]")
ax3t.plot(   days, grh_values, color="darkblue", lw=1, ls=":", label="$g_{RH}$")
axes[2].set_ylabel("RH [%]", color="b")
ax3t.set_ylabel("$g_{RH}$ [-]", color="darkblue")
ax3t.set_ylim(-0.1, 1.1)
axes[2].set_title("Humidity & $g_{RH}$")
axes[2].grid(True, alpha=0.3)

ax4t = axes[3].twinx()
axes[3].plot(days, S_data,    "orange",        lw=2,   label="S [W/m²]")
ax4t.plot(   days, gs_values, color="darkorange", lw=1, ls=":", label="$g_S$")
axes[3].set_ylabel("Solar [W/m²]", color="orange")
ax4t.set_ylabel("$g_S$ [-]", color="darkorange")
ax4t.set_ylim(-0.1, 1.1)
axes[3].set_title("Solar Radiation & $g_S$")
axes[3].grid(True, alpha=0.3)

axes[4].plot(days, G_values, "k-", lw=2)
axes[4].set_ylabel("G [-]")
axes[4].set_xlabel("Time [days]")
axes[4].set_title("Comprehensive Environmental Index $G$")
axes[4].set_ylim(-1, 1.1)
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_DIR / "Miyauchi_AlgaeModel_Results.pdf", dpi=300, bbox_inches="tight")
plt.show()


# ── Save results ──────────────────────────────────────────────────────────────

pd.DataFrame({
    "Hour [h]":                  hours,
    "Temperature_Index_gT":      gt_values,
    "Humidity_Index_gRH":        grh_values,
    "Solar_Index_gS":            gs_values,
    "Comprehensive_Index_G":     G_values,
    "Algal_Population [cell/m2]": N_population,
}).to_csv(OUTPUT_DIR / "Miyauchi_AlgaeModel_Results.csv", index=False)
