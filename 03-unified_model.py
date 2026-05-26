from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .algae_model import (
        r0, P_ref, R_ref, A_max, Y_0,
        calculate_K_temperature, calculate_A_temperature, calculate_t1,
        load_climate_data, TEST_DATA_PATH,
    )
    from .response_functions import g_A_T, g_A_RH, g_A_solar
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from algae_model import (
        r0, P_ref, R_ref, A_max, Y_0,
        calculate_K_temperature, calculate_A_temperature, calculate_t1,
        load_climate_data, TEST_DATA_PATH,
    )
    from response_functions import g_A_T, g_A_RH, g_A_solar


# ── Run configuration ────────────────────────────────────────────────────────
# RUN_MODE   : "all"    → run all three models (exponential, logistic, Avrami)
#              "single" → run the model specified by MODEL_TYPE
# MODEL_TYPE : "exponential" | "logistic" | "avrami"  (only for RUN_MODE="single")
RUN_MODE   = "single"
MODEL_TYPE = "avrami"
# ─────────────────────────────────────────────────────────────────────────────


def solve_growth_exponential_aligned(
    model_type: str,
    T_data: np.ndarray,
    RH_data: np.ndarray,
    S_data: np.ndarray,
    hours: np.ndarray,
    Y_initial: float | None = None,
    g_T_func=None,
    g_RH_func=None,
    g_solar_func=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler-integrated ODE solver for exponential or logistic growth (Miyauchi framework).

    Inputs:
        model_type  : 'exponential' or 'logistic'
        T_data      : air temperature [°C]
        RH_data     : relative humidity [%]
        S_data      : solar irradiance [W/m²]
        hours       : integer hour indices [h]
        Y_initial   : initial relative saturation [-]  (default Y_0)
        g_T/RH/solar_func : override response functions (default: g_A_T, g_A_RH, g_A_solar)

    Outputs:
        Y_values  : relative saturation [-]
        R_values  : instantaneous growth rate [1/s]
        G_values  : environmental driving function [-]

    Assumptions:
        - dY/dt = R·Y (exponential) or R·Y·(1−Y) (logistic)
        - R(t) = r0·G(t); G = g_T·g_RH·g_solar
        - Euler forward integration; time step = 1 h
    """
    if Y_initial is None:
        Y_initial = Y_0

    _gT  = g_T_func     if g_T_func     is not None else g_A_T
    _gRH = g_RH_func    if g_RH_func    is not None else g_A_RH
    _gS  = g_solar_func if g_solar_func is not None else g_A_solar

    G_values = _gT(T_data) * _gRH(RH_data) * _gS(S_data)
    R_values = r0 * G_values

    n = len(hours)
    Y = np.zeros(n)
    Y[0] = Y_initial

    for i in range(1, n):
        dt = (hours[i] - hours[i - 1]) * 3600
        R  = R_values[i - 1]
        y  = Y[i - 1]

        if model_type == "exponential":
            dY_dt = R * y
        elif model_type == "logistic":
            dY_dt = R * y * (1 - y)
        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. Use 'exponential' or 'logistic'."
            )

        Y[i] = y + dY_dt * dt

    return Y, R_values, G_values


def solve_avrami_aligned(
    T_data: np.ndarray,
    RH_data: np.ndarray,
    S_data: np.ndarray,
    P: float = P_ref,
    R: float = R_ref,
    g_T_func=None,
    g_RH_func=None,
    g_solar_func=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Avrami growth model with active-time clock alignment.

    Inputs:
        T_data  : air temperature [°C]
        RH_data : relative humidity [%]
        S_data  : solar irradiance [W/m²]
        P       : material porosity [-]
        R       : surface roughness [μm]
        g_T/RH/solar_func : override response functions

    Outputs:
        Y_avrami : relative saturation Y = X/A_max [-]
        R_avrami : effective growth rate K_eff [1/day⁴]
        G_values : environmental driving function [-]

    Assumptions:
        - Active time t* advances only when G(t) > 0
        - X(t*) = A(T,P,R)·[1 − exp(−K_eff·(t*−t1)⁴)]
        - K_eff = K_material·G(t);  K_material = max_T K(T,P,R)
        - Growth is monotonically non-decreasing (X never rolled back)
    """
    _gT  = g_T_func     if g_T_func     is not None else g_A_T
    _gRH = g_RH_func    if g_RH_func    is not None else g_A_RH
    _gS  = g_solar_func if g_solar_func is not None else g_A_solar

    _T_cal     = np.linspace(0, 40, 401)
    K_material = float(np.max(calculate_K_temperature(_T_cal, P, R)))
    t1         = calculate_t1(P, R)

    n        = len(T_data)
    X        = np.zeros(n)
    R_avrami = np.zeros(n)
    G        = np.zeros(n)
    Timehour = 0

    for i in range(n):
        G[i]        = _gT(T_data[i]) * _gRH(RH_data[i]) * _gS(S_data[i])
        K_eff       = K_material * G[i]
        A_i         = calculate_A_temperature(T_data[i], P, R)
        R_avrami[i] = K_eff

        if G[i] > 0:
            Timehour += 1
            Timeday   = Timehour / 24

            if Timehour == 1:
                try:
                    growth_term = max(0.0, Timeday - t1)
                    X[i] = A_i * (1 - np.exp(-K_eff * growth_term ** 4))
                    X[i] = max(0.0, min(A_i, X[i]))
                except (ValueError, RuntimeWarning):
                    X[i] = 0.0
            else:
                if i > 0:
                    if X[i - 1] < A_i:
                        try:
                            ratio    = max(0.0, min(0.999, X[i - 1] / A_i))
                            log_term = np.log(1 - ratio)
                            if log_term >= 0:
                                new_time = t1
                            else:
                                if K_eff > 0:
                                    power_term = (-log_term / K_eff) ** 0.25
                                    new_time   = t1 + power_term
                                else:
                                    new_time = t1
                            growth_term = max(0.0, new_time + 1 / 24 - t1)
                            new_X = A_i * (1 - np.exp(-K_eff * growth_term ** 4))
                            X[i]  = max(X[i - 1], min(A_i, new_X))
                        except (ValueError, RuntimeWarning, ZeroDivisionError, OverflowError):
                            X[i] = X[i - 1]
                    else:
                        X[i] = X[i - 1]
                else:
                    X[i] = 0.0
        else:
            X[i] = X[i - 1] if i > 0 else 0.0

    Y_avrami = X / A_max
    return Y_avrami, R_avrami, G


def run_model(
    T_data: np.ndarray,
    RH_data: np.ndarray,
    S_data: np.ndarray,
    hours: np.ndarray,
    model_type: str = "logistic",
    P: float = P_ref,
    R: float = R_ref,
) -> dict:
    """
    Run a single growth model.

    Inputs:
        T_data, RH_data, S_data : hourly climate series [°C, %, W/m²]
        hours      : integer hour indices [h]
        model_type : 'exponential', 'logistic', or 'avrami'
        P          : material porosity [-]  (Avrami only)
        R          : surface roughness [μm] (Avrami only)

    Outputs (dict):
        model_type key : Y(t) [-]
        'G'            : environmental driving function [-]
        'days'         : time axis [days]
    """
    if model_type == "avrami":
        Y, _, G = solve_avrami_aligned(T_data, RH_data, S_data, P=P, R=R)
    elif model_type in ("exponential", "logistic"):
        Y, _, G = solve_growth_exponential_aligned(model_type, T_data, RH_data, S_data, hours)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Use 'exponential', 'logistic', or 'avrami'.")
    return {model_type: Y, "G": G, "days": hours / 24.0}


def run_all_models(
    T_data: np.ndarray,
    RH_data: np.ndarray,
    S_data: np.ndarray,
    hours: np.ndarray,
    P: float = P_ref,
    R: float = R_ref,
) -> dict:
    """
    Run all three models (exponential, logistic, Avrami) and return combined results.

    Inputs:
        T_data, RH_data, S_data : hourly climate series [°C, %, W/m²]
        hours : integer hour indices [h]
        P     : material porosity [-]
        R     : surface roughness [μm]

    Outputs (dict):
        'exponential', 'logistic', 'avrami' : Y(t) [-]
        'G'    : environmental driving function [-]
        'days' : time axis [days]
    """
    Y_exp, _, G = solve_growth_exponential_aligned("exponential", T_data, RH_data, S_data, hours)
    Y_log, _, _ = solve_growth_exponential_aligned("logistic",    T_data, RH_data, S_data, hours)
    Y_avr, _, _ = solve_avrami_aligned(T_data, RH_data, S_data, P=P, R=R)
    return {
        "exponential": Y_exp,
        "logistic":    Y_log,
        "avrami":      Y_avr,
        "G":           G,
        "days":        hours / 24.0,
    }


def save_results(results: dict, output_path: str | Path) -> None:
    """
    Save model time series to CSV.

    Inputs:
        results     : dict from run_all_models() or run_model()
        output_path : destination file path (.csv)

    Outputs:
        CSV with columns: time_days [days], Y_exponential, Y_logistic, Y_avrami [-]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _cols = {"exponential": "Y_exponential", "logistic": "Y_logistic", "avrami": "Y_avrami"}
    data  = {"time_days": results["days"]}
    for key, col in _cols.items():
        if key in results:
            data[col] = results[key]
    pd.DataFrame(data).to_csv(output_path, index=False)


def plot_growth_curves(
    results: dict,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot relative saturation Y(t) for all models present in results.

    Inputs:
        results     : dict from run_all_models() or run_model()
        output_path : save path (.pdf/.svg/.png); None = do not save

    Outputs:
        matplotlib Figure — Y [-] vs time [years]
    """
    years = results["days"] / 365.25

    _style = {
        "exponential": ("purple",     "solid",  "Exponential"),
        "logistic":    ("royalblue",  "solid",  "Logistic"),
        "avrami":      ("darkorange", "dashed", "Avrami"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, (color, ls, label) in _style.items():
        if key in results:
            ax.plot(years, results[key], color=color, lw=2.0, linestyle=ls, label=label)
    ax.axhline(1.0, color="gray", lw=1.0, ls=":", alpha=0.8, label="Saturation (Y = 1)")
    ax.set_xlabel("Time [years]", fontsize=12)
    ax.set_ylabel("Relative Saturation $Y$ [−]", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def _main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       default=str(TEST_DATA_PATH), metavar="CSV")
    parser.add_argument("--output-dir", default="output",            metavar="DIR")
    parser.add_argument("--porosity",   type=float, default=P_ref,   metavar="P")
    parser.add_argument("--roughness",  type=float, default=R_ref,   metavar="R")
    args = parser.parse_args()

    T_data, RH_data, S_data, hours = load_climate_data(args.data)
    if RUN_MODE == "single":
        results = run_model(T_data, RH_data, S_data, hours,
                            model_type=MODEL_TYPE, P=args.porosity, R=args.roughness)
    else:
        results = run_all_models(T_data, RH_data, S_data, hours,
                                 P=args.porosity, R=args.roughness)

    out_dir = Path(args.output_dir)
    save_results(results, out_dir / "growth_results.csv")
    plot_growth_curves(results, out_dir / "growth_curves.pdf")
    plt.show()


if __name__ == "__main__":
    _main()
