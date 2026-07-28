import numpy as np

# Temperature – CTMI
T_min_card: float = 0.0
T_opt_card: float = 22.0
T_max_card: float = 35.0

# Relative humidity – linear ramp
RH_min: float = 94.0
RH_sat: float = 98.0

# Solar irradiance – Monod with photoinhibition
I_c: float         = 7.5
PAR_to_global_ratio = 0.45
K_I_W: float       = 45.59/PAR_to_global_ratio
K_I_prime_W: float = 729.39/PAR_to_global_ratio

I_opt_W: float     = float(np.sqrt(K_I_W * K_I_prime_W))

# Blanchard temperature shape parameter
beta_blanch: float = 1.76


def _f_monod(I) -> np.ndarray:
    I_arr = np.atleast_1d(np.asarray(I, dtype=float))
    return I_arr / (K_I_W + I_arr + I_arr**2 / K_I_prime_W)


_f_opt  = _f_monod(I_opt_W)
_f_comp = _f_monod(I_c)


def g_A_T(T) -> float | np.ndarray:
    T_arr  = np.atleast_1d(np.asarray(T, dtype=float))
    result = np.zeros_like(T_arr)
    mask   = (T_arr > T_min_card) & (T_arr < T_max_card)
    Tv     = T_arr[mask]
    numer  = (Tv - T_max_card) * (Tv - T_min_card) ** 2
    bracket = (
        (T_opt_card - T_min_card) * (Tv - T_opt_card)
        - (T_opt_card - T_max_card) * (T_opt_card + T_min_card - 2 * Tv)
    )
    denom  = (T_opt_card - T_min_card) * bracket
    safe   = np.abs(denom) > 1e-15
    result[mask] = np.clip(
        np.where(safe, numer / np.where(safe, denom, 1.0), 0.0), 0.0, 1.0
    )
    return float(result[0]) if np.isscalar(T) else result


def g_CTMI(T) -> float | np.ndarray:
    return g_A_T(T)


def g_Blanchard(T) -> float | np.ndarray:
    T_arr  = np.atleast_1d(np.asarray(T, dtype=float))
    result = np.zeros_like(T_arr)
    mask   = T_arr < T_max_card
    Tv     = T_arr[mask]
    ratio  = (T_max_card - Tv) / (T_max_card - T_opt_card)
    result[mask] = ratio ** beta_blanch * np.exp(-beta_blanch * (ratio - 1))
    return float(result[0]) if np.isscalar(T) else result


def g_A_RH(RH) -> float | np.ndarray:
    RH_arr = np.atleast_1d(np.asarray(RH, dtype=float))
    result = np.clip((RH_arr - RH_min) / (RH_sat - RH_min), 0.0, 1.0)
    return float(result[0]) if np.isscalar(RH) else result


def g_A_solar(S) -> float | np.ndarray:
    S_arr  = np.atleast_1d(np.asarray(S, dtype=float))
    result = np.maximum(0.0, (_f_monod(S_arr) - _f_comp) / (_f_opt - _f_comp))
    return float(result[0]) if np.isscalar(S) else result


def calculate_G(T, RH, S) -> float | np.ndarray:
    return g_A_T(T) * g_A_RH(RH) * g_A_solar(S)
