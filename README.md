# Algae Growth Model

A Python implementation of three biofilm/algae growth models driven by hourly outdoor climate data (temperature, RH, solar irradiance).  
The primary model for use by collaborators is **`model_init.py`**.

---

## File Overview

| File | Description |
|------|-------------|
| `model_init.py` | **Main model** — run single or all growth models |
| `algae_model.py` | Material constants and Avrami calibration parameters |
| `response_functions.py` | Environmental response functions *g*(*T*), *g*(RH), *g*(*S*) |
| `02-Original_Avrami_Quagliarini.py` | Original Avrami model (Quagliarini et al., 2021) — reference |
| `03-Original_Exponential_Miyauchi.py` | Original exponential model (Miyauchi et al.) — reference |
| `input/TestData_ZG.csv` | Test climate dataset (hourly, brick ZG) |

---
## 📊 Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib
```

### 2. Configure run mode

Open `01-unified_model.py` and set the two variables near the top of the file:

```python
RUN_MODE   = "all"      # "all"  → run all three models
                        # "single" → run only the model specified below
MODEL_TYPE = "logistic" # "exponential" | "logistic" | "avrami"
```

### 3. Run

```bash
python 01-unified_model.py
```

Results are written to `output/`:

- `growth_results.csv` — time series of *Y*(*t*) [−] for each model
- `growth_curves.pdf` — figure of relative saturation vs. time

---

## Input Data

The default dataset is `input/TestData_ZG.csv`.  
To use your own climate data, replace the path in `model_init.py`:

```python
TEST_DATA_PATH: Path = Path(__file__).parent.parent / "input" / "TestData_ZG.csv"
```

The CSV must contain the following columns (hourly resolution):

| Column | Unit |
|--------|------|
| `T [C]` | °C |
| `RH [%]` | % |
| `Short_wave_global [W/m2]` | W/m² |

---

## Material Parameters

Material-specific constants are defined in `model_init.py`.  
The default values correspond to brick ZG (Quagliarini et al., 2021):

```python
P_ref: float = 0.2909   # porosity [-]
R_ref: float = 5.5      # surface roughness [μm]
```

These are passed to `run_model()` or `run_all_models()` via the `--porosity` and `--roughness` CLI flags, or directly as function arguments:

```python
from unified_model import run_all_models, load_climate_data

T, RH, S, hours = load_climate_data("input/TestData_ZG.csv")
results = run_all_models(T, RH, S, hours, P=0.30, R=6.0)
```

The Avrami rate constant *K*(*T*, *P*, *R*) and saturation limit *A*(*T*, *P*, *R*) are evaluated from the polynomial coefficient matrices in `model_init.py` (Quagliarini et al., 2021). The ODE growth rate *r*₀ for the exponential/logistic models is also set there:

```python
r0: float = 9.26e-6   # [1/s]
```

---

## Environmental Response Functions

The default response functions in `02-environmental_response_functions.py` are aligned versions based on published formulations:

| Function | Formulation | Reference |
|----------|------------|-----------|
| *g*(*T*) | CTMI (Cardinal Temperature Model with Inflection) | Rosso et al. (1993) |
| *g*(RH) | Linear ramp (94 %–98 %) | Nakajima et al, Xie et al, Quagliarini et al |
| *g*(*S*) | Monod with photoinhibition | P-I curves |

### Replacing a response function

All three solver functions (`solve_growth_exponential_aligned`, `solve_avrami_aligned`, `run_model`) accept optional keyword arguments to override any response function:

```python
from unified_model import run_model
import numpy as np

def my_gT(T):
    """Custom temperature response. Returns values in [0, 1]."""
    return np.clip((T - 5) / (30 - 5), 0.0, 1.0)

results = run_model(T, RH, S, hours, model_type="logistic",
                    g_T_func=my_gT)
```

The interface is identical for `g_RH_func` and `g_solar_func`.  
All custom functions must accept a NumPy array and return a NumPy array of the same shape with values in [0, 1].

---

## Models

### Unified model (`model_init.py`)

Three growth models share a common environmental driving function *G*(*t*) = *g*(*T*)·*g*(RH)·*g*(*S*):

| Model | Equation | Notes |
|-------|----------|-------|
| Exponential | d*Y*/d*t* = *r*₀ · *G* · *Y* | Euler integration |
| Logistic | d*Y*/d*t* = *r*₀ · *G* · *Y* · (1 − *Y*) | Euler integration; *new* |
| Avrami | *X*(*t*\*) = *A*(*T*,*P*,*R*) · [1 − exp(−*K*_eff · (*t*\* − *t*₁)⁴)] | Active-time clock alignment |

*Y* is the relative saturation [−]; *t*\* is active time (advances only when *G* > 0).

### Reference models

- `02-Original_Avrami_Quagliarini.py`
- `03-Original_Exponential_Miyauchi.py`

These scripts are provided for comparison and transparency.
