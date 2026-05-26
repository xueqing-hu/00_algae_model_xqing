"""
Algae Growth Model Web Application
藻类生长模型网页应用

This Streamlit app allows users to:
- Select from three growth models (Exponential, Logistic, Avrami)
- Upload custom climate data files
- Adjust material parameters (porosity, roughness)
- Visualize and download simulation results
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import streamlit as st

# Import model functions
try:
    from model_init import (
        r0, P_ref, R_ref, A_max, Y_0,
        calculate_K_temperature, calculate_A_temperature, calculate_t1,
        load_climate_data, TEST_DATA_PATH,
    )
    from response_functions import g_A_T, g_A_RH, g_A_solar
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model_init import (
        r0, P_ref, R_ref, A_max, Y_0,
        calculate_K_temperature, calculate_A_temperature, calculate_t1,
        load_climate_data, TEST_DATA_PATH,
    )
    from response_functions import g_A_T, g_A_RH, g_A_solar


# ============================================================================
# Model functions (imported from 01-unified_model.py)
# ============================================================================

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
    """Euler-integrated ODE solver for exponential or logistic growth."""
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
            raise ValueError(f"Unknown model_type: '{model_type}'")

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
    """Avrami growth model with active-time clock alignment."""
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


def run_models(
    T_data: np.ndarray,
    RH_data: np.ndarray,
    S_data: np.ndarray,
    hours: np.ndarray,
    selected_models: list[str],
    P: float = P_ref,
    R: float = R_ref,
) -> dict:
    """Run selected growth models."""
    results = {"days": hours / 24.0}
    
    for model in selected_models:
        if model == "avrami":
            Y, _, G = solve_avrami_aligned(T_data, RH_data, S_data, P=P, R=R)
            results[model] = Y
            results["G"] = G
        elif model in ("exponential", "logistic"):
            Y, _, G = solve_growth_exponential_aligned(model, T_data, RH_data, S_data, hours)
            results[model] = Y
            results["G"] = G
    
    return results


def plot_growth_curves(results: dict) -> plt.Figure:
    """Plot relative saturation Y(t) for all models."""
    years = results["days"] / 365.25

    _style = {
        "exponential": ("purple",     "solid",  "Exponential"),
        "logistic":    ("royalblue",  "solid",  "Logistic"),
        "avrami":      ("darkorange", "dashed", "Avrami"),
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, (color, ls, label) in _style.items():
        if key in results:
            ax.plot(years, results[key], color=color, lw=2.5, linestyle=ls, label=label)
    
    ax.axhline(1.0, color="gray", lw=1.0, ls=":", alpha=0.8, label="Saturation (Y = 1)")
    ax.set_xlabel("Time [years]", fontsize=13)
    ax.set_ylabel("Relative Saturation $Y$ [−]", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


# ============================================================================
# Streamlit App Configuration
# ============================================================================

st.set_page_config(
    page_title="Algae Growth Model",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# Main App
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🌿 Algae Growth Model Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">', unsafe_allow_html=True)
    
    # Sidebar - Model Configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        st.markdown("---")
        
        # Model Selection
        st.subheader("1. Select Models")
        model_options = {
            "Exponential (Miyauchi)": "exponential",
            "Logistic (Xueqing)": "logistic",
            "Avrami (Quagliarini)": "avrami"
        }
        
        selected_model_names = st.multiselect(
            "Choose model(s) to run:",
            options=list(model_options.keys()),
            default=["Avrami (Quagliarini)"]
        )
        selected_models = [model_options[name] for name in selected_model_names]
        
        st.markdown("---")
        
        # Material Parameters (for Avrami model)
        if "avrami" in selected_models:
            st.subheader("2. Material Parameters")
            st.caption("(For Avrami model only)")
            
            porosity = st.slider(
                "Porosity (P)",
                min_value=0.1,
                max_value=0.5,
                value=float(P_ref),
                step=0.01,
                format="%.3f",
                help="Material porosity [-]"
            )
            
            roughness = st.slider(
                "Surface Roughness (R) [μm]",
                min_value=1.0,
                max_value=15.0,
                value=float(R_ref),
                step=0.1,
                format="%.1f",
                help="Surface roughness [μm]"
            )
        else:
            porosity = P_ref
            roughness = R_ref
        
        st.markdown("---")
        
        # Data Upload
        st.subheader("3. Climate Data")
        use_example_data = st.checkbox("Use example data", value=True)
        
        if not use_example_data:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=['csv'],
                help="CSV must contain columns: 'T [C]', 'RH [%]', 'Short_wave_global [W/m2]'"
            )
        else:
            uploaded_file = None
        
        st.markdown("---")
        
        # Run Button
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Main Content Area
    if not selected_models:
        st.warning("⚠️ Please select at least one model from the sidebar.")
        return
    
    # Show model descriptions
    with st.expander("📖 Model Descriptions"):
        st.markdown("""
        ### Growth Models
        
        **1. Exponential Model (Miyauchi)**
        - Simple exponential growth: dY/dt = r₀·G(t)·Y
        - No saturation limit
        - Suitable for early-stage growth
        
        **2. Logistic Model (Miyauchi)**
        - Logistic growth with saturation: dY/dt = r₀·G(t)·Y·(1-Y)
        - Self-limiting growth approaching Y = 1
        - Classical S-shaped growth curve
        
        **3. Avrami Model (Quagliarini et al.)**
        - Active-time clock model: X(t*) = A(T,P,R)·[1 - exp(-K_eff·(t*-t1)⁴)]
        - Temperature and material-dependent
        - Accounts for porosity and surface roughness
        - Based on Quagliarini et al. (2021) formulation
        
        ### Environmental Response Functions
        - **Temperature (T)**: Optimal range 15-25°C
        - **Relative Humidity (RH)**: Growth increases with RH > 70%
        - **Solar Irradiance (S)**: Threshold-based activation
        """)
    
    # Run simulation
    if run_button:
        with st.spinner("Running simulation..."):
            try:
                # Load data
                if use_example_data or uploaded_file is None:
                    T_data, RH_data, S_data, hours = load_climate_data()
                    st.info("ℹ️ Using example climate data from TestData_ZG.csv")
                else:
                    # Save uploaded file temporarily
                    temp_file = Path("temp_upload.csv")
                    with open(temp_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    T_data, RH_data, S_data, hours = load_climate_data(temp_file)
                    temp_file.unlink()  # Delete temp file
                    st.success(f"✅ Loaded data from {uploaded_file.name}")
                
                # Run models
                results = run_models(
                    T_data, RH_data, S_data, hours,
                    selected_models=selected_models,
                    P=porosity,
                    R=roughness
                )
                
                # Display results
                st.success("✅ Simulation completed successfully!")
                
                # Create tabs for results
                tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Data Table", "📄 Summary"])
                
                with tab1:
                    st.subheader("Growth Curves")
                    fig = plot_growth_curves(results)
                    st.pyplot(fig)
                    
                    # Download plot
                    buf = io.BytesIO()
                    fig.savefig(buf, format='pdf', dpi=300, bbox_inches='tight')
                    buf.seek(0)
                    st.download_button(
                        label="📥 Download Plot (PDF)",
                        data=buf,
                        file_name="growth_curves.pdf",
                        mime="application/pdf"
                    )
                    plt.close(fig)
                
                with tab2:
                    st.subheader("Simulation Results")
                    
                    # Prepare DataFrame
                    df_results = pd.DataFrame()
                    df_results["Time [days]"] = results["days"]
                    df_results["Time [years]"] = results["days"] / 365.25
                    
                    for model_name in selected_model_names:
                        model_key = model_options[model_name]
                        if model_key in results:
                            df_results[f"Y_{model_key}"] = results[model_key]
                    
                    if "G" in results:
                        df_results["Environmental_Factor_G"] = results["G"]
                    
                    st.dataframe(df_results, use_container_width=True, height=400)
                    
                    # Download CSV
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="growth_results.csv",
                        mime="text/csv"
                    )
                
                with tab3:
                    st.subheader("Simulation Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Simulation Duration", f"{results['days'][-1]:.1f} days")
                        st.metric("Time Steps", f"{len(results['days'])} hours")
                        if "avrami" in selected_models:
                            st.metric("Porosity (P)", f"{porosity:.3f}")
                            st.metric("Roughness (R)", f"{roughness:.1f} μm")
                    
                    with col2:
                        for model_name in selected_model_names:
                            model_key = model_options[model_name]
                            if model_key in results:
                                final_Y = results[model_key][-1]
                                st.metric(f"{model_name} - Final Y", f"{final_Y:.4f}")
                    
                    # Display model parameters
                    with st.expander("Model Parameters"):
                        st.write(f"**r₀** (base growth rate): {r0:.6e} s⁻¹")
                        st.write(f"**Y₀** (initial saturation): {Y_0:.6f}")
                        st.write(f"**A_max** (max coverage): {A_max:.4f}")
                        if "avrami" in selected_models:
                            K_val = calculate_K_temperature(20, porosity, roughness)
                            t1_val = calculate_t1(porosity, roughness)
                            st.write(f"**K** (at 20°C, P={porosity:.3f}, R={roughness:.1f}): {K_val:.6e}")
                            st.write(f"**t₁** (lag time): {t1_val:.4f} days")
                
            except Exception as e:
                st.error(f"❌ Error during simulation: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888; font-size: 0.9rem;'>
            Developed by Xueqing Hu | UGent PhD Project<br>
            Based on Miyauchi et al. and Quagliarini et al. models
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
