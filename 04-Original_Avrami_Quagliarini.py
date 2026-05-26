# Original model by Quagliarini E., et al. 2021, doi: 10.1016/j.jobe.2021.102965" A modified Avrami aglae model
# Copyright (c) 2025 Xueqing Hu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Defalut values of P (porosity, default value = 0.2909) and R (surface roughness, default value = 5.5) employ the mean values from Quagliarini's experiments
# Please reivse values of P and R if material properties are available
def calculate_algae_growth(csv_file="TestData_ZG.csv", P=0.2909, R=5.5):
    """Main function to calculate algae growth based on temperature and RH."""
    # Load data
    trh_data = pd.read_csv(csv_file)
    Hours = len(trh_data)
    Days = Hours / 24
    
    # Read T and RH values directly from dataframe
    T_values = trh_data['T [C]'].values
    RH_values = trh_data['RH [%]'].values
    
    # Initialize arrays
    Xt_values = np.zeros(Hours)
    A_values = np.zeros(Hours)
    K_values = np.zeros(Hours)
    effective_time_values = np.zeros(Hours)
    
    # Define coefficient matrices
    A_coeff_matrix = np.array([
        [-3.419, 9.2e-2, -5.7e-3],
        [8.798e-1, -3.1032e-2, 2.16e-3],
        [-3.98e-2, 2.8023e-3, -2.184e-4],
        [5e-4, -5.21e-5, 4.198e-6]
    ])
    K_coeff_matrix = np.array([
        [-1.018137e-6, 2.638435e-5, -1.109344e-3],
        [3.832828e-7, -1.056712e-5, 2.729806e-4],
        [-3.978387e-8, 1.173803e-6, -1.081282e-5],
        [7.709831e-10, -2.315303e-8, 1.169671e-7]
    ])
    t1_coeff_matrix = np.array([4.73e-5, -2.88e-4, -2.66e-4])
    
    # Pre-calculate constant vectors
    P_R_A_vector = np.array([P**2, R, R**2])
    P_R_K_vector = np.array([P, P**3, R**-8])
    P_R_t1_vector = np.array([P**-8, R, R**2])
    
    # Calculate latency period
    t1 = t1_coeff_matrix @ P_R_t1_vector
    print(f"Latency period (t1): {t1:.2f} days")
    
    # Initialize tracking variables
    time_shift = 0
    Timehour = 0
    last_effective_time = 0
    
    # Simulate growth hour by hour
    for i in range(Hours):
        T = T_values[i]
        RH = RH_values[i]
        
        # Check if conditions support algae growth
        if RH >= 98 and 5 <= T <= 40:
            Timehour += 1
            Timeday = Timehour / 24
            
            # Calculate A and K values with safeguards
            T_vector = np.array([1, T, T**2, T**3])
            A_intermediate = A_coeff_matrix @ P_R_A_vector
            A_values[i] = max(1e-10, np.dot(A_intermediate, T_vector))  # Ensure A is positive
            
            K_intermediate = K_coeff_matrix @ P_R_K_vector
            K_values[i] = max(1e-10, np.dot(K_intermediate, T_vector))  # Ensure K is positive
            
            # Record effective time for this timestep
            effective_time_values[i] = Timeday
            
            if Timehour == 1:
                # First hour with growth conditions
                try:
                    growth_term = max(0, Timeday - t1)
                    Xt_values[i] = A_values[i] * (1 - np.exp(-1 * K_values[i] * (growth_term)**4))
                    Xt_values[i] = max(0, min(A_values[i], Xt_values[i]))  # Ensure in valid range
                except (ValueError, RuntimeWarning) as e:
                    print(f"Warning at hour {i}: {e}")
                    Xt_values[i] = 0
            else:
                # Subsequent hours - check if growth is possible
                if i > 0:
                    if Xt_values[i-1] < A_values[i]:
                        try:
                            # Calculate the "equivalent time" based on current coverage
                            ratio = max(0, min(0.999, Xt_values[i-1]/A_values[i]))
                            log_term = np.log(1 - ratio)
                            if log_term >= 0:  # Safety check
                                new_time = t1  # Default to latency period
                            else:
                                power_term = (-log_term / K_values[i])**0.25
                                new_time = t1 + power_term
                                
                            # Calculate new coverage after one more hour
                            growth_term = max(0, new_time + 1/24 - t1)
                            new_Xt = A_values[i] * (1 - np.exp(-1 * K_values[i] * growth_term**4))
                            Xt_values[i] = max(Xt_values[i-1], min(A_values[i], new_Xt))
                        except (ValueError, RuntimeWarning, ZeroDivisionError, OverflowError) as e:
                            print(f"Warning at hour {i}: {e}")
                            Xt_values[i] = Xt_values[i-1]  # Keep previous value on error
                    else:
                        # Current coverage already at or above asymptotic value
                        Xt_values[i] = Xt_values[i-1]
        else:
            # Conditions don't support algae growth
            A_values[i] = 0
            K_values[i] = 0
            # Maintain previous coverage when conditions don't support growth
            Xt_values[i] = Xt_values[i-1] if i > 0 else 0
            effective_time_values[i] = effective_time_values[i-1] if i > 0 else 0
    
    # Print summary stats
    print(f"Max A value: {np.max(A_values):.4f}")
    print(f"Max K value: {np.max(K_values):.6f}")
    print(f"Final algae coverage: {Xt_values[-1]:.4f}")
    
    # Plot results
    plot_results(Hours, Days, time_days=np.linspace(0, Days, Hours),
                 T_values=T_values, RH_values=RH_values, Xt_values=Xt_values, 
                 P=P, R=R)
    
    # Save results
    save_results(Hours, Days, T_values, RH_values, Xt_values, A_values, K_values, effective_time_values)
    
    return Xt_values, A_values, K_values

def plot_results(Hours, Days, time_days, T_values, RH_values, Xt_values, P, R):
    """Plot temperature, humidity and algae growth results."""
    plt.figure(figsize=(12, 10))
    
    # Temperature plot
    plt.subplot(3, 1, 1)
    plt.plot(time_days, T_values, color='red')
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature and RH Conditions")
    plt.grid(True)
    
    # RH plot
    plt.subplot(3, 1, 2)
    plt.plot(time_days, RH_values, color='blue')
    plt.ylabel("Relative Humidity (%)")
    plt.grid(True)
    
    # Algae growth plot
    plt.subplot(3, 1, 3)
    plt.plot(time_days, Xt_values, label="X(t)", color='green')
    plt.xlabel("Days")
    plt.ylabel("Algae Coverage")
    plt.title(f"Algae Growth Model (P={P}, R={R})")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def save_results(Hours, Days, T_values, RH_values, Xt_values, A_values, K_values, effective_time_values):
    """Save simulation results to CSV file."""
    time_days = np.linspace(0, Days, Hours)
    results_df = pd.DataFrame({
        'Day': time_days,
        'Algae_Coverage': Xt_values,
        'A_value': A_values,
        'K_value': K_values,
        'Effective_Time': effective_time_values
    })
    results_df.to_csv('algae_Avrami.csv', index=False)
    print("Results saved to algae_Avrami.csv")

if __name__ == "__main__":
    calculate_algae_growth()