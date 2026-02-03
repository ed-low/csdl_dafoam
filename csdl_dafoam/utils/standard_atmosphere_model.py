import csdl_alpha as csdl


def if_below_then_else(value, upper_bound, then_value, else_value, rho=1.):
    then_weight = 0.5 * (csdl.tanh((upper_bound - value) * rho) + 1)
    else_weight = 1 - then_weight
    return then_weight * then_value + else_weight * else_value


def compute_ambient_conditions_group(h_m):	
    # Constants
    T0_K            = 288.15    # Sea level standard temperature
    T1_K            = 216.65    # Temperature at the tropopause
    L_K_m           = 0.0065    # Temperature lapse rate in K/m
    h_t_m           = 11000.0   # Tropopause altitude in meters
    P0_Pa           = 101325.0  # Sea level standard pressure
    P1_Pa           = 22632.06  # Pressure at the tropopause
    g_m_s2          = 9.80665   # Standard acceleration due to gravity
    R_m2_s2_K       = 287.05    # Specific gas constant for dry air
    gamma           = 1.4       # Specific heat ratio of air
    mu_ref_kg_m_s   = 1.716e-5  # Reference viscosity
    T_ref_K         = 273.15    # Reference temperature for Sutherland's law
    S_K             = 110.4     # Sutherland's constant

    T_K = if_below_then_else(h_m, h_t_m, T0_K - L_K_m * h_m, T1_K)

    P_below_Pa  = P0_Pa * (T_K / T0_K) ** (g_m_s2 / (L_K_m * R_m2_s2_K))        # Pressure below tropopause
    P_above_Pa  = P1_Pa * csdl.exp(-g_m_s2 * (h_m - h_t_m) / (R_m2_s2_K * T_K)) # Pressure above tropopause
    P_Pa        = if_below_then_else(h_m, h_t_m, P_below_Pa, P_above_Pa)        # Minimum pressure

    rho_kg_m3 = P_Pa / R_m2_s2_K / T_K

    a_m_s = (gamma * R_m2_s2_K * T_K) ** 0.5

    mu_kg_m_s = mu_ref_kg_m_s * (T_K / T_ref_K) ** 1.5 * (T_ref_K + S_K) / (T_K + S_K)
    
    nu_m2_s = mu_kg_m_s / rho_kg_m3
    
    ambient_conditions_group            = csdl.VariableGroup()
    ambient_conditions_group.T_K        = T_K
    ambient_conditions_group.P_Pa       = P_Pa
    ambient_conditions_group.rho_kg_m3  = rho_kg_m3
    ambient_conditions_group.a_m_s      = a_m_s
    ambient_conditions_group.mu_kg_m_s  = mu_kg_m_s
    ambient_conditions_group.nu_m2_s    = nu_m2_s
    
    return ambient_conditions_group	


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    recorder = csdl.Recorder()
    recorder.start()

    h_km = csdl.Variable(value=3.)
    h_m = h_km * 1000.

    ambient_conditions_group = compute_ambient_conditions_group(h_m)

    recorder.stop()

    sim = csdl.experimental.PySimulator(recorder)

    num = 100
    h_km_range = np.linspace(0, 15., num)
    T_km_range = np.zeros(num)
    P_Pa_range = np.zeros(num)
    rho_kg_m3_range = np.zeros(num)
    a_m_s_range = np.zeros(num)
    mu_kg_m_s_range = np.zeros(num)
    nu_m2_s_range = np.zeros(num)
    for ind, h_km_value in enumerate(h_km_range):
        sim[h_km] = h_km_value
        sim.run()
        T_km_range[ind] = sim[ambient_conditions_group.T_K]
        P_Pa_range[ind] = sim[ambient_conditions_group.P_Pa]
        rho_kg_m3_range[ind] = sim[ambient_conditions_group.rho_kg_m3]
        a_m_s_range[ind] = sim[ambient_conditions_group.a_m_s]
        mu_kg_m_s_range[ind] = sim[ambient_conditions_group.mu_kg_m_s]
        nu_m2_s_range[ind] = sim[ambient_conditions_group.nu_m2_s]

    fig = plt.figure(figsize=(10, 7))

    num_rows = 3
    num_columns = 2
    
    fig.add_subplot(num_rows, num_columns, 1)
    plt.plot(h_km_range, T_km_range)
    plt.xlabel("Altitude (km)")
    plt.title("Temperature (K)")

    fig.add_subplot(num_rows, num_columns, 2)
    plt.plot(h_km_range, P_Pa_range)
    plt.xlabel("Altitude (km)")
    plt.title("Pressure (Pa)")

    fig.add_subplot(num_rows, num_columns, 3)
    plt.plot(h_km_range, rho_kg_m3_range)
    plt.xlabel("Altitude (km)")
    plt.title("Density (kg/m^3)")
    
    fig.add_subplot(num_rows, num_columns, 4)
    plt.plot(h_km_range, a_m_s_range)
    plt.xlabel("Altitude (km)")
    plt.title("Speed of sound (m/s)")

    fig.add_subplot(num_rows, num_columns, 5)
    plt.plot(h_km_range, mu_kg_m_s_range)
    plt.xlabel("Altitude (km)")
    plt.title("Dynamic viscosity (kg/m/s)")

    fig.add_subplot(num_rows, num_columns, 6)
    plt.plot(h_km_range, nu_m2_s_range)
    plt.xlabel("Altitude (km)")
    plt.title("Kinematic viscosity (m^2/s)")

    plt.tight_layout()
    plt.show()