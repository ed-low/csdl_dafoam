import csdl_alpha as csdl
import numpy as np
# from VortexAD.utils.csdl_switch import switch_func

def switch_func(x, funcs_list, bounds_list, scale=10.):
    f_0, f_end = funcs_list[0], funcs_list[-1]
    x_0, x_end = bounds_list[0], bounds_list[-1]

    y = f_0*(0.5*csdl.tanh(scale*(x_0-x)) + 0.5)
    
    for i in range(len(bounds_list) - 1):
        f_i = funcs_list[i+1]

        x_l, x_h = bounds_list[i], bounds_list[i+1]

        y = y + f_i*(0.5*(csdl.tanh(scale*(x-x_l)) - csdl.tanh(scale*(x-x_h))))

    y = y + f_end*(0.5*csdl.tanh(scale*(x-x_end)) + 0.5)

    return y

def PG_correction(CL, CDi, M):
    beta = (1-M**2)**0.5
    CL_PG = CL/beta
    CDi_PG = CDi/beta

    return CL_PG, CDi_PG

def compute_static_margin(alpha_list, CL_list, CM_list):
    # SM = (NP-CG)/MAC
    # neutral point behind cg indicates stable flight
    dalpha = (alpha_list[1] - alpha_list[0]) * 1.
    dCL_dalpha = (CL_list[1]-CL_list[0])/dalpha
    dCM_dalpha = (CM_list[1]-CM_list[0])/dalpha

    SM = -dCM_dalpha/dCL_dalpha
    
    return SM

def estimate_fuel_burn(R, TSFC, V, L_D, W2):
    # TSFC input is in lb/(hr*lbf)
    # want to convert to kg/(s*N)
    TSFC_SI = TSFC * 0.45359237/60.**2/4.44822162
    g = 9.81
    C = csdl.exp(R*g*TSFC_SI/(V*L_D))
    W1 = W2*C
    Wf = W1 - W2
    return Wf, W1

def estimate_fuel_burn_w_reserve(R, TSFC, V, L_D, W_bar, res_fuel_frac=0.2):
    # TSFC input is in lb/(hr*lbf)
    # want to convert to kg/(s*N)
    rff = res_fuel_frac
    TSFC_SI = TSFC * 0.45359237/60.**2/4.44822162
    g = 9.81
    C = csdl.exp(R*g*TSFC_SI/(V*L_D))
    Wf = W_bar*(C-1)/(1+rff-C*rff)
    W1 = W_bar + (1+rff)*Wf
    W2 = W1-Wf
    return Wf, W1, W2

def estimate_fuel_volume(Wf_newton):
    fuel_density = 800 # kg/m^3
    Vf = Wf_newton/9.81/fuel_density # m^3 --> 1000L = 1m^3
    return Vf

def estimate_CDw(t_c, CL, M):
    kappa = 0.87
    # 0.87 for conventional airfoils, 0.95 for supercritical

    MDD = kappa - t_c - 0.1*CL
    Mcr = MDD - (0.1/80.)**(1/3)
    CDw = 20*(M-Mcr)**4
    return CDw

def estimate_sectional_CDw(t_c_for_each_strip : csdl.Variable, CL: csdl.Variable, 
                 M: csdl.Variable, sweep_for_each_strip: csdl.Variable, 
                 planform_area_strips: csdl.Variable, planform_area: csdl.Variable):
    kappa = 0.87
    # 0.87 for conventional airfoils, 0.95 for supercritical

    cos_sweep_for_each_strip = csdl.cos(sweep_for_each_strip)
    tech_component = kappa / cos_sweep_for_each_strip
    # thickness_component = t_c_for_each_strip / cos_sweep_for_each_strip**2
    # lift_component = CL / (10 * cos_sweep_for_each_strip**3)
    thickness_component = t_c_for_each_strip / cos_sweep_for_each_strip
    lift_component = CL / (10 * cos_sweep_for_each_strip)
    MDD = tech_component - thickness_component - lift_component
    Mcr = MDD - (0.1 / 80)**(1/3)
    activation_factor = 100.
    mach_violation = 1/activation_factor*csdl.softplus(activation_factor*(csdl.expand(M, Mcr.shape) - Mcr))
    CDw_for_each_strip = 20*mach_violation**4
    CDw = csdl.sum(CDw_for_each_strip*planform_area_strips/planform_area)
    return CDw, CDw_for_each_strip, mach_violation, Mcr, MDD, tech_component, thickness_component, lift_component



def estimate_Cf(Re,M):
    # Cf = 0.455/((csdl.log(Re,10))**2.58*(1+0.144*M**2)**0.65)
    # Cf = 0.074/Re**(1/5) # Prandtl (low Re)
    Cf = 0.0725/Re**(1/5) # Prandtl (high Re)
    # Cf = 0.455/csdl.log(Re)**(2.58) # Schlichting Compressible
    return Cf

def atmos_model(h):
    '''
    outputs density and speed of sound based on altitude
    input: h (in km)
    '''
    gamma = 1.4
    R = 287 # m^2/(s^2K)

    h_t = 11 # km
    T0 = 288.16 # K
    T1 = 216.65 # K
    L = 6.5 # K/km

    p0 = 101325 # Pa
    p1 = 22632 # Pa
    g = 9.81 # m/s^2

    T_funcs = [
        T0 - L*h,
        T1
    ]
    T_bounds = [h_t]
    T = switch_func(h, T_funcs, T_bounds, scale=100)
    
    p_funcs = [
        p0*(T/T0)**(g/(1e-3*L*R)),
        p1*csdl.exp(g*(h_t-h)/(R*T1))
    ]
    p_bounds = [h_t]
    p = switch_func(h, p_funcs, p_bounds, scale=100)
    
    rho = p/(R*T)
    a = (gamma*R*T)**0.5

    mu_0 = 1.716e-5 # Pa*s
    T_s = 273.15 # K
    C = 110.4 # K

    mu = mu_0*(T/T_s)**(3/2)*(T_s+C)/(T+C)
    return rho, a, mu


# These are functions to calculate takeoff and landing distances
# based on the equations from Anderson's "Introduction to Flight". 
def takeoff(W, # weight (N)
            T, # thrust (N)
            S, # wing area (m^2)
            rho=0.91, # air density (kg/m^3) 0.91 at 9000 ft
            g=9.81, # gravity (m/s^2)
            CL_max=1.5, # max lift coefficient
            ):
 
    # Anderson Intro to Flight equation 6.104
 
    S_LO = 1.44 * W**2 / (g * rho * S * CL_max * T)
 
    return S_LO
 
 
def landing(W, # weight (N)
            S, # wing area (m^2)
            V_T=70, # touchdown speed (m/s)
            CD_0=0.02, # zero-lift drag coefficient
            L=0, # lift (N) fair to assume lift is zero if spoilers are deployed
            rho=0.91, # air density (kg/m^3) 0.91 at 9000 ft
            g=9.81, # gravity (m/s^2)
            CL_max=1.5, # max lift coefficient
            mu_r=0.4, # runway friction coefficient with brakes
            ):
 
    # Anderson Intro to Flight equation 6.111
 
    D = 0.5 * rho * V_T**2 * S * CD_0 / 2
 
    denominator = g * rho * S * CL_max * (D + mu_r * (W - L))
 
    S_L = 1.69 * W**2 / denominator
 
    return S_L

def prop_weight(THRUST, # rated thrust per engine (lbf)
                NENG=2, # number of engines
                THRSO=48000, # rated thrust of each baseline engine (lbf)
                EEXP=1.15, # engine weight scaling parameter
                WENGB=8780, # reference engine weight (lb)
                ):
    
    # reference engine is a GE CF6-80A
    # 48000 lbf thrust
    # 8780 lb per engine

    if WENGB is None:
        WENGB = THRSO / 5.5 # transport
        # WENGB = THRSO / 8 # fighter
        # WENGB = THRSO / 10.5 # general aviation

    WENGP = WENGB * (THRUST / THRSO) ** EEXP # EEXP >= 0.3
    # WENGP = WENGB + (THRUST - THRSO) * EEXP # EEXP < 0.3

    WENG = NENG * WENGP


    # thrust reverser weight
    WTHR = 0.034 * THRUST * NENG # lbs

    # engine controls weight
    WEC = 0.26 * NENG * THRUST ** 0.5 # transport category (lbs)

    return WENG + WTHR + WEC # lbs


if __name__ == '__main__':
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    h_ft = csdl.Variable(value=np.array([18,24,30,36,38])) * 1e3
    h_km = h_ft/3280.84
    print(h_km.value)
    # h_km = 11

    rho, a, mu = atmos_model(h_km)
    print(rho.value)
    print(a.value)

    # approx. 737 numbers
    W = 90000 * 9.81 # N
    T = 133500 # N
    S = 130 # m^2
 
    S_lo = takeoff(W, T, S)
    S_lo_ft = S_lo * 3.28084 # convert to feet
    print(f"Takeoff distance: {S_lo:.2f} m")
    print(f"Takeoff distance: {S_lo_ft:.2f} ft")
 
    S_landing = landing(W, S)
    S_landing_ft = S_landing * 3.28084 # convert to feet
    print(f"Landing distance: {S_landing:.2f} m")
    print(f"Landing distance: {S_landing_ft:.2f} ft")

    # Example usage
    NENG = 2
    THRUST = 60000
    WENG = prop_weight(NENG, THRUST)
    print(f"Estimated propulsion weight: {WENG:.2f} lb")
