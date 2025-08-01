# ===============================
# region PACKAGES
# ===============================
import numpy as np
import sys
import os
import time
import pickle
from pathlib import Path

# CSDL packages
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo

# BWB specific
from bwb_helper_functions import setup_geometry, read_geometry_pickle, write_geometry_pickle, gather_array_to_rank0, read_simple_pickle, write_simple_pickle

# Plotting
from vedo import Points, show, Plotter
import matplotlib.pyplot as plt


#---- DEBUGGING TOOLS ----
import faulthandler
faulthandler.enable()
os.environ["PETSC_OPTIONS"] = "-malloc_debug"
#-------------------------



# ===============================
# region USER INPUT
# ===============================
# Geometry
geometry_directory        =  os.path.join(os.getcwd(), 'bwb_geometry/')
stp_file_name             = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
geometry_pickle_file_name = 'bwb_stored_refit.pickle'

# Timing
TIMING_ENABLED = True  # True if we want timing printed for the CSDL operations




# ===============================
# region HELPER FUNCTIONS
# ===============================
# TIMER
from contextlib import contextmanager
# Use this to print the timings for certain lines
timings = {}  # Optional: for logging total times

@contextmanager
def Timer(name):
    if TIMING_ENABLED:
        print(f'{name}...', flush=True)
        start = time.time()
        yield
        elapsed = time.time() - start
        print(f'{name} elapsed time: {elapsed:.3f} s')
        timings[name] = elapsed
    else:
        yield



# ===============================
# region SETUP
# ===============================
# region File paths
geometry_pickle_file_path         = Path(geometry_directory)/geometry_pickle_file_name
stp_file_path                     = Path(geometry_directory)/stp_file_name



# ===============================
# region CSDL RECORDER
# ===============================
# recorder 
recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()


#region Geometry setup I
if geometry_pickle_file_path.is_file():
    with Timer(f'reading geometry'):
        geometry = read_geometry_pickle(geometry_pickle_file_path)
        
else:
    print('No geometry pickle file found.')
    with Timer('importing geometry'):
        geometry = lsdo_geo.import_geometry(stp_file_path,
                                            parallelize=False)

    # These are hardcoded indices?
    oml_indices                 = [key for key in geometry.functions.keys()]
    wing_c_indices              = [0,1,8,9]
    wing_r_transition_indices   = [2,3]
    wing_r_indices              = [4,5,6,7]
    wing_l_transition_indices   = [10,11]
    wing_l_indices              = [12,13,14,15] 

    with Timer('declaring geometry components'):
        left_wing_transition    = geometry.declare_component(wing_l_transition_indices)
        left_wing               = geometry.declare_component(wing_l_indices)
        right_wing_transition   = geometry.declare_component(wing_r_transition_indices)
        right_wing              = geometry.declare_component(wing_r_indices)
        center_wing             = geometry.declare_component(wing_c_indices)
        oml = geometry.declare_component(oml_indices)

    wing_parameterization   = 15
    num_v                   = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]
    
    with Timer('BSplineSpace'):
        wing_refit_bspline      = lfs.BSplineSpace(num_parametric_dimensions=2, degree=1, coefficients_shape=(wing_parameterization, num_v))

    with Timer('left wing refit'):
        left_wing_function_set  = left_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))

    with Timer('right wing refit'):
        right_wing_function_set = right_wing.refit(wing_refit_bspline, grid_resolution=(100,1000))

    with Timer('allocating left wing functions'):
        for i, function in left_wing_function_set.functions.items():
            geometry.functions[i]   = function
            left_wing.functions[i]  = function

    with Timer('allocating right wing functions'):
        for i, function in right_wing_function_set.functions.items():
            geometry.functions[i]   = function
            right_wing.functions[i] = function

    with Timer('pickling geometry'):
        write_geometry_pickle(geometry, geometry_pickle_file_path)  


original_geo = geometry.plot(show=False, opacity=0.5)

# region Design variables
# ============================ Design variables ===========================
root_twist  = csdl.Variable(shape=(1,), value=np.array([0*np.pi/180.]))
tip_twist   = csdl.Variable(shape=(1,), value=np.array([0*np.pi/180.]))
mid_twist   = csdl.Variable(shape=(2,), value=np.array([0*np.pi/180., 0*np.pi/180.]))
wing_twists = csdl.concatenate((root_twist, mid_twist, tip_twist))
wing_twists.flatten()

percent_change_in_thickness_dof_wing        = csdl.Variable(shape=(8,4), value=0.)
percent_change_in_thickness_dof_body        = csdl.Variable(shape=(8,4), value=0.)
percent_change_in_thickness_dof             = csdl.concatenate(
                                                (percent_change_in_thickness_dof_wing,
                                                 percent_change_in_thickness_dof_body), axis=1)

normalized_percent_camber_change_dof_wing   = csdl.Variable(shape=(6,4), value=0.)
normalized_percent_camber_change_dof_body   = csdl.Variable(shape=(6,4), value=0.)
normalized_percent_camber_change_dof        = csdl.concatenate(
                                                (normalized_percent_camber_change_dof_wing,
                                                 normalized_percent_camber_change_dof_body), axis=1)

centerbody_chord_stretches                  = csdl.Variable(shape=(3,), value=0.)
wing_chord_stretching_b_spline_coefficients = csdl.Variable(shape=(2,), value=np.array([0., 0.]))
centerbody_span                             = csdl.Variable(value=10.)
transition_span                             = csdl.Variable(value=4.891)
wing_span                                   = csdl.Variable(value=25.852 - 9.891)
wing_sweep_translation                      = csdl.Variable(value=0.)
centerbody_dihedral_translations            = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]))
wing_dihedral_translation_b_spline_coefficients = csdl.Variable(shape=(2,), value=np.array([0., 0.]))
centerbody_twists                           = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]))
# wing_twists                                 = csdl.Variable(shape=(4,), value=np.array([0., 0., 0., 0.]))
# percent_change_in_thickness_dof             = csdl.Variable(shape=(8,8), value=0.)
# normalized_percent_camber_change_dof        = csdl.Variable(shape=(6,8), value=0.)


geometry_values_dict = {
    'centerbody_chord_stretches': centerbody_chord_stretches,
    'wing_chord_stretching_b_spline_coefficients': wing_chord_stretching_b_spline_coefficients,
    'centerbody_span': centerbody_span,
    'transition_span': transition_span,
    'wing_span': wing_span,
    'wing_sweep_translation': wing_sweep_translation,
    'centerbody_dihedral_translations': centerbody_dihedral_translations,
    'wing_dihedral_translation_b_spline_coefficients': wing_dihedral_translation_b_spline_coefficients,
    'centerbody_twists': centerbody_twists,
    'wing_twists': wing_twists,
    'percent_change_in_thickness_dof': percent_change_in_thickness_dof,
    'normalized_percent_camber_change_dof': normalized_percent_camber_change_dof,
}


geometry = setup_geometry(geometry, geometry_values_dict)


p = Plotter()
p.show(original_geo, geometry.plot(show=False, opacity=0.5, color='#e01f07'), axes=1)