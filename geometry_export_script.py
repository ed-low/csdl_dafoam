import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import lsdo_geo
from stp_export import export_geo2step
from bwb_helper_functions import read_geometry_pickle, write_geometry_pickle
from pathlib import Path
import os

from vedo import Points, Arrows, Mesh, show

# File paths
geometry_directory          =  os.path.join(os.getcwd(), 'bwb_geometry/')
stp_file_name               = 'bwbv2_no_wingtip_coarse_refined_flat.stp'
geometry_pickle_file_name   = 'bwb_stored_refit.pickle'
geometry_pickle_file_path   = Path(geometry_directory)/geometry_pickle_file_name



stp_file_path               = Path(geometry_directory)/stp_file_name

recorder = csdl.Recorder(inline=True, debug=True)
recorder.start()


geometry      = lsdo_geo.import_geometry(stp_file_path, parallelize=False)
geometry_plot = geometry.plot(show=False, color='blue', opacity=0.5)

oml_indices                 = [key for key in geometry.functions.keys()]
wing_c_indices              = [0,1,8,9]
wing_r_transition_indices   = [2,3]
wing_r_indices              = [4,5,6,7]
wing_l_transition_indices   = [10,11]
wing_l_indices              = [12,13,14,15] 

print('declaring geometry components')
left_wing_transition    = geometry.declare_component(wing_l_transition_indices)
left_wing               = geometry.declare_component(wing_l_indices)
right_wing_transition   = geometry.declare_component(wing_r_transition_indices)
right_wing              = geometry.declare_component(wing_r_indices)
center_wing             = geometry.declare_component(wing_c_indices)
oml = geometry.declare_component(oml_indices)

wing_parameterization   = 15
num_v                   = left_wing.functions[wing_l_indices[0]].coefficients.shape[1]

print('BSplineSpace')
wing_refit_bspline      = lfs.BSplineSpace(num_parametric_dimensions=2, degree=1, coefficients_shape=(wing_parameterization, num_v))

print('left wing refit')
left_wing_function_set  = left_wing.refit(wing_refit_bspline, grid_resolution=(30,300))

print('right wing refit')
right_wing_function_set = right_wing.refit(wing_refit_bspline, grid_resolution=(30,300))

print('allocating left wing functions')
for i, function in left_wing_function_set.functions.items():
    geometry.functions[i]   = function
    left_wing.functions[i]  = function

print('allocating right wing functions')
for i, function in right_wing_function_set.functions.items():
    geometry.functions[i]   = function
    right_wing.functions[i] = function

print('pickling geometry')
write_geometry_pickle(geometry, geometry_pickle_file_path)

geometry_refit      = read_geometry_pickle(geometry_pickle_file_path)
geometry_refit_plot = geometry_refit.plot(show=False, color='red', opacity=0.5)

show(geometry_plot, geometry_refit_plot)

export_geo2step(Path(geometry_directory)/'reduced_refit_geometry.pickle', geometry_refit)