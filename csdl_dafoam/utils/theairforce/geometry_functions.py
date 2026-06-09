import numpy as np
import csdl_alpha as csdl
import lsdo_function_spaces as lfs
from lsdo_geo.core.parameterization.free_form_deformation_functions import (
    construct_tight_fit_ffd_block,construct_ffd_block_around_entities,construct_ffd_block_from_corners
)
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables
import time
import lsdo_geo as lg

def setup_geometry(geometry: lg.Geometry, geometry_values_dict, make_video=False):

    centerline_LE = np.array([-1.1, 0, 0])
    centerline_TE = np.array([30.5, 0, 0])
    centerbody_mid_LE_R = np.array([3.899, 2.5, 0])
    centerbody_mid_TE_R = np.array([30.5, 2.5, 0.])
    C_t_joint_LE_R = np.array([9.813, 5, 0])
    C_t_joint_TE_R = np.array([30.5, 5, 0])
    transition_mid_LE_R = np.array([16.539, 8.617, 0.841])
    transition_mid_TE_R = np.array([24.006, 8.617, 0.860])
    wing_root_LE_R = np.array([17.815, 9.891, 1.040])
    wing_root_TE_R = np.array([23.815, 9.891, 1.040])
    wing_sect1_LE_R = np.array([20.931, 14.33, 1.341])
    wing_sect1_TE_R = np.array([26.096, 14.33, 1.377])
    wing_sect2_LE_R = np.array([25.903, 21.413, 1.822])
    wing_sect2_TE_R = np.array([29.735, 21.413, 1.916])
    wing_tip_LE_R = np.array([29.019, 25.852, 2.123])
    wing_tip_TE_R = np.array([32.016, 25.852, 2.254])


    dz_cl = np.array([0, 0, 2.25])
    dz_cm = np.array([0, 0, 2.25])
    dz_ctj = np.array([0, 0, 1.514025])
    dz_tm = np.array([0, 0, 1.514025])
    dz_wr = np.array([0, 0, 0.5])
    dz_ws1 = np.array([0, 0, 0.5])
    dz_ws2 = np.array([0, 0, 0.5])
    dz_wt = np.array([0, 0, 0.3])

    x_val_half = np.array([
        [centerline_LE[0], centerline_TE[0]+1.],
        [centerbody_mid_LE_R[0]-1, centerbody_mid_TE_R[0]+1],
        [C_t_joint_LE_R[0]-0.5, C_t_joint_TE_R[0]+1.],
        [transition_mid_LE_R[0]-0.5, transition_mid_TE_R[0]+0.5],
        [wing_root_LE_R[0]-0.5, wing_root_TE_R[0]+0.5],
        [wing_sect1_LE_R[0]-0.5, wing_sect1_TE_R[0]+0.5],
        [wing_sect2_LE_R[0]-0.5, wing_sect2_TE_R[0]+0.5],
        [wing_tip_LE_R[0]-0.5, wing_tip_TE_R[0]+0.5],
    ])
    x_val_full = np.concatenate(
        (x_val_half[1:,:][::-1,:], x_val_half)
    )

    z_val_half = np.array([
        [centerline_LE[2], centerline_TE[2]],
        [centerbody_mid_LE_R[2], centerbody_mid_TE_R[2]],
        [C_t_joint_LE_R[2], C_t_joint_TE_R[2]],
        [transition_mid_LE_R[2], transition_mid_TE_R[2]],
        [wing_root_LE_R[2], wing_root_TE_R[2]],
        [wing_sect1_LE_R[2], wing_sect1_TE_R[2]],
        [wing_sect2_LE_R[2], wing_sect2_TE_R[2]],
        [wing_tip_LE_R[2], wing_tip_TE_R[2]],
    ])
    z_delta_half = np.array([
        [-dz_cl[2]-0.5,dz_cl[2]+0.5],  
        [-dz_cm[2]-0.5,dz_cm[2]+0.5],  
        [-dz_ctj[2]-0.2,dz_ctj[2]+0.2],  
        [-dz_tm[2]-0.2,dz_tm[2]+0.2],  
        [-dz_wr[2]-0.2,dz_wr[2]+0.2],  
        [-dz_ws1[2]-0.2,dz_ws1[2]+0.2],  
        [-dz_ws2[2]-0.2,dz_ws2[2]+0.2],  
        [-dz_wt[2]-0.2,dz_wt[2]+0.2], 
    ])
    
    z_val_full = np.concatenate(
        (z_val_half[1:,:][::-1,:] + z_delta_half[1:,:][::-1,:], z_val_half + z_delta_half)
    )

    # y_half = np.array([0, 5, 9.891, 25.852 + 0.25])
    y_half = np.array([0, 2.5, 5, 8.617, 9.891, 14.33, 21.413, 25.852 + 0.25])
    # y_half = np.array([0, 25.852])
    y_full = np.concatenate((-y_half[1:][::-1], y_half))
    num_ffd_sections = x_val_full.shape[0]
    FFD_block_points = np.zeros((2,num_ffd_sections,2,3))
    for i in range(num_ffd_sections):
        x_val = x_val_full[i]
        z_val = z_val_full[i]
        x_grid, z_grid = np.meshgrid(x_val, z_val, indexing='ij')
        y_val = y_full[i]

        cross_section = np.zeros((2,2,3))
        cross_section[:,:,1] = y_val
        cross_section[:,:,0] = x_grid
        cross_section[:,:,2] = z_grid

        FFD_block_points[:,i,:] = cross_section

    num_ffd_coefficients_chordwise = 8

    BWB_ffd_block = construct_ffd_block_from_corners(
        entities=geometry,
        corners=FFD_block_points,
        num_coefficients=(num_ffd_coefficients_chordwise,2,2),
        degree=(2,2,1)
    )
    if False:
        asdf = BWB_ffd_block.plot()
        import vedo
        plotter = vedo.Plotter(offscreen=True)
        cam = dict(
            pos=(16.7104, -0.720840, 117.391),
            focal_point=(16.7104, -0.720840, 0),
            viewup=(0, 1.00000, 0),
            roll=0,
            distance=117.391,
            clipping_range=(109.809, 127.214),
        )

        plotter.show(asdf, camera=cam, interactive=False).screenshot('BWB_FFD.pdf')
        exit()
    center_section_index = num_ffd_sections // 2
    num_centerbody_ffd_sections = 3
    num_wing_ffd_sections = 4

    ffd_sectional_parameterization = VolumeSectionalParameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=BWB_ffd_block.coefficients,
        principal_parametric_dimension=1,
    )
    # ffd_sectional_parameterization.plot()

    space_of_linear_2_dof_b_splines = lfs.BSplineSpace(num_parametric_dimensions=1, degree=1, coefficients_shape=(2,))

    wing_chord_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='wing_chord_stretching_b_spline_coefficients')

    centerbody_span_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='centerbody_span_stretching_b_spline_coefficients')

    wing_span_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='wing_span_stretching_b_spline_coefficients')

    wing_sweep_translation_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='wing_sweep_translation_b_spline_coefficients')

    wing_dihedral_translation_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                                coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='dihedral_translation_b_spline_coefficients')

    # twist_b_spline = lfs.Function(space=space_of_linear_3_dof_b_splines,
    #                                 coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])*np.pi/180), name='twist_b_spline_coefficients')

    # Extract DOFs

    centerbody_span = csdl.Variable(value=10., name='centerbody_span')
    transition_span = csdl.Variable(value=4.891, name='transition_span')
    wing_span = csdl.Variable(value=25.852 - 9.891, name='wing_span')

    sectional_span = geometry_values_dict['sectional span'] # these are sectional half-span
    centerbody_span = 2*sectional_span[0]
    transition_span = sectional_span[1]
    wing_span = sectional_span[2]

    initial_centerbody_span = 10.
    initial_transition_span = 4.891
    initial_wing_span = 25.852 - 9.891

    wing_sweep_translation = csdl.Variable(value=0., name='wing_sweep_translation')
    wing_sweep_0 = 35. * np.pi/180
    wing_sweep_position_0 = csdl.tan(wing_sweep_0)*initial_wing_span
    wing_sweep = geometry_values_dict['sweep'] * np.pi/180.
    wing_sweep_translation = csdl.tan(wing_sweep)*wing_span - wing_sweep_position_0
    wing_sweep_translation_b_spline.coefficients = wing_sweep_translation_b_spline.coefficients.set(
        csdl.slice[1], wing_sweep_translation
    )
    
    centerbody_outer_span_translation = (centerbody_span - initial_centerbody_span)/2   # Divide by 2 because halfspan
    centerbody_span_stretching_b_spline.coefficients = centerbody_span_stretching_b_spline.coefficients.set(
        csdl.slice[1], centerbody_outer_span_translation
    )
    transition_outer_span_translation = centerbody_outer_span_translation + (transition_span - initial_transition_span)
    wing_span_stretching_b_spline.coefficients = wing_span_stretching_b_spline.coefficients.set(
        csdl.slice[0], transition_outer_span_translation
    )
    wing_span_stretching_b_spline.coefficients = wing_span_stretching_b_spline.coefficients.set(
        csdl.slice[1], transition_outer_span_translation + (wing_span - initial_wing_span)
        # csdl.slice[1], transition_outer_span_translation + ((wing_sweep_translation+wing_sweep_position_0)/csdl.tan(wing_sweep) - initial_wing_span)
    )

    centerbody_dihedral_translations = csdl.Variable(shape=(num_centerbody_ffd_sections,), value=0., name='centerbody_dihedral_translations')

    wing_twists = csdl.Variable(shape=(num_wing_ffd_sections,), value=0., name='wing_twists')
    centerbody_twists = csdl.Variable(shape=(num_centerbody_ffd_sections,), value=0., name='centerbody_twists')

    wing_twists = geometry_values_dict['wing twist']*np.pi/180.
    centerbody_twists_noCL = geometry_values_dict['centerbody twist']*np.pi/180.
    centerbody_twists = centerbody_twists.set(csdl.slice[1:], centerbody_twists_noCL)

    # endregion Create Parameterization Objects

    # region Evaluate Inner Parameterization Map To Define Forward Model For Parameterization Solver
    parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_ffd_sections).reshape((-1, 1))
    # chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(parametric_b_spline_inputs)
    # wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(parametric_b_spline_inputs)
    # sweep_translation_sectional_parameters = sweep_translation_b_spline.evaluate(parametric_b_spline_inputs)
    # twist_sectional_parameters = twist_b_spline.evaluate(parametric_b_spline_inputs)
    chord_stretches = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    centerbody_chord_stretches = csdl.Variable(shape=(num_centerbody_ffd_sections,), value=0., name='centerbody_chord_stretches')
    wing_chord_stretches = wing_chord_stretching_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))

    centerbody_chord_dist = geometry_values_dict['centerbody chord']
    wing_chord_dist = geometry_values_dict['wing chord']

    centerbody_chord_dist_0 = csdl.Variable(value=np.array([30., 30-3.899, 30-9.813]))
    wing_chord_dist_0 = csdl.Variable(value=np.array([6, 5.165, 3.832,3]))

    centerbody_chord_stretches = centerbody_chord_dist - centerbody_chord_dist_0
    wing_chord_stretches = wing_chord_dist - wing_chord_dist_0
    
    wing_translation_stretch = 0.9*wing_chord_stretches[0] + 0.1*centerbody_chord_stretches[-1]
    chord_stretches = chord_stretches.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                            centerbody_chord_stretches)
    chord_stretches = chord_stretches.set(csdl.slice[-num_wing_ffd_sections:], wing_chord_stretches)
    chord_stretches = chord_stretches.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                            centerbody_chord_stretches[::-1])
    chord_stretches = chord_stretches.set(csdl.slice[:num_wing_ffd_sections],
                                            wing_chord_stretches[::-1])
    chord_stretches = chord_stretches.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                            wing_translation_stretch)
    chord_stretches = chord_stretches.set(csdl.slice[num_wing_ffd_sections],
                                            wing_translation_stretch)

    centerbody_span_translations = centerbody_span_stretching_b_spline.evaluate(np.linspace(0., 1., num_centerbody_ffd_sections))
    wing_span_translations = wing_span_stretching_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
    wing_transition_section_translation = 1.1*wing_span_translations[0] + (-0.1)*wing_span_translations[-1]

    span_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    span_translations = span_translations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                              centerbody_span_translations)
    span_translations = span_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_span_translations)

    span_translations = span_translations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                              -centerbody_span_translations[::-1])
    span_translations = span_translations.set(csdl.slice[:num_wing_ffd_sections],
                                              -wing_span_translations[::-1])
    
    span_translations = span_translations.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                              wing_transition_section_translation)
    span_translations = span_translations.set(csdl.slice[(num_wing_ffd_sections)],
                                                -wing_transition_section_translation)
    
    wing_sweep_translations = wing_sweep_translation_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
    transition_sweep_translation = 0.9*wing_sweep_translations[0] # ORIGINAL
    # transition_sweep_translation = -0.1*wing_sweep_translations[-1] # NEW
    
    sweep_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    sweep_translations = sweep_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_sweep_translations)
    sweep_translations = sweep_translations.set(csdl.slice[:num_wing_ffd_sections], wing_sweep_translations[::-1])

    sweep_translations = sweep_translations.set(csdl.slice[-(num_wing_ffd_sections+1)],
                                                transition_sweep_translation)
    sweep_translations = sweep_translations.set(csdl.slice[num_wing_ffd_sections],
                                                transition_sweep_translation)
    
    wing_dihedral_translations = wing_dihedral_translation_b_spline.evaluate(np.linspace(0., 1., num_wing_ffd_sections))
    transition_dihedral = 0.9*wing_dihedral_translations[0] + 0.1*centerbody_dihedral_translations[-1]
    dihedral_translations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    dihedral_translations = dihedral_translations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections],
                                                      centerbody_dihedral_translations)
    dihedral_translations = dihedral_translations.set(csdl.slice[-num_wing_ffd_sections:], wing_dihedral_translations)
    dihedral_translations = dihedral_translations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1],
                                                      centerbody_dihedral_translations[::-1])
    dihedral_translations = dihedral_translations.set(csdl.slice[:num_wing_ffd_sections], wing_dihedral_translations[::-1])
    dihedral_translations = dihedral_translations.set(csdl.slice[-(num_wing_ffd_sections+1)], transition_dihedral)
    dihedral_translations = dihedral_translations.set(csdl.slice[num_wing_ffd_sections], transition_dihedral)

    transition_twist = 0.9*wing_twists[0] + 0.1*centerbody_twists[-1]
    
    twist_rotations = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    twist_rotations = twist_rotations.set(csdl.slice[center_section_index:center_section_index+num_centerbody_ffd_sections], centerbody_twists)
    twist_rotations = twist_rotations.set(csdl.slice[-num_wing_ffd_sections:], wing_twists)
    twist_rotations = twist_rotations.set(csdl.slice[center_section_index-num_centerbody_ffd_sections+1:center_section_index+1], centerbody_twists[::-1])
    twist_rotations = twist_rotations.set(csdl.slice[:num_wing_ffd_sections], wing_twists[::-1])
    twist_rotations = twist_rotations.set(csdl.slice[-(num_wing_ffd_sections+1)], transition_twist)
    twist_rotations = twist_rotations.set(csdl.slice[num_wing_ffd_sections], transition_twist)

    sectional_parameters = VolumeSectionalParameterizationInputs()
    sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretches)
    sectional_parameters.add_sectional_translation(axis=csdl.Variable(value=np.array([0., 1., 0.])), translation=span_translations)
    sectional_parameters.add_sectional_translation(axis=0, translation=sweep_translations)
    sectional_parameters.add_sectional_translation(axis=2, translation=dihedral_translations)
    sectional_parameters.add_sectional_rotation(axis=csdl.Variable(value=np.array([0., 1., 0.])), rotation=twist_rotations)

    ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    # NOTE: set plot to True above to see ffd block
    # Apply shape variables
    original_block_thickness = BWB_ffd_block.coefficients.value[:, :, 1, 2] - BWB_ffd_block.coefficients.value[:, :, 0, 2]

    percent_change_in_thickness = csdl.Variable(shape=(num_ffd_coefficients_chordwise,num_ffd_sections), value=0.)
    # percent_change_in_thickness_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise, num_ffd_sections//2+1), value=0.,
    #                                                 name='percent_change_in_thickness')
    percent_change_in_thickness_dof = geometry_values_dict['thickness change']
    percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[:,:num_ffd_sections//2+1], percent_change_in_thickness_dof)
    percent_change_in_thickness = percent_change_in_thickness.set(csdl.slice[:,num_ffd_sections//2+1:], percent_change_in_thickness_dof[:,-2::-1])
    delta_block_thickness = (percent_change_in_thickness / 100) * original_block_thickness
    thickness_upper_translation = 1/2 * delta_block_thickness
    thickness_lower_translation = -thickness_upper_translation
    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,1,2], ffd_coefficients[:,:,1,2] + thickness_upper_translation)
    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,0,2], ffd_coefficients[:,:,0,2] + thickness_lower_translation)

    # Parameterize camber change as normalized by the original block (kind of like chord) length
    block_length = BWB_ffd_block.coefficients.value[1, :, 0, 0] - BWB_ffd_block.coefficients.value[0, :, 0, 0]
    block_length = csdl.expand(block_length, (num_ffd_coefficients_chordwise, num_ffd_sections), 'j->ij')

    normalized_percent_camber_change = csdl.Variable(shape=(num_ffd_coefficients_chordwise,num_ffd_sections), value=0.)
    # normalized_percent_camber_change_dof = csdl.Variable(shape=(num_ffd_coefficients_chordwise-2, num_ffd_sections//2+1), value=0.,
    #                                                      name='normalized_percent_camber_change')
    normalized_percent_camber_change_dof = geometry_values_dict['camber change']
    normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1,:num_ffd_sections//2+1],
                                                                            normalized_percent_camber_change_dof)
    normalized_percent_camber_change = normalized_percent_camber_change.set(csdl.slice[1:-1,num_ffd_sections//2+1:], 
                                                                            normalized_percent_camber_change_dof[:,-2::-1])
    camber_change = (normalized_percent_camber_change / 100) * block_length
    ffd_coefficients = ffd_coefficients.set(csdl.slice[:,:,:,2], 
                                            ffd_coefficients[:,:,:,2] + 
                                            csdl.expand(camber_change, (num_ffd_coefficients_chordwise, num_ffd_sections, 2), 'ij->ijk'))

    # """
    elevator_translation = csdl.Variable(value=0., name='elevator_translation')
    ffd_coefficients = ffd_coefficients.set(csdl.slice[-1,5:10,:,2], 
                                            ffd_coefficients[-1,5:10,:,2] + 
                                            csdl.expand(elevator_translation, (5, 2), 'ij->ijk'))
    
    # elevator_rotation = csdl.Variable(value=0.*np.pi/180, name='elevator_rotation')
    # elevator_rotation = csdl.Variable(value=45.*np.pi/180, name='elevator_rotation')
    # elevator_rotation = csdl.Variable(value=geometry_values_dict['elevator_rotation'], name='elevator_rotation')
    elevator_rotation = geometry_values_dict['elevator rotation']
    elevator_rotation_origin = csdl.average(ffd_coefficients[-1], axes=(0,1))
    elevator_rotation_axis = np.array([0., 1., 0.])
    ffd_coefficients_rotated = lg.rotate(ffd_coefficients[-1,5:10,:,:], rotation_origin=elevator_rotation_origin,
                                         axis_vector=elevator_rotation_axis, angles=-elevator_rotation).reshape((5,2,3))
    ffd_coefficients = ffd_coefficients.set(csdl.slice[-1,5:10,:,:], ffd_coefficients_rotated)
    
    # """
    geometry_coefficients = BWB_ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    geometry.set_coefficients(geometry_coefficients) # type: ignore
    

    # ffd_plot = BWB_ffd_block.plot(show=False)
    # geometry.plot(additional_plotting_elements=ffd_plot) # NOTE: comment and uncomment this line to see geometry
    # exit()

    # spanwise_direction_left = geometry.evaluate(quarter_chord_left) - geometry.evaluate(quarter_chord_center)
    # spanwise_direction_right = geometry.evaluate(quarter_chord_right) - geometry.evaluate(quarter_chord_center)
    # sweep_angle_left = csdl.arctan(-spanwise_direction_left[0] / spanwise_direction_left[1]) # type: ignore
    # sweep_angle_right = csdl.arctan(spanwise_direction_right[0] / spanwise_direction_right[1]) # type: ignore

    if make_video:
        jax_inputs = [centerbody_chord_stretches, wing_chord_stretching_b_spline.coefficients, centerbody_span, transition_span, wing_span, 
                    wing_sweep_translation, centerbody_dihedral_translations, wing_dihedral_translation_b_spline.coefficients,
                    centerbody_twists, wing_twists, percent_change_in_thickness_dof, normalized_percent_camber_change_dof]
        # jax outputs is a list containing all the geometry coefficients (geometry.functions[:].coefficients)
        jax_outputs = [geometry_function.coefficients for geometry_function in geometry.functions.values()] \
                    + [BWB_ffd_block.coefficients]

        recorder = csdl.get_current_recorder()
        jax_sim = csdl.experimental.JaxSimulator(
            recorder=recorder,
            additional_inputs=jax_inputs,
            additional_outputs=jax_outputs,
            gpu=False
        )

        jax_sim.run()

        # Geometry Variables Video
        import vedo

        # video = vedo.Video(name="Y1_demo/BWB_Geometry_Variables.mp4", fps=11, backend='opencv')
        video = vedo.Video(name="BWB_Geometry_Variables.mp4", fps=11, backend='imageio')
        # video = vedo.Video(name="Y1_demo/BWB_Geometry_Variables.mp4", duration=20, backend='opencv')

        wing_chord_stretching_b_spline.coefficients.add_name('wing_chord_stretching_b_spline_coefficients')
        wing_dihedral_translation_b_spline.coefficients.add_name('wing_dihedral_translation_b_spline_coefficients')

        parameters_with_offsets = [
            (centerbody_chord_stretches, 10.),
            (wing_chord_stretching_b_spline.coefficients, 3.),
            (centerbody_span, centerbody_span.value), 
            (transition_span, transition_span.value), 
            (wing_span, wing_span.value*0.3),
            (wing_sweep_translation, 10.),
            (centerbody_dihedral_translations, 3.),
            (wing_dihedral_translation_b_spline.coefficients, 3.),
            (centerbody_twists, 30*np.pi/180),
            (wing_twists, 30*np.pi/180), # 21 dv not including shape variables
            (percent_change_in_thickness_dof, 100.), # 64 of these
            (normalized_percent_camber_change_dof, 100.)    # 48 of these
        ]

        camera = {
            # 'pos': (4*num_bodies + 1, 0, -1*num_bodies/2),
            'pos': (-50, -50, 50),
            # 'pos': (5*num_bodies + 1, 0, 0),
            'focalPoint': (10, -10, 0),
            # 'focalPoint': (0, 0, 0),
            'viewup': (0, 0, 1),
        }

        for parameter, offset in parameters_with_offsets:
            if len(parameter.value.shape) == 1:
                for i in range(len(parameter.value)):
                    values = np.hstack((np.linspace(parameter.value[i], parameter.value[i] + offset, 11).flatten(),
                                    np.linspace(parameter.value[i] + offset, parameter.value[i], 11).flatten()))
                    for value in values:
                        parameter_value = parameter.value
                        parameter_value[i] = value
                        jax_sim[parameter] = parameter_value
                        jax_sim.run()
                        frame = geometry.plot(show=False)
                        ffd_block_frame = BWB_ffd_block.plot(show=False, plot_embedded_points=False)
                        if len(parameter.value) == 1:
                            text = f"Parameter: {parameter.name}, Value: {value:.2f}"
                        else:
                            text = f"Parameter: {parameter.name}[{i}], Value: {value:.2f}"
                        vedo_text = vedo.Text2D(text, pos="bottom-left", s=2, c='black')
                        video_plotter = vedo.Plotter(offscreen=True, title="BWB Geometry Variables",
                                                    size=(1920, 1200))
                        # video_plotter.show(frame, viewup='z')
                        video_plotter.show(frame + ffd_block_frame + [vedo_text], camera=camera)
                        video.add_frame()
            elif len(parameter.value.shape) == 2:
                for i in range(parameter.value.shape[0]):
                    for j in range(parameter.value.shape[1]):
                        values = np.hstack((np.linspace(parameter.value[i, j], parameter.value[i, j] + offset, 11).flatten(),
                                        np.linspace(parameter.value[i, j] + offset, parameter.value[i, j], 11).flatten()))
                        for value in values:
                            parameter_value = parameter.value
                            parameter_value[i, j] = value
                            jax_sim[parameter] = parameter_value
                            jax_sim.run()
                            frame = geometry.plot(show=False)
                            ffd_block_frame = BWB_ffd_block.plot(show=False, plot_embedded_points=False)
                            text = f"Parameter: {parameter.name}[{i,j}], Value: {value:.2f}"
                            vedo_text = vedo.Text2D(text, pos="bottom-left", s=2, c='black')
                            video_plotter = vedo.Plotter(offscreen=True, title="BWB Geometry Variables",
                                                        size=(1920, 1200))
                            # video_plotter.show(frame, viewup='z')
                            video_plotter.show(frame + ffd_block_frame + [vedo_text], camera=camera)
                            video.add_frame()
        
        video.close()
        exit()
    return geometry, BWB_ffd_block

def project_centerbody_volume_points(geometry, centerbody_LE_pts, centerbody_TE_pts):
    LE_points = centerbody_LE_pts
    TE_points = centerbody_TE_pts
    nc = 25
    ns = 5

    upper_surface_pts_to_project = np.zeros((nc,ns,3))
    lower_surface_pts_to_project = np.zeros((nc,ns,3))

    # upper_surface_pts_to_project[4:,:,2] = 2.5
    # upper_surface_pts_to_project[:4,:,2] = np.array([1, 1, 1.5, 2.25])
    # lower_surface_pts_to_project[4:,:,2] = -2.5
    # lower_surface_pts_to_project[:4,:,2] = np.array([1, 1, 1.5, 2.25])
    for i in range(ns):
        upper_surface_pts_to_project[:,i,1] = LE_points[i,1]
        lower_surface_pts_to_project[:,i,1] = LE_points[i,1]
        upper_surface_pts_to_project[:,i,0] = np.linspace(LE_points[i,0], TE_points[i,0], nc)
        lower_surface_pts_to_project[:,i,0] = np.linspace(LE_points[i,0], TE_points[i,0], nc)

        upper_surface_pts_to_project[6:,i,2] = 2.5
        upper_surface_pts_to_project[:6,i,2] = np.array([1, 1, 1.5, 1.5, 2., 2.25])
        lower_surface_pts_to_project[6:,i,2] = -2.5
        lower_surface_pts_to_project[:6,i,2] = -np.array([1, 1, 1.5, 1.5, 2., 2.25])

        upper_surface_pts_to_project[0,i] = LE_points[i,:]
        upper_surface_pts_to_project[-1,i] = TE_points[i,:]
        lower_surface_pts_to_project[0,i] = LE_points[i,:]
        lower_surface_pts_to_project[-1,i] = TE_points[i,:]

    # LE_points_proj = geometry.project(LE_points, plot=True)
    # TE_points_proj = geometry.project(TE_points, plot=True)

    upper_surf_projected = geometry.project(upper_surface_pts_to_project, plot=False)
    lower_surf_projected = geometry.project(lower_surface_pts_to_project, plot=False)
    volume_projection_points = [
        upper_surf_projected,
        lower_surf_projected
    ]
    return volume_projection_points

def project_transition_volume_points(geometry, transition_LE_pts, transition_TE_pts):
    LE_points = transition_LE_pts
    TE_points = transition_TE_pts

    nc = 25
    ns = 5

    upper_surface_pts_to_project = np.zeros((nc,ns,3))
    lower_surface_pts_to_project = np.zeros((nc,ns,3))

    # upper_surface_pts_to_project[4:,:,2] = 2.5
    # upper_surface_pts_to_project[:4,:,2] = np.array([1, 1, 1.5, 2.25])
    # lower_surface_pts_to_project[4:,:,2] = -2.5
    # lower_surface_pts_to_project[:4,:,2] = np.array([1, 1, 1.5, 2.25])
    lower_surf_z_disp = -np.linspace(1.25,-1,5)
    for i in range(ns):
        upper_surface_pts_to_project[:,i,1] = LE_points[i,1]
        lower_surface_pts_to_project[:,i,1] = LE_points[i,1]
        upper_surface_pts_to_project[:,i,0] = np.linspace(LE_points[i,0], TE_points[i,0], nc)
        lower_surface_pts_to_project[:,i,0] = np.linspace(LE_points[i,0], TE_points[i,0], nc)

        upper_surface_pts_to_project[6:,i,2] = 2
        upper_surface_pts_to_project[:6,i,2] = np.array([1, 1, 1.5, 1.5, 2., 2.25])
        lower_surface_pts_to_project[6:,i,2] = lower_surf_z_disp[i]
        # lower_surface_pts_to_project[:6,i,2] = np.array([1, 1, 1.5, 1.5, 2., 2.25])
        lower_surface_pts_to_project[:6,i,2] = np.linspace(LE_points[i,2], lower_surf_z_disp[i], 6)

        upper_surface_pts_to_project[0,i] = LE_points[i,:]
        upper_surface_pts_to_project[-1,i] = TE_points[i,:]
        lower_surface_pts_to_project[0,i] = LE_points[i,:]
        lower_surface_pts_to_project[-1,i] = TE_points[i,:]

    # LE_points_proj = geometry.project(LE_points, plot=True)
    # TE_points_proj = geometry.project(TE_points, plot=True)

    upper_surf_projected = geometry.project(upper_surface_pts_to_project, plot=False)
    lower_surf_projected = geometry.project(lower_surface_pts_to_project, plot=False)
    volume_projection_points = [
        upper_surf_projected,
        lower_surf_projected
    ]
    return volume_projection_points

def compute_volume(geometry, volume_projection_points):
    # LE_points = np.array([
    #     [0., 0., 0.],
    #     [1.433, 1.25, 0.],
    #     [3.899, 2.5, 0.],
    #     [6.748, 3.75, 0.],
    #     [9.813, 5., 0.],
    # ])
    # TE_points = np.array([
    #     [30., 0., 0.],
    #     [30., 1.25, 0.],
    #     [30., 2.5, 0.],
    #     [30., 3.75, 0.],
    #     [30., 5., 0.],
    # ])
    nc = 25
    ns = 5

    # upper_surface_pts_to_project = np.zeros((nc,ns,3))
    # lower_surface_pts_to_project = np.zeros((nc,ns,3))

    # # upper_surface_pts_to_project[4:,:,2] = 2.5
    # # upper_surface_pts_to_project[:4,:,2] = np.array([1, 1, 1.5, 2.25])
    # # lower_surface_pts_to_project[4:,:,2] = -2.5
    # # lower_surface_pts_to_project[:4,:,2] = np.array([1, 1, 1.5, 2.25])
    # for i in range(ns):
    #     upper_surface_pts_to_project[:,i,1] = LE_points[i,1]
    #     lower_surface_pts_to_project[:,i,1] = LE_points[i,1]
    #     upper_surface_pts_to_project[:,i,0] = np.linspace(LE_points[i,0], TE_points[i,0], nc)
    #     lower_surface_pts_to_project[:,i,0] = np.linspace(LE_points[i,0], TE_points[i,0], nc)

    #     upper_surface_pts_to_project[6:,i,2] = 2.5
    #     upper_surface_pts_to_project[:6,i,2] = np.array([1, 1, 1.5, 1.5, 2., 2.25])
    #     lower_surface_pts_to_project[6:,i,2] = -2.5
    #     lower_surface_pts_to_project[:6,i,2] = -np.array([1, 1, 1.5, 1.5, 2., 2.25])

    #     upper_surface_pts_to_project[0,i] = LE_points[i,:]
    #     upper_surface_pts_to_project[-1,i] = TE_points[i,:]
    #     lower_surface_pts_to_project[0,i] = LE_points[i,:]
    #     lower_surface_pts_to_project[-1,i] = TE_points[i,:]

    # # LE_points_proj = geometry.project(LE_points, plot=True)
    # # TE_points_proj = geometry.project(TE_points, plot=True)

    # upper_surf_projected = geometry.project(upper_surface_pts_to_project, plot=True)
    # lower_surf_projected = geometry.project(lower_surface_pts_to_project, plot=True)

    upper_surf_projected = volume_projection_points[0]
    lower_surf_projected = volume_projection_points[1]

    upper_surface = geometry.evaluate(upper_surf_projected).reshape((nc, ns, 3))
    lower_surface = geometry.evaluate(lower_surf_projected).reshape((nc, ns, 3))

    # computing average height of each panel
    # height = csdl.norm(upper_surface-lower_surface, axes=(2,)) # nc, ns, 3 
    height = upper_surface[:,:,2]-lower_surface[:,:,2] # nc, ns, 3 
    avg_height = (height[:-1,:-1]+height[1:,:-1]+height[1:,1:]+height[:-1,1:])/4

    # computing average width of each panel
    # upper_width_pc = csdl.norm(upper_surface[:,1:] - upper_surface[:,:-1], axes=(2,))
    upper_width_pc = ((upper_surface[:,1:,1] - upper_surface[:,:-1,1])**2)**0.5
    upper_width = (upper_width_pc[:-1,:] + upper_width_pc[1:,:])/2
    # lower_width_pc = csdl.norm(lower_surface[:,1:] - lower_surface[:,:-1], axes=(2,))
    lower_width_pc = ((lower_surface[:,1:,1] - lower_surface[:,:-1,1])**2)**0.5
    lower_width = (lower_width_pc[:-1,:] + lower_width_pc[1:,:])/2
    avg_width = (upper_width + lower_width)/2

    # computing average length of each panel
    # upper_length_pc = csdl.norm(upper_surface[1:,:] - upper_surface[:-1,:], axes=(2,))
    upper_length_pc = ((upper_surface[1:,:,0] - upper_surface[:-1,:,0])**2)**0.5
    upper_length = (upper_length_pc[:,:-1] + upper_length_pc[:,1:])/2
    # lower_length_pc = csdl.norm(lower_surface[1:,:] - lower_surface[:-1,:], axes=(2,))
    lower_length_pc = ((lower_surface[1:,:,0] - lower_surface[:-1,:,0])**2)**0.5
    lower_length = (lower_length_pc[:,:-1] + lower_length_pc[:,1:])/2
    avg_length = (upper_length + lower_length)/2

    cell_volume = avg_height*avg_width*avg_length
    volume = csdl.sum(cell_volume) # only modeling half the body here; 2x is done outside

    return volume
