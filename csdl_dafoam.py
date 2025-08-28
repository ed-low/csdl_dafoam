from dafoam import PYDAFOAM
import numpy as np
import csdl_alpha as csdl
from petsc4py import PETSc
import os
from mpi4py import MPI



USE_CHANGE_DIRECTORY_WORKAROUND = True

# region INSTANTIATEDAFOAM
def instantiateDAFoam(options, comm, run_directory=None, mesh_options=None):
    # CHANGE DIRECTORY WORKAROUND 1/5
    # NOTE: This workaround was implemented so that we
    # can change back and forth from the run_directory to
    # whatever directory was initially in place. This serves
    # to allow us to jump to the DAFoam directory and back to
    # Some other directory, such as geometry
    if USE_CHANGE_DIRECTORY_WORKAROUND:
        current_directory = os.getcwd()
    # -------------------------------

    if not run_directory:
        run_directory = os.getcwd()
    else:
        os.chdir(run_directory)
    
    dafoam_instance = PYDAFOAM(options=options, comm=comm)

    # Add IDWarp
    if mesh_options is not None:
        from idwarp import USMesh
        mesh = USMesh(options=mesh_options, comm=comm)
        dafoam_instance.setMesh(mesh)

    # CHANGE DIRECTORY WORKAROUND 2/4
    if USE_CHANGE_DIRECTORY_WORKAROUND:
        dafoam_instance.run_directory = run_directory
        os.chdir(current_directory)
    # -------------------------------

    return dafoam_instance



# region DAFOAMSOLVER
class DAFoamSolver(csdl.experimental.CustomImplicitOperation):
    def __init__(self, dafoam_instance):
        super().__init__()

        self.dafoam_instance    = dafoam_instance
        self.solution_counter   = 1

        # initialize the dRdWT matrix-free matrix in DASolver
        dafoam_instance.solverAD.initializedRdWTMatrixFree()

        # create the adjoint vector
        self.num_local_state_elements = dafoam_instance.getNLocalAdjointStates()
        self.psi = PETSc.Vec().create(comm=dafoam_instance.comm)
        self.psi.setSizes((self.num_local_state_elements, PETSc.DECIDE), bsize=1)
        self.psi.setFromOptions()
        self.psi.zeroEntries()

        # If true, we need to compute the coloring
        if dafoam_instance.getOption("adjEqnSolMethod") == "fixedPoint":
            self.runColoring = False
        else:
            self.runColoring = True 

        # Saving last successful primal result (in case primal fails)
        self.last_successful_primal_states = dafoam_instance.getStates()
        self.last_time_converged = True


    # region evaluate
    def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup):
        # Read daOptions to set proper inputs
        input_dict = self.dafoam_instance.getOption("inputInfo")
        for input_name in input_dict.keys():
            if "solver" in input_dict[input_name]["components"]:
                self.declare_input(input_name, getattr(dafoam_input_variables_group, input_name))   

        # Set outputs
        dafoam_solver_states = self.create_output('dafoam_solver_states', (self.num_local_state_elements,))

        return dafoam_solver_states

    # region solve_residual_equations
    def solve_residual_equations(self, input_vals, output_vals):
        dafoam_instance = self.dafoam_instance

        # set the solver input, including mesh, boundary etc.
        dafoam_instance.set_solver_input(input_vals)

        # before running the primal, we need to check if the mesh
        # quality is good

        #*****************
        meshOK = 1#dafoam_instance.solver.checkMesh()
        if dafoam_instance.rank == 0:
            print('SKIPPING MESH CHECK. BE SURE TO CHANGE THIS BACK WHEN DONE DEBUGGING')
        #*****************

        # if the mesh is not OK, do not run the primal
        if meshOK != 1:
            dafoam_instance.solver.writeFailedMesh()
            if dafoam_instance.rank == 0:
                print('Mesh is not OK!')
            return

        # Revert solver to last successful primal state (to help with convergence on next iteration)
        # (This helps when we had an unconverged run last - placing here instead of at end in unconverged
        #  scenario allows us to still used the unconverged results from the solver if necessary) 
        if not self.last_time_converged:
            if dafoam_instance.rank == 0:
                print('Initializing DAFoam solver with last converged primal state')  
            dafoam_instance.setStates(self.last_successful_primal_states)

        # Run primal
        dafoam_instance()

        # After solving the primal, we need to print its residual info
        if dafoam_instance.getOption("useAD")["mode"] == "forward":
            dafoam_instance.solverAD.calcPrimalResidualStatistics("print")
        else:
            dafoam_instance.solver.calcPrimalResidualStatistics("print")

        # Use this to assign the computed flow states
        states = dafoam_instance.getStates()

        # Unconverged case - return NaN for CSDL, flag to revert to last successful state for DAFoam
        if dafoam_instance.primalFail != 0:
            if dafoam_instance.rank == 0:
                print('Primal solution failed!')

            # If we didn't converge, send the optimizer a NaN solution
            output_vals['dafoam_solver_states'] = np.full((self.num_local_state_elements, ), np.nan)

            # Save convergence flag
            self.last_time_converged            = False

        # Converged case - return states and update the last successful primal state with current
        else:
            if dafoam_instance.rank == 0:
                print('Primal solution converged!')
                print('Caching successful primal state...')
            output_vals['dafoam_solver_states'] = states
            self.last_successful_primal_states  = states
            self.last_time_converged            = True

        # We also need to just calculate the residual for the AD mode to initialize vars like URes
        # We do not print the residual for AD, though
        dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")


    # region apply_inverse_jacobian
    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        dafoam_instance = self.dafoam_instance

        # CHANGE DIRECTORY WORKAROUND 3/5
        if USE_CHANGE_DIRECTORY_WORKAROUND:
            previous_directory = os.getcwd()
            os.chdir(dafoam_instance.run_directory)
        # -------------------------------

        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamSolver')

        adjEqnSolMethod = dafoam_instance.getOption("adjEqnSolMethod")

        # right hand side array from d_outputs
        dFdWArray = d_outputs['dafoam_solver_states']

        # Check for NaNs - exit early if found
        if has_global_nan_or_inf(dFdWArray, dafoam_instance.comm):
            if dafoam_instance.rank == 0:
                print('DAFoamSolver.apply_inverse_jacobian: Found NaN in dFdWArray! Returning NaNs for d_residuals')
            
            d_residuals['dafoam_solver_states'] = np.nan*np.ones((self.num_local_state_elements, ))
            
            # Have to change the directory back 
            # CHANGE DIRECTORY WORKAROUND 4/5
            if USE_CHANGE_DIRECTORY_WORKAROUND:
                os.chdir(previous_directory)
            # -------------------------------
            return

        # convert the array to vector
        dFdW = dafoam_instance.array2Vec(dFdWArray)

        # run coloring
        if self.dafoam_instance.getOption("adjUseColoring") and self.runColoring:
            self.dafoam_instance.solver.runColoring()
            self.runColoring = False

        if adjEqnSolMethod == "Krylov":
            # solve the adjoint equation using the Krylov method
            # if writeMinorIterations=True, we rename the solution in pyDAFoam.py. So we don't recompute the PC
            if dafoam_instance.getOption("writeMinorIterations"):
                if dafoam_instance.dRdWTPC is None or dafoam_instance.ksp is None:
                    dafoam_instance.dRdWTPC = PETSc.Mat().create(dafoam_instance.comm)
                    dafoam_instance.solver.calcdRdWT(1, dafoam_instance.dRdWTPC)
                    dafoam_instance.ksp = PETSc.KSP().create(dafoam_instance.comm)
                    dafoam_instance.solverAD.createMLRKSPMatrixFree(dafoam_instance.dRdWTPC, dafoam_instance.ksp)

            # otherwise, we need to recompute the PC mat based on adjPCLag
            else:
                # NOTE: this function will be called multiple times (one time for one obj func) in each opt iteration
                # so we don't want to print the total info and recompute PC for each obj, we need to use renamed
                # to check if a recompute is needed. In other words, we only recompute the PC for the first obj func
                # adjoint solution
                solutionTime, renamed = dafoam_instance.renameSolution(self.solution_counter)

                if renamed:
                    if dafoam_instance.comm.rank == 0:
                        print("Driver total derivatives for iteration: %d" % self.solution_counter, flush=True)
                        print("---------------------------------------------", flush=True)
                    self.solution_counter += 1

                # compute the preconditioner matrix for the adjoint linear equation solution
                # and initialize the ksp object. We reinitialize them every adjPCLag
                adjPCLag = dafoam_instance.getOption("adjPCLag")
                if dafoam_instance.dRdWTPC is None or dafoam_instance.ksp is None or (self.solution_counter - 1) % adjPCLag == 0:
                    if renamed:
                        # calculate the PC mat
                        if dafoam_instance.dRdWTPC is not None:
                            dafoam_instance.dRdWTPC.destroy()
                        dafoam_instance.dRdWTPC = PETSc.Mat().create(dafoam_instance.comm)
                        dafoam_instance.solver.calcdRdWT(1, dafoam_instance.dRdWTPC)
                        # reset the KSP
                        if dafoam_instance.ksp is not None:
                            dafoam_instance.ksp.destroy()
                        dafoam_instance.ksp = PETSc.KSP().create(dafoam_instance.comm)
                        dafoam_instance.solverAD.createMLRKSPMatrixFree(dafoam_instance.dRdWTPC, dafoam_instance.ksp)

            # if useNonZeroInitGuess is False, we will manually reset self.psi to zero
            # this is important because we need the correct psi to update the KSP tolerance
            # in the next line
            if not self.dafoam_instance.getOption("adjEqnOption")["useNonZeroInitGuess"]:
                self.psi.set(0)
            else:
                # if useNonZeroInitGuess is True, we will assign the OM's psi to self.psi
                self.psi = dafoam_instance.array2Vec(d_residuals['dafoam_solver_states'].copy())

            if self.dafoam_instance.getOption("adjEqnOption")["dynAdjustTol"]:
                # if we want to dynamically adjust the tolerance, call this function. This is mostly used
                # in the block Gauss-Seidel method in two discipline coupling
                # update the KSP tolerances the coupled adjoint before solving
                self._updateKSPTolerances(self.psi, dFdW, dafoam_instance.ksp)

            # actually solving the adjoint linear equation using Petsc
            fail = dafoam_instance.solverAD.solveLinearEqn(dafoam_instance.ksp, dFdW, self.psi)

        elif adjEqnSolMethod == "fixedPoint":
            solutionTime, renamed = dafoam_instance.renameSolution(self.solution_counter)
            if renamed:
                # write the deformed FFD for post-processing
                # dafoam_instance.writeDeformedFFDs(self.solution_counter)
                # print the solution counter
                if dafoam_instance.comm.rank == 0:
                    print("Driver total derivatives for iteration: %d" % self.solution_counter, flush=True)
                    print("---------------------------------------------", flush=True)
                self.solution_counter += 1
            # solve the adjoint equation using the fixed-point adjoint approach
            fail = dafoam_instance.solverAD.runFPAdj(dFdW, self.psi)

        else:
            raise RuntimeError("adjEqnSolMethod=%s not valid! Options are: Krylov or fixedPoint" % adjEqnSolMethod)

        # optionally write the adjoint vector as OpenFOAM field format for post-processing
        psi_array = dafoam_instance.vec2Array(self.psi)
        solTimeFloat = (self.solution_counter - 1) / 1e4
        dafoam_instance.writeAdjointFields("function", solTimeFloat, psi_array)

        if not fail:
            # convert the solution vector to array and assign it to d_residuals
            d_residuals['dafoam_solver_states'] = dafoam_instance.vec2Array(self.psi)

        # if the adjoint solution fail, we return NaN and let the optimizer handle it
        else:
            if dafoam_instance.rank == 0:
                print("Adjoint solution failed! Returning NANs")

            d_residuals['dafoam_solver_states'] = np.nan*np.ones((self.num_local_state_elements, ))
        
        # CHANGE DIRECTORY WORKAROUND 5/5
        if USE_CHANGE_DIRECTORY_WORKAROUND:
            os.chdir(previous_directory)
        # -------------------------------


    # region compute_jacvec_product
    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
        dafoam_instance = self.dafoam_instance
        comm            = dafoam_instance.comm
        rank            = dafoam_instance.rank

        # assign the states in outputs to the OpenFOAM flow fields
        # NOTE: this is not quite necessary because setStates have been called before in the solve_nonlinear
        # here we call it just be on the safe side
        # Check if states contain any NaN values (NaNs would be passed from optimizer)
        states = output_vals['dafoam_solver_states']

        # Update solver states only if no NaNs exist
        if not has_global_nan_or_inf(states, comm):
            dafoam_instance.setStates(states)
        else:
            if rank == 0:
                print('DAFoamSolver.compute_jacvec_product: Detected NaN(s) in input_vals. Skipping DAFoam setStates')

        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamSolver')

        if 'dafoam_solver_states' in d_residuals:
             # get the reverse mode AD seed from d_residuals
            seed = d_residuals['dafoam_solver_states']
 
            # loop over all inputs keys and compute the matrix-vector products accordingly
            input_dict = dafoam_instance.getOption("inputInfo")
            for input_name in list(input_vals.keys()):
                input_type = input_dict[input_name]["type"]
                jac_input = input_vals[input_name].copy()
                product = np.zeros_like(jac_input)
                dafoam_instance.solverAD.calcJacTVecProduct(
                    input_name,
                    input_type,
                    jac_input,
                    "aero_residuals",
                    "residual",
                    seed,
                    product,
                )
                d_inputs[input_name] += product


    # region _updateKSPTolerances
    def _updateKSPTolerances(self, psi, dFdW, ksp):
        # Here we need to manually update the KSP tolerances because the default
        # relative tolerance will always want to converge the adjoint to a fixed
        # tolerance during the LINGS adjoint solution. However, what we want is
        # to converge just a few orders of magnitude. Here we need to bypass the
        # rTol in Petsc and manually calculate the aTol.

        dafoam_instance = self.dafoam_instance
        # calculate the initial residual for the adjoint before solving
        rArray = np.zeros(self.num_local_state_elements)
        jac_input = dafoam_instance.getStates()
        seed = dafoam_instance.vec2Array(psi)
        dafoam_instance.solverAD.calcJacTVecProduct(
            'dafoam_solver_states',
            "stateVar",
            jac_input,
            'aero_residuals',
            "residual",
            seed,
            rArray,
        )
        rVec = dafoam_instance.array2Vec(rArray)
        rVec.axpy(-1.0, dFdW)
        # NOTE, this is the norm for the global vec
        rNorm = rVec.norm()

        # read the rTol and aTol from DAOption
        rTol0 = self.dafoam_instance.getOption("adjEqnOption")["gmresRelTol"]
        aTol0 = self.dafoam_instance.getOption("adjEqnOption")["gmresAbsTol"]
        # calculate the new absolute tolerance that gives you rTol residual drop
        aTolNew = rNorm * rTol0
        # if aTolNew is smaller than aTol0, assign aTol0 to aTolNew
        if aTolNew < aTol0:
            aTolNew = aTol0
        # assign the atolNew and distable rTol
        ksp.setTolerances(rtol=0.0, atol=aTolNew, divtol=None, max_it=None)



# region DAFOAMFUNCTIONS
class DAFoamFunctions(csdl.CustomExplicitOperation):
    def __init__(self, dafoam_instance):
        super().__init__()
        self.dafoam_instance = dafoam_instance


    # region evaluate
    def evaluate(self, dafoam_solver_states:csdl.Variable, dafoam_input_variables_group:csdl.VariableGroup):
        # Solver states is the easy one
        self.declare_input("dafoam_solver_states", dafoam_solver_states)
        
        # Read daOptions to set proper inputs
        input_dict = self.dafoam_instance.getOption("inputInfo")
        for input_name in input_dict.keys():
            if "function" in input_dict[input_name]["components"]:
                self.declare_input(input_name, getattr(dafoam_input_variables_group, input_name))  
        
        # Initialize output
        dafoam_function_output = csdl.VariableGroup()

        # Read daOptions to get outputs
        outputDict = self.dafoam_instance.getOption("function")
        for outputName in outputDict.keys():
            setattr(dafoam_function_output, outputName, self.create_output(outputName, (1, )))

        return dafoam_function_output

    # region compute
    def compute(self, input_vals, output_vals):
        dafoam_instance = self.dafoam_instance
        comm            = dafoam_instance.comm
        rank            = dafoam_instance.rank

        # Check if states contain any NaN values (NaNs would be passed from optimizer)
        # TODO: Maybe check for Inf as well?
        states = input_vals['dafoam_solver_states']

        # Update solver states only if no NaNs exist
        has_nan_or_inf = has_global_nan_or_inf(states, comm)
        if not has_nan_or_inf:
            dafoam_instance.setStates(states)
        else:
            if rank == 0:
                print('DAFoamFunctions.compute: Detected NaN(s) in input_vals. Skipping DAFoam setStates')

        # Read daOptions to get outputs, and assign them to respective outputs.
        # Assign NaN if NaN existed in state
        output_dict = self.dafoam_instance.getOption("function")
        for output_name in output_dict.keys():
            function_val = dafoam_instance.solver.calcFunction(output_name)
            if has_nan_or_inf:
                output_vals[output_name] = np.nan*function_val
            else:
                output_vals[output_name] = function_val


    # region compute_jacvec_product
    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
        dafoam_instance = self.dafoam_instance
        comm            = dafoam_instance.comm
        rank            = dafoam_instance.rank

        # Check if states contain any NaN values (NaNs would be passed from optimizer)
        # TODO: Maybe check for Inf as well?
        states = input_vals['dafoam_solver_states']

        # Update solver states only if no NaNs exist
        if not has_global_nan_or_inf(states, comm):
            dafoam_instance.setStates(states)
        else:
            if rank == 0:
                print('DAFoamFunctions.compute_jacvec_product: Detected NaN(s) in input_vals. Skipping DAFoam setStates')

        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamFunctions')
        
        input_dict = dafoam_instance.getOption("inputInfo")
        for functionName in list(d_outputs.keys()):

            seed = d_outputs[functionName]

            # if the seed is zero, do not compute
            if abs(seed) < 1e-12:
                continue

            for input_name in list(d_inputs.keys()):
                # compute dFdW * seed
                if input_name == 'dafoam_solver_states':
                    jac_input = input_vals['dafoam_solver_states']
                    product = np.zeros_like(jac_input)
                    dafoam_instance.solverAD.calcJacTVecProduct(
                        'dafoam_solver_states',
                        "stateVar",
                        jac_input,
                        functionName,
                        "function",
                        seed,
                        product,
                    )
                    d_inputs['dafoam_solver_states'] += product
                else:
                    input_type = input_dict[input_name]["type"]
                    jac_input = input_vals[input_name]
                    product = np.zeros_like(jac_input)
                    dafoam_instance.solverAD.calcJacTVecProduct(
                        input_name,
                        input_type,
                        jac_input,
                        functionName,
                        "function",
                        seed,
                        product,
                    )
                    d_inputs[input_name] += product



# region COMPUTE_DAFOAM_INPUT_VARIABLES
def compute_dafoam_input_variables(dafoam_instance, ambient_conditions_group:csdl.VariableGroup, flight_conditions_group:csdl.VariableGroup, aerodynamic_volume_coordinates:csdl.Variable):
    # Currently expect the ambient_conditions_group to, at minimum, contain the following variables:
    # T_K   (Temperature [K])
    # P_Pa  (Pressure [Pa])
    # a_m_s (Speed of sound [m/s])

    # Currently expect the flight_conditions_group to, at minimum, contain the following variables:
    # airspeed_m_s         (airspeed [m/s])
    # angle_of_attack_deg  (Angle of attack [deg])
    # OR
    # mach_number          (Mach number)
    # angle_of_attack_deg  (Angle of attack [deg])
    
    # Initialize dafoam variable group
    dafoam_input_variables_group = csdl.VariableGroup()

    # Read inputInfo listed in daOptions (these should be set by user)
    dafoam_input_dict = dafoam_instance.getOption("inputInfo")

    for input_name in dafoam_input_dict.keys():
        input_type = dafoam_input_dict[input_name]["type"]

        # If the type is volCoord, we get the coordinates and assign it to the CSDL variable group
        if input_type == "volCoord":
            # TODO: Logic for checking dimensions
            setattr(dafoam_input_variables_group, input_name, aerodynamic_volume_coordinates)

        # If the type is patchVelocity, we get the velocity and angle of attack and assign it to a CSDL variable
        elif input_type == "patchVelocity":
            # Compute the airspeed if not in the flight conditions group, and add to it
            if not hasattr(flight_conditions_group, "airspeed_m_s"):
                flight_conditions_group.airspeed_m_s = flight_conditions_group.mach_number*ambient_conditions_group.a_m_s

            # Create the patchVelocity input
            patchVelocity = csdl.concatenate((flight_conditions_group.airspeed_m_s, flight_conditions_group.angle_of_attack_deg))
            setattr(dafoam_input_variables_group, input_name, patchVelocity)
        
        # If the type is a patchVar, we assign the appropriate value to the CSDL variable
        elif input_type == "patchVar":
            input_variable_name = dafoam_input_dict[input_name]["varName"]

            # Pressure case
            if input_variable_name == "p":
                setattr(dafoam_input_variables_group, input_name, ambient_conditions_group.P_Pa)
            
            # Temperature case
            elif input_variable_name == "T":
                setattr(dafoam_input_variables_group, input_name, ambient_conditions_group.T_K)

            # Not implemented case
            else:
                raise NotImplementedError(f'unable to create csdl variable for "{input_name}" (patchVar for "{input_variable_name}" interpretation not implemented)')

        # Not implemented case
        else:
            raise NotImplementedError(f'unable to create csdl variable for "{input_name}" (type "{input_type}" interpretation not implemented)')

    return dafoam_input_variables_group



# region HAS_GLOBAL_NAN_OR_INF
def has_global_nan_or_inf(arr, comm):
    """
    Check if a distributed array contains any NaN or Inf values across all MPI ranks.

    Parameters
    ----------
    arr : np.ndarray
        Local portion of the array.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    bool
        True if any NaN or Inf exists in the *global* array, else False.
    """
    local_has_nan_or_inf  = np.any(np.isnan(arr) | np.isinf(arr))
    global_has_nan_or_inf = comm.allreduce(local_has_nan_or_inf, op=MPI.LOR)
    return global_has_nan_or_inf



# region DAFOAMROM
class DAFoamROM(csdl.experimental.CustomImplicitOperation):
    def __init__(self, 
                 dafoam_instance,
                 pod_modes:np.array      =None,
                 fom_ref_state:np.array  =None, 
                 tolerance               =1e-6, 
                 max_iters:int           =100, 
                 update_jac_frequency:int=0, 
                 fom_states_ref:np.array =None,
                 weights:np.array        =None,
                 scaling_factors:np.array=None):

        super().__init__()
        self.dafoam_instance        = dafoam_instance
        self.tolerance              = tolerance
        self.max_iters              = max_iters
        self.solution_counter       = 1
        self.update_jac_frequency   = update_jac_frequency if update_jac_frequency > 0 else np.nan
        self.reduced_jacobian       = None
        self.solution_counter       = 1

        # Do some checks here to assign default values if none passed. Alert user if none are passed.
        if weights is None:
            print('No weights passed to DAFoamROM. Assuming no weighting used in POD mode computation...')
            self.weights = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
        else:
            self.weights = weights

        if scaling_factors is None:
            print('No weights passed to DAFoamROM. Assuming no scaling factor used in POD mode computation...')
            self.scaling_factors = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
        else:
            self.scaling_factors = scaling_factors

        if fom_ref_state is None:
            print('No FOM reference state passed to DAFoamROM. Assuming no mean/reference subtraction used in POD mode computation...')
            self.fom_ref_state = np.ones((dafoam_instance.getNLocalAdjointStates(), ))
        else:
            self.fom_ref_state = fom_ref_state

        if pod_modes is None:
            print('No POD modes passed to DAFoamROM. Expecting POD modes as CSDL variable for the evaluate method...')
        else:
            print('POD modes passed to DAFoamROM. Assuming constant/global POD modes unless modes are passed to the evaluate method...')
        
        self.pod_modes = pod_modes

        
    # region evaluate
    def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup, pod_modes:csdl.Variable=None):
        
        # Read daOptions to set proper inputs
        inputDict = self.dafoam_instance.getOption("inputInfo")
        for inputName in inputDict.keys():
            if "solver" in inputDict[inputName]["components"]:
                self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))

        # Set our basis as an input variable if not already set to a constant
        if pod_modes is not None:
            if self.pod_modes is None:
                print('POD modes passed to DAFoamROM evalute method. Neglecting constant/global modes passed in initialization.')
            self.declare_input('pod_modes', pod_modes)
            num_modes = pod_modes.value.shape[1]
            self.pod_modes_is_csdl_var = True
        else:
            if self.pod_modes is None:
                TypeError('No POD modes assigned to ROM. Please pass a constant POD mode set during initialization, or a POD mode in the evaluate section.')
            else:
                num_modes = self.pod_modes.shape[1]
                self.pod_modes_is_csdl_var = False
        
        # Set outputs
        dafoam_rom_states = self.create_output('dafoam_rom_states', (num_modes,))

        return dafoam_rom_states


    # region solve_residual_equations
    def solve_residual_equations(self, input_vals, output_vals):

        # Pull values from self
        dafoam_instance       = self.dafoam_instance
        max_iters             = self.max_iters
        update_jac_frequency  = self.update_jac_frequency
        tolerance             = self.tolerance
        pod_modes_is_csdl_var = self.pod_modes_is_csdl_var

        # Some important quanitites
        y_fom_ref    = self.fom_ref_state
        W            = self.weights
        D            = self.scaling_factors

        # Determine if our basis is constant, or an input value
        if pod_modes_is_csdl_var:
            phi      = input_vals['pod_modes']
        else:
            phi      = self.pod_modes

        num_modes    = phi.shape[1]
        
        # MPI stuff
        # TODO: will have to make this all MPI compatible eventually
        rank = dafoam_instance.rank

        # Make sure solver is updated with the most recent input values
        dafoam_instance.set_solver_input(input_vals)

        # We also need to just calculate the residual for the AD mode to initialize vars like URes
        # We do not print the residual for AD, though
        # NOTE: This was taken from DAFoamSolver.solve_residual_equation
        dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")

        # Initialize our newton convergence flag
        converged = False

        # Initialize our ROM/FOM states, solver, and ROM/FOM residuals 
        y_rom = np.zeros((num_modes, ))
        y_fom = y_fom_ref + D*(phi@y_rom)
        dafoam_instance.setStates(y_fom)
        r_fom = dafoam_instance.getResiduals()
        r_rom = phi.T@(W*r_fom)
        r_rom_norm = np.linalg.norm(r_rom)

        # Some plotting for diagnostics
        #----------------------
        import matplotlib.pyplot as plt
        # Plotting coefficients
        # Initialize figure
        plt.ion()  # Interactive mode on
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,6))

        # --- subplot 1: ROM state progression ---
        lines = [ax1.plot([], [], label=f'element {i}')[0] for i in range(num_modes)]

        # Store data
        x_data = []                     # iteration index
        y_data = [[] for _ in range(num_modes) ]  # data for each element

        # --- subplot 2: FOM update ---
        fom_state_line,    = ax2.plot(np.zeros_like(y_fom_ref), 'r-')

        # --- subplot 3: residual update ---
        fom_residual_line, = ax3.plot(np.zeros_like(y_fom_ref), 'b')

        x_data.append(0)
        for i, line in enumerate(lines):
            y_data[i].append(y_rom[i])
            line.set_data(x_data, y_data[i])
            plt.draw()
            plt.pause(0.1)
        #-------------------------

        # Main Newton iteration loop
        for iter in range(max_iters):
            print('\n')
            print('-------------------------------------')
            print(f'ROM Newton iteration {iter}')
            print('-------------------------------------')

            # (Re)compute Jacobian on first iteration or if not reusing the same Jacobian
            if iter%update_jac_frequency == 0 or iter == 0:
                print('Computing reduced Jacobian...')
                J_rom = self._compute_reduced_jacobian(phi, y_fom, W, D, 'fom_jTvp')
                self.reduced_jacobian = J_rom

            # Compute step (robust linear solve with lstsq fallback)
            print('Updating ROM and FOM states...')
            try:
                delta_y_rom = np.linalg.solve(J_rom, -r_rom)
            except np.linalg.LinAlgError:
                # fallback to least-squares if matrix singular or near-singular
                delta_y_rom, *_ = np.linalg.lstsq(J_rom, -r_rom, rcond=1e-12)
                delta_y_rom = delta_y_rom.reshape(-1,)
                print('\tWarning: reduced Jacobian singular; used least-squares fallback.')

            accepted = False
            alpha = 1
            while alpha > 1e-4:
                print(f'Line search: alpha = {alpha}')
                y_rom_test = y_rom + alpha*delta_y_rom
                y_fom_test = y_fom_ref + D*(phi@y_rom_test)
                dafoam_instance.setStates(y_fom_test)
                r_fom_test = dafoam_instance.getResiduals()
                r_rom_test = phi.T@(W*r_fom_test)
                r_rom_test_norm = np.linalg.norm(r_rom_test)

                if r_rom_test_norm < r_rom_norm:
                    y_rom = y_rom_test
                    y_fom = y_fom_test
                    r_rom = r_rom_test
                    r_fom = r_fom_test
                    r_rom_norm = r_rom_test_norm
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                print('Line search failed! Taking small step...')
                y_rom = y_rom + 1e-4*delta_y_rom
                y_fom = y_fom_ref + D*(phi@y_rom)
                dafoam_instance.setStates(y_fom)
                r_fom = dafoam_instance.getResiduals()
                r_rom = phi.T@(W*r_fom)
                r_rom_norm = np.linalg.norm(r_rom)

            # Save the initial value of the reduced residual
            if iter == 0:
                r_rom_norm0 = r_rom_norm.copy()

            # Will print out the relevant residual statistics here
            print('Residual summary:')
            print('\t{:<30} : {:.4E}'.format('FOM residual norm', np.linalg.norm(r_fom)))
            print('\t{:<30} : {:.4E}'.format('ROM residual norm', r_rom_norm))
            print('\t{:<30} : {:.4E}'.format('Start/current ROM norm ratio', r_rom_norm/r_rom_norm0))

            # Exit condition
            if r_rom_norm < tolerance:
                converged = True
                break

            # Upate plots
            x_data.append(iter+1)
            for i, line in enumerate(lines):
                y_data[i].append(y_rom[i])
                line.set_data(x_data, y_data[i])
            ax1.relim()
            ax1.autoscale_view()

            fom_state_line.set_ydata(y_fom)
            ax2.relim()
            ax2.autoscale_view()

            fom_residual_line.set_ydata(r_fom)
            ax3.relim()
            ax3.autoscale_view()
            plt.draw()
            plt.pause(0.1)
        
        # Return NaN array as output if failed (for the optimizer), otherwise return rom states
        if converged:
            print('\n\n Specified tolerance reached!')
            # Update reduced jacobain to most recent state
            print('Updating Jacobian to most recent state value...')
            J_rom = self._compute_reduced_jacobian(phi, y_fom, W, D)
            self.reduced_jacobian = J_rom

            output_vals['dafoam_rom_states'] = y_rom
        else:
            print('Warning: ROM iteration limit reached!')
            output_vals['dafoam_rom_states'] = np.full((num_modes, ), np.nan)


    # region apply_inverse_jacobian
    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamROM')
        
        J_romT = self.reduced_jacobian.T
        v      = d_outputs['dafoam_rom_states']
        d_residuals['dafoam_rom_states'] = np.linalg.solve(J_romT, v)

        # Write solution to file
        dafoam_instance = self.dafoam_instance
        D               = self.scaling_factors
        W               = self.weights
        y_fom_ref           = self.fom_ref_state
        # Determine if our basis is constant, or an input value
        if self.pod_modes_is_csdl_var:
            phi      = input_vals['pod_modes']
        else:
            phi      = self.pod_modes
        y_rom = output_vals['dafoam_rom_states']
        y_fom = y_fom_ref + D*(phi@y_rom)

        n_points = dafoam_instance.solver.getNLocalPoints()
        points0  = np.zeros(n_points*3)
        dafoam_instance.solver.getOFMeshPoints(points0)
        dafoam_instance.solver.writeMeshPoints(points0, self.solution_counter)
        dafoam_instance.solver.writeAdjointFields('sol', self.solution_counter, y_fom)
        self.solution_counter += 1
        

    # region compute_jacvec_product
    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
        dafoam_instance = self.dafoam_instance
        comm            = dafoam_instance.comm
        rank            = dafoam_instance.rank

        D         = self.scaling_factors
        W         = self.weights
        phi       = self.pod_modes
        y_fom_ref = self.fom_ref_state

        dafoam_scaling_factors = dafoam_instance.getStateScalingFactors()

        # assign the states in outputs to the OpenFOAM flow fields
        # NOTE: this is not quite necessary because setStates have been called before in the solve_nonlinear
        # here we call it just be on the safe side
        # Check if states contain any NaN values (NaNs would be passed from optimizer)
        y_rom = output_vals['dafoam_rom_states']
        y_fom = y_fom_ref + D*(phi@y_rom)

        # Update solver states only if no NaNs exist
        if not has_global_nan_or_inf(y_fom, comm):
            dafoam_instance.setStates(y_fom)
        else:
            if rank == 0:
                print('DAFoamSolver.compute_jacvec_product: Detected NaN(s) in input_vals. Skipping DAFoam setStates')

        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamROM')

        if 'dafoam_rom_states' in d_residuals:
             # get the reverse mode AD seed from d_residuals and expand to FOM size
            seed_r = d_residuals['dafoam_rom_states']
            seed   = W*(phi@seed_r)
 
            # loop over all inputs keys and compute the matrix-vector products accordingly
            input_dict = dafoam_instance.getOption("inputInfo")
            for input_name in list(input_vals.keys()):
                input_type = input_dict[input_name]["type"]
                jac_input = input_vals[input_name].copy()
                product = np.zeros_like(jac_input)
                dafoam_instance.solverAD.calcJacTVecProduct(
                    input_name,
                    input_type,
                    jac_input,
                    "aero_residuals",
                    "residual",
                    seed,
                    product,
                )
                d_inputs[input_name] += product


    # region _compute_reduced_jacobian
    def _compute_reduced_jacobian(self, phi, y_fom, W, D, mode='fom_jTvp', fd_eps=1e-6):
        dafoam_instance = self.dafoam_instance
        num_modes       = phi.shape[1]

        if mode == 'fom_jTvp': # Computing using DAFoamSolver's calcJacTVecProduct column by column
            J_romT_W_phi           = np.zeros_like(phi)
            W_phi                  = W[:, None]*phi
            dafoam_scaling_factors = dafoam_instance.getStateScalingFactors()

            # Update solver for desired states
            dafoam_instance.setStates(y_fom)
            
            # Loop over columns of phi
            for i in range(num_modes):
                seed    = np.ascontiguousarray(W_phi[:, i])
                product = np.zeros_like(seed)
                dafoam_instance.solverAD.calcJacTVecProduct(
                    'dafoam_solver_states',
                    "stateVar",
                    y_fom,
                    'aero_residuals',
                    "residual",
                    seed,
                    product,
                )

                J_romT_W_phi[:, i] = product/dafoam_scaling_factors
                
            J_romT = phi.T@(D[:, None]*J_romT_W_phi)
            J_rom   = J_romT.T

            return J_rom

        elif mode == 'fd': # Finite difference method in reduced space
            J_rom  = np.zeros((num_modes, num_modes))
            
            # Get reduced residual
            dafoam_instance.setStates(y_fom)
            r_fom  = dafoam_instance.getResiduals()
            r_rom0 = phi.T@(W*r_fom)

            for j in range(num_modes):         
                # Perturb the reduced coordinates
                e_j = np.zeros(num_modes)
                e_j[j] = 1.0
                y_fom_pert = y_fom + fd_eps*D*(phi@e_j)

                # Get perturbed reduced residuals
                r_rom_pert = self._compute_reduced_residual(phi, y_fom_pert, W, D)

                # Add to respective reduced jacobian column
                J_rom[:, j] = (r_rom_pert - r_rom0)/fd_eps

            # Reset states in DAFoam
            dafoam_instance.setStates(y_fom)

            return J_rom

        else:
            raise ValueError(f"Unknown J_rom computation mode '{mode}'. Choose from 'fom_jTvp' or 'fd'")
    

    # This is just a helper function for convenience
    def _compute_reduced_residual(self, phi, y_fom, W, D):
        dafoam_instance = self.dafoam_instance

        dafoam_instance.setStates(y_fom)
        r_fom = dafoam_instance.getResiduals()
        r_rom = phi.T@(W*r_fom)

        return r_rom







# # ORIGINAL VERSION
# # region solve_residual_equations
#     def solve_residual_equations(self, input_vals, output_vals):

#         # Pull values from self
#         dafoam_instance       = self.dafoam_instance
#         max_iters             = self.max_iters
#         update_jac_frequency  = self.update_jac_frequency
#         tolerance             = self.tolerance
#         pod_modes_is_csdl_var = self.pod_modes_is_csdl_var

#         # Some important quanitites
#         y_fom_ref    = self.fom_ref_state
#         W            = self.weights
#         D            = self.scaling_factors

#         # Determine if our basis is constant, or an input value
#         if pod_modes_is_csdl_var:
#             phi      = input_vals['pod_modes']
#         else:
#             phi      = self.pod_modes

#         num_modes    = phi.shape[1]
        
#         # MPI stuff
#         # TODO: will have to make this all MPI compatible eventually
#         rank = dafoam_instance.rank

#         # Make sure solver is updated with the most recent input values
#         dafoam_instance.set_solver_input(input_vals)

#         # We also need to just calculate the residual for the AD mode to initialize vars like URes
#         # We do not print the residual for AD, though
#         # NOTE: This was taken from DAFoamSolver.solve_residual_equation
#         dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")

#         # Initialize our newton convergence flag
#         converged = False

#         # Initialize our ROM/FOM states, solver, and ROM/FOM residuals 
#         y_rom = np.zeros((num_modes, ))
#         y_fom = y_fom_ref + D*(phi@y_rom)
#         dafoam_instance.setStates(y_fom)
#         r_fom = dafoam_instance.getResiduals()
#         r_rom = phi.T@(W*r_fom)
#         r_rom_norm = np.linalg.norm(r_rom)

#         # Some plotting for diagnostics
#         #----------------------
#         import matplotlib.pyplot as plt
#         # Plotting coefficients
#         # Initialize figure
#         plt.ion()  # Interactive mode on
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,6))

#         # --- subplot 1: ROM state progression ---
#         lines = [ax1.plot([], [], label=f'element {i}')[0] for i in range(num_modes)]

#         # Store data
#         x_data = []                     # iteration index
#         y_data = [[] for _ in range(num_modes) ]  # data for each element

#         # --- subplot 2: FOM update ---
#         fom_state_line,    = ax2.plot(np.zeros_like(y_fom_ref), 'r-')

#         # --- subplot 3: residual update ---
#         fom_residual_line, = ax3.plot(np.zeros_like(y_fom_ref), 'b')

#         x_data.append(0)
#         for i, line in enumerate(lines):
#             y_data[i].append(y_rom[i])
#             line.set_data(x_data, y_data[i])
#             plt.draw()
#             plt.pause(0.1)
#         #-------------------------

#         # Main Newton iteration loop
#         for iter in range(max_iters):
#             print('\n')
#             print('-------------------------------------')
#             print(f'ROM Newton iteration {iter}')
#             print('-------------------------------------')

#             # (Re)compute Jacobian on first iteration or if not reusing the same Jacobian
#             if iter%update_jac_frequency == 0 or iter == 0:
#                 print('Computing reduced Jacobian...')
#                 J_rom = self._compute_reduced_jacobian(phi, y_fom, W, D)
#                 self.reduced_jacobian = J_rom

#             # Compute step (robust linear solve with lstsq fallback)
#             print('Updating ROM and FOM states...')
#             try:
#                 delta_y_rom = np.linalg.solve(J_rom, -r_rom)
#             except np.linalg.LinAlgError:
#                 # fallback to least-squares if matrix singular or near-singular
#                 delta_y_rom, *_ = np.linalg.lstsq(J_rom, -r_rom, rcond=1e-12)
#                 delta_y_rom = delta_y_rom.reshape(-1,)
#                 print('\tWarning: reduced Jacobian singular; used least-squares fallback.')

#             y_rom  += delta_y_rom
#             y_fom = y_fom_ref + D*(phi@y_rom)

#             # evaluate residual at candidate state
#             dafoam_instance.setStates(y_fom)
#             r_fom = dafoam_instance.getResiduals()
#             r_rom = phi.T@(W*r_fom)
#             r_rom_norm = np.linalg.norm(r_rom)

#             # Save the initial value of the reduced residual
#             if iter == 0:
#                 r_rom_norm0 = r_rom_norm.copy()

#             # Will print out the relevant residual statistics here
#             print('Residual summary:')
#             print('\t{:<30} : {:.4E}'.format('FOM residual norm', np.linalg.norm(r_fom)))
#             print('\t{:<30} : {:.4E}'.format('ROM residual norm', r_rom_norm))
#             print('\t{:<30} : {:.4E}'.format('Start/current ROM norm ratio', r_rom_norm/r_rom_norm0))

#             # Exit condition
#             if r_rom_norm < tolerance:
#                 converged = True
#                 break

#             # Upate plots
#             x_data.append(iter+1)
#             for i, line in enumerate(lines):
#                 y_data[i].append(y_rom[i])
#                 line.set_data(x_data, y_data[i])
#             ax1.relim()
#             ax1.autoscale_view()

#             fom_state_line.set_ydata(y_fom)
#             ax2.relim()
#             ax2.autoscale_view()

#             fom_residual_line.set_ydata(r_fom)
#             ax3.relim()
#             ax3.autoscale_view()
#             plt.draw()
#             plt.pause(0.1)
        
#         # Return NaN array as output (for the optimizer), otherwise return rom states
#         if converged:
#             print('\n\n Specified tolerance reached!')
#             # Update reduced jacobain to most recent state
#             print('Updating Jacobian to most recent state value...')
#             J_rom = self._compute_reduced_jacobian(phi, y_fom, W, D)
#             self.reduced_jacobian = J_rom

#             output_vals['dafoam_rom_states'] = y_rom
#         else:
#             print('Warning: ROM iteration limit reached!')
#             output_vals['dafoam_rom_states'] = np.full((num_modes, ), np.nan)