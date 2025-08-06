from dafoam import PYDAFOAM
import numpy as np
import csdl_alpha as csdl
from petsc4py import PETSc
import os


USE_CHANGE_DIRECTORY_WORKAROUND = True

def instantiateDAFoam(options, comm, run_directory=None, mesh_options=None):
    # CHANGE DIRECTORY WORKAROUND 1/4
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



class DAFoamSolver(csdl.experimental.CustomImplicitOperation):
    def __init__(self, dafoam_instance):
        super().__init__()

        self.dafoam_instance    = dafoam_instance
        self.solution_counter   = 1

        # initialize the dRdWT matrix-free matrix in DASolver
        dafoam_instance.solverAD.initializedRdWTMatrixFree()

        # create the adjoint vector
        self.num_state_elements = dafoam_instance.getNLocalAdjointStates()
        self.psi = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.psi.setSizes((self.num_state_elements, PETSc.DECIDE), bsize=1)
        self.psi.setFromOptions()
        self.psi.zeroEntries()

        # If true, we need to compute the coloring
        if dafoam_instance.getOption("adjEqnSolMethod") == "fixedPoint":
            self.runColoring = False
        else:
            self.runColoring = True 


    def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup):
        # Read daOptions to set proper inputs
        inputDict = self.dafoam_instance.getOption("inputInfo")
        for inputName in inputDict.keys():
            if "solver" in inputDict[inputName]["components"]:
                self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))   

        # Set outputs
        dafoam_solver_states = self.create_output('dafoam_solver_states', (self.num_state_elements,))

        return dafoam_solver_states


    def solve_residual_equations(self, input_vals, output_vals):
        dafoam_instance = self.dafoam_instance

        # set the solver input, including mesh, boundary etc.
        dafoam_instance.set_solver_input(input_vals)

        # before running the primal, we need to check if the mesh
        # quality is good

        #*****************
        meshOK = 1#dafoam_instance.solver.checkMesh()
        print('SKIPPING MESH CHECK. BE SURE TO CHANGE THIS BACK WHEN DONE DEBUGGING')
        #*****************

        # if the mesh is not OK, do not run the primal
        if meshOK != 1:
            dafoam_instance.solver.writeFailedMesh()
            print('Mesh is not OK!')
            return

        # Run primal
        dafoam_instance()

        # if the primal fails, do not set states and return
        if dafoam_instance.primalFail != 0:
            print('Primal solution failed!')
            # TODO: Remove return, instead assign nans to state vector and return

        # after solving the primal, we need to print its residual info
        if dafoam_instance.getOption("useAD")["mode"] == "forward":
            dafoam_instance.solverAD.calcPrimalResidualStatistics("print")
        else:
            dafoam_instance.solver.calcPrimalResidualStatistics("print")

        # assign the computed flow states to outputs
        states = dafoam_instance.getStates()
        if dafoam_instance.primalFail != 0:
            print('Primal solution failed!')
            states = np.nan

        output_vals['dafoam_solver_states'] = states

        # set states
        dafoam_instance.setStates(states)

        # We also need to just calculate the residual for the AD mode to initialize vars like URes
        # We do not print the residual for AD, though
        dafoam_instance.solverAD.calcPrimalResidualStatistics("calc")


    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        dafoam_instance = self.dafoam_instance

        # CHANGE DIRECTORY WORKAROUND 3/4
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

        # convert the solution vector to array and assign it to d_residuals
        d_residuals['dafoam_solver_states'] = dafoam_instance.vec2Array(self.psi)

        # if the adjoint solution fail, we return analysisError and let the optimizer handle it
        if fail:
            #raise AnalysisError("Adjoint solution failed!")

            #*****************
            print("----------- !!!ADJOINT SOLUTION FAILED!!! -----------")
            print("Continuing anyways! (Fix this in code later)")
            #*****************
        
        # CHANGE DIRECTORY WORKAROUND 4/4
        if USE_CHANGE_DIRECTORY_WORKAROUND:
            os.chdir(previous_directory)
        # -------------------------------


    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
        dafoam_instance = self.dafoam_instance

        # assign the states in outputs to the OpenFOAM flow fields
        # NOTE: this is not quite necessary because setStates have been called before in the solve_nonlinear
        # here we call it just be on the safe side
        dafoam_instance.setStates(output_vals['dafoam_solver_states'])

        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamSolver')

        if 'dafoam_solver_states' in d_residuals:
             # get the reverse mode AD seed from d_residuals
            seed = d_residuals['dafoam_solver_states']
 
            # loop over all inputs keys and compute the matrix-vector products accordingly
            inputDict = dafoam_instance.getOption("inputInfo")
            for inputName in list(input_vals.keys()):
                inputType = inputDict[inputName]["type"]
                jacInput = input_vals[inputName].copy()
                product = np.zeros_like(jacInput)
                dafoam_instance.solverAD.calcJacTVecProduct(
                    inputName,
                    inputType,
                    jacInput,
                    "aero_residuals",
                    "residual",
                    seed,
                    product,
                )
                d_inputs[inputName] += product


    def _updateKSPTolerances(self, psi, dFdW, ksp):
        # Here we need to manually update the KSP tolerances because the default
        # relative tolerance will always want to converge the adjoint to a fixed
        # tolerance during the LINGS adjoint solution. However, what we want is
        # to converge just a few orders of magnitude. Here we need to bypass the
        # rTol in Petsc and manually calculate the aTol.

        dafoam_instance = self.dafoam_instance
        # calculate the initial residual for the adjoint before solving
        rArray = np.zeros(self.num_state_elements)
        jacInput = dafoam_instance.getStates()
        seed = dafoam_instance.vec2Array(psi)
        dafoam_instance.solverAD.calcJacTVecProduct(
            'dafoam_solver_states',
            "stateVar",
            jacInput,
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



class DAFoamFunctions(csdl.CustomExplicitOperation):
    def __init__(self, dafoam_instance):
        super().__init__()
        self.dafoam_instance = dafoam_instance


    def evaluate(self, dafoam_solver_states:csdl.Variable, dafoam_input_variables_group:csdl.VariableGroup):
        # Solver states is the easy one
        self.declare_input("dafoam_solver_states", dafoam_solver_states)
        
        # Read daOptions to set proper inputs
        inputDict = self.dafoam_instance.getOption("inputInfo")
        for inputName in inputDict.keys():
            if "function" in inputDict[inputName]["components"]:
                self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))  
        
        # Initialize output
        dafoam_function_output = csdl.VariableGroup()

        # Read daOptions to get outputs
        outputDict = self.dafoam_instance.getOption("function")
        for outputName in outputDict.keys():
            setattr(dafoam_function_output, outputName, self.create_output(outputName, (1, )))

        return dafoam_function_output


    def compute(self, input_vals, output_vals):
        dafoam_instance = self.dafoam_instance

        # Update solver states (I don't think this is really necessary... Only the function values from the
        # latest evaluation is retrieved - for CL and CD, at least.)
        dafoam_instance.setStates(input_vals['dafoam_solver_states'])

        # Initialize and evaluate
        funcs = {}
        dafoam_instance.evalFunctions(funcs)

        # Read daOptions to get outputs, and assign them to respective outputs
        outputDict = self.dafoam_instance.getOption("function")
        for outputName in outputDict.keys():
            output_vals[outputName] = funcs[outputName]
            output_vals[outputName] = funcs[outputName]


    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
        dafoam_instance = self.dafoam_instance

        # Update quantities
        dafoam_instance.setStates(input_vals['dafoam_solver_states'])

        # Can't do forward mode
        if mode == 'fwd':
            raise NotImplementedError('forward mode has not been implemented for DAFoamFunctions')
        
        inputDict = dafoam_instance.getOption("inputInfo")
        for functionName in list(d_outputs.keys()):

            seed = d_outputs[functionName]

            # if the seed is zero, do not compute
            if abs(seed) < 1e-12:
                continue

            for inputName in list(d_inputs.keys()):
                # compute dFdW * seed
                if inputName == 'dafoam_solver_states':
                    jacInput = input_vals['dafoam_solver_states']
                    product = np.zeros_like(jacInput)
                    dafoam_instance.solverAD.calcJacTVecProduct(
                        'dafoam_solver_states',
                        "stateVar",
                        jacInput,
                        functionName,
                        "function",
                        seed,
                        product,
                    )
                    d_inputs['dafoam_solver_states'] += product
                else:
                    inputType = inputDict[inputName]["type"]
                    jacInput = input_vals[inputName]
                    product = np.zeros_like(jacInput)
                    dafoam_instance.solverAD.calcJacTVecProduct(
                        inputName,
                        inputType,
                        jacInput,
                        functionName,
                        "function",
                        seed,
                        product,
                    )
                    d_inputs[inputName] += product



def compute_dafoam_input_variables(dafoam_instance, ambient_conditions_group:csdl.VariableGroup, flight_conditions_group:csdl.VariableGroup, aerodynamic_volume_coordinates:csdl.Variable):
    # Currently expect the ambient_conditions_group to, at minimum, contain the following variables:
    # T_K   (Temperature [K])
    # P_Pa  (Pressure [Pa])
    # a_m_s (Speed of sound [m/s])

    # Currently expect the flight_conditions_group to, at minimum, contain the following variables:
    # airspeed_m_s     (airspeed [m/s])
    # angle_of_attack  (Angle of attack [deg])
    # OR
    # mach_number      (Mach number)
    # angle_of_attack  (Angle of attack [deg])
    
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
            patchVelocity = csdl.concatenate((flight_conditions_group.airspeed_m_s, flight_conditions_group.angle_of_attack))
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



class DAFoamROM(csdl.experimental.CustomImplicitOperation):
    def __init__(self, dafoam_instance, reduced_residual_norm=1e-6, phi=None, states_ref=None):
        super().__init__()
        self.dafoam_instance        = dafoam_instance
        self.solution_counter       = 1
        self.reduced_residual_norm  = reduced_residual_norm
        
        # Check if user supplies a constant phi array (this cooresponds to a global basis)
        if phi is not None:
            self.constant_phi  = True
            self.phi           = constant_phi
        else:
            self.constant_phi  = False

        # Check if user supplies a constant reference state (this cooresponds to a global reference)
        if states_ref is not None:
            self.constant_states_ref = True
            self.states_ref    = states_ref
        else:
            self.constant_states_ref = False


    def evaluate(self, dafoam_input_variables_group:csdl.VariableGroup):
        # Read daOptions to set proper inputs
        inputDict = self.dafoam_instance.getOption("inputInfo")
        for inputName in inputDict.keys():
            if "solver" in inputDict[inputName]["components"]:
                self.declare_input(inputName, getattr(dafoam_input_variables_group, inputName))   

        # Set outputs
        dafoam_solver_states = self.create_output('dafoam_solver_states', (self.num_state_elements,))

        return dafoam_solver_states


    def solve_residual_equations(self, input_vals, output_vals):
        dafoam_instance = self.dafoam_instance
        max_iter        = 10

        # Make sure solver is updated with the most recent input values
        dafaom_instance.set_solver_input(input_vals)

        # Determine if our basis is constant, or an input value
        if self.constant_phi:
            phi = self.phi
        else:
            phi = input_vals['phi']

        # Determine if our reference state is constant, or an input value
        if self.constant_states_ref:
            states_ref = self.states_ref
        else:
            states_ref = input_vals['states_ref']

        # Main Newton iteration loop
        iter = 0
        while iter < max_iter and not converged:
            states = states+ref + phi@abs
            if iter == 0 or not reuse_jac:
                # Update states and update solver
                states = states_ref + phi@a
                dafoam_instance.setStates(states)

                # Compute residual at this new states value
                res         = dafoam_instance.getResiduals()

                # Reduce the residual and check if the tolerance is satisfied
                res_reduced = phi.T@res
                res_reduced_norm = np.linalg.norm(res_reduced)
                if res_reduced_norm < tol:
                    break
            
                if not reuse_jac or iter == 0:
                    jac_reduced = _compute_reduced_jacobian(phi, states)
                





        else:
            raise NotImplementedError('DAFoamRomSolver solve_residual_equations not implemented yet')

    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        raise NotImplementedError('DAFoamRomSolver solve_residual_equations not implemented yet')

    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
        raise NotImplementedError('DAFoamRomSolver solve_residual_equations not implemented yet')



    def _compute_reduced_jacobian(phi, states):
        dafoam_instance     = self.dafoam_instance
        phi_shape           = phi.shape
        full_jac_times_phi  = np.zeros_like(phi)

        # Update solver for desired states
        dafoam_instance.setStates(states)
        
        # Loop over columns of phi
        for i in range(phi_shape[1]):
            seed    = phi[:, i]
            product = np.zeros()
            dafoam_instance.solverAD.calcJacTVecProduct(
                'dafoam_solver_states',
                "stateVar",
                dafoam_instance.getStates(),
                'aero_residuals',
                "residual",
                seed,
                product,
            )

            full_jac_times_phi[:, i] = product

        reduced_jac = phi.T @ full_jac_times_phi

        return reduced_jac


