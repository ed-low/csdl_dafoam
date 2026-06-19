import csdl_alpha as csdl
import numpy as np
from csdl_dafoam.core.rom.rom_models import BaseModel
from csdl_dafoam.core.rom.rom_solver import BaseSolver, SolverResult


# region CSDLROMWrapper
class CSDLROMWrapper(csdl.experimental.CustomImplicitOperation):
    def __init__(self, model:BaseModel, solver:BaseSolver, constant_initial_rom_state:np.ndarray|csdl.Variable=None, write_unconverged_solutions_to_file:bool=False):
        super().__init__()
        solver.model    = model
        self.model      = model
        self.solver     = solver
        self.print_fn   = model.print_fn
        self.write_unconverged_solutions_to_file = write_unconverged_solutions_to_file
        self.constant_intitial_rom_state         = constant_initial_rom_state

        self._cached_result    = None # Will be set and updated during solve_residual_equations
        # self._input_info       = {}   # Set up during evaluate
        self._state_name       = None # Set up during evaluate
        self._state_info       = None # Set up during evaluate

        # When the initial ROM state is a CSDL Variable we declare it as a (passive) input
        self._initial_state_input_name  = "_initial_rom_state"
        self._initial_state_is_variable = isinstance(constant_initial_rom_state, csdl.Variable)


    # region evaluate
    def evaluate(self):
        # Let model return the inputs and output dimensions
        # Input dict will be {name:variable}, output will be {name:str, shape:tuple}
        (csdl_input_declaration_dict, 
         csdl_output_creation_dict) = self.model.evaluate_input_output()

        for name, csdl_var in csdl_input_declaration_dict.items():
            self.declare_input(name, csdl_var)
            # self._input_info[name] = {"shape":csdl_var.shape} # Update the output shape dict

        # Declare the initial ROM state as a passive input (only when it's a Variable),
        if self._initial_state_is_variable:
            self.declare_input(self._initial_state_input_name, self.constant_intitial_rom_state)

        # Update output info
        self._state_name  = csdl_output_creation_dict["name"]
        self._state_shape = csdl_output_creation_dict["shape"]

        # Expecting the output to be the ROM states
        rom_state = self.create_output(self._state_name, self._state_shape)
        
        return rom_state
        
    
    # region solve_residual_equaitons
    def solve_residual_equations(self, input_vals, output_vals):
        model  = self.model
        solver = self.solver

        output_name  = self._state_name
        output_shape = self._state_shape

        # Let the model handle the input update
        model.update_from_input_vals(input_vals=input_vals)

        # Use cached state, otherwise start at zero
        rom_state0_user = self.constant_intitial_rom_state
        if rom_state0_user is None:
            rom_state0 = np.zeros(output_shape) if self._cached_result is None else self._cached_result.rom_state.copy()
        elif self._initial_state_is_variable:
            # Pull the freshly-computed value from input_vals (graph-ordered), not .value.
            rom_state0 = np.asarray(input_vals[self._initial_state_input_name]).reshape(output_shape)
        else:
            rom_state0 = rom_state0_user

        result     = solver.solve(initial_state=rom_state0)
        rom_state  = result.rom_state #.copy() Might need the copy?

        # Set solution (to NaN if failed)
        output_vals[output_name] = rom_state if result.converged else np.full_like(rom_state, np.nan)

        # Cache result
        self._cached_result = result
        

    # region apply_inverse_jacobian
    def apply_inverse_jacobian(self, input_vals, output_vals, d_outputs, d_residuals, mode):
        model       = self.model
        solver      = self.solver
        output_name = self._state_name
        rom_state   = output_vals[output_name]
        vec         = d_outputs[output_name]

        if np.linalg.norm(rom_state - self._cached_result.rom_state) / np.linalg.norm(self._cached_result.rom_state) > 1e-6:
            model.print_fn("WARNING: Cached ROM state seems to be different than supplied state in apply_inverse_jacobian?")

        if mode == "fwd":
            raise NotImplementedError("Forward mode not yet implemented for CSDL ROM class.")
        
        lam = solver.adjoint_solve(result=self._cached_result, rhs=vec, mode=mode)

        # Write solution to file if the model has the capability
        # We'll write the cached solution if we have NaNs
        if any(np.isnan(rom_state)) and self.write_unconverged_solutions_to_file:
                model.write_solution(rom_state=self._cached_result.rom_state)
        else:
            model.write_solution(rom_state=rom_state)

        d_residuals[output_name] += lam

    
    # region compute_jacvec_product
    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, d_residuals, mode):
        model       = self.model
        output_name = self._state_name
        rom_state   = output_vals[output_name]
        vec         = d_residuals[output_name]

        if mode == "fwd":
            raise NotImplementedError("Forward mode not yet implemented for CSDL ROM class.")
        
        input_sensitivities = model.input_jacvec_transpose(rom_state=rom_state, vec=vec, input_vals=input_vals, mode=mode)

        for name, value in input_sensitivities.items():
            if name in d_inputs:
                d_inputs[name] += value 


    # region evaluate_residuals
    def evaluate_residuals(self, input_vals, output_vals, residual_vals):
        self.model.update_from_input_vals(input_vals=input_vals)
        output_name = self._state_name
        rom_state   = output_vals[output_name]
        residual_vals[output_name] = self.model.evaluate_residuals(rom_state=rom_state)
