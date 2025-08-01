import numpy as np
import csdl_alpha as csdl

# =====================================
# (1) IDWarp Custom Explicit Operation
#     motivated by Line788: DAFoamWarper in mphys_dafoam.py
# =====================================
class DAFoamMeshWarper(csdl.CustomExplicitOperation):
    def __init__(self, dafoam_instance):
        super().__init__()
        self.dafoam_instance = dafoam_instance
        
    
    def evaluate(self, x_surf):
        self.declare_input('x_surf', x_surf)
        solver_grid_size = self.dafoam_instance.mesh.getSolverGrid().shape
        x_vol = self.create_output('x_vol', solver_grid_size)
        return x_vol

    def compute(self, inputs, outputs):
        """
        Forward-mode evaluation: mesh warping; x_surf -> x_vol
        """
        dafoam_instance = self.dafoam_instance

        x_surf = inputs['x_surf']
        
        dafoam_instance.setSurfaceCoordinates(x_surf.reshape(-1,3))
        dafoam_instance.mesh.warpMesh()                         # Mesh warp
        outputs['x_vol'] = dafoam_instance.mesh.getSolverGrid() # Take the updated volume mesh        

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode): 
        # Reverse-mode sensitivity: w = (dx_s / dx_v)^T (v)   v:dxv, w:dxs, 
        dafoam_instance = self.dafoam_instance
        
        if mode == 'rev':
            if 'x_vol' in d_outputs:
                if 'x_surf' in d_inputs:
                    dxV = d_outputs['x_vol']
                    dafoam_instance.mesh.warpDeriv(dxV)
                    dxS = dafoam_instance.mesh.getdXs()
                    dxS = dafoam_instance.mapVector(dxS, dafoam_instance.allWallsGroup, dafoam_instance.designSurfacesGroup)
                    # d_inputs['x_surf'] += dxS.flatten()
                    d_inputs['x_surf'] = dxS.flatten()
        
        
        
