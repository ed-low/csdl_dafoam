from petsc4py import PETSc
import numpy as np

# Open the binary viewer for reading
viewer = PETSc.Viewer().createBinary("openfoam_airfoil/dRdWTPC.bin", PETSc.Viewer.Mode.READ)

# Load the matrix
LT = PETSc.Mat().load(viewer)

# Create numpy vectors
u = np.random.rand(LT.getSize()[0])
v = np.random.rand(LT.getSize()[0])

# Convert numpy array -> PETSc Vec
u_petsc = PETSc.Vec().createSeq(len(u))
u_petsc.setArray(u)

v_petsc = PETSc.Vec().createSeq(len(v))
v_petsc.setArray(v)

Lu_petsc = LT.createVecLeft()   # allocate the result vector with correct size
LT.multTranspose(u_petsc, Lu_petsc)

# Convert back to numpy
Lu = Lu_petsc.getArray()


vTLu = v_petsc.dot(Lu_petsc)

print(vTLu)


LTv_petsc = LT.createVecLeft() 
LT.mult(v_petsc, LTv_petsc)

LTv = LTv_petsc.getArray()

LTv_Tu = u_petsc.dot(LTv_petsc)
print(LTv_Tu)