import lsdo_geo

def export_geo2step(filename, geometry: lsdo_geo.Geometry):
    """
    Exports a full geometry of B-spline surfaces to a .step file.

    Parameters:
        filename (str): The name of the .step file to export.
        geometry (lsdo_geo.Geometry): The geometry containing B-spline surfaces.
    """
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCC.Core.Geom import Geom_BSplineSurface, Geom_BezierSurface
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static_SetCVal
    from OCC.Core.gp import gp_Pnt
    # Initialize STEP writer
    step_writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP203")

    # Iterate through all functions in the geometry
    for geo_fn in geometry.functions.values():
        space: lfs.BSplineSpace = geo_fn.space

        # Create a B-spline surface
        num_u, num_v = space.coefficients_shape
        control_points = geo_fn.coefficients.value.reshape(num_u, num_v, 3)

        # Convert control points to TColgp_Array2OfPnt
        ctrl_pts = TColgp_Array2OfPnt(1, num_u, 1, num_v)
        for i in range(num_u):
            for j in range(num_v):
                x, y, z = control_points[i, j]
                ctrl_pts.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))

        # convert knots to TColStd_Array1OfReal
        num_unique_knots_u = len(space.knots[space.knot_indices[0]]) - space.degree[0]*2
        num_unique_knots_v = len(space.knots[space.knot_indices[1]]) - space.degree[1]*2

        knots_u = TColStd_Array1OfReal(1, num_unique_knots_u)
        for i in range(num_unique_knots_u):
            knots_u.SetValue(i + 1, space.knots[space.knot_indices[0]][i + space.degree[0]])
        knots_v = TColStd_Array1OfReal(1, num_unique_knots_v)
        for i in range(num_unique_knots_v):
            print(space.knots[space.knot_indices[1]][i + space.degree[1]])
            knots_v.SetValue(i + 1, space.knots[space.knot_indices[1]][i + space.degree[1]])

        # make multiciplicity vector (all 1s, except ends are 1+degree) of type TColStd_Array1OfInteger

        mult_u = TColStd_Array1OfInteger(1, num_unique_knots_u)
        mult_v = TColStd_Array1OfInteger(1, num_unique_knots_v)
        for i in range(num_unique_knots_u):
            if i == 0 or i == num_unique_knots_u - 1:
                mult_u.SetValue(i + 1, space.degree[0] + 1)
            else:
                mult_u.SetValue(i + 1, 1)
        for i in range(num_unique_knots_v):
            if i == 0 or i == num_unique_knots_v - 1:
                mult_v.SetValue(i + 1, space.degree[1] + 1)
            else:
                mult_v.SetValue(i + 1, 1)


        # print(space.knots[space.knot_indices[0]].tolist())
        # exit()

        # bspline_surface = Geom_BezierSurface(ctrl_pts)

        # Create the B-spline surface
        bspline_surface = Geom_BSplineSurface(
            ctrl_pts,
            knots_u,
            knots_v,
            mult_u,
            mult_v,
            space.degree[0],
            space.degree[1],
        )

        # Create a face from the B-spline surface
        face = BRepBuilderAPI_MakeFace(bspline_surface, 1e-6).Face()

        # Add the face to the STEP writer
        step_writer.Transfer(face, STEPControl_AsIs)

    # Write the STEP file
    status = step_writer.Write(filename)
    if status != 0:
        raise RuntimeError(f"Failed to write STEP file: {filename}")