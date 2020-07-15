"""
This file provides kernels for par loops.
These are contained in this file as functions so that they can be tested separately.
"""

import numpy as np
from firedrake import dx
from firedrake.parloops import par_loop, READ, INC, WRITE
from pyop2 import ON_TOP, ON_BOTTOM


class GaussianElimination(object):
    """
    A kernel for performing Gaussian elimination locally in each element
    for the BoundaryRecoverer procedure.

    The apply method takes a scalar field in DG1, the coordinates of the DoFs
    and the effective" coordinates of the DoFs. These "effective" coordinates
    correspond to the locations of the DoFs from the original space (i.e.
    the one being recovered from).

    This kernel expands the field in the cell as a local Taylor series,
    keeping only the linear terms. By assuming that the field would be
    correct for the effective coordinates, we can extrapolate to find
    what the field would be at the actual coordinates, which involves
    inverting a local matrix -- which is done by this kernel using
    Gaussian elimination.

    :arg DG1: A 1st order discontinuous Galerkin FunctionSpace.
    """
    def __init__(self, DG1):

        shapes = {"nDOFs": DG1.finat_element.space_dimension(),
                  "dim": DG1.mesh().topological_dimension()}

        # EFF_COORDS are the effective coordinates
        # ACT_COORDS are the actual coordinates
        # DG1_OLD is the original field
        # DG1 is the target field
        # NUM_EXT is the field containing number of exterior nodes

        # In 1D EFF_COORDS and ACT_COORDS have only a single index
        # We can't generalise the expression without causing an error
        # So here we write special bits of code for the 1D case vs multi-dimensional
        if shapes['dim'] == 1:
            eff_coord_expr = (
                """
                A[i,0] = 1.0
                A[i,1] = EFF_COORDS[i]
                """)
            act_coord_expr = (
                """
                DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii]
                """)
        else:
            eff_coord_expr = (
                """
                A[i,0] = 1.0
                A[i,1] = EFF_COORDS[i,0]
                if {nDOFs} == 3
                    A[i,2] = EFF_COORDS[i,1]
                elif {nDOFs} == 4
                    A[i,2] = EFF_COORDS[i,1]
                    A[i,3] = EFF_COORDS[i,0]*EFF_COORDS[i,1]
                elif {nDOFs} == 6
                """
                # N.B we use {nDOFs} - 1 to access the z component in 3D cases
                # Otherwise loopy tries to search for this component in 2D cases, raising an error
                """
                    A[i,2] = EFF_COORDS[i,1]
                    A[i,3] = EFF_COORDS[i,{dim}-1]
                    A[i,4] = EFF_COORDS[i,0]*EFF_COORDS[i,{dim}-1]
                    A[i,5] = EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                elif {nDOFs} == 8
                    A[i,2] = EFF_COORDS[i,1]
                    A[i,3] = EFF_COORDS[i,0]*EFF_COORDS[i,1]
                    A[i,4] = EFF_COORDS[i,{dim}-1]
                    A[i,5] = EFF_COORDS[i,0]*EFF_COORDS[i,{dim}-1]
                    A[i,6] = EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                    A[i,7] = EFF_COORDS[i,0]*EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                end
                """).format(**shapes)

            act_coord_expr = (
                """
                if {nDOFs} == 2
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0]
                elif {nDOFs} == 3
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1]
                elif {nDOFs} == 4
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1]
                elif {nDOFs} == 6
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,{dim}-1] + a[4]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,{dim}-1] + a[5]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1]
                elif {nDOFs} == 8
                    DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1] + a[4]*ACT_COORDS[iiii,{dim}-1] + a[5]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,{dim}-1] + a[6]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1] + a[7]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1]
                end
                """).format(**shapes)

        shapes['act_coord_expr'] = act_coord_expr
        shapes['eff_coord_expr'] = eff_coord_expr

        domain = (
            """
            {{[i, ii_loop, jj_loop, kk, ll_loop, mm, iii_loop, kkk_loop, iiii]:
            0 <= i < {nDOFs} and 0 <= ii_loop < {nDOFs} and
            0 <= jj_loop < {nDOFs} and 0 <= kk < {nDOFs} and
            0 <= ll_loop < {nDOFs} and 0 <= mm < {nDOFs} and
            0 <= iii_loop < {nDOFs} and 0 <= kkk_loop < {nDOFs} and
            0 <= iiii < {nDOFs}}}
            """).format(**shapes)

        instrs = (
            """
            <int> ii = 0
            <int> jj = 0
            <int> ll = 0
            <int> iii = 0
            <int> jjj = 0
            <int> i_max = 0
            <float64> A_max = 0.0
            <float64> temp_f = 0.0
            <float64> temp_A = 0.0
            <float64> c = 0.0
            <float64> f[{nDOFs}] = 0.0
            <float64> a[{nDOFs}] = 0.0
            <float64> A[{nDOFs},{nDOFs}] = 0.0
            """
            # We are aiming to find the vector a that solves A*a = f, for matrix A and vector f.
            # This is done by performing row operations (swapping and scaling) to obtain A in upper diagonal form.
            # N.B. several for loops must be executed in numerical order (loopy does not necessarily do this).
            # For these loops we must manually iterate the index.
            """
            if NUM_EXT[0] > 0.0
            """
            # only do Gaussian elimination for elements with effective coordinates
            """
                for i
            """
            # fill f with the original field values and A with the effective coordinate values
            """
                    f[i] = DG1_OLD[i]
                    {eff_coord_expr}
                end
            """
            # now loop through rows/columns of A
            """
                for ii_loop
                    A_max = abs(A[ii,ii])
                    i_max = ii
            """
            # loop to find the largest value in the ii-th column
            # set i_max as the index of the row with this largest value.
            """
                    jj = ii + 1
                    for jj_loop
                        if jj < {nDOFs}
                            if abs(A[jj,ii]) > A_max
                                i_max = jj
                            end
                            A_max = fmax(A_max, abs(A[jj,ii]))
                        end
                        jj = jj + 1
                    end
            """
            # if the max value in the ith column isn't in the ii-th row, we must swap the rows
            """
                    if i_max != ii
            """
            # swap the elements of f
            """
                        temp_f = f[ii]  {{id=set_temp_f}}
                        f[ii] = f[i_max]  {{id=set_f_imax, dep=set_temp_f}}
                        f[i_max] = temp_f  {{id=set_f_ii, dep=set_f_imax}}
            """
            # swap the elements of A
            # N.B. kk runs from ii to (nDOFs-1) as elements below diagonal should be 0
            """
                        for kk
                            if kk > ii - 1
                                temp_A = A[ii,kk]  {{id=set_temp_A}}
                                A[ii, kk] = A[i_max, kk]  {{id=set_A_ii, dep=set_temp_A}}
                                A[i_max, kk] = temp_A  {{id=set_A_imax, dep=set_A_ii}}
                            end
                        end
                    end
            """
            # scale the rows below the ith row
            """
                    ll = ii + 1
                    for ll_loop
                        if ll > ii
                            if ll < {nDOFs}
            """
            # find scaling factor
            """
                                c = - A[ll,ii] / A[ii,ii]
                                for mm
                                    A[ll, mm] = A[ll, mm] + c * A[ii,mm]
                                end
                                f[ll] = f[ll] + c * f[ii]
                            end
                        end
                        ll = ll + 1
                    end
                    ii = ii + 1
                end
            """
            # do back substitution of upper diagonal A to obtain a
            """
                iii = 0
                for iii_loop
            """
            # jjj starts at the bottom row and works upwards
            """
                    jjj = {nDOFs} - iii - 1  {{id=assign_jjj}}
                    a[jjj] = f[jjj]   {{id=set_a, dep=assign_jjj}}
                    for kkk_loop
                        if kkk_loop > {nDOFs} - iii_loop - 1
                            a[jjj] = a[jjj] - A[jjj,kkk_loop] * a[kkk_loop]
                        end
                    end
                    a[jjj] = a[jjj] / A[jjj,jjj]
                    iii = iii + 1
                end
            end
            """
            # Do final loop to assign output values
            """
            for iiii
            """
            # Having found a, this gives us the coefficients for the Taylor expansion with the actual coordinates.
            """
                if NUM_EXT[0] > 0.0
                {act_coord_expr}
            """
            # if element is not external, just use old field values.
            """
                else
                    DG1[iiii] = DG1_OLD[iiii]
                end
            end
            """).format(**shapes)

        self._kernel = (domain, instrs)

    def apply(self, v_DG1_old, v_DG1, act_coords, eff_coords, num_ext):
        """
        Performs the par loop for the Gaussian elimination kernel.

        :arg v_DG1_old: the originally recovered field in DG1.
        :arg v_DG1: the new target field in DG1.
        :arg act_coords: the actual coordinates in vec DG1.
        :arg eff_coords: the effective coordinates of the recovery in vec DG1.
        :arg num_ext: the number of exterior DOFs in the cell, in DG0.
        """

        par_loop(self._kernel, dx,
                 {"DG1_OLD": (v_DG1_old, READ),
                  "DG1": (v_DG1, WRITE),
                  "ACT_COORDS": (act_coords, READ),
                  "EFF_COORDS": (eff_coords, READ),
                  "NUM_EXT": (num_ext, READ)},
                 is_loopy_kernel=True)


class Average(object):
    """
    A kernel for the Averager object.

    For vertices shared between cells, it computes the average
    value from the neighbouring cells.

    :arg V: The FunctionSpace of the target field for the Averager.
    """

    def __init__(self, V):

        shapes = {"nDOFs": V.finat_element.space_dimension(),
                  "dim": np.prod(V.shape, dtype=int)}

        domain = "{{[i, j]: 0 <= i < {nDOFs} and 0 <= j < {dim}}}".format(**shapes)

        # Loop over node extent and dof extent
        # vo is v_out, v is the function in, w is the weight
        # NOTE: Any bcs on the function v should just work.
        instrs = (
            """
            for i
                for j
                    vo[i,j] = vo[i,j] + v[i,j] / w[i,j]
                end
            end
            """)

        self._kernel = (domain, instrs)

    def apply(self, v_out, weighting, v_in):
        """
        Perform the averaging par loop.

        :arg v_out: the continuous target field.
        :arg weighting: the weights to be used for the averaging.
        :arg v_in: the input field.
        """

        par_loop(self._kernel, dx,
                 {"vo": (v_out, INC),
                  "w": (weighting, READ),
                  "v": (v_in, READ)},
                 is_loopy_kernel=True)


class NodeMultiplicity(object):
    """
    A kernel for finding the weights for the Averager object.
    These are simply the multiplicity of nodes shared between cells.

    :arg V: The FunctionSpace of the target field for the Averager.
    """

    def __init__(self, V, coords_from_V0=None, coords=None):

        shapes = {"nDOFs": V.finat_element.space_dimension(),
                  "dim": np.prod(V.shape, dtype=int)}

        domain = "{{[i, j]: 0 <= i < {nDOFs} and 0 <= j < {dim}}}".format(**shapes)

        # w is the weights
        instrs = (
            """
            for i
                for j
                    w[i,j] = w[i,j] + 1.0
                end
            end
            """)

        self._kernel = (domain, instrs)

    def apply(self, w):
        """
        Perform the par loop for calculating the weightings for the Averager.

        :arg w: the field to store the weights in.
        """

        par_loop(self._kernel, dx,
                 {"w": (w, INC)},
                 is_loopy_kernel=True)


class AverageWeightings(object):
    """
    A kernel for finding the weights for the Averager object.

    :arg V: The FunctionSpace of the target field for the Averager.
    """

    def __init__(self, V):

        shapes = {"nDOFs": V.finat_element.space_dimension(),
                  "dim": np.prod(V.shape, dtype=int)}

        domain = "{{[i, j]: 0 <= i < {nDOFs} and 0 <= j < {dim}}}".format(**shapes)

        # w is the weights
        instrs = (
            """
            for i
                for j
                    w[i,j] = w[i,j] + 1.0 / DIST[i,j]
                end
            end
            """)

        self._kernel = (domain, instrs)

    def apply(self, w, distances):
        """
        Perform the par loop for calculating the weightings for the Averager.

        :arg w: the field to store the weights in.
        :arg distances: the distances providing the weights.
        """

        par_loop(self._kernel, dx,
                 {"w": (w, INC), "DIST": (distances, READ)},
                 is_loopy_kernel=True)


class PhysicsRecoveryTop():
    """
    A kernel for fixing the physics recovery method at the top boundary.
    This takes a variable from the lowest order density space to the lowest
    order temperature space.
    """

    def __init__(self):

        domain = ("{[i]: 0 <= i < 1}")

        # CG1 is the uncorrected field that has been originally recovered
        # DG1 is the corrected output field
        instrs = (
            """
            DG1[0] = CG1[0]
            DG1[1] = -CG1[0] + 2 * CG1[1]
            """)

        self._kernel = (domain, instrs)

    def apply(self, v_DG1, v_CG1):
        """
        Performs the par loop.

        :arg v_DG1: the target field to correct.
        :arg v_CG1: the initially recovered uncorrected field.
        """

        par_loop(self._kernel, dx,
                 args={"DG1": (v_DG1, WRITE),
                       "CG1": (v_CG1, READ)},
                 is_loopy_kernel=True,
                 iterate=ON_TOP)


class PhysicsRecoveryBottom():
    """
    A kernel for fixing the physics recovery method at the bottom boundary.
    This takes a variable from the lowest order density space to the lowest
    order temperature space.
    """

    def __init__(self):

        domain = ("{[i]: 0 <= i < 1}")

        # CG1 is the uncorrected field that has been originally recovered
        # DG1 is the corrected output field
        instrs = (
            """
            DG1[0] = 2 * CG1[0] - CG1[1]
            DG1[1] = CG1[1]
            """)

        self._kernel = (domain, instrs)

    def apply(self, v_DG1, v_CG1):
        """
        Performs the par loop.

        :arg v_DG1: the target field to correct.
        :arg v_CG1: the initially recovered uncorrected field.
        """

        par_loop(self._kernel, dx,
                 args={"DG1": (v_DG1, WRITE),
                       "CG1": (v_CG1, READ)},
                 is_loopy_kernel=True,
                 iterate=ON_BOTTOM)
