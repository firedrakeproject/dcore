"""
The recovery operators used for lowest-order advection schemes.
"""
from firedrake import (expression, function, Function, FunctionSpace, Projector,
                       VectorFunctionSpace, SpatialCoordinate, as_vector,
                       dx, Interpolator, BrokenElement, interval, Constant,
                       TensorProductElement, FiniteElement, DirichletBC,
                       VectorElement, conditional, max_value)
from firedrake.utils import cached_property
from firedrake.parloops import par_loop, READ, INC, WRITE
from gusto.configuration import logger
from pyop2 import ON_TOP, ON_BOTTOM
import ufl
import numpy as np
from enum import Enum

__all__ = ["Averager", "Boundary_Method", "Boundary_Recoverer", "Recoverer"]


class Averager(object):
    """
    An object that 'recovers' a low order field (e.g. in DG0)
    into a higher order field (e.g. in CG1).
    The code is essentially that of the Firedrake Projector
    object, using the "average" method, and could possibly
    be replaced by it if it comes into the master branch.

    :arg v: the :class:`ufl.Expr` or
         :class:`.Function` to project.
    :arg v_out: :class:`.Function` to put the result in.
    """

    def __init__(self, v, v_out):

        if isinstance(v, expression.Expression) or not isinstance(v, (ufl.core.expr.Expr, function.Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(v))

        # Check shape values
        if v.ufl_shape != v_out.ufl_shape:
            raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' % (v.ufl_shape, v_out.ufl_shape))

        self._same_fspace = (isinstance(v, function.Function) and v.function_space() == v_out.function_space())
        self.v = v
        self.v_out = v_out
        self.V = v_out.function_space()

        # Check the number of local dofs
        if self.v_out.function_space().finat_element.space_dimension() != self.v.function_space().finat_element.space_dimension():
            raise RuntimeError("Number of local dofs for each field must be equal.")

        # NOTE: Any bcs on the function self.v should just work.
        # Loop over node extent and dof extent
        self.shapes = {"nDOFs": self.V.finat_element.space_dimension(),
                       "dim": np.prod(self.V.shape, dtype=int)}
        # Averaging kernel
        average_domain = "{{[i, j]: 0 <= i < {nDOFs} and 0 <= j < {dim}}}".format(**self.shapes)
        average_instructions = ("""
                                for i
                                    for j
                                        vo[i,j] = vo[i,j] + v[i,j] / w[i,j]
                                    end
                                end
                                """)
        self._average_kernel = (average_domain, average_instructions)

    @cached_property
    def _weighting(self):
        """
        Generates a weight function for computing a projection via averaging.
        """
        w = Function(self.V)
        weight_domain = "{{[i, j]: 0 <= i < {nDOFs} and 0 <= j < {dim}}}".format(**self.shapes)
        weight_instructions = ("""
                               for i
                                   for j
                                      w[i,j] = w[i,j] + 1.0
                                   end
                               end
                               """)
        _weight_kernel = (weight_domain, weight_instructions)
        par_loop(_weight_kernel, dx, {"w": (w, INC)}, is_loopy_kernel=True)
        return w

    def project(self):
        """
        Apply the recovery.
        """

        # Ensure that the function being populated is zeroed out
        self.v_out.dat.zero()
        par_loop(self._average_kernel, dx, {"vo": (self.v_out, INC),
                                            "w": (self._weighting, READ),
                                            "v": (self.v, READ)},
                 is_loopy_kernel=True)
        return self.v_out


class Boundary_Method(Enum):
    """
    An Enum object storing the two types of boundary method:
    dynamics -- which corrects a field recovered into CG1.
    physics -- which corrects a field recovered into the temperature space.
    """

    dynamics = 0
    physics = 1


class Boundary_Recoverer(object):
    """
    An object that performs a `recovery` process at the domain
    boundaries that has second order accuracy. This is necessary
    because the :class:`Averager` object does not recover a field
    with sufficient accuracy at the boundaries.

    The strategy is to minimise the curvature of the function in
    the boundary cells, subject to the constraints of conserved
    mass and continuity on the interior facets. The quickest way
    to perform this is by using the analytic solution and a parloop.

    Currently this is only implemented for the (DG0, DG1, CG1)
    set of spaces, and only on a `PeriodicIntervalMesh` or
    'PeriodicUnitIntervalMesh` that has been extruded.

    :arg v_CG1: the continuous function after the first recovery
             is performed. Should be in CG1. This is correct
             on the interior of the domain.
    :arg v_DG1: the function to be output. Should be in DG1.
    :arg method: a Boundary_Method Enum object.
    :arg coords_to_adjust: a DG1 field containing 1 at locations of
                           coords that must be adjusted to give they
                           effective coords.
    """

    def __init__(self, v_CG1, v_DG1, method=Boundary_Method.physics, coords_to_adjust=None):

        self.v_DG1 = v_DG1
        self.v_CG1 = v_CG1
        self.v_DG1_old = Function(v_DG1.function_space())
        self.coords_to_adjust = coords_to_adjust

        self.method = method
        mesh = v_CG1.function_space().mesh()
        VDG0 = FunctionSpace(mesh, "DG", 0)
        VCG1 = FunctionSpace(mesh, "CG", 1)
        VDG1 = FunctionSpace(mesh, "DG", 1)

        self.num_ext = Function(VDG0)

        # check function spaces of functions
        if self.method == Boundary_Method.dynamics:
            if v_CG1.function_space() != VCG1:
                raise NotImplementedError("This boundary recovery method requires v1 to be in CG1.")
            if v_DG1.function_space() != VDG1:
                raise NotImplementedError("This boundary recovery method requires v_out to be in DG1.")
            # check whether mesh is valid
            if mesh.topological_dimension() == 2:
                # if mesh is extruded then we're fine, but if not needs to be quads
                if not VDG0.extruded and mesh.ufl_cell().cellname() != 'quadrilateral':
                    raise NotImplementedError('For 2D meshes this recovery method requires that elements are quadrilaterals')
            elif mesh.topological_dimension() == 3:
                # assume that 3D mesh is extruded
                if mesh._base_mesh.ufl_cell().cellname() != 'quadrilateral':
                    raise NotImplementedError('For 3D extruded meshes this recovery method requires a base mesh with quadrilateral elements')
            elif mesh.topological_dimension() != 1:
                raise NotImplementedError('This boundary recovery is implemented only on certain classes of mesh.')
            if coords_to_adjust is None:
                raise ValueError('Need coords_to_adjust field for dynamics boundary methods')

        elif self.method == Boundary_Method.physics:
            # check that mesh is valid -- must be an extruded mesh
            if not VDG0.extruded:
                raise NotImplementedError('The physics boundary method only works on extruded meshes')
            # base spaces
            cell = mesh._base_mesh.ufl_cell().cellname()
            w_hori = FiniteElement("DG", cell, 0)
            w_vert = FiniteElement("CG", interval, 1)
            # build element
            theta_element = TensorProductElement(w_hori, w_vert)
            # spaces
            Vtheta = FunctionSpace(mesh, theta_element)
            Vtheta_broken = FunctionSpace(mesh, BrokenElement(theta_element))
            if v_CG1.function_space() != Vtheta:
                raise ValueError("This boundary recovery method requires v_CG1 to be in DG0xCG1 TensorProductSpace.")
            if v_DG1.function_space() != Vtheta_broken:
                raise ValueError("This boundary recovery method requires v_DG1 to be in the broken DG0xCG1 TensorProductSpace.")
        else:
            raise ValueError("Boundary method should be a Boundary Method Enum object.")

        VuDG1 = VectorFunctionSpace(VDG0.mesh(), "DG", 1)
        x = SpatialCoordinate(VDG0.mesh())
        self.interpolator = Interpolator(self.v_CG1, self.v_DG1)

        if self.method == Boundary_Method.dynamics:

            # STRATEGY
            # obtain a coordinate field for all the nodes
            VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
            self.act_coords = Function(VuDG1).project(x)  # actual coordinates
            self.eff_coords = Function(VuDG1).project(x)  # effective coordinates

            shapes = {"nDOFs": self.v_DG1.function_space().finat_element.space_dimension(),
                      "dim": np.prod(VuDG1.shape, dtype=int)}

            num_ext_domain = ("{{[i]: 0 <= i < {nDOFs}}}").format(**shapes)
            num_ext_instructions = ("""
            <float64> SUM_EXT = 0
            for i
                SUM_EXT = SUM_EXT + EXT_V1[i]
            end

            NUM_EXT[0] = SUM_EXT
            """)

            coords_domain = ("{{[i, j, k, ii, jj, kk, ll, mm, iii, kkk]: "
                             "0 <= i < {nDOFs} and "
                             "0 <= j < {nDOFs} and 0 <= k < {dim} and "
                             "0 <= ii < {nDOFs} and 0 <= jj < {nDOFs} and "
                             "0 <= kk < {dim} and 0 <= ll < {dim} and "
                             "0 <= mm < {dim} and 0 <= iii < {nDOFs} and "
                             "0 <= kkk < {dim}}}").format(**shapes)
            coords_insts = ("""
                            <float64> sum_V1_ext = 0
                            <int> index = 100
                            <float64> dist = 0.0
                            <float64> max_dist = 0.0
                            <float64> min_dist = 0.0
                            """
                            # only do adjustment in cells with at least one DOF to adjust
                            """
                            if NUM_EXT[0] > 0
                            """
                            # find the maximum distance between DOFs in this cell, to serve as starting point for finding min distances
                            """
                                for i
                                    for j
                                        dist = 0.0
                                        for k
                                            dist = dist + pow(ACT_COORDS[i,k] - ACT_COORDS[j,k], 2.0)
                                        end
                                        dist = pow(dist, 0.5) {{id=sqrt_max_dist, dep=*}}
                                        max_dist = fmax(dist, max_dist) {{id=max_dist, dep=sqrt_max_dist}}
                                    end
                                end
                            """
                            # loop through cells and find which ones to adjust
                            """
                                for ii
                                    if EXT_V1[ii] > 0.5
                            """
                            # find closest interior node
                            """
                                        min_dist = max_dist
                                        index = 100
                                        for jj
                                            if EXT_V1[jj] < 0.5
                                                dist = 0.0
                                                for kk
                                                    dist = dist + pow(ACT_COORDS[ii,kk] - ACT_COORDS[jj,kk], 2)
                                                end
                                                dist = pow(dist, 0.5)
                                                if dist <= min_dist
                                                    index = jj
                                                end
                                                min_dist = fmin(min_dist, dist)
                                                for ll
                                                    EFF_COORDS[ii,ll] = 0.5 * (ACT_COORDS[ii,ll] + ACT_COORDS[index,ll])
                                                end
                                            end
                                        end
                                    else
                            """
                            # for DOFs that aren't exterior, use the original coordinates
                            """
                                        for mm
                                            EFF_COORDS[ii, mm] = ACT_COORDS[ii, mm]
                                        end
                                    end
                                end
                            else
                            """
                            # for interior elements, just use the original coordinates
                            """
                                for iii
                                    for kkk
                                        EFF_COORDS[iii, kkk] = ACT_COORDS[iii, kkk]
                                    end
                                end
                            end
                            """).format(**shapes)

            elimin_domain = ("{{[i, ii_loop, jj_loop, kk, ll_loop, mm, iii_loop, kkk_loop, iiii, iiiii]: "
                             "0 <= i < {nDOFs} and 0 <= ii_loop < {nDOFs} and "
                             "ii_loop + 1 <= jj_loop < {nDOFs} and ii_loop <= kk < {nDOFs} and "
                             "ii_loop + 1 <= ll_loop < {nDOFs} and ii_loop <= mm < {nDOFs} + 1 and "
                             "0 <= iii_loop < {nDOFs} and {nDOFs} - iii_loop <= kkk_loop < {nDOFs} + 1 and "
                             "0 <= iiii < {nDOFs} and 0 <= iiiii < {nDOFs}}}").format(**shapes)
            elimin_insts = ("""
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
                                    A[i,0] = 1.0
                                    A[i,1] = EFF_COORDS[i,0]
                                    if {nDOFs} > 3
                                        A[i,2] = EFF_COORDS[i,1]
                                        A[i,3] = EFF_COORDS[i,0]*EFF_COORDS[i,1]
                                        if {nDOFs} > 7
                                            A[i,4] = EFF_COORDS[i,{dim}-1]
                                            A[i,5] = EFF_COORDS[i,0]*EFF_COORDS[i,{dim}-1]
                                            A[i,6] = EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                                            A[i,7] = EFF_COORDS[i,0]*EFF_COORDS[i,1]*EFF_COORDS[i,{dim}-1]
                                        end
                                    end
                                end
                            """
                            # now loop through rows/columns of A
                            """
                                for ii_loop
                                    A_max = fabs(A[ii,ii])
                                    i_max = ii
                            """
                            # loop to find the largest value in the ith column
                            # set i_max as the index of the row with this largest value.
                            """
                                    jj = ii + 1
                                    for jj_loop
                                        if fabs(A[jj,ii]) > A_max
                                            i_max = jj
                                        end
                                        A_max = fmax(A_max, fabs(A[jj,ii]))
                                        jj = jj + 1
                                    end
                            """
                            # if the max value in the ith column isn't in the ith row, we must swap the rows
                            """
                                    if i_max != ii
                            """
                            # swap the elements of f
                            """
                                        temp_f = f[ii]  {{id=set_temp_f, dep=*}}
                                        f[ii] = f[i_max]  {{id=set_f_imax, dep=set_temp_f}}
                                        f[i_max] = temp_f  {{id=set_f_ii, dep=set_f_imax}}
                            """
                            # swap the elements of A
                            # N.B. kk runs from ii to (nDOFs-1) as elements below diagonal should be 0
                            """
                                        for kk
                                            temp_A = A[ii,kk]  {{id=set_temp_A, dep=*}}
                                            A[ii, kk] = A[i_max, kk]  {{id=set_A_ii, dep=set_temp_A}}
                                            A[i_max, kk] = temp_A  {{id=set_A_imax, dep=set_A_ii}}
                                        end
                                    end
                            """
                            # scale the rows below the ith row
                            """
                                    ll = ii + 1
                                    for ll_loop
                                        if ll > ii
                            """
                            # find scaling factor
                            """
                                            c = - A[ll,ii] / A[ii,ii]
                            """
                            # N.B. mm runs from ii to (nDOFs-1) as elements below diagonal should be 0
                            """
                                            for mm
                                                A[ll, mm] = A[ll, mm] + c * A[ii,mm]
                                            end
                                            f[ll] = f[ll] + c * f[ii]
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
                                    jjj = {nDOFs} - iii - 1  {{id=assign_jjj, dep=*}}
                                    a[jjj] = f[jjj]   {{id=set_a, dep=assign_jjj}}
                                    for kkk_loop
                                        a[jjj] = a[jjj] - A[jjj,kkk_loop] * a[kkk_loop]
                                    end
                                    a[jjj] = a[jjj] / A[jjj,jjj]
                                    iii = iii + 1
                                end
                            """
                            # Having found a, this gives us the coefficients for the Taylor expansion with the actual coordinates.
                            """
                                for iiii
                                    if {nDOFs} == 2
                                        DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0]
                                    elif {nDOFs} == 4
                                        DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1]
                                    elif {nDOFs} == 8
                                        DG1[iiii] = a[0] + a[1]*ACT_COORDS[iiii,0] + a[2]*ACT_COORDS[iiii,1] + a[3]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1] + a[4]*ACT_COORDS[iiii,{dim}-1] + a[5]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,{dim}-1] + a[6]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1] + a[7]*ACT_COORDS[iiii,0]*ACT_COORDS[iiii,1]*ACT_COORDS[iiii,{dim}-1]
                                    end
                                end
                            """
                            # if element is not external, just use old field values.
                            """
                            else
                                for iiiii
                                    DG1[iiiii] = DG1_OLD[iiiii]
                                end
                            end
                            """).format(**shapes)

            _num_ext_kernel = (num_ext_domain, num_ext_instructions)
            _eff_coords_kernel = (coords_domain, coords_insts)
            self._gaussian_elimination_kernel = (elimin_domain, elimin_insts)

            # find number of external DOFs per cell
            par_loop(_num_ext_kernel, dx,
                     {"NUM_EXT": (self.num_ext, WRITE),
                      "EXT_V1": (self.coords_to_adjust, READ)},
                     is_loopy_kernel=True)

            # find effective coordinates
            logger.warning('Finding effective coordinates for boundary recovery. This could give unexpected results for deformed meshes over very steep topography.')
            par_loop(_eff_coords_kernel, dx,
                     {"EFF_COORDS": (self.eff_coords, WRITE),
                      "ACT_COORDS": (self.act_coords, READ),
                      "NUM_EXT": (self.num_ext, READ),
                      "EXT_V1": (self.coords_to_adjust, READ)},
                     is_loopy_kernel=True)

        elif self.method == Boundary_Method.physics:
            top_bottom_domain = ("{[i]: 0 <= i < 1}")
            bottom_instructions = ("""
                                   DG1[0] = 2 * CG1[0] - CG1[1]
                                   DG1[1] = CG1[1]
                                   """)
            top_instructions = ("""
                                DG1[0] = CG1[0]
                                DG1[1] = -CG1[0] + 2 * CG1[1]
                                """)

            self._bottom_kernel = (top_bottom_domain, bottom_instructions)
            self._top_kernel = (top_bottom_domain, top_instructions)

    def apply(self):
        self.interpolator.interpolate()
        if self.method == Boundary_Method.physics:
            par_loop(self._bottom_kernel, dx,
                     args={"DG1": (self.v_DG1, WRITE),
                           "CG1": (self.v_CG1, READ)},
                     is_loopy_kernel=True,
                     iterate=ON_BOTTOM)

            par_loop(self._top_kernel, dx,
                     args={"DG1": (self.v_DG1, WRITE),
                           "CG1": (self.v_CG1, READ)},
                     is_loopy_kernel=True,
                     iterate=ON_TOP)
        else:
            self.v_DG1_old.assign(self.v_DG1)
            par_loop(self._gaussian_elimination_kernel, dx,
                     args={"DG1_OLD": (self.v_DG1_old, READ),
                           "DG1": (self.v_DG1, WRITE),
                           "ACT_COORDS": (self.act_coords, READ),
                           "EFF_COORDS": (self.eff_coords, READ),
                           "NUM_EXT": (self.num_ext, READ)},
                     is_loopy_kernel=True)


class Recoverer(object):
    """
    An object that 'recovers' a field from a low order space
    (e.g. DG0) into a higher order space (e.g. CG1). This encompasses
    the process of interpolating first to a the right space before
    using the :class:`Averager` object, and also automates the
    boundary recovery process. If no boundary method is specified,
    this simply performs the action of the :class: `Averager`.

    :arg v_in: the :class:`ufl.Expr` or
         :class:`.Function` to project. (e.g. a VDG0 function)
    :arg v_out: :class:`.Function` to put the result in. (e.g. a CG1 function)
    :arg VDG: optional :class:`.FunctionSpace`. If not None, v_in is interpolated
         to this space first before recovery happens.
    :arg boundary_method: an Enum object, .
    """

    def __init__(self, v_in, v_out, VDG=None, boundary_method=None):

        # check if v_in is valid
        if isinstance(v_in, expression.Expression) or not isinstance(v_in, (ufl.core.expr.Expr, function.Function)):
            raise ValueError("Can only recover UFL expression or Functions not '%s'" % type(v_in))

        self.v_in = v_in
        self.v_out = v_out
        self.V = v_out.function_space()
        if VDG is not None:
            self.v = Function(VDG)
            self.interpolator = Interpolator(v_in, self.v)
        else:
            self.v = v_in
            self.interpolator = None

        self.VDG = VDG
        self.boundary_method = boundary_method
        self.averager = Averager(self.v, self.v_out)

        # check boundary method options are valid
        if boundary_method is not None:
            if boundary_method != Boundary_Method.dynamics and boundary_method != Boundary_Method.physics:
                raise ValueError("Boundary method must be a Boundary_Method Enum object.")
            if VDG is None:
                raise ValueError("If boundary_method is specified, VDG also needs specifying.")

            # now specify things that we'll need if we are doing boundary recovery
            if boundary_method == Boundary_Method.physics:
                # check dimensions
                if self.V.value_size != 1:
                    raise ValueError('This method only works for scalar functions.')
                self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v, method=Boundary_Method.physics)
            else:

                mesh = self.V.mesh()
                # this ensures we get the pure function space, not an indexed function space
                V0 = FunctionSpace(mesh, self.v_in.function_space().ufl_element())
                VDG1 = FunctionSpace(mesh, "DG", 1)
                VCG1 = FunctionSpace(mesh, "CG", 1)

                if self.V.value_size == 1:
                    coords_to_adjust = find_coords_to_adjust(V0, VDG1)

                    self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v,
                                                                 coords_to_adjust=coords_to_adjust,
                                                                 method=Boundary_Method.dynamics)
                else:
                    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
                    coords_to_adjust = find_coords_to_adjust(V0, VuDG1)

                    # now, break the problem down into components
                    v_scalars = []
                    v_out_scalars = []
                    self.boundary_recoverers = []
                    self.project_to_scalars_CG = []
                    self.extra_averagers = []
                    coords_to_adjust_list = []
                    for i in range(self.V.value_size):
                        v_scalars.append(Function(VDG1))
                        v_out_scalars.append(Function(VCG1))
                        coords_to_adjust_list.append(Function(VDG1).project(coords_to_adjust[i]))
                        self.project_to_scalars_CG.append(Projector(self.v_out[i], v_out_scalars[i]))
                        self.boundary_recoverers.append(Boundary_Recoverer(v_out_scalars[i], v_scalars[i],
                                                                           method=Boundary_Method.dynamics,
                                                                           coords_to_adjust=coords_to_adjust_list[i]))
                        # need an extra averager that works on the scalar fields rather than the vector one
                        self.extra_averagers.append(Averager(v_scalars[i], v_out_scalars[i]))

                    # the boundary recoverer needs to be done on a scalar fields
                    # so need to extract component and restore it after the boundary recovery is done
                    # do an interpolate, as projector is unreliable on Jenkins
                    self.interpolate_to_vector = Interpolator(as_vector(v_out_scalars), self.v_out)

    def project(self):
        """
        Perform the fully specified recovery.
        """

        if self.interpolator is not None:
            self.interpolator.interpolate()
        self.averager.project()
        if self.boundary_method is not None:
            if self.V.value_size > 1:
                for i in range(self.V.value_size):
                    self.project_to_scalars_CG[i].project()
                    self.boundary_recoverers[i].apply()
                    self.extra_averagers[i].project()
                self.interpolate_to_vector.interpolate()
            else:
                self.boundary_recoverer.apply()
                self.averager.project()
        return self.v_out


def find_coords_to_adjust(V0, DG1):
    """
    This function finds the coordinates that need to be adjusted
    for the recovery at the boundary. These are assigned by a 1,
    while all coordinates to be left unchanged are assigned a 0.
    This field is returned as a DG1 field.
    Fields can be scalar or vector.

    :arg V0: the space of the original field (before recovery).
    :arg DG1: a DG1 space, in which the boundary recovery will happen.
    """

    # check that spaces are correct
    mesh = DG1.mesh()
    scalar_DG1 = FunctionSpace(mesh, "DG", 1)
    # check DG1 field is correct
    if type(DG1.ufl_element()) == VectorElement:
        if DG1 != VectorFunctionSpace(mesh, "DG", 1):
            raise ValueError('The function space entered as vector DG1 is not vector DG1.')
    elif DG1 != FunctionSpace(mesh, "DG", 1):
        raise ValueError('The function space entered as DG1 is not DG1.')

    # STRATEGY
    # We need to pass the boundary recoverer a field denoting the location
    # of nodes on the boundary, which denotes the coordinates to adjust to be new effective
    # coords. This field will be 1 for these coords and 0 otherwise.
    # How do we do this?
    # 1. Obtain a DG1 field which is 1 at all exterior DOFs by applying Dirichlet
    #    boundary conditions. i.e. for cells in the bottom right corner of a domain:
    #    ------- 0 ------- 0 ------- 1
    #            |         |         ||
    #            |         |         ||
    #            |         |         ||
    #    ======= 1 ======= 1 ======= 1
    # 2. Obtain a field in DG1 that is 1 at exterior DOFs adjacent to the exterior
    #    DOFs of V0 (i.e. the original space). For V0=DG0 there will be no exterior
    #    DOFs, but could be if velocity is in RT or if there is a temperature space.
    #    This is done by applying topological boundary conditions to a field in V0,
    #    before interpolating these into DG1.
    #    For instance, marking V0 DOFs with x, for rho and theta spaces this would give
    #    ------- 0 ------- 0 ------- 0          ---x--- 0 ---x--- 0 ---x--- 0
    #            |         |         ||                 |         |         ||
    #       x    |    x    |    x    ||                 |         |         ||
    #            |         |         ||                 |         |         ||
    #    ======= 0 ======= 0 ======= 0          ===x=== 1 ===x=== 1 ===x=== 1
    # 3. Obtain a field that is 1 at corners in 2D or along edges in 3D.
    #    We do this by using that corners in 2D and edges in 3D are intersections
    #    of edges/faces respectively. In 2D, this means that if a field which is 1 on a
    #    horizontal edge is summed with a field that is 1 on a vertical edge, the
    #    corner value will be 2. Subtracting the exterior DG1 field from step 1 leaves
    #    a field that is 1 in the corner. This is generalised to 3D.
    #    ------- 0 ------- 0    ------- 0 ------- 1    ------- 0 ------- 1    ------- 0 ------- 0
    #            |         ||           |         ||           |         ||            |         ||
    #            |         ||  +        |         ||  -        |         ||  =         |         ||
    #            |         ||           |         ||           |         ||            |         ||
    #    ======= 1 ======= 1    ======= 0 ======= 1    ======= 1 ======= 1     ======= 0 ======= 1
    # 4. The field of coords to be adjusted is then found by the following formula:
    #                            f1 + f3 - f2
    #    where f1, f2 and f3 are the DG1 fields obtained from steps 1, 2 and 3.

    # make DG1 field with 1 at all exterior coords
    all_ext_in_DG1 = Function(DG1)
    bcs = [DirichletBC(DG1, Constant(1.0), "on_boundary", method="geometric")]

    if DG1.extruded:
        bcs.append(DirichletBC(DG1, Constant(1.0), "top", method="geometric"))
        bcs.append(DirichletBC(DG1, Constant(1.0), "bottom", method="geometric"))

    for bc in bcs:
        bc.apply(all_ext_in_DG1)

    # make DG1 field with 1 at coords surrounding exterior coords of V0
    # first do topological BCs to get V0 function which is 1 at DOFs on edges
    all_ext_in_V0 = Function(V0)
    bcs = [DirichletBC(V0, Constant(1.0), "on_boundary", method="topological")]

    if V0.extruded:
        bcs.append(DirichletBC(V0, Constant(1.0), "top", method="topological"))
        bcs.append(DirichletBC(V0, Constant(1.0), "bottom", method="topological"))

    for bc in bcs:
        bc.apply(all_ext_in_V0)

    if DG1.value_size > 1:
        # for vector valued functions, DOFs aren't pointwise evaluation. We break into components and use a conditional interpolation to get values of 1
        V0_ext_in_DG1_components = []
        for i in range(DG1.value_size):
            V0_ext_in_DG1_components.append(Function(scalar_DG1).interpolate(conditional(abs(all_ext_in_V0[i]) > 0.0, 1.0, 0.0)))
        V0_ext_in_DG1 = Function(DG1).project(as_vector(V0_ext_in_DG1_components))
    else:
        # for scalar functions (where DOFs are pointwise evaluation) we can simply interpolate to get these values
        V0_ext_in_DG1 = Function(DG1).interpolate(all_ext_in_V0)

    corners_in_DG1 = Function(DG1)
    if DG1.mesh().topological_dimension() == 2:
        if DG1.extruded:
            DG1_ext_hori = Function(DG1)
            DG1_ext_vert = Function(DG1)
            hori_bcs = [DirichletBC(DG1, Constant(1.0), "top", method="geometric"),
                        DirichletBC(DG1, Constant(1.0), "bottom", method="geometric")]
            vert_bc = DirichletBC(DG1, Constant(1.0), "on_boundary", method="geometric")
            for bc in hori_bcs:
                bc.apply(DG1_ext_hori)

            vert_bc.apply(DG1_ext_vert)
            corners_in_DG1.assign(DG1_ext_hori + DG1_ext_vert - all_ext_in_DG1)

        else:
            # we don't know whether its periodic or in how many directions
            DG1_ext_x = Function(DG1)
            DG1_ext_y = Function(DG1)
            x_bcs = [DirichletBC(DG1, Constant(1.0), 1, method="geometric"),
                     DirichletBC(DG1, Constant(1.0), 2, method="geometric")]
            y_bcs = [DirichletBC(DG1, Constant(1.0), 3, method="geometric"),
                     DirichletBC(DG1, Constant(1.0), 4, method="geometric")]

            # there is no easy way to know if the mesh is periodic or in which
            # directions, so we must use a try here
            # LookupError is the error for asking for a boundary number that doesn't exist
            try:
                for bc in x_bcs:
                    bc.apply(DG1_ext_x)
            except LookupError:
                pass
            try:
                for bc in y_bcs:
                    bc.apply(DG1_ext_y)
            except LookupError:
                pass

            corners_in_DG1.assign(DG1_ext_x + DG1_ext_y - all_ext_in_DG1)

    elif DG1.mesh().topological_dimension() == 3:
        DG1_vert_x = Function(DG1)
        DG1_vert_y = Function(DG1)
        DG1_hori = Function(DG1)
        x_bcs = [DirichletBC(DG1, Constant(1.0), 1, method="geometric"),
                 DirichletBC(DG1, Constant(1.0), 2, method="geometric")]
        y_bcs = [DirichletBC(DG1, Constant(1.0), 3, method="geometric"),
                 DirichletBC(DG1, Constant(1.0), 4, method="geometric")]
        hori_bcs = [DirichletBC(DG1, Constant(1.0), "top", method="geometric"),
                    DirichletBC(DG1, Constant(1.0), "bottom", method="geometric")]

        # there is no easy way to know if the mesh is periodic or in which
        # directions, so we must use a try here
        # LookupError is the error for asking for a boundary number that doesn't exist
        try:
            for bc in x_bcs:
                bc.apply(DG1_vert_x)
        except LookupError:
            pass

        try:
            for bc in y_bcs:
                bc.apply(DG1_vert_y)
        except LookupError:
            pass

        for bc in hori_bcs:
            bc.apply(DG1_hori)

        corners_in_DG1.assign(DG1_vert_x + DG1_vert_y + DG1_hori - all_ext_in_DG1)

    # we now combine the different functions. We use max_value to avoid getting 2s or 3s at corners/edges
    # we do this component-wise because max_value only works component-wise
    if DG1.value_size > 1:
        coords_to_correct_components = []
        for i in range(DG1.value_size):
            coords_to_correct_components.append(Function(scalar_DG1).interpolate(max_value(corners_in_DG1[i], all_ext_in_DG1[i] - V0_ext_in_DG1[i])))
        coords_to_correct = Function(DG1).project(as_vector(coords_to_correct_components))
    else:
        coords_to_correct = Function(DG1).interpolate(max_value(corners_in_DG1, all_ext_in_DG1 - V0_ext_in_DG1))

    return coords_to_correct
