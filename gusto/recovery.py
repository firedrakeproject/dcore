"""
The recovery operators used for lowest-order advection schemes.
"""
from firedrake import (expression, function, Function, FunctionSpace, Projector,
                       VectorFunctionSpace, SpatialCoordinate, as_vector,
                       dx, Interpolator, BrokenElement, interval, Constant,
                       TensorProductElement, FiniteElement, DirichletBC,
                       VectorElement, conditional, max_value)
from firedrake.utils import cached_property
from firedrake.parloops import par_loop, READ, INC, WRITE, RW
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
    set of spaces.

    :arg v_CG1: the continuous function after the first recovery
             is performed. Should be in CG1. This is correct
             on the interior of the domain.
    :arg v_DG1: the function to be output. Should be in DG1.
    :arg method: a Boundary_Method Enum object.
    :arg eff_coords: a vector DG1 field giving the effective coordinates
                     for the boundary recovery. Necessary for the Dynamics
                     Boundary_Method.
    """

    def __init__(self, v_CG1, v_DG1, method=Boundary_Method.physics, eff_coords=None):

        self.v_DG1 = v_DG1
        self.v_CG1 = v_CG1
        self.v_DG1_old = Function(v_DG1.function_space())

        self.method = method
        mesh = v_CG1.function_space().mesh()
        VDG0 = FunctionSpace(mesh, "DG", 0)
        VCG1 = FunctionSpace(mesh, "CG", 1)
        VDG1 = FunctionSpace(mesh, "DG", 1)

        VuDG1 = VectorFunctionSpace(VDG0.mesh(), "DG", 1)
        VuCG1 = VectorFunctionSpace(VDG0.mesh(), "CG", 1)
        x = SpatialCoordinate(VDG0.mesh())
        self.interpolator = Interpolator(self.v_CG1, self.v_DG1)

        self.num_ext = Function(VDG0)

        # check function spaces of functions
        if self.method == Boundary_Method.dynamics:
            if v_CG1.function_space() != VCG1:
                raise NotImplementedError("This boundary recovery method requires v1 to be in CG1")
            if v_DG1.function_space() != VDG1:
                raise NotImplementedError("This boundary recovery method requires v_out to be in DG1")
            if eff_coords == None:
                raise ValueError('For dynamics boundary recovery, need to specify effective coordinates')
            elif eff_coords.function_space() != VuDG1:
                raise NotImplementedError("Need effective coordinates to be in vector DG1 function space")
            # check whether mesh is valid
            if mesh.topological_dimension() != mesh.geometric_dimension():
                raise NotImplementedError('This boundary recovery is implemented only on certain classes of mesh.')

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


        if self.method == Boundary_Method.dynamics:

            # STRATEGY
            # obtain a coordinate field for all the nodes
            VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
            self.act_coords = Function(VuDG1).project(x)  # actual coordinates
            self.eff_coords = eff_coords  # effective coordinates in DG1
            self.output = Function(VDG1)

            shapes = {"nDOFs": self.v_DG1.function_space().finat_element.space_dimension(),
                      "dim": np.prod(VuDG1.shape, dtype=int)}

            num_ext_domain = ("{{[i, j, k, ii, kk]: 0 <= i < {nDOFs} and 0 <= j < {nDOFs} and "
                              "0 <= k < {dim} and 0 <= ii < {nDOFs} and 0 <= kk < {dim}}}").format(**shapes)
            num_ext_instructions = ("""
            <float64> SUM_EXT = 0
            <float64> dist = 0.0
            <float64> min_dist = (ACT_COORDS[0,0] - ACT_COORDS[1,0]) ** 2
            <float64> act_eff_dist = 0.0
            <float64> dist_i[{dim}] = 0.0
            <float64> max_dist_i[{dim}] = 0.0
            """
            # first find minimum distance between DOFs within the cell
            # also find max distance in each dimension within that cell
            """
            for i
                for j
                    dist = 0.0
                    for k
                        dist = dist + (ACT_COORDS[i,k] - ACT_COORDS[j,k]) ** 2
                        dist_i[k] = (ACT_COORDS[i,k] - ACT_COORDS[j,k]) ** 2
                        max_dist_i[k] = fmax(dist_i[k], max_dist_i[k])
                    end
                    min_dist = fmin(dist, min_dist)
                end
            end
            """
            # measure distance between actual and effective coordinates
            # if greater than some tolerance (min_dist / 100), then coordinates have been adjusted
            """
            for ii
                act_eff_dist = 0.0
                for kk
                    act_eff_dist = act_eff_dist + (ACT_COORDS[ii,kk] - EFF_COORDS[ii,kk]) ** 2
                    dist_i[kk] = (ACT_COORDS[ii,kk] - EFF_COORDS[ii,kk]) ** 2
                end
                if act_eff_dist > min_dist / 100
                    SUM_EXT = SUM_EXT + 1
                end
            end
            NUM_EXT[0] = SUM_EXT
            """).format(**shapes)


            elimin_domain = ("{{[i, ii_loop, jj_loop, kk, ll_loop, mm, iii_loop, kkk_loop, iiii]: "
                             "0 <= i < {nDOFs} and 0 <= ii_loop < {nDOFs} and "
                             "0 <= jj_loop < {nDOFs} and 0 <= kk < {nDOFs} and "
                             "0 <= ll_loop < {nDOFs} and 0 <= mm < {nDOFs} and "
                             "0 <= iii_loop < {nDOFs} and 0 <= kkk_loop < {nDOFs} and "
                             "0 <= iiii < {nDOFs}}}").format(**shapes)
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
                                end
                            """
                            # now loop through rows/columns of A
                            """
                                for ii_loop
                                    A_max = fabs(A[ii,ii])
                                    i_max = ii
                            """
                            # loop to find the largest value in the ii-th column
                            # set i_max as the index of the row with this largest value.
                            """
                                    jj = ii + 1
                                    for jj_loop
                                        if jj < {nDOFs}
                                            if fabs(A[jj,ii]) > A_max
                                                i_max = jj
                                            end
                                            A_max = fmax(A_max, fabs(A[jj,ii]))
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
                            """
                            # if element is not external, just use old field values.
                            """
                                else
                                    DG1[iiii] = DG1_OLD[iiii]
                                end
                            end
                            """).format(**shapes)

            _num_ext_kernel = (num_ext_domain, num_ext_instructions)
            self._gaussian_elimination_kernel = (elimin_domain, elimin_insts)

            # find number of external DOFs per cell
            par_loop(_num_ext_kernel, dx,
                     {"NUM_EXT": (self.num_ext, WRITE),
                      "ACT_COORDS": (self.act_coords, READ),
                      "EFF_COORDS": (self.eff_coords, READ)},
                     is_loopy_kernel=True)

            #for act, eff in zip(self.act_coords.dat.data[:], self.eff_coords.dat.data[:]):
                #print('act [%.2f %.2f %.2f], eff [%.2f %.2f %.2f]' % (act[0], act[1], act[2], eff[0], eff[1], eff[2]))


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
                     {"DG1_OLD": (self.v_DG1_old, READ),
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
                eff_coords = find_eff_coords(V0)

                if self.V.value_size == 1:

                    self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v,
                                                                 method=Boundary_Method.dynamics,
                                                                 eff_coords=eff_coords)
                else:
                    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)

                    # now, break the problem down into components
                    v_scalars = []
                    v_out_scalars = []
                    self.boundary_recoverers = []
                    self.project_to_scalars_CG = []
                    self.extra_averagers = []
                    for i in range(self.V.value_size):
                        v_scalars.append(Function(VDG1))
                        v_out_scalars.append(Function(VCG1))
                        self.project_to_scalars_CG.append(Projector(self.v_out[i], v_out_scalars[i]))
                        self.boundary_recoverers.append(Boundary_Recoverer(v_out_scalars[i], v_scalars[i],
                                                                           method=Boundary_Method.dynamics,
                                                                           eff_coords=eff_coords[i]))
                        # need an extra averager that works on the scalar fields rather than the vector one
                        self.extra_averagers.append(Averager(v_scalars[i], v_out_scalars[i]))

                    # the boundary recoverer needs to be done on a scalar fields
                    # so need to extract component and restore it after the boundary recovery is done
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


def find_eff_coords(V0):
    """
    Takes a function in a field V0 and returns the effective coordindates,
    in a vector DG1 spacem of recovery into a CG1 field. This is for use with the
    Boundary_Recoverer, as it facilitates the Gaussian elimination used to get
    second-order recovery at boundaries.

    If V0 is a vector function space, this returns an array of coordinates for
    each component.

    :arg V0: the original function space.
    """


    mesh = V0.mesh()
    VuCG1 = VectorFunctionSpace(mesh, "CG", 1)
    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
    x = SpatialCoordinate(mesh)

    if V0.ufl_element().value_size() > 1:
        eff_coords_list = []
        V0_coords_list = []

        # treat this separately for each component
        for i in range(V0.ufl_element().value_size()):
            # fill an d-dimensional list with i-th coordinate
            x_list = [x[i] for j in range(V0.ufl_element().value_size())]

            # the i-th element in V0_coords_list is a vector with all components the i-th coord
            ith_V0_coords = Function(V0).project(as_vector(x_list))
            V0_coords_list.append(ith_V0_coords)

        for i in range(V0.ufl_element().value_size()):
            # slice through V0_coords_list to obtain the coords of the DOFs for that component
            x_list = [V0_coords[i] for V0_coords in V0_coords_list]

            # average these to find effective coords in CG1
            V0_coords_in_DG1 = Function(VuDG1).interpolate(as_vector(x_list))
            eff_coords_in_CG1 = Function(VuCG1)
            eff_coords_averager = Averager(V0_coords_in_DG1, eff_coords_in_CG1).project()

            # obtain these in DG1
            eff_coords_in_DG1 = Function(VuDG1).interpolate(eff_coords_in_CG1)
            eff_coords_list.append(correct_eff_coords(eff_coords_in_DG1))

        return eff_coords_list

    else:
        # find the coordinates at DOFs in V0
        VuV0 = VectorFunctionSpace(mesh, V0.ufl_element())
        V0_coords = Function(VuV0).project(x)

        # average these to find effective coords in CG1
        V0_coords_in_DG1 = Function(VuDG1).interpolate(V0_coords)
        eff_coords_in_CG1 = Function(VuCG1)
        eff_coords_averager = Averager(V0_coords_in_DG1, eff_coords_in_CG1).project()

        # obtain these in DG1
        eff_coords_in_DG1 = Function(VuDG1).interpolate(eff_coords_in_CG1)

        return correct_eff_coords(eff_coords_in_DG1)


def correct_eff_coords(eff_coords):
    """
    Correct the effective coordinates calculated by simply averaging
    which will not be correct at periodic boundaries.

    :arg eff_coords: the effective coordinates in VuDG1 space.
    """

    mesh = eff_coords.function_space().mesh()
    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)
    VuCG1 = VectorFunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)

    if eff_coords.function_space() != VuDG1:
        raise ValueError('eff_coords needs to be in the vector DG1 space')

    # obtain different coords in DG1
    DG1_coords = Function(VuDG1).interpolate(x)
    CG1_coords_from_DG1 = Function(VuCG1)
    averager = Averager(DG1_coords, CG1_coords_from_DG1).project()
    DG1_coords_from_averaged_CG1 = Function(VuDG1).interpolate(CG1_coords_from_DG1)
    DG1_coords_diff = Function(VuDG1).interpolate(DG1_coords - DG1_coords_from_averaged_CG1)

    # interpolate coordinates, adjusting those different coordinates
    adjusted_coords = Function(VuDG1)
    adjusted_coords.interpolate(eff_coords + DG1_coords_diff)

    return adjusted_coords
