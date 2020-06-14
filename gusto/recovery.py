"""
The recovery operators used for lowest-order advection schemes.
"""
from firedrake import (expression, function, Function, FunctionSpace, Projector,
                       VectorFunctionSpace, SpatialCoordinate, as_vector,
                       Interpolator, BrokenElement, interval, Constant,
                       TensorProductElement, FiniteElement, DirichletBC)
from firedrake.utils import cached_property
from gusto import kernels, coord_transforms
import ufl
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

        self.average_kernel = kernels.Average(self.V)

    @cached_property
    def _weighting(self):
        """
        Generates a weight function for computing a projection via averaging.
        """
        w = Function(self.V)

        weight_kernel = kernels.AverageWeightings(self.V)
        weight_kernel.apply(w)

        return w

    def project(self):
        """
        Apply the recovery.
        """

        # Ensure that the function being populated is zeroed out
        self.v_out.dat.zero()
        self.average_kernel.apply(self.v_out, self._weighting, self.v)
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
    :arg eff_coords: the effective coordinates of the iniital recovery.
                     This must be provided for the dynamics Boundary_Method.
    """

    def __init__(self, v_CG1, v_DG1, method=Boundary_Method.physics, eff_coords=None):

        self.v_DG1 = v_DG1
        self.v_CG1 = v_CG1
        self.v_DG1_old = Function(v_DG1.function_space())
        self.eff_coords = eff_coords

        self.method = method
        mesh = v_CG1.function_space().mesh()
        DG0 = FunctionSpace(mesh, "DG", 0)
        CG1 = FunctionSpace(mesh, "CG", 1)

        if DG0.extruded:
            cell = mesh._base_mesh.ufl_cell().cellname()
            DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
            DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
            DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
        else:
            cell = mesh.ufl_cell().cellname()
            DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1 = FunctionSpace(mesh, DG1_element)

        self.num_ext = find_domain_boundaries(mesh)

        # check function spaces of functions
        if self.method == Boundary_Method.dynamics:
            if v_CG1.function_space() != CG1:
                raise NotImplementedError("This boundary recovery method requires v1 to be in CG1.")
            if v_DG1.function_space() != DG1:
                raise NotImplementedError("This boundary recovery method requires v_out to be in DG1.")
            if eff_coords is None:
                raise ValueError('Need eff_coords field for dynamics boundary methods')

        elif self.method == Boundary_Method.physics:
            # check that mesh is valid -- must be an extruded mesh
            if not DG0.extruded:
                raise NotImplementedError('The physics boundary method only works on extruded meshes')
            # base spaces
            cell = mesh._base_mesh.ufl_cell().cellname()
            w_hori = FiniteElement("DG", cell, 0, variant="equispaced")
            w_vert = FiniteElement("CG", interval, 1, variant="equispaced")
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

        vec_DG1 = VectorFunctionSpace(DG0.mesh(), DG1_element)
        x = SpatialCoordinate(DG0.mesh())
        self.interpolator = Interpolator(self.v_CG1, self.v_DG1)

        if self.method == Boundary_Method.dynamics:

            # STRATEGY
            # obtain a coordinate field for all the nodes
            self.act_coords = Function(vec_DG1).project(x)  # actual coordinates
            self.eff_coords = eff_coords  # effective coordinates
            self.output = Function(DG1)
            self.on_exterior = find_domain_boundaries(mesh)

            self.gaussian_elimination_kernel = kernels.GaussianElimination(DG1)

        elif self.method == Boundary_Method.physics:

            self.bottom_kernel = kernels.PhysicsRecoveryBottom()
            self.top_kernel = kernels.PhysicsRecoveryTop()

    def apply(self):
        self.interpolator.interpolate()
        if self.method == Boundary_Method.physics:
            self.bottom_kernel.apply(self.v_DG1, self.v_CG1)
            self.top_kernel.apply(self.v_DG1, self.v_CG1)

        else:
            self.v_DG1_old.assign(self.v_DG1)
            self.gaussian_elimination_kernel.apply(self.v_DG1_old,
                                                   self.v_DG1,
                                                   self.act_coords,
                                                   self.eff_coords,
                                                   self.num_ext)


class Recoverer(object):
    """
    An object that 'recovers' a field from a low order space
    (e.g. DG0) into a higher order space (e.g. CG1). This encompasses
    the process of interpolating first to a the right space before
    using the :class:`Averager` object, and also automates the
    boundary recovery process. If no boundary method is specified,
    this simply performs the action of the :class: `Averager`.

    :arg v_in: the :class:`ufl.Expr` or
         :class:`.Function` to project. (e.g. a DG0 function)
    :arg v_out: :class:`.Function` to put the result in. (e.g. a CG1 function)
    :arg VDG: optional :class:`.FunctionSpace`. If not None, v_in is interpolated
         to this space first before recovery happens.
    :arg boundary_method: an Enum object, which sets whether or not to do an
         additional boundary recovery step using the BoundaryRecoverer, and which
         method (physics or dynamics) to use. The default value is None.
    :arg polar_transform: a Boolean setting whether to transform to polar
         coordinates to do averaging. Only valid for vector fields. Default is False.
    """

    def __init__(self, v_in, v_out, VDG=None,
                 boundary_method=None, polar_transform=False):

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
        self.polar_transform = polar_transform
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
                CG1 = FunctionSpace(mesh, "CG", 1)
                eff_coords = find_eff_coords(V0)

                if V0.extruded:
                    cell = mesh._base_mesh.ufl_cell().cellname()
                    DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
                    DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
                    DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
                else:
                    cell = mesh.ufl_cell().cellname()
                    DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")
                DG1 = FunctionSpace(mesh, DG1_element)

                if self.V.value_size == 1:

                    self.boundary_recoverer = Boundary_Recoverer(self.v_out, self.v,
                                                                 method=Boundary_Method.dynamics,
                                                                 eff_coords=eff_coords)
                else:

                    # now, break the problem down into components
                    v_scalars = []
                    v_out_scalars = []
                    self.boundary_recoverers = []
                    self.project_to_scalars_CG = []
                    self.extra_averagers = []
                    for i in range(self.V.value_size):
                        v_scalars.append(Function(DG1))
                        v_out_scalars.append(Function(CG1))
                        self.project_to_scalars_CG.append(Projector(self.v_out[i], v_out_scalars[i]))
                        self.boundary_recoverers.append(Boundary_Recoverer(v_out_scalars[i], v_scalars[i],
                                                                           method=Boundary_Method.dynamics,
                                                                           eff_coords=eff_coords[i]))
                        # need an extra averager that works on the scalar fields rather than the vector one
                        self.extra_averagers.append(Averager(v_scalars[i], v_out_scalars[i]))

                    # the boundary recoverer needs to be done on a scalar fields
                    # so need to extract component and restore it after the boundary recovery is done
                    self.interpolate_to_vector = Interpolator(as_vector(v_out_scalars), self.v_out)

        # Now set polar transform options
        if polar_transform:
            # First check if polar transform is valid
            mesh = self.VDG.mesh()
            if self.VDG.shape != (mesh.geometric_dimension(),):
                raise ValueError("""
                                 The polar transform option is not valid when field
                                 has shape %s and geometric dimension of space is %i
                                 """ % (self.VDG.shape, mesh.geometric_dimension()))
            elif mesh.geometric_dimension() not in [2, 3]:
                raise ValueError('The polar transform option is not valid for a geometric dimension of %i'
                                 % mesh.geometric_dimension())

            # Converts vector components into polar coordinates for the averaging step
            #!! I'm not really sure what these spaces should be, but bear in mind we want
            # coordinates for RTCF and BDM spaces!!
            # Really it should be something like a list of vector function spaces for each point...
            vec_V_in = self.v_in.function_space()
            vec_V_out = self.v_out.function_space()
            if mesh.geometric_dimension() == 2:
                xy_coords = SpatialCoordinate(mesh)
                rphi_coords = coord_transforms.rphi_from_xy(xy_coords[0], xy_coords[1])
                coords_v_in_cartesian = Function(vec_V_in).interpolate(as_vector(xy_coords))
                coords_VDG_cartesian = Function(self.VDG).interpolate(coords_v_in_cartesian)
                coords_v_out_polar = Function(vec_V_out).interpolate(as_vector(rphi_coords))
                self.convert_to_polar = Interpolator(as_vector(coord_transforms.rphi_vector_from_xy(self.v,
                                                                                                    coords_VDG_cartesian)),
                                                     self.v)

                self.convert_to_cartesian = Interpolator(as_vector(coord_transforms.xy_vector_from_rphi(self.v_out,
                                                                                                        coords_v_out_polar)),
                                                         self.v_out)

            elif mesh.geometric_dimension() == 3:
                raise NotImplementedError('I need to implement finding the coordinates of RTCF and BDM spaces')



    def project(self):
        """
        Perform the fully specified recovery.
        """

        if self.interpolator is not None:
            self.interpolator.interpolate()
        if self.polar_transform:
            self.convert_to_polar.interpolate()
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
        if self.polar_transform:
            self.convert_to_cartesian.interpolate()

        return self.v_out


def find_eff_coords(V0):
    """
    Takes a function in a field V0 and returns the effective coordinates,
    in a vector DG1 space, of a recovery into a CG1 field. This is for use with the
    Boundary_Recoverer, as it facilitates the Gaussian elimination used to get
    second-order recovery at boundaries.
    If V0 is a vector function space, this returns an array of coordinates for
    each component.
    :arg V0: the original function space.
    """

    mesh = V0.mesh()
    if V0.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")

    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)
    vec_DG1 = VectorFunctionSpace(mesh, DG1_element)
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
            V0_coords_in_DG1 = Function(vec_DG1).interpolate(as_vector(x_list))
            eff_coords_in_CG1 = Function(vec_CG1)
            eff_coords_averager = Averager(V0_coords_in_DG1, eff_coords_in_CG1)
            eff_coords_averager.project()

            # obtain these in DG1
            eff_coords_in_DG1 = Function(vec_DG1).interpolate(eff_coords_in_CG1)
            eff_coords_list.append(correct_eff_coords(eff_coords_in_DG1))

        return eff_coords_list

    else:
        # find the coordinates at DOFs in V0
        vec_V0 = VectorFunctionSpace(mesh, V0.ufl_element())
        V0_coords = Function(vec_V0).project(x)

        # average these to find effective coords in CG1
        V0_coords_in_DG1 = Function(vec_DG1).interpolate(V0_coords)
        eff_coords_in_CG1 = Function(vec_CG1)
        eff_coords_averager = Averager(V0_coords_in_DG1, eff_coords_in_CG1)
        eff_coords_averager.project()

        # obtain these in DG1
        eff_coords_in_DG1 = Function(vec_DG1).interpolate(eff_coords_in_CG1)

        return correct_eff_coords(eff_coords_in_DG1)


def correct_eff_coords(eff_coords):
    """
    Correct the effective coordinates calculated by simply averaging
    which will not be correct at periodic boundaries.
    :arg eff_coords: the effective coordinates in vec_DG1 space.
    """

    mesh = eff_coords.function_space().mesh()
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    if vec_CG1.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_element = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_element = FiniteElement("DG", cell, 1, variant="equispaced")

    vec_DG1 = VectorFunctionSpace(mesh, DG1_element)

    x = SpatialCoordinate(mesh)

    if eff_coords.function_space() != vec_DG1:
        raise ValueError('eff_coords needs to be in the vector DG1 space')

    # obtain different coords in DG1
    DG1_coords = Function(vec_DG1).interpolate(x)
    CG1_coords_from_DG1 = Function(vec_CG1)
    averager = Averager(DG1_coords, CG1_coords_from_DG1)
    averager.project()
    DG1_coords_from_averaged_CG1 = Function(vec_DG1).interpolate(CG1_coords_from_DG1)
    DG1_coords_diff = Function(vec_DG1).interpolate(DG1_coords - DG1_coords_from_averaged_CG1)

    # interpolate coordinates, adjusting those different coordinates
    adjusted_coords = Function(vec_DG1)
    adjusted_coords.interpolate(eff_coords + DG1_coords_diff)

    return adjusted_coords


def find_domain_boundaries(mesh):
    """
    Makes a scalar DG0 function whose values are 0. everywhere except for in
    cells on the boundary of the domain, where the values are 1.0.
    This allows boundary cells to be identified easily.
    :arg mesh: the mesh.
    """

    DG0 = FunctionSpace(mesh, "DG", 0)
    CG1 = FunctionSpace(mesh, "CG", 1)

    on_exterior_DG0 = Function(DG0)
    on_exterior_CG1 = Function(CG1)

    # we get values in CG1 initially as DG0 will not work for triangular elements
    bc_codes = ['on_boundary', 'top', 'bottom']
    bcs = [DirichletBC(CG1, Constant(1.0), bc_code, method='geometric') for bc_code in bc_codes]

    for bc in bcs:
        try:
            bc.apply(on_exterior_CG1)
        except ValueError:
            pass

    on_exterior_DG0.interpolate(on_exterior_CG1)

    return on_exterior_DG0
