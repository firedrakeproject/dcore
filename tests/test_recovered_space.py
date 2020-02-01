from gusto import *
from firedrake import (as_vector, Constant, PeriodicIntervalMesh,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace,
                       Function, conditional, sqrt)

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then advected by a prescribed advection scheme


def setup_recovered_space(dirname):

    # declare grid shape, with length L and height H
    L = 400.
    H = 400.
    nlayers = int(H / 20.)
    ncolumns = int(L / 20.)

    # make mesh
    m = PeriodicIntervalMesh(ncolumns, L)
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=(H / nlayers))
    x, z = SpatialCoordinate(mesh)

    fieldlist = ['u', 'rho']
    timestepping = TimesteppingParameters(dt=1.0, maxk=4, maxi=1)
    output = OutputParameters(dirname=dirname+"/recovered_space_test",
                              dumpfreq=5,
                              dumplist=['u'])
    parameters = CompressibleParameters()

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist)

    # declare initial fields
    u0 = state.fields("u")

    # spaces
    Vpsi = FunctionSpace(mesh, "CG", 2)
    VDG0 = FunctionSpace(mesh, "DG", 0)
    VDG1 = state.spaces("DG1")
    VCG1 = FunctionSpace(mesh, "CG", 1)

    # set up tracer field
    tracer0 = state.fields("tracer", VDG0)

    # make a gradperp
    gradperp = lambda u: as_vector([-u.dx(1), u.dx(0)])

    # set up bubble
    xc = 200.
    zc = 200.
    rc = 100.
    tracer0.interpolate(conditional(sqrt((x - xc) ** 2.0) < rc,
                                    conditional(sqrt((z - zc) ** 2.0) < rc,
                                                Constant(0.2),
                                                Constant(0.0)), Constant(0.0)))

    # set up velocity field
    u_max = Constant(10.0)
    psi_expr = - u_max * z
    psi0 = Function(Vpsi).interpolate(psi_expr)
    u0.project(gradperp(psi0))

    state.initialise([('u', u0),
                      ('tracer', tracer0)])

    # set up advection schemes
    recovered_opts = RecoveredOptions(embedding_space=VDG1,
                                      recovered_space=VCG1,
                                      broken_space=VDG0,
                                      boundary_method=Boundary_Method.dynamics)
    tracereqn = EmbeddedDGAdvection(state, VDG0, equation_form="continuity",
                                    options=recovered_opts)

    # build advection dictionary
    advected_fields = []
    advected_fields.append(('tracer', SSPRK3(state, tracer0, tracereqn)))

    # build time stepper
    stepper = AdvectionDiffusion(state, advected_fields)

    return stepper, 100.0


def run_recovered_space(dirname):

    stepper, tmax = setup_recovered_space(dirname)
    stepper.run(t=0, tmax=tmax)


def test_recovered_space_setup(tmpdir):

    dirname = str(tmpdir)
    run_recovered_space(dirname)
