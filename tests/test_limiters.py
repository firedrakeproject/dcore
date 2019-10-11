from os import path
from gusto import *
from firedrake import (as_vector, PeriodicIntervalMesh, interval, FiniteElement, pi,
                       SpatialCoordinate, ExtrudedMesh, FunctionSpace, Function,
                       conditional, sqrt, BrokenElement, TensorProductElement)
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from netCDF4 import Dataset

# This setup creates a sharp bubble of warm air in a vertical slice
# This bubble is then advected by a prescribed advection scheme
# If the limiter is working, the advection should have produced
# no new maxima or minima. Advection is a solid body rotation.


def setup_limiters(dirname):

    dt = 0.01
    Ld = 1.
    tmax = 0.2
    rotations = 0.1
    m = PeriodicIntervalMesh(20, Ld)
    mesh = ExtrudedMesh(m, layers=20, layer_height=(Ld/20))
    output = OutputParameters(dirname=dirname, dumpfreq=1, dumplist=['u', 'chemical', 'moisture_higher', 'moisture_lower'])
    parameters = CompressibleParameters()
    timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)
    fieldlist = ['u', 'rho', 'theta', 'chemical', 'moisture_higher', 'moisture_lower']
    diagnostic_fields = []
    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    x, z = SpatialCoordinate(mesh)

    Vr = state.spaces("DG")
    Vt = state.spaces("HDiv_v")
    Vpsi = FunctionSpace(mesh, "CG", 2)

    cell = mesh._base_mesh.ufl_cell().cellname()
    DG0_element = FiniteElement("DG", cell, 0)
    CG1_element = FiniteElement("CG", interval, 1)
    Vt0_element = TensorProductElement(DG0_element, CG1_element)
    Vt0 = FunctionSpace(mesh, Vt0_element)
    Vt0_brok = FunctionSpace(mesh, BrokenElement(Vt0_element))
    VCG1 = FunctionSpace(mesh, "CG", 1)

    u = state.fields("u", dump=True)
    chemical = state.fields("chemical", Vr, dump=True)
    moisture_higher = state.fields("moisture_higher", Vt, dump=True)
    moisture_lower = state.fields("moisture_lower", Vt0, dump=True)

    x_lower = 2 * Ld / 5
    x_upper = 3 * Ld / 5
    z_lower = 6 * Ld / 10
    z_upper = 8 * Ld / 10
    bubble_expr_1 = conditional(x > x_lower,
                                conditional(x < x_upper,
                                            conditional(z > z_lower,
                                                        conditional(z < z_upper, 1.0, 0.0),
                                                        0.0),
                                            0.0),
                                0.0)

    bubble_expr_2 = conditional(x > z_lower,
                                conditional(x < z_upper,
                                            conditional(z > x_lower,
                                                        conditional(z < x_upper, 1.0, 0.0),
                                                        0.0),
                                            0.0),
                                0.0)

    chemical.assign(1.0)
    moisture_higher.assign(280.)
    chem_pert_1 = Function(Vr).interpolate(bubble_expr_1)
    chem_pert_2 = Function(Vr).interpolate(bubble_expr_2)
    moist_h_pert_1 = Function(Vt).interpolate(bubble_expr_1)
    moist_h_pert_2 = Function(Vt).interpolate(bubble_expr_2)
    moist_l_pert_1 = Function(Vt0).interpolate(bubble_expr_1)
    moist_l_pert_2 = Function(Vt0).interpolate(bubble_expr_2)

    chemical.assign(chemical + chem_pert_1 + chem_pert_2)
    moisture_higher.assign(moisture_higher + moist_h_pert_1 + moist_h_pert_2)
    moisture_lower.assign(moisture_lower + moist_l_pert_1 + moist_l_pert_2)

    # set up solid body rotation for advection
    # we do this slightly complicated stream function to make the velocity 0 at edges
    # thus we avoid odd effects at boundaries
    xc = Ld / 2
    zc = Ld / 2
    r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
    omega = rotations * 2 * pi / tmax
    r_out = 9 * Ld / 20
    r_in = 2 * Ld / 5
    A = omega * r_in / (2 * (r_in - r_out))
    B = - omega * r_in * r_out / (r_in - r_out)
    C = omega * r_in ** 2 * r_out / (r_in - r_out) / 2
    psi_expr = conditional(r < r_in,
                           omega * r ** 2 / 2,
                           conditional(r < r_out,
                                       A * r ** 2 + B * r + C,
                                       A * r_out ** 2 + B * r_out + C))
    psi = Function(Vpsi).interpolate(psi_expr)

    gradperp = lambda v: as_vector([-v.dx(1), v.dx(0)])
    u.project(gradperp(psi))

    state.initialise([('u', u),
                      ('chemical', chemical),
                      ('moisture_higher', moisture_higher),
                      ('moisture_lower', moisture_lower)])

    # set up advection schemes
    dg_opts = EmbeddedDGOptions()
    recovered_opts = RecoveredOptions(embedding_space=Vr,
                                      recovered_space=VCG1,
                                      broken_space=Vt0_brok,
                                      boundary_method=Boundary_Method.dynamics)

    chemeqn = AdvectionEquation(state, Vr, equation_form="advective")
    moisteqn_higher = EmbeddedDGAdvection(state, Vt, equation_form="advective", options=dg_opts)
    moisteqn_lower = EmbeddedDGAdvection(state, Vt0, equation_form="advective", options=recovered_opts)

    # build advection dictionary
    advected_fields = []
    advected_fields.append(('chemical', SSPRK3(state, chemical, chemeqn, limiter=VertexBasedLimiter(Vr))))
    advected_fields.append(('moisture_higher', SSPRK3(state, moisture_higher, moisteqn_higher, limiter=ThetaLimiter(Vt))))
    advected_fields.append(('moisture_lower', SSPRK3(state, moisture_lower, moisteqn_lower, limiter=VertexBasedLimiter(Vr))))

    # build time stepper
    stepper = AdvectionDiffusion(state, advected_fields)

    return stepper, tmax


def run_limiters(dirname):

    stepper, tmax = setup_limiters(dirname)
    stepper.run(t=0, tmax=tmax)
    # print(stepper.state.fields('moisture').dat.data[:])
    return


def test_limiters(tmpdir):

    dirname = str(tmpdir)
    run_limiters(dirname)
    filename = path.join(dirname, "diagnostics.nc")
    data = Dataset(filename, "r")

    chem_data = data.groups["chemical"]
    max_chem = chem_data.variables["max"]
    min_chem = chem_data.variables["min"]

    moist_h_data = data.groups["moisture_higher"]
    max_moist_h = moist_h_data.variables["max"]
    min_moist_h = moist_h_data.variables["min"]

    moist_l_data = data.groups["moisture_lower"]
    max_moist_l = moist_l_data.variables["max"]
    min_moist_l = moist_l_data.variables["min"]

    tolerance = 1e-8

    # check that maxima and minima do not exceed previous maxima and minima
    # however provide a small amount of tolerance
    assert max_chem[-1] <= max_chem[0] + (max_chem[0] - min_chem[0]) * tolerance
    assert min_chem[-1] >= min_chem[0] - (max_chem[0] - min_chem[0]) * tolerance
    assert max_moist_l[-1] <= max_moist_l[0] + (max_moist_l[0] - min_moist_l[0]) * tolerance
    assert min_moist_l[-1] >= min_moist_l[0] - (max_moist_l[0] - min_moist_l[0]) * tolerance
    assert max_moist_h[-1] <= max_moist_h[0] + (max_moist_h[0] - min_moist_h[0]) * tolerance
    assert min_moist_h[-1] >= min_moist_h[0] - (max_moist_h[0] - min_moist_h[0]) * tolerance
