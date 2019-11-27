from gusto import *
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, exp, sin, Function, as_vector)
import numpy as np
import itertools


def setup_sk(dirname):
    nlayers = 10  # horizontal layers
    columns = 30  # number of columns
    L = 1.e5
    m = PeriodicIntervalMesh(columns, L)
    dt = 6.0

    # build volume mesh
    H = 1.0e4  # Height position of the model top
    mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)

    # Set up points for output at the centre of the domain, edges and corners.
    # The point at x=L*(13.0/30) is in the halo region for a two-way MPI decomposition
    points_x = [0.0, L*(13.0/30), L/2.0, L]
    points_z = [0.0, H/2.0, H]
    points = np.array([p for p in itertools.product(points_x, points_z)])

    fieldlist = ['u', 'rho', 'theta']
    timestepping = TimesteppingParameters(dt=dt)
    output = OutputParameters(dirname=dirname+"/sk_nonlinear", dumplist=['u'], dumpfreq=5, log_level=INFO,
                              point_data=[('rho', points), ('u', points)])
    parameters = CompressibleParameters()
    diagnostic_fields = [CourantNumber()]

    state = State(mesh, vertical_degree=1, horizontal_degree=1,
                  family="CG",
                  timestepping=timestepping,
                  output=output,
                  parameters=parameters,
                  fieldlist=fieldlist,
                  diagnostic_fields=diagnostic_fields)

    # Initial conditions
    u0 = state.fields("u")
    rho0 = state.fields("rho")
    theta0 = state.fields("theta")

    # spaces
    Vu = u0.function_space()
    Vt = theta0.function_space()
    Vr = rho0.function_space()

    # Thermodynamic constants required for setting initial conditions
    # and reference profiles
    g = parameters.g
    N = parameters.N

    # N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
    x, z = SpatialCoordinate(mesh)
    Tsurf = 300.
    thetab = Tsurf*exp(N**2*z/g)

    theta_b = Function(Vt).interpolate(thetab)
    rho_b = Function(Vr)

    # Calculate hydrostatic Pi
    compressible_hydrostatic_balance(state, theta_b, rho_b)

    a = 5.0e3
    deltaTheta = 1.0e-2
    theta_pert = deltaTheta*sin(np.pi*z/H)/(1 + (x - L/2)**2/a**2)
    theta0.interpolate(theta_b + theta_pert)
    rho0.assign(rho_b)
    u0.project(as_vector([20.0, 0.0]))

    state.initialise([('u', u0),
                      ('rho', rho0),
                      ('theta', theta0)])
    state.set_reference_profiles([('rho', rho_b),
                                  ('theta', theta_b)])

    # Set up advection schemes
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = SUPGAdvection(state, Vt)
    advected_fields = []
    advected_fields.append(("u", ThetaMethod(state, u0, ueqn)))
    advected_fields.append(("rho", SSPRK3(state, rho0, rhoeqn)))
    advected_fields.append(("theta", SSPRK3(state, theta0, thetaeqn)))

    # Set up linear solver
    linear_solver = CompressibleSolver(state)

    # Set up forcing
    compressible_forcing = CompressibleForcing(state)

    # build time stepper
    stepper = CrankNicolson(state, advected_fields, linear_solver,
                            compressible_forcing)

    return stepper, 2*dt


def test_checkpointing(tmpdir):

    dirname = str(tmpdir)
    stepper, tmax = setup_sk(dirname)
    stepper.run(t=0., tmax=tmax)
    dt = stepper.state.timestepping.dt
    stepper.run(t=0, tmax=2*tmax+dt, pickup=True)
