from gusto import *
from gusto import thermodynamics
from firedrake import (PeriodicIntervalMesh, ExtrudedMesh,
                       SpatialCoordinate, conditional, cos, pi, sqrt, NonlinearVariationalProblem,
                       NonlinearVariationalSolver, TestFunction, dx, TrialFunction, Constant, Function,
                       LinearVariationalProblem, LinearVariationalSolver, DirichletBC,
                       FunctionSpace, BrokenElement, VectorFunctionSpace)
import sys

if '--recovered' in sys.argv:
    recovered = True
else:
    recovered = False
if '--limit' in sys.argv:
    limit = True
else:
    limit = False
if '--diffusion' in sys.argv:
    diffusion = True
else:
    diffusion = False

dt = 1.0
if '--running-tests' in sys.argv:
    tmax = 10.
    deltax = 1000.
else:
    deltax = 100. if recovered else 200
    tmax = 1000.


L = 10000.
H = 10000.
nlayers = int(H/deltax)
ncolumns = int(L/deltax)

m = PeriodicIntervalMesh(ncolumns, L)
mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
degree = 0 if recovered else 1

fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt, maxk=4, maxi=1)

dirname = 'moist_bf'

if recovered:
    dirname += '_recovered'
if limit:
    dirname += '_limit'
if diffusion:
    dirname += '_diffusion'

output = OutputParameters(dirname=dirname,
                          dumpfreq=20,
                          dumplist=['u'],
                          perturbation_fields=[],
                          log_level='INFO')

params = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [Theta_e(), InternalEnergy(), Perturbation('InternalEnergy'), PotentialEnergy()]

state = State(mesh, vertical_degree=degree, horizontal_degree=degree,
              family="CG",
              timestepping=timestepping,
              output=output,
              parameters=params,
              diagnostics=diagnostics,
              fieldlist=fieldlist,
              diagnostic_fields=diagnostic_fields)

# Initial conditions
u0 = state.fields("u")
rho0 = state.fields("rho")
theta0 = state.fields("theta")
water_v0 = state.fields("water_v", theta0.function_space())
water_c0 = state.fields("water_c", theta0.function_space())
moisture = ["water_v", "water_c"]

# spaces
Vu = u0.function_space()
Vt = theta0.function_space()
Vr = rho0.function_space()
x, z = SpatialCoordinate(mesh)
quadrature_degree = (4, 4)
dxp = dx(degree=(quadrature_degree))

# Define constant theta_e and water_t
Tsurf = 320.0
total_water = 0.02
theta_e = Function(Vt).assign(Tsurf)
water_t = Function(Vt).assign(total_water)

# Calculate hydrostatic fields
saturated_hydrostatic_balance(state, theta_e, water_t)

# make mean fields
theta_b = Function(Vt).assign(theta0)
rho_b = Function(Vr).assign(rho0)
water_vb = Function(Vt).assign(water_v0)
water_cb = Function(Vt).assign(water_t - water_vb)
pibar = thermodynamics.pi(state.parameters, rho_b, theta_b)
Tb = thermodynamics.T(state.parameters, theta_b, pibar, r_v=water_vb)
Ibar = state.fields("InternalEnergybar", Vt, dump=False)
Ibar.interpolate(thermodynamics.internal_energy(state.parameters, rho_b, Tb, r_v=water_vb, r_l=water_cb))

# define perturbation
xc = L / 2
zc = 2000.
rc = 2000.
Tdash = 2.0
r = sqrt((x - xc) ** 2 + (z - zc) ** 2)
theta_pert = Function(Vt).interpolate(conditional(r > rc,
                                                  0.0,
                                                  Tdash * (cos(pi * r / (2.0 * rc))) ** 2))

# define initial theta
theta0.assign(theta_b * (theta_pert / 300.0 + 1.0))

# find perturbed rho
gamma = TestFunction(Vr)
rho_trial = TrialFunction(Vr)
a = gamma * rho_trial * dxp
L = gamma * (rho_b * theta_b / theta0) * dxp
rho_problem = LinearVariationalProblem(a, L, rho0)
rho_solver = LinearVariationalSolver(rho_problem)
rho_solver.solve()

physics_boundary_method = Boundary_Method.physics if recovered else None

# find perturbed water_v
w_v = Function(Vt)
phi = TestFunction(Vt)
rho_averaged = Function(Vt)
rho_recoverer = Recoverer(rho0, rho_averaged,
                          VDG=FunctionSpace(mesh, BrokenElement(Vt.ufl_element())),
                          boundary_method=physics_boundary_method)
rho_recoverer.project()

pi = thermodynamics.pi(state.parameters, rho_averaged, theta0)
p = thermodynamics.p(state.parameters, pi)
T = thermodynamics.T(state.parameters, theta0, pi, r_v=w_v)
w_sat = thermodynamics.r_sat(state.parameters, T, p)

w_functional = (phi * w_v * dxp - phi * w_sat * dxp)
w_problem = NonlinearVariationalProblem(w_functional, w_v)
w_solver = NonlinearVariationalSolver(w_problem)
w_solver.solve()

water_v0.assign(w_v)
water_c0.assign(water_t - water_v0)

# initialise fields
state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0),
                  ('water_v', water_v0),
                  ('water_c', water_c0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b),
                              ('water_v', water_vb)])


# set up limiter
if limit:
    if recovered:
        limiter = VertexBasedLimiter(VDG1)
    else:
        limiter = ThetaLimiter(Vt)
else:
    limiter = None


# Set up advection schemes
if recovered:
    VDG1 = state.spaces("DG1")
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
    Vu_DG1 = VectorFunctionSpace(mesh, VDG1.ufl_element())
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    u_opts = RecoveredOptions(embedding_space=Vu_DG1,
                              recovered_space=Vu_CG1,
                              broken_space=Vu,
                              boundary_method=Boundary_Method.dynamics)
    rho_opts = RecoveredOptions(embedding_space=VDG1,
                                recovered_space=VCG1,
                                broken_space=Vr,
                                boundary_method=Boundary_Method.dynamics)
    theta_opts = RecoveredOptions(embedding_space=VDG1,
                                  recovered_space=VCG1,
                                  broken_space=Vt_brok)

    ueqn = EmbeddedDGAdvection(state, Vu, equation_form="advective", options=u_opts)
    rhoeqn = EmbeddedDGAdvection(state, Vr, equation_form="continuity", options=rho_opts)
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", options=theta_opts)
else:
    ueqn = EulerPoincare(state, Vu)
    rhoeqn = AdvectionEquation(state, Vr, equation_form="continuity")
    thetaeqn = EmbeddedDGAdvection(state, Vt, equation_form="advective", options=EmbeddedDGOptions())

u_advection = ('u', SSPRK3(state, u0, ueqn)) if recovered else ('u', ThetaMethod(state, u0, ueqn))
euler_poincare = False if recovered else True

advected_fields = [u_advection,
                   ('rho', SSPRK3(state, rho0, rhoeqn)),
                   ('theta', SSPRK3(state, theta0, thetaeqn, limiter=limiter)),
                   ('water_v', SSPRK3(state, water_v0, thetaeqn, limiter=limiter)),
                   ('water_c', SSPRK3(state, water_c0, thetaeqn, limiter=limiter))]

# Set up linear solver
linear_solver = CompressibleSolver(state, moisture=moisture)

# Set up forcing
compressible_forcing = CompressibleForcing(state, moisture=moisture, euler_poincare=euler_poincare)

# diffusion
bcs = [DirichletBC(Vu, 0.0, "bottom"),
       DirichletBC(Vu, 0.0, "top")]

diffused_fields = []

if diffusion:
    diffused_fields.append(('u', InteriorPenalty(state, Vu, kappa=Constant(60.),
                                                 mu=Constant(10./deltax), bcs=bcs)))

# define condensation
physics_list = [Condensation(state)]

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing, physics_list=physics_list,
                        diffused_fields=diffused_fields)

stepper.run(t=0, tmax=tmax)
