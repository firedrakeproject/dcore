from gusto import *
from firedrake import (FunctionSpace, as_vector, VectorFunctionSpace,
                       PeriodicIntervalMesh, ExtrudedMesh, SpatialCoordinate,
                       exp, pi, cos, Function, conditional, Mesh, sin, op2,
                       BrokenElement, sqrt, DirichletBC, conditional)
import sys
import numpy as np

dt = 8.0
if '--running-tests' in sys.argv:
    tmax = dt
else:
    tmax = 5*60*60

if '--recovered' in sys.argv:
    recovered = True
else:
    recovered = False

smooth_z = False
through_mountain = False
dirname = 'schar_mountain_weighted'

degree = 0 if recovered else 1
L = 100000.
H = 30000.  # Height position of the model top
dx = 500 if recovered else 1000
dz = 500 if recovered else 1000
ncolumns = L/dx
nlayers = H/dz
m = PeriodicIntervalMesh(ncolumns, L)

# build volume mesh
ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
coord = SpatialCoordinate(ext_mesh)
x = Function(Vc).interpolate(as_vector([coord[0], coord[1]]))
a = 5000.
Lambda = 4000.
x, z = SpatialCoordinate(ext_mesh)
xc = L/2
hm = 250.
zs = hm*exp(-((x-xc)/a)**2)*(cos(pi*(x-xc)/Lambda))**2

if through_mountain:
    dirname += '_through_mountain'
if recovered:
    dirname += '_recovered'
if smooth_z:
    dirname += '_smootherz'
    zh = 5000.
    xexpr = as_vector([x, conditional(z < zh, z + cos(0.5*pi*z/zh)**6*zs, z)])
else:
    xexpr = as_vector([x, z + ((H-z)/H)*zs])

new_coords = Function(Vc).interpolate(xexpr)
mesh = Mesh(new_coords)

# sponge function
W_DG = FunctionSpace(mesh, "DG", 2)
x, z = SpatialCoordinate(mesh)
zc = H-10000.
mubar = 1.2/dt
mu_top = conditional(z <= zc, 0.0, mubar*sin((pi/2.)*(z-zc)/(H-zc))**2)
mu = Function(W_DG).interpolate(mu_top)
fieldlist = ['u', 'rho', 'theta']
timestepping = TimesteppingParameters(dt=dt)

output = OutputParameters(dirname=dirname,
                          dumpfreq=tmax/(1*dt),
                          dumplist=['u', 'theta'],
                          perturbation_fields=['theta', 'rho'],
                          log_level='INFO')

parameters = CompressibleParameters()
diagnostics = Diagnostics(*fieldlist)
diagnostic_fields = [VelocityZ(), ExnerPi()]

state = State(mesh, vertical_degree=degree, horizontal_degree=degree,
              family="CG",
              sponge_function=mu,
              timestepping=timestepping,
              output=output,
              parameters=parameters,
              diagnostics=diagnostics,
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
p_0 = parameters.p_0
c_p = parameters.cp
R_d = parameters.R_d
kappa = parameters.kappa

# N^2 = (g/theta)dtheta/dz => dtheta/dz = theta N^2g => theta=theta_0exp(N^2gz)
Tsurf = 288.
N = 0.01
thetab = Tsurf*exp(N**2*z/g)
theta_b = Function(Vt).interpolate(thetab)

# Calculate hydrostatic Pi
Pi = Function(Vr)
rho_b = Function(Vr)

piparams = {'ksp_type': 'gmres',
            'ksp_monitor_true_residual': None,
            'pc_type': 'python',
            'mat_type': 'matfree',
            'pc_python_type': 'gusto.VerticalHybridizationPC',
            # Vertical trace system is only coupled vertically in columns
            # block ILU is a direct solver!
            'vert_hybridization': {'ksp_type': 'preonly',
                                   'pc_type': 'bjacobi',
                                   'sub_pc_type': 'ilu'}}

compressible_hydrostatic_balance(state, theta_b, rho_b, Pi,
                                 top=True, pi_boundary=0.5,
                                 params=piparams)


def maximum(f):
    fmax = op2.Global(1, [-1], dtype=float)
    op2.par_loop(op2.Kernel("""
static void maxify(double *a, double *b) {
a[0] = a[0] < fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "maxify"), f.dof_dset.set, fmax(op2.MAX), f.dat(op2.READ))
    return fmax.data[0]

def minimum(f):
    fmin = op2.Global(1, [1000], dtype=float)
    op2.par_loop(op2.Kernel("""
static void minify(double *a, double *b) {
    a[0] = a[0] > fabs(b[0]) ? fabs(b[0]) : a[0];
}
""", "minify"), f.dof_dset.set, fmin(op2.MIN), f.dat(op2.READ))
    return fmin.data[0]


p0 = maximum(Pi)
compressible_hydrostatic_balance(state, theta_b, rho_b, Pi,
                                 top=True, params=piparams)
p1 = maximum(Pi)
alpha = 2.*(p1-p0)
beta = p1-alpha
pi_top = (1.-beta)/alpha
compressible_hydrostatic_balance(state, theta_b, rho_b, Pi,
                                 top=True, pi_boundary=pi_top, solve_for_rho=True,
                                 params=piparams)

theta0.assign(theta_b)
rho0.assign(rho_b)

if through_mountain:
    u0.project(as_vector([10, 0]))
else:
    M = hm * exp(-((x-xc)/a)**2) * (cos(pi*(x-xc)/Lambda))**2
    dM_dx = -2*hm * exp(-((x-xc)/a)**2)*cos(pi*(x-xc)/Lambda)*((x-xc)/a**2*cos(pi*(x-xc)/Lambda)+pi/Lambda*sin(pi*(x-xc)/Lambda))
    if smooth_z:
        dzs_dx = conditional(z < zh, (cos(pi*z/(2*zh)))**6*dM_dx, 0)
        dzs_dz = conditional(z < zh, 1 - 3*pi/(2*zh)*(cos(pi*z/(2*zh)))**5 * sin(pi*z/(2*zh))*M, 1)
    else:
        dzs_dx = (1-z/H)*dM_dx
        dzs_dz = 1 - M/H
    C = sqrt(dzs_dx**2 + dzs_dz**2)
    F = 1/C * dzs_dz
    G = 1/C * dzs_dx
    u0.project(as_vector([10*F, 10*G]))

state.initialise([('u', u0),
                  ('rho', rho0),
                  ('theta', theta0)])
state.set_reference_profiles([('rho', rho_b),
                              ('theta', theta_b)])

# Set up advection schemes
if recovered:
    VDG1 = FunctionSpace(mesh, "DG", 1)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    Vt_brok = FunctionSpace(mesh, BrokenElement(Vt.ufl_element()))
    Vu_DG1 = VectorFunctionSpace(mesh, "DG", 1)
    Vu_CG1 = VectorFunctionSpace(mesh, "CG", 1)
    Vu_brok = FunctionSpace(mesh, BrokenElement(Vu.ufl_element()))

    u_opts = RecoveredOptions(embedding_space=Vu_DG1,
                              recovered_space=Vu_CG1,
                              broken_space=Vu_brok,
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

advected_fields = [('rho', SSPRK3(state, rho0, rhoeqn)),
                   ('theta', SSPRK3(state, theta0, thetaeqn))]
if recovered:
    advected_fields.append(('u', SSPRK3(state, u0, ueqn)))
else:
    advected_fields.append(('u', ThetaMethod(state, u0, ueqn)))

if recovered:
    compressible_forcing = CompressibleForcing(state, euler_poincare=False)
else:
    compressible_forcing = CompressibleForcing(state)

# Set up linear solver
linear_solver = CompressibleSolver(state)

# build time stepper
stepper = CrankNicolson(state, advected_fields, linear_solver,
                        compressible_forcing)

stepper.run(t=0, tmax=tmax)
