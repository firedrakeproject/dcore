from firedrake import as_vector
from gusto import *


def run(setup):

    state = setup.state
    tmax = setup.tmax
    f_init = setup.f_init
    V = state.spaces("DG", "DG", 1)

    equation = AdvectionDiffusionEquation(state, V, "f", ufamily=setup.family,
                                          udegree=setup.degree, kappa=1., mu=5)
    problem = [(equation, ((SSPRK3(state), advection),
                           (BackwardEuler(state), diffusion)))]
    state.fields("f").interpolate(f_init)
    state.fields("u").project(as_vector([10, 0.]))

    timestepper = PrescribedAdvection(state, problem)

    timestepper.run(0, tmax=tmax)


def test_advection_diffusion(tmpdir, tracer_setup):

    setup = tracer_setup(tmpdir, geometry="slice", blob=True)
    run(setup)