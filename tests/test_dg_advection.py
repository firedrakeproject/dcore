from firedrake import norm, VectorFunctionSpace, as_vector
from gusto import *
import pytest


def run(state, advection_schemes, tmax, f_end):
    timestepper = PrescribedAdvection(state, advection_schemes)
    timestepper.run(0, tmax)
    return norm(state.fields("f") - f_end)


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
def test_dg_advection_scalar(tmpdir, geometry, equation_form, scheme,
                             tracer_setup):
    setup = tracer_setup(tmpdir, geometry)
    state = setup.state
    V = state.spaces("DG")
    f = state.fields("f", V)
    f.interpolate(setup.f_init)
    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f")
    else:
        eqn = ContinuityEquation(state, V, "f")
    if scheme == "ssprk":
        advection_schemes = [(eqn, SSPRK3(state))]
    elif scheme == "implicit_midpoint":
        advection_schemes = [(eqn, ImplicitMidpoint(state))]
    assert run(state, advection_schemes, setup.tmax, setup.f_end) < setup.tol


@pytest.mark.parametrize("geometry", ["slice", "sphere"])
@pytest.mark.parametrize("equation_form", ["advective", "continuity"])
@pytest.mark.parametrize("scheme", ["ssprk", "implicit_midpoint"])
def test_dg_advection_vector(tmpdir, geometry, equation_form, scheme,
                             tracer_setup):
    setup = tracer_setup(tmpdir, geometry)
    state = setup.state
    V = VectorFunctionSpace(state.mesh, "DG", 1)
    f = state.fields("f", V)
    gdim = state.mesh.geometric_dimension()
    f_init = as_vector((setup.f_init, *[0.]*(gdim-1)))
    f.interpolate(f_init)
    if equation_form == "advective":
        eqn = AdvectionEquation(state, V, "f")
    else:
        eqn = ContinuityEquation(state, V, "f")
    if scheme == "ssprk":
        advection_schemes = [(eqn, SSPRK3(state))]
    elif scheme == "implicit_midpoint":
        advection_schemes = [(eqn, ImplicitMidpoint(state))]
    f_end = as_vector((setup.f_end, *[0.]*(gdim-1)))
    assert run(state, advection_schemes, setup.tmax, f_end) < setup.tol