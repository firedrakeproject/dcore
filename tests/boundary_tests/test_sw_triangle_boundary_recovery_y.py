"""
This short program applies the boundary recoverer operation to check
the boundary values under some analytic forms.
"""
from gusto import *
from firedrake import (as_vector, PeriodicRectangleMesh, SpatialCoordinate,
                       FunctionSpace, Function, errornorm,
                       VectorFunctionSpace, norm)
import numpy as np


def setup_2d_recovery(dirname):

    L = 100.
    W = 100.

    deltax = L / 5.
    deltay = W / 5.
    ncolumnsx = int(L/deltax)
    ncolumnsy = int(W/deltay)

    mesh = PeriodicRectangleMesh(ncolumnsx, ncolumnsy, L, W, direction='y', quadrilateral=False)
    x, y = SpatialCoordinate(mesh)

    # spaces
    VDG0 = FunctionSpace(mesh, "DG", 0)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    VDG1 = FunctionSpace(mesh, "DG", 1)
    Vu = FunctionSpace(mesh, "BDM", 1)
    VuCG1 = VectorFunctionSpace(mesh, "CG", 1)
    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)

    # set up initial conditions
    np.random.seed(0)
    expr = np.random.randn() + np.random.randn() * x

    # our actual theta and rho and v
    rho_CG1_true = Function(VCG1).interpolate(expr)
    v_CG1_true = Function(VuCG1).interpolate(as_vector([expr, expr]))

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(VDG0).interpolate(expr)
    rho_CG1 = Function(VCG1)
    v_Vu = Function(Vu).project(as_vector([expr, expr]))
    v_CG1 = Function(VuCG1)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, VDG=VDG1, boundary_method=Boundary_Method.dynamics)
    v_recoverer = Recoverer(v_Vu, v_CG1, VDG=VuDG1, boundary_method=Boundary_Method.dynamics)

    rho_recoverer.project()
    v_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)
    v_diff = errornorm(v_CG1, v_CG1_true) / norm(v_CG1_true)

    return (rho_diff, v_diff)


def run_2d_recovery(dirname):

    (rho_diff, v_diff) = setup_2d_recovery(dirname)
    return (rho_diff, v_diff)


def test_2d_boundary_recovery(tmpdir):

    dirname = str(tmpdir)
    rho_diff, v_diff = run_2d_recovery(dirname)

    tolerance = 1e-7
    assert rho_diff < tolerance
    assert v_diff < tolerance
