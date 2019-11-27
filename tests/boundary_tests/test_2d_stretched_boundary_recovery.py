"""
This short program applies the boundary recoverer operation to check
the boundary values under some analytic forms.
"""
from gusto import *
from firedrake import (as_vector, IntervalMesh, SpatialCoordinate,
                       ExtrudedMesh, FunctionSpace, Function, errornorm,
                       VectorFunctionSpace, interval, TensorProductElement,
                       FiniteElement, HDiv, norm, BrokenElement, Mesh, File)
import numpy as np


def setup_2d_recovery(dirname):

    L = 5.
    H = 5.

    deltax = L / 5.
    deltay = H / 5.
    nlayers = int(H/deltay)
    ncolumns = int(L/deltax)

    m = IntervalMesh(ncolumns, L)
    ext_mesh = ExtrudedMesh(m, layers=nlayers, layer_height=H/nlayers)
    Vc = VectorFunctionSpace(ext_mesh, "DG", 2)
    x, z = SpatialCoordinate(ext_mesh)
    mesh_expr = as_vector([x, z*(1 + 0.1*z)])
    new_coords = Function(Vc).interpolate(mesh_expr)
    mesh = Mesh(new_coords)
    x, z = SpatialCoordinate(mesh)

    # horizontal base spaces
    cell = mesh._base_mesh.ufl_cell().cellname()
    u_hori = FiniteElement("CG", cell, 1)
    w_hori = FiniteElement("DG", cell, 0)

    # vertical base spaces
    u_vert = FiniteElement("DG", interval, 0)
    w_vert = FiniteElement("CG", interval, 1)

    # build elements
    u_element = HDiv(TensorProductElement(u_hori, u_vert))
    w_element = HDiv(TensorProductElement(w_hori, w_vert))
    theta_element = TensorProductElement(w_hori, w_vert)
    v_element = u_element + w_element

    # spaces
    VDG0 = FunctionSpace(mesh, "DG", 0)
    VCG1 = FunctionSpace(mesh, "CG", 1)
    VDG1 = FunctionSpace(mesh, "DG", 1)
    Vt = FunctionSpace(mesh, theta_element)
    Vt_brok = FunctionSpace(mesh, BrokenElement(theta_element))
    Vu = FunctionSpace(mesh, v_element)
    VuCG1 = VectorFunctionSpace(mesh, "CG", 1)
    VuDG1 = VectorFunctionSpace(mesh, "DG", 1)

    # set up initial conditions
    np.random.seed(0)
    expr = np.random.randn() + np.random.randn() * x + np.random.randn() * z + np.random.randn() * x * z

    # our actual theta and rho and v
    rho_CG1_true = Function(VCG1, name='true').interpolate(expr)
    theta_CG1_true = Function(VCG1).interpolate(expr)
    v_CG1_true = Function(VuCG1).interpolate(as_vector([expr, expr]))
    rho_Vt_true = Function(Vt).interpolate(expr)

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(VDG0).interpolate(expr)
    rho_CG1 = Function(VCG1, name='recovered')
    theta_Vt = Function(Vt).interpolate(expr)
    theta_CG1 = Function(VCG1)
    v_Vu = Function(Vu).project(as_vector([expr, expr]))
    v_CG1 = Function(VuCG1)
    rho_Vt = Function(Vt)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, VDG=VDG1, boundary_method=Boundary_Method.dynamics, weighting=True)
    theta_recoverer = Recoverer(theta_Vt, theta_CG1, VDG=VDG1, boundary_method=Boundary_Method.dynamics)
    v_recoverer = Recoverer(v_Vu, v_CG1, VDG=VuDG1, boundary_method=Boundary_Method.dynamics)
    rho_Vt_recoverer = Recoverer(rho_DG0, rho_Vt, VDG=Vt_brok, boundary_method=Boundary_Method.physics)

    rho_recoverer.project()
    theta_recoverer.project()
    v_recoverer.project()
    rho_Vt_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)
    theta_diff = errornorm(theta_CG1, theta_CG1_true) / norm(theta_CG1_true)
    v_diff = errornorm(v_CG1, v_CG1_true) / norm(v_CG1_true)
    rho_Vt_diff = errornorm(rho_Vt, rho_Vt_true) / norm(rho_Vt_true)

    output = File('results/recovery_output.pvd')
    output.write(rho_CG1, rho_CG1_true)

    return (rho_diff, theta_diff, v_diff, rho_Vt_diff)


def run_2d_recovery(dirname):

    (rho_diff, theta_diff, v_diff, rho_Vt_diff) = setup_2d_recovery(dirname)
    return (rho_diff, theta_diff, v_diff, rho_Vt_diff)


def test_2d_boundary_recovery(tmpdir):

    dirname = str(tmpdir)
    rho_diff, theta_diff, v_diff, rho_Vt_diff = run_2d_recovery(dirname)

    tolerance = 1e-7
    assert rho_diff < tolerance
    assert theta_diff < tolerance
    assert v_diff < tolerance
    assert rho_Vt_diff < tolerance
