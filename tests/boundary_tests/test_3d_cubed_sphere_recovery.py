"""
This short program applies the boundary recoverer operation to check
the boundary values under some analytic forms.
"""
from gusto import *
from firedrake import (as_vector, CubedSphereMesh, SpatialCoordinate,
                       FunctionSpace, Function, errornorm, FiniteElement,
                       VectorFunctionSpace, norm, ExtrudedMesh, BrokenElement,
                       interval, HDiv, TensorProductElement, sqrt, File)
import numpy as np


def setup_3d_spherical_recovery(dirname):

    radius = 1.

    m = CubedSphereMesh(radius, refinement_level=3)
    mesh = ExtrudedMesh(m, layers=3, layer_height=0.1, extrusion_type='radial')
    x = SpatialCoordinate(mesh)
    # mesh.init_cell_orientations(x)

    cell = mesh._base_mesh.ufl_cell().cellname()
    u_hori = FiniteElement("RTCF", cell, 1)
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
    expr = sqrt(x[0]**2 + x[1]**2 + x[2]**2)

    # our actual theta and rho and v
    rho_CG1_true = Function(VCG1, name='rho true').interpolate(expr)
    v_CG1_true = Function(VuCG1).interpolate(as_vector([-x[1], x[0], 0.0]))

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(VDG0, name='rho DG0').interpolate(expr)
    rho_CG1 = Function(VCG1, name='rho CG1')
    v_Vu = Function(Vu).project(as_vector([-x[1], x[0], 0.0]))
    v_CG1 = Function(VuCG1)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, VDG=VDG1, boundary_method=Boundary_Method.dynamics)
    v_recoverer = Recoverer(v_Vu, v_CG1, VDG=VuDG1, spherical_transformation=True, boundary_method=Boundary_Method.dynamics)

    rho_recoverer.project()
    output = File('3d_spherical_recovery.pvd')
    output.write(rho_CG1_true, rho_DG0, rho_CG1)
    v_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)
    v_diff = errornorm(v_CG1, v_CG1_true) / norm(v_CG1_true)

    return rho_diff, v_diff


def run_3d_spherical_recovery(dirname):

    rho_diff, v_diff = setup_3d_spherical_recovery(dirname)
    return rho_diff, v_diff


def test_3d_spherical_recovery(tmpdir):

    dirname = str(tmpdir)
    rho_diff, v_diff = run_3d_spherical_recovery(dirname)

    tolerance = 1e-2
    assert rho_diff < tolerance
    assert v_diff < tolerance
