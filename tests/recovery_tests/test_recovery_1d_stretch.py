"""
Test whether the boundary recovery is working in on a stretched 1D mesh.
To be working, a linearly varying field should be exactly recovered.

This is tested for:
- the lowest-order density space recovered to DG1
"""

from firedrake import (PeriodicIntervalMesh, IntervalMesh, VectorFunctionSpace,
                       SpatialCoordinate, FiniteElement, FunctionSpace,
                       Function, norm, errornorm, Mesh, as_vector)
from gusto import *
import numpy as np
import pytest

def test_1D_recovery():

    Lx = 100.
    deltax = Lx / 5.
    ncolumnsx = int(Lx/deltax)

    # Set up initial regular equispaced mesh
    regular_mesh = IntervalMesh(ncolumnsx, Lx)

    cell = regular_mesh.ufl_cell().cellname()
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")

    x_reg, = SpatialCoordinate(regular_mesh)
    vec_DG1_reg = VectorFunctionSpace(regular_mesh, DG1_elt)
    stretched_coords = Function(vec_DG1_reg).interpolate(as_vector([x_reg**2]))

    # make stretched mesh
    stretched_mesh = Mesh(stretched_coords)
    x, = SpatialCoordinate(stretched_mesh)

    expr = 1 + 2 * x

    # DG1
    DG1 = FunctionSpace(stretched_mesh, DG1_elt)

    # spaces
    DG0 = FunctionSpace(stretched_mesh, "DG", 0)
    CG1 = FunctionSpace(stretched_mesh, "CG", 1)

    # our actual theta and rho and v
    rho_CG1_true = Function(CG1).interpolate(expr)

    # make the initial fields by projecting expressions into the lowest order spaces
    rho_DG0 = Function(DG0).interpolate(expr)
    rho_CG1 = Function(CG1)

    # make the recoverers and do the recovery
    rho_recoverer = Recoverer(rho_DG0, rho_CG1, VDG=DG1, weighted=True,
                              boundary_method=Boundary_Method.dynamics)

    rho_recoverer.project()

    rho_diff = errornorm(rho_CG1, rho_CG1_true) / norm(rho_CG1_true)

    tolerance = 1e-7
    error_message = ("""
                     Incorrect recovery for {variable} with {boundary} boundary method
                     on stretched 1D domain
                     """)
    assert rho_diff < tolerance, error_message.format(variable='rho', boundary='dynamics')
