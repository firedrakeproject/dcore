"""
Test whether the polar recovery is working in on a circular 1D mesh.
To be working, a constant vector field should be exactly recovered.

This is tested for:
- the lowest-order density space recovered to DG1
"""

from firedrake import (CircleManifoldMesh, SpatialCoordinate, FiniteElement,
                       VectorFunctionSpace, as_vector, Function, Constant,
                       norm, errornorm)
from gusto import coord_transforms, Recoverer
import numpy as np
import pytest

def test_circle_recovery():

    mesh = CircleManifoldMesh(5, radius=7)

    x, y = SpatialCoordinate(mesh)

    # horizontal base spaces
    cell = mesh.ufl_cell().cellname()

    # DG1
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    Vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)

    # spaces
    Vec_DG0 = VectorFunctionSpace(mesh, "DG", 0)
    Vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)


    # We want the field to have constant angular and radial components
    polar_coordinates = coord_transforms.rphi_from_xy(x, y)
    polar_components = [Constant(3.0), Constant(11.0)]
    xy_components = coord_transforms.xy_vector_from_rphi(polar_components, polar_coordinates)

    # initialise Vec DG0 field and true Vec DG1 field
    initial_DG0_field = Function(Vec_DG0).interpolate(as_vector(xy_components))
    true_CG1_field = Function(Vec_CG1).interpolate(as_vector(xy_components))

    # make the initial fields by projecting expressions into the lowest order spaces
    final_CG1_field = Function(Vec_CG1)

    # make the recoverers and do the recovery
    recoverer = Recoverer(initial_DG0_field, final_CG1_field, VDG=Vec_DG1,
                          polar_transform=True)

    recoverer.project()

    diff = errornorm(final_CG1_field, true_CG1_field) / norm(true_CG1_field)

    tolerance = 1e-7
    error_message = ("""
                     Incorrect recovery for circular 1D domain
                     """)
    assert diff < tolerance, error_message
