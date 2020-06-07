"""
Test whether the spherical polar recovery is working in on 2D spherical meshes.
To be working, a constant vector field should be exactly recovered.

This is tested for:
- the lowest-order density space recovered to DG1
"""

from firedrake import (IcosahedralSphereMesh, CubedSphereMesh, SpatialCoordinate,
                       FiniteElement, VectorFunctionSpace, as_vector, Function,
                       Constant, norm, errornorm)
from gusto import coord_transforms, Recoverer
import numpy as np
import pytest

@pytest.fixture
def mesh(geometry):

    radius = 7.

    if geometry == "CubedSphere":
        m = CubedSphereMesh(radius, refinement_level=0, degree=2)
    elif geometry == "Icosahedron":
        m = IcosahedralSphereMesh(radius, refinement_level=0, degree=2)

    return m

@pytest.mark.parametrize("geometry", ["Icosahedron", "CubedSphere"])
def test_sphere_recovery(geometry, mesh):

    x, y, z = SpatialCoordinate(mesh)

    # horizontal base spaces
    cell = mesh.ufl_cell().cellname()

    # DG1
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    Vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)

    # spaces
    Vec_DG0 = VectorFunctionSpace(mesh, "DG", 0)
    Vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # We want the field to have constant angular and radial components
    rlonlat_coordinates = coord_transforms.rlonlat_from_xyz(x, y, z)
    rlonlat_components = [Constant(3.0), Constant(11.0), Constant(-0.4)]
    xyz_components = coord_transforms.xyz_vector_from_rlonlat(rlonlat_components,
                                                              rlonlat_coordinates)

    # initialise Vec DG0 field and true Vec DG1 field
    initial_DG0_field = Function(Vec_DG0).interpolate(as_vector(xyz_components))
    true_CG1_field = Function(Vec_CG1).interpolate(as_vector(xyz_components))

    # make the initial fields by projecting expressions into the lowest order spaces
    final_CG1_field = Function(Vec_CG1)

    # make the recoverers and do the recovery
    recoverer = Recoverer(initial_DG0_field, final_CG1_field, VDG=Vec_DG1,
                          polar_transform=True)

    recoverer.project()

    diff = errornorm(final_CG1_field, true_CG1_field) / norm(true_CG1_field)

    tolerance = 1e-7
    error_message = ("""
                     Incorrect recovery on 2D sphere with {geometry} mesh
                     """)
    assert diff < tolerance, error_message.format(geometry=geometry)
