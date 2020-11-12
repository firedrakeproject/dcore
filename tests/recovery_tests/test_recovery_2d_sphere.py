"""
Test whether the spherical polar recovery is working in on 2D spherical meshes.
To be working, a constant vector field should be exactly recovered.

This is tested for:
- vector DG0 recovered to vector DG1
- lowest-order velocity spaces recovered to vector DG1
"""

from firedrake import (IcosahedralSphereMesh, CubedSphereMesh, SpatialCoordinate,
                       FiniteElement, VectorFunctionSpace, as_vector, Function,
                       Constant, norm, errornorm, FunctionSpace, File)
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

    x = SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)

    # horizontal base spaces
    cell = mesh.ufl_cell().cellname()
    family = "BDM" if geometry == "Icosahedron" else "RTCF"

    # DG1
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    Vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)

    # spaces
    Vu = FunctionSpace(mesh, family, 1)
    Vec_DG0 = VectorFunctionSpace(mesh, "DG", 0)
    Vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # We want the field to have constant angular and radial components
    rlonlat_coordinates = coord_transforms.rlonlat_from_xyz(x[0], x[1], x[2])
    rlonlat_components = [Constant(0.0), Constant(11.0), Constant(-0.4)]
    xyz_components = coord_transforms.xyz_vector_from_rlonlat(rlonlat_components,
                                                              rlonlat_coordinates)

    # initialise Vec DG0 field and true Vec DG1 field
    initial_DG0_field = Function(Vec_DG0).interpolate(as_vector(xyz_components))
    initial_v_field = Function(Vu).project(as_vector(xyz_components))
    true_CG1_field = Function(Vec_CG1).interpolate(as_vector(xyz_components))

    # make the initial fields by projecting expressions into the lowest order spaces
    DG0_CG1_field = Function(Vec_CG1)
    v_CG1_field = Function(Vec_CG1)

    # make the recoverers and do the recovery
    DG0_recoverer = Recoverer(initial_DG0_field, DG0_CG1_field, VDG=Vec_DG1,
                             polar_transform=True)

    if geometry == "Icosahedron":
        v_CG1_field.interpolate(initial_v_field)

    else:
        v_recoverer = Recoverer(initial_v_field, v_CG1_field, VDG=Vec_DG1)
        v_recoverer.project()

    DG0_recoverer.project()


    DG0_diff = errornorm(DG0_CG1_field, true_CG1_field) / norm(true_CG1_field)
    v_diff = errornorm(v_CG1_field, true_CG1_field) / norm(true_CG1_field)

    print('DGO_diff', DG0_diff)

    tolerance = 1e-7
    error_message = ("""
                     Incorrect recovery for {variable} on 2D sphere with {geometry} mesh
                     """)
    assert DG0_diff < tolerance, error_message.format(variable='DG0', geometry=geometry)
    assert v_diff < tolerance, error_message.format(variable='v', geometry=geometry)
