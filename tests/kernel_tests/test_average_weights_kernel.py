"""
A test of the AverageWeightings kernel used for the Averager.
"""

from firedrake import (IntervalMesh, Function, RectangleMesh,
                       VectorFunctionSpace, Mesh, as_vector,
                       FiniteElement, SpatialCoordinate)

from gusto import kernels
import pytest


@pytest.fixture
def mesh(geometry):

    # For 1D do a stretched mesh
    # For 2D just use the original mesh (but show how to stretch it)

    L = 3.0
    n = 3

    if geometry == "1D":
        regular_mesh = IntervalMesh(n, L)
    elif geometry == "2D":
        regular_mesh = RectangleMesh(n, n, L, L, quadrilateral=True)

    cell = regular_mesh.ufl_cell().cellname()
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(regular_mesh, DG1_elt)
    new_coords = Function(vec_DG1)

    if geometry == "1D":
        x, = SpatialCoordinate(regular_mesh)
        new_coords.interpolate(as_vector([x**2]))
    elif geometry == "2D":
        x, y = SpatialCoordinate(regular_mesh)
        new_coords.interpolate(as_vector([x, y]))

    new_mesh = Mesh(new_coords)

    return new_mesh


def setup_values(geometry, true_values):

    # The true values can be determined by the number of elements
    # that the DoF is shared between.

    if geometry == "1D":
        # The numbering of DoFs for DG1 in this mesh is
        #  |      |  DG1  |       |
        #  1-----0-3-----2-5------4

        # Cell lengths
        #  |   1  |   3   |   5   |

        # edges of the domain
        true_values.dat.data[1] = 1.0
        true_values.dat.data[4] = 1.0

        # internal values
        # these are vol_i * sum_i (1/vol_i)
        true_values.dat.data[0] = 1.0 * (1.0/1.0 + 1.0/3.0)
        true_values.dat.data[3] = 3.0 * (1.0/1.0 + 1.0/3.0)
        true_values.dat.data[2] = 3.0 * (1.0/3.0 + 1.0/5.0)
        true_values.dat.data[5] = 5.0 * (1.0/3.0 + 1.0/5.0)

    elif geometry == "2D":
        # The numbering of DoFs for DG1 near origin
        # Just focus on these 4 DoFs
        #   |     DG1
        #   |      |
        #   |------|--
        #   |1    3|
        #   |      |
        #   |0    2|
        #   ----------

        # List indices for corners
        corner_indices = [0]
        edge_indices = [1, 2]
        internal_indices = [3]

        for index in corner_indices:
            true_values.dat.data[index] = [1.0, 1.0]
        for index in edge_indices:
            true_values.dat.data[index] = [2.0, 2.0]
        for index in internal_indices:
            true_values.dat.data[index] = [4.0, 4.0]

    return true_values


@pytest.mark.parametrize("geometry", ["1D", "2D"])
def test_average(geometry, mesh):

    cell = mesh.ufl_cell().cellname()
    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)
    DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)

    weights = Function(vec_DG1)
    true_values = Function(vec_DG1)

    true_values = setup_values(geometry, true_values)

    kernel = kernels.AverageWeightings(vec_CG1, vec_DG1)
    kernel.apply(weights)

    tolerance = 1e-12
    if geometry == "1D":
        for i, (weight, true) in enumerate(zip(weights.dat.data[:], true_values.dat.data[:])):
            assert abs(weight - true) < tolerance, "Weight not correct at position %i" % i
    elif geometry == "2D":
        # Only look at DoFs for first cell
        for i in range(4):
            weight = weights.dat.data[i]
            true = true_values.dat.data[i]
            for weight_j, true_j in zip(weight, true):
                assert abs(weight_j - true_j) < tolerance, "Weight not correct at position %i" % i
