"""
A test of the NodeMultiplicity kernel used for the Averager.
"""

from firedrake import (IntervalMesh, Function, RectangleMesh,
                       VectorFunctionSpace)

from gusto import kernels
import pytest


@pytest.fixture
def mesh(geometry):

    L = 3.0
    n = 3

    if geometry == "1D":
        m = IntervalMesh(n, L)
    elif geometry == "2D":
        m = RectangleMesh(n, n, L, L, quadrilateral=True)

    return m


def setup_values(geometry, true_values):

    # The true values can be determined by the number of elements
    # that the DoF is shared between.

    if geometry == "1D":
        # The numbering of DoFs for CG1 in this mesh is
        #  |      |  CG1 |       |
        #  0------1------2-------3

        edge_indices = [0, 3]
        internal_indices = [1, 2]

        for index in edge_indices:
            true_values.dat.data[index] = 1.0
        for index in internal_indices:
            true_values.dat.data[index] = 2.0

    elif geometry == "2D":
        # The numbering of DoFs for DG1 and CG1 near the origin in this mesh is
        #   |     CG1     |
        #   11-----12-----14-----15
        #   |      |      |      |
        #   |      |      |      |
        #   |      |      |      |
        #   6------7------10-----13
        #   |      |      |      |
        #   |      |      |      |
        #   |      |      |      |
        #   1------2------4------8
        #   |      |      |      |
        #   |      |      |      |
        #   |      |      |      |
        #   0------3------5------9

        # List indices for corners
        corner_indices = [0, 9, 11, 15]
        edge_indices = [1, 3, 5, 6, 8, 12, 13, 14]
        internal_indices = [2, 4, 7, 10]

        for index in corner_indices:
            true_values.dat.data[index] = [1.0, 1.0]
        for index in edge_indices:
            true_values.dat.data[index] = [2.0, 2.0]
        for index in internal_indices:
            true_values.dat.data[index] = [4.0, 4.0]

    return true_values


@pytest.mark.parametrize("geometry", ["1D", "2D"])
def test_average(geometry, mesh):

    vec_CG1 = VectorFunctionSpace(mesh, "CG", 1)

    # We will fill DG_field with values, and average them to CG_field
    weights = Function(vec_CG1)
    true_values = Function(vec_CG1)

    true_values = setup_values(geometry, true_values)

    kernel = kernels.NodeMultiplicity(vec_CG1)
    kernel.apply(weights)

    tolerance = 1e-12
    if geometry == "1D":
        for i, (weight, true) in enumerate(zip(weights.dat.data[:], true_values.dat.data[:])):
            assert abs(weight - true) < tolerance, "Weight not correct at position %i" % i
    elif geometry == "2D":
        for i, (weight, true) in enumerate(zip(weights.dat.data[:], true_values.dat.data[:])):
            for weight_j, true_j in zip(weight, true):
                assert abs(weight_j - true_j) < tolerance, "Weight not correct at position %i" % i
