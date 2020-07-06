"""
Test whether the HDiv_Coords function works.

This is tested for:
- a 2D cartesian plane
"""

from firedrake import (PeriodicRectangleMesh, RectangleMesh, as_vector,
                       SpatialCoordinate, FiniteElement, FunctionSpace,
                       Function, norm, errornorm, BrokenElement)
from gusto import *
import numpy as np
import pytest

@pytest.fixture
def mesh(element):

    Lx = 100.
    Ly = 100.

    deltax = Lx / 3.
    deltay = Ly / 3.
    ncolumnsy = int(Ly/deltay)
    ncolumnsx = int(Lx/deltax)

    quadrilateral = True if element == "quadrilateral" else False

    m = RectangleMesh(ncolumnsx, ncolumnsy, Lx, Ly, quadrilateral=quadrilateral)

    return m


@pytest.mark.parametrize("element", ["quadrilateral", "triangular"])
def test_2D_cartesian_recovery(element, mesh):

    family = "RTCF" if element == "quadrilateral" else "BDM"

    cell = mesh.ufl_cell().cellname()
    elt = FiniteElement(family, cell, 1, variant="equispaced")
    V = FunctionSpace(mesh, elt)

    x,y = SpatialCoordinate(mesh)

    coord_list = HDiv_Coords(V)

    tol = 1e-8

    #   Coordinates of quadrilateral and triangle
    #    --(16.7,33.3)--            |      \
    #    |            |          (0,22.2)  (11.1,22.2)
    # (0,16.7)    (33.3,16.7)       |            \
    #    |            |          (0,11.1)         (22.2,11.1)
    #    ---(16.7,0)---             |                 \
    #                               |---(11.1,0)--(22.2,0)---\

    if element == "quadrilateral":
        true_coordinates = [[0.0, 100.0/6],
                            [100.0/3, 100.0/6],
                            [100.0/6, 0.0],
                            [100.0/6, 100.0/3]]
    else:
        true_coordinates = [[100.0/9, 200.0/9],
                            [200.0/9, 100.0/9],
                            [100.0/9, 0.0],
                            [200.0/9, 0.0],
                            [0.0, 100.0/9],
                            [0.0, 200.0/9]]


    # Check whether coordinates for first cell are correct
    for i, (true_coords, coord_x, coord_y) in enumerate(zip(true_coordinates,
                                                            coord_list[0].dat.data[:],
                                                            coord_list[1].dat.data[:])):
        assert abs(true_coords[0] - coord_x) < tol, ('x-coord of DoF %i not accurate, ' % i
                                                     +'got %.2f but expected %.2f'
                                                     % (coord_x, true_coords[0]))
        assert abs(true_coords[1] - coord_y) < tol, ('y-coord of DoF %i not accurate, ' % i
                                                     +'got %.2f but expected %.2f'
                                                     % (coord_y, true_coords[1]))
